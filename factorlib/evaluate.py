"""Standard evaluation harness, identical for every factor.

The point of a shared harness is that factor comparisons mean something. Every
factor in this library is scored on the same universe, with the same timing
convention and the same portfolio construction, so a Sharpe difference reflects
the signal rather than a difference in how it was measured.

Canonical conventions:
- factor at close D -> position at close D -> return D->D+1 (daily rebalance)
- cross-sectional quintiles within that day's universe
- LS   : rank-weighted long/short (weights = quantile - mean, gross 1)
- LSiv : LS x inverse-volatility, long and short legs normalised 50/50
- costs and funding are excluded here on purpose: this layer ranks signals,
  the portfolio layer is where they have to survive fees
"""

import numpy as np
import pandas as pd

N_BINS = 5
SIZING_VOLW = 20

# Periods per year of the return stream being evaluated. Daily = 365 (the
# canonical default; crypto trades every day). A stream at X steps/day -> 365*X.
PERIODS_PER_YEAR = 365


def forward_returns(px_mat, days=(1, 7)):
    """Forward returns per horizon (in INDEX STEPS, not necessarily days)."""
    return {h: px_mat.shift(-h) / px_mat - 1 for h in days}


def quantile_returns(F, px_mat, q=N_BINS, sizing_volw=SIZING_VOLW, fwd_step=1,
                     min_coins=None):
    """Per-step returns: Q1..Qq (equal-weight), LS (rank), LSiv (rank x inv-vol).

    fwd_step : forward return horizon IN INDEX STEPS (1 = next step for daily
    data, 24 = +24h on an hourly index).
    F must already be masked by the universe (NaN outside it).
    min_coins : minimum cross-section width (default 2*q). A day with fewer
    scored names returns NaN everywhere -- a long/short over 2-6 coins is no
    longer the factor, it is noise, and it silently inflates backtest stats.
    """
    ret = px_mat.pct_change(fill_method=None)
    r1 = px_mat.shift(-fwd_step) / px_mat - 1 if fwd_step != 1 else ret.shift(-1)
    vol_sizing = ret.rolling(sizing_volw).std()
    b = np.minimum((F.rank(axis=1, pct=True) * q).fillna(-1).astype(int), q - 1)
    out = {}
    for k in range(q):
        w = (b == k).astype(float)
        out[f"Q{k+1}"] = (r1 * w.div(w.sum(axis=1), axis=0)).sum(axis=1)
    wr = (b + 1.0 - (q + 1) / 2).where(b >= 0)
    out["LS"] = (r1 * wr.div(wr.abs().sum(axis=1), axis=0)).sum(axis=1)
    wv = wr / vol_sizing
    pos, neg = wv.clip(lower=0), wv.clip(upper=0)
    wiv = (pos.div(pos.sum(axis=1), axis=0) * 0.5
           + neg.div(neg.abs().sum(axis=1), axis=0) * 0.5)
    out["LSiv"] = (r1 * wiv).sum(axis=1)
    qr = pd.DataFrame(out)
    n_ok = F.notna().sum(axis=1).reindex(qr.index).fillna(0)
    return qr.where(n_ok >= (2 * q if min_coins is None else min_coins))


def perf_stats(r, periods_per_year=PERIODS_PER_YEAR):
    """ann %, cagr %, vol %, sharpe, maxDD %, hit % over non-NaN steps."""
    r = r.dropna()
    if len(r) == 0 or r.std() == 0:
        return dict(ann=np.nan, cagr=np.nan, vol=np.nan, sharpe=np.nan,
                    maxdd=np.nan, hit=np.nan)
    eq = (1 + r).cumprod()
    # CAGR = the compounded annual rate that reproduces the final equity.
    cagr = (eq.iloc[-1] ** (periods_per_year / len(r)) - 1) * 100 if eq.iloc[-1] > 0 else np.nan
    return dict(
        ann=r.mean() * periods_per_year * 100,
        cagr=cagr,
        vol=r.std() * np.sqrt(periods_per_year) * 100,
        sharpe=r.mean() / r.std() * np.sqrt(periods_per_year),
        maxdd=(eq / eq.cummax() - 1).min() * 100,
        hit=(r > 0).mean() * 100)


def yearly_sharpe(r, periods_per_year=PERIODS_PER_YEAR):
    """Sharpe per calendar year -- the first thing to check for a decaying edge."""
    r = r.dropna()
    return r.groupby(r.index.year).apply(
        lambda x: x.mean() / x.std() * np.sqrt(periods_per_year) if x.std() > 0 else np.nan)


def rolling_sharpe(r, window=90, periods_per_year=PERIODS_PER_YEAR):
    """Annualised rolling Sharpe over `window` steps."""
    mu, sd = r.rolling(window).mean(), r.rolling(window).std()
    return (mu / sd * np.sqrt(periods_per_year)).where(sd > 0)


def conditional_perf(r, cond, n_bins=3, lag=0, labels=None,
                     periods_per_year=PERIODS_PER_YEAR):
    """Performance of r bucketed by quantile of a regime variable `cond`.

    Convention: r is indexed at the DECISION date (position at close D, return
    D->D+1), so `cond` at close D is observable -> lag=0 is not lookahead.
    lag>0 shifts `cond` by extra steps (more conservative).
    Returns (DataFrame bins x stats, t-stat of the top-minus-bottom mean).
    """
    both = pd.concat({"r": r, "c": cond.shift(lag)}, axis=1).dropna()
    b = pd.qcut(both["c"], n_bins, duplicates="drop")
    cats = b.cat.categories
    names = labels if labels and len(labels) == len(cats) \
        else [f"T{i+1}" for i in range(len(cats))]
    b = b.cat.rename_categories(names)
    rows = {}
    for k, grp in both.groupby(b, observed=True)["r"]:
        st = perf_stats(grp, periods_per_year)
        rows[k] = dict(n=len(grp), ann=st["ann"], vol=st["vol"],
                       sharpe=st["sharpe"], hit=st["hit"])
    lo, hi = both["r"][b == names[0]], both["r"][b == names[-1]]
    se = np.sqrt(lo.var() / len(lo) + hi.var() / len(hi))
    t = (hi.mean() - lo.mean()) / se if se > 0 else np.nan
    return pd.DataFrame(rows).T, t


def updown_beta(r, mkt, periods_per_year=PERIODS_PER_YEAR):
    """Regress r = alpha + beta_up * mkt+ + beta_dn * mkt- (beta asymmetry).

    r and mkt must share the same holding step and indexing. A strongly
    negative beta_dn with beta_up ~ 0 flags a short-volatility profile: the
    strategy is quietly selling crash insurance, which a plain Sharpe hides.
    """
    df = pd.concat({"r": r, "m": mkt}, axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["m"].clip(lower=0),
                         df["m"].clip(upper=0)])
    coef, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ coef
    cov = resid.var(ddof=3) * np.linalg.inv(X.T @ X)
    t = coef / np.sqrt(np.diag(cov))
    return dict(alpha_ann=coef[0] * periods_per_year * 100, t_alpha=t[0],
                beta_up=coef[1], t_up=t[1], beta_dn=coef[2], t_dn=t[2],
                n=len(df))


def top_drawdowns(r, n=5):
    """The n worst drawdown episodes: depth, peak/trough/recovery dates."""
    eq = (1 + r.dropna()).cumprod()
    dd = eq / eq.cummax() - 1
    under = dd < 0
    ep = (under != under.shift()).cumsum()
    rows = []
    for _, seg in dd[under].groupby(ep[under]):
        i0, i1 = dd.index.get_loc(seg.index[0]), dd.index.get_loc(seg.index[-1])
        rows.append(dict(
            depth=round(seg.min() * 100, 1),
            peak=dd.index[max(i0 - 1, 0)].strftime("%Y-%m-%d"),
            trough=seg.idxmin().strftime("%Y-%m-%d"),
            recovery=(dd.index[i1 + 1].strftime("%Y-%m-%d")
                      if i1 + 1 < len(dd.index) else "ongoing"),
            days=len(seg)))
    return pd.DataFrame(rows).nsmallest(n, "depth").reset_index(drop=True)


def ic_series(F, fwd_h, min_names=20):
    """Daily rank IC (Spearman): factor -> forward return, cross-sectionally."""
    both = F.notna() & fwd_h.notna()
    dates = F.index[both.sum(axis=1) >= min_names]
    return pd.Series(
        {d: F.loc[d, both.loc[d]].rank().corr(fwd_h.loc[d, both.loc[d]].rank()) for d in dates}
    ).dropna()


def ic_stats(ic, periods_per_year=PERIODS_PER_YEAR):
    """IC mean, IR (mean/std) and its t-stat -- signal quality, sizing-free."""
    ic = ic.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return dict(ic=np.nan, icir=np.nan, t=np.nan, n=len(ic))
    icir = ic.mean() / ic.std()
    return dict(ic=ic.mean(), icir=icir * np.sqrt(periods_per_year),
                t=icir * np.sqrt(len(ic)), n=len(ic))


def binned_forward(F, fwd_h, n_bins=10, demean=True):
    """Mean forward return (%) per cross-sectional percentile bin (pooled).

    Monotonicity across bins is the honest test of a factor: a good Sharpe can
    come from one extreme bucket alone, which is far more fragile.
    """
    fw = fwd_h.where(F.notna())
    if demean:
        fw = fw.sub(fw.mean(axis=1), axis=0)
    b = np.minimum((F.rank(axis=1, pct=True) * n_bins).fillna(-1).astype(int), n_bins - 1)
    stacked = pd.DataFrame({"b": b.stack(), "r": fw.stack()}).query("b >= 0").dropna()
    return stacked.groupby("b")["r"].mean() * 100


def sharpe_grid(px_mat, in_univ, factor_fn, spans, volws, start, leg="LSiv", **qr_kwargs):
    """Sharpe of the L/S leg for every (span x volw) -- the robustness test.

    A single tuned parameter pair proves nothing. What matters is whether the
    whole neighbourhood is positive and varies smoothly; a lone bright cell in
    a noisy grid is an overfit.
    """
    grid = pd.DataFrame(index=[f"vol{v}" if v else "raw" for v in volws],
                        columns=spans, dtype=float)
    for volw in volws:
        for span in spans:
            F = factor_fn(px_mat, span=span, volw=volw).where(in_univ).loc[start:]
            r = quantile_returns(F, px_mat, **qr_kwargs)[leg].loc[start:]
            grid.loc[f"vol{volw}" if volw else "raw", span] = perf_stats(r)["sharpe"]
    return grid
