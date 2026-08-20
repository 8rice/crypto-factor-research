"""Portfolio construction: combining factors, sizing, costs, leverage.

This is the layer where a signal has to become a tradable book. It is also
where most research-grade Sharpes die: turnover is free in an evaluation
harness and expensive in reality, so `gross_net` and `breakeven_bps` are the
functions that decide whether a factor is worth anything.

Same timing convention as `evaluate`: factor at close D -> position at close D
-> return D->D+1.
"""

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import evaluate

ANN = 365


def transform_scores(F, method="gauss"):
    """Cross-sectional score that is comparable across factors.

    - "gauss" : gaussianised rank, inverse-normal of ((r - 0.5) / n). Robust to
      outliers while keeping the extremes differentiated. Default.
    - "pct"   : raw percentile rank. Flattens the tails completely.
    - "z"     : z-score clipped at +/-3. Avoid when a factor has heavy tails --
      a handful of names then dominates the composite.
    """
    if method == "pct":
        return F.rank(axis=1, pct=True)
    if method == "gauss":
        r, n = F.rank(axis=1), F.notna().sum(axis=1)
        return pd.DataFrame(sps.norm.ppf(r.sub(0.5).div(n, axis=0)),
                            index=F.index, columns=F.columns)
    if method == "z":
        return F.sub(F.mean(axis=1), axis=0).div(F.std(axis=1), axis=0).clip(-3, 3)
    raise ValueError(method)


def composite_signal(factor_values, weights=None, transform="gauss",
                     require_all=True, smooth=1):
    """Composite = weighted mean of transformed scores.

    factor_values : list of date x asset DataFrames (already universe-masked).
    weights       : per-factor weights (default equal). Equal weighting is a
                    deliberately hard baseline to beat -- optimised weights on
                    a handful of correlated factors overfit readily.
    require_all   : True = require every signal on a given asset/date;
                    False = average whatever is available (wider cross-section,
                    but the composite's meaning drifts with coverage).
    smooth        : rolling mean of the composite, in days (1 = none). The
                    cheapest turnover reduction available.
    """
    scores = [transform_scores(F, transform) for F in factor_values]
    if weights is None:
        composite = pd.concat(scores).groupby(level=0).mean()
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        composite = sum(s * wi for s, wi in zip(scores, w))
    min_count = len(scores) if require_all else 1
    n_sig = pd.concat([s.notna().astype(int) for s in scores]).groupby(level=0).sum()
    composite = composite.where(n_sig >= min_count)
    if smooth > 1:
        composite = composite.rolling(smooth, min_periods=1).mean()
    return composite


def lsiv_weights(F, px_mat, q=evaluate.N_BINS, sizing_volw=evaluate.SIZING_VOLW):
    """Daily LSiv weights -- reproduces evaluate.quantile_returns exactly
    (rank-weighted quintiles x inverse-vol, legs 50/50, gross 1).

    Deliberately duplicated logic: the evaluation path returns returns, this one
    returns the book. Keeping them byte-for-byte consistent is what makes the
    backtest and the live target the same object.
    """
    vol_sizing = px_mat.pct_change(fill_method=None).rolling(sizing_volw).std()
    b = np.minimum((F.rank(axis=1, pct=True) * q).fillna(-1).astype(int), q - 1)
    wr = (b + 1.0 - (q + 1) / 2).where(b >= 0)
    wv = wr / vol_sizing
    pos, neg = wv.clip(lower=0), wv.clip(upper=0)
    return pos.div(pos.sum(axis=1), axis=0) * 0.5 + neg.div(neg.abs().sum(axis=1), axis=0) * 0.5


def turnover(w, start=None):
    """Mean daily turnover: sum |delta weight|, as a fraction of gross."""
    to = w.fillna(0).diff().abs().sum(axis=1)
    return (to.loc[start:] if start is not None else to).mean()


def gross_net(F, px_mat, fee_bps=5.0, start=None, funding=None):
    """Gross and net returns of the LSiv portfolio on signal F.

    Cost = sum |delta weight| x fee_bps, applied between consecutive targets
    (intraday drift is second order).

    `funding` (date x asset, daily summed rates): funding accrued by the book
    is a COMPONENT of the return, not a fee -- a long pays it, a short earns
    it, exactly like a dividend. So it enters BOTH gross and net, and the
    gross-net gap stays pure fee drag. Getting this wrong is a classic way to
    double-count a carry edge.

    Returns (gross, net, turnover) as daily series.
    """
    w = lsiv_weights(F, px_mat)
    r1 = px_mat.pct_change(fill_method=None).shift(-1)
    gross = (r1 * w).sum(axis=1)
    to = w.fillna(0).diff().abs().sum(axis=1)
    if funding is not None:
        gross = gross + _funding_accrual(w, funding)
    net = gross - to * fee_bps / 1e4
    if start is not None:
        gross, net, to = gross.loc[start:], net.loc[start:], to.loc[start:]
    return gross.dropna(), net.dropna(), to


def _funding_accrual(w, funding):
    """Daily funding PnL of book w: positions at close D x rate over D->D+1,
    matching the price return alignment. Long (w>0) pays a positive rate, short
    earns it -> pnl = -w * f. Assets with no funding data -> 0, not NaN, so a
    missing venue does not void the whole net series."""
    f = (funding.reindex(index=w.index, columns=w.columns)
         .shift(-1).fillna(0.0))
    return -(w * f).sum(axis=1)


def breakeven_bps(F, px_mat, start=None, funding=None):
    """Round-trip cost (bps) at which the strategy's net return hits zero.

    The single most useful number for a high-turnover signal: compare it to the
    spread + fees you actually pay. A 1.5 Sharpe with a 2 bps breakeven is not
    a strategy, it is a measurement of the fee schedule.
    """
    gross, _, to = gross_net(F, px_mat, fee_bps=0.0, start=start, funding=funding)
    to = to.reindex(gross.index).dropna()
    gross = gross.reindex(to.index)
    denom = to.mean()
    return np.nan if denom == 0 else gross.mean() / denom * 1e4


def apply_leverage(r, mode="static", target_vol=0.20, cap=3.0,
                   vol_window=60, kelly_window=180, kelly_fraction=1.0,
                   overlay=None):
    """Leverage policy applied to a gross-1 return stream.

    No lookahead: every rolling estimate is shifted by one day.
    Returns (levered returns, leverage series).

    - "static"     : constant leverage target_vol / full-sample vol. Note the
      full-sample vol is only known ex post -- fine for comparing, not a live
      policy.
    - "vol_target" : leverage = target_vol / rolling `vol_window` vol, capped.
      Usually the winner, because volatility is far more forecastable than
      returns.
    - "kelly"      : f* = mu/sigma^2 over `kelly_window`, x kelly_fraction,
      clipped to [0, cap]. In practice it saturates the cap for a decent-Sharpe
      strategy and adds noise from estimating mu.

    overlay : Series of multipliers composed ON TOP of the chosen mode -- the
    mode sets the risk scale, the overlay modulates exposure by context. Dates
    missing from the overlay default to 1.
    """
    r = r.dropna()
    if mode == "static":
        lev = pd.Series(target_vol / (r.std() * np.sqrt(ANN)), index=r.index)
    elif mode == "vol_target":
        vol = r.rolling(vol_window).std().shift(1) * np.sqrt(ANN)
        lev = (target_vol / vol).clip(upper=cap)
    elif mode == "kelly":
        mu = r.rolling(kelly_window).mean().shift(1)
        var = r.rolling(kelly_window).var().shift(1)
        lev = (mu / var).clip(lower=0, upper=cap) * kelly_fraction
    else:
        raise ValueError(mode)
    if overlay is not None:
        lev = lev * overlay.reindex(r.index).fillna(1.0)
    return (lev * r).dropna(), lev


def correlation_matrix(streams):
    """Correlation of factor return streams -- {name: Series} -> DataFrame.

    The number that decides whether a new factor is additive or is just an
    expensive re-parameterisation of one you already trade.
    """
    return pd.DataFrame(streams).dropna(how="all").corr()
