"""Canonical universe: a rolling liquidity-ranked cross-section, built to be
survivorship-free.

Every factor analysis uses THIS universe so results are comparable. Two
decisions matter more than the rest:

1. Delisted names stay in. A universe rebuilt today from currently-listed
   perpetuals silently deletes every coin that died, and most factor edges in
   crypto look far better when the losers are gone. Each asset gets a life
   window, and it leaves the universe when it actually stopped trading.
2. Membership is a rolling top-N by dollar volume, not a fixed list. It moves
   with the market, which is realistic -- but note the caveat in the README:
   entering the universe is itself correlated with recent momentum.
"""

import pandas as pd

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE"}

TOP_N = 40
LIQ_WINDOW = 30


def life_windows(first_last, delisted=None, end_sentinel="2100-01-01"):
    """Build {asset: (first_date, last_date)} life windows.

    first_last : {asset: (first_candle, last_candle)} as observed in the data.
    delisted   : set of assets known to be delisted. A live asset gets an open
                 window (sentinel far future) so it never drops out by accident;
                 a delisted one closes at its last observed candle.

    Kept separate from any exchange API so the logic is testable and the data
    source is swappable.
    """
    delisted = delisted or set()
    out = {}
    for asset, (first, last) in first_last.items():
        if asset in STABLES:
            continue
        end = pd.Timestamp(last) if asset in delisted else pd.Timestamp(end_sentinel)
        out[asset] = (pd.Timestamp(first).normalize(), end)
    return out


def alive_mask(index, life):
    """Bool date x asset matrix: is the asset tradable on that date?"""
    return pd.DataFrame(
        {a: (index >= life[a][0]) & (index <= life[a][1]) for a in life},
        index=index)


def rolling_topn(dv_mat, alive=None, top_n=TOP_N, liq_window=LIQ_WINDOW):
    """Bool date x asset mask: top-N by mean dollar volume over `liq_window` days.

    Returns (mask, smoothed liquidity) -- the second is useful for diagnostics
    (universe turnover, how tight the N-th rank cut actually is).
    """
    liq = dv_mat.rolling(liq_window, min_periods=liq_window // 2).mean()
    if alive is not None:
        liq = liq.where(alive)
    return liq.rank(axis=1, ascending=False) <= top_n, liq


def build_universe(px_mat, dv_mat, life, top_n=TOP_N, liq_window=LIQ_WINDOW,
                   min_start=None):
    """Assemble the canonical universe from price/volume matrices + life windows.

    min_start : hard floor for the backtest start. Exchange APIs happily
    backfill candles from before the venue existed; without a floor the early
    sample is fiction. Returns a dict with px_mat, dv_mat, in_univ, liq, start.
    """
    alive = alive_mask(px_mat.index, life)
    in_univ, liq = rolling_topn(dv_mat, alive=alive, top_n=top_n, liq_window=liq_window)
    start = in_univ.sum(axis=1).ge(top_n).idxmax()
    if min_start is not None:
        start = max(start, pd.Timestamp(min_start))
    return dict(px_mat=px_mat, dv_mat=dv_mat, in_univ=in_univ, liq=liq,
                start=start, life=life)


def universe_stats(in_univ):
    """Diagnostics: members per day, entries/exits, membership turnover."""
    n = in_univ.sum(axis=1)
    entries = (in_univ & ~in_univ.shift(1).fillna(False)).sum(axis=1)
    exits = (~in_univ & in_univ.shift(1).fillna(False)).sum(axis=1)
    return pd.DataFrame({"n_members": n, "entries": entries, "exits": exits})
