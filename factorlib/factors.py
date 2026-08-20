"""Factor definitions.

Only price/volume factors live in this public repository. The signature is
always `f(px_mat, **params) -> DataFrame` (date x asset), so any factor drops
straight into `evaluate.quantile_returns` and `evaluate.sharpe_grid`.

Scope note: the private research lab this harness comes from also covers
positioning, flow and liquidity families built on funding, open interest and
order-book data. Those definitions are not published. What is here are standard
price/volume constructions, enough to exercise the whole pipeline end to end.
"""

import numpy as np


def trend_sharpe(px_mat, span=40, volw=20):
    """Distance from an EMA trend, expressed in units of the asset's own vol.

        factor = log(px / EMA_span(px)) / sigma_volw(daily log returns)

    A "trend Sharpe". The intuition is cross-sectional: an asset trading well
    above its own trend RELATIVE TO ITS USUAL NOISE tends to keep outperforming
    its peers over a short horizon.

    The volatility normalisation is doing real work and is not cosmetic. It
    makes assets comparable (a 10% move means something very different for BTC
    and for a small-cap), and it equalises risk contribution across the
    cross-section. Empirically it improves the Sharpe at essentially every span
    see the parameter grid in the README.

    span : EMA span in days. volw : vol window in days (0 or None = raw,
    un-normalised).
    """
    logp = np.log(px_mat)
    dist = logp - np.log(px_mat.ewm(span=span, min_periods=span // 2).mean())
    if not volw:
        return dist
    vol = logp.diff().rolling(volw, min_periods=volw // 2).std()
    return dist / vol.replace(0, np.nan)


def momentum(px_mat, lookback=30, skip=0):
    """Plain cross-sectional momentum: return over `lookback` days, optionally
    skipping the most recent `skip` days to sidestep short-term reversal.

    The baseline any fancier trend factor has to beat.
    """
    p = px_mat.shift(skip)
    return p / p.shift(lookback) - 1


def realised_vol(px_mat, window=20, annualise=True):
    """Rolling realised volatility of daily log returns. Useful both as a
    standalone (low-vol) factor and as a sizing input."""
    v = np.log(px_mat).diff().rolling(window, min_periods=window // 2).std()
    return v * np.sqrt(365) if annualise else v


def liquidity_rank(dv_mat, window=30):
    """Cross-sectional percentile of smoothed dollar volume -- a size/liquidity
    control. Most crypto factors carry a large implicit size tilt, and checking
    a signal against this one is how you find out whether you have discovered
    anything beyond "small caps move more"."""
    return dv_mat.rolling(window, min_periods=window // 2).mean().rank(axis=1, pct=True)


REGISTRY = {
    "trend_sharpe": trend_sharpe,
    "momentum": momentum,
    "realised_vol": realised_vol,
}
