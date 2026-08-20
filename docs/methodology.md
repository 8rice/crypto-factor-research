# Methodology

How a candidate signal gets from "interesting idea" to "something I would put
money behind", and, more often, how it gets killed. The order matters: each
stage is cheaper than the one after it, so anything that fails early saves the
expensive work.

## 1. The universe comes first

Almost every inflated crypto backtest I have seen traces back to the universe
rather than the signal.

**Survivorship.** A universe rebuilt today from currently-listed perpetuals has
silently deleted every asset that went to zero. Since most factors are long
winners and short losers, deleting the losers flatters the short leg and the
long leg at once. The fix is a life window per asset, first candle to last
traded candle, with delisted names kept in until they actually died
(`universe.life_windows`).

**Backfilled candles.** Exchange APIs will happily serve candles from before
the venue existed. If you trust listing dates naively, your early sample is
fiction. Every backtest is floored at a date the venue can actually support
(`min_start`).

**Liquidity, rolling.** Membership is a rolling top-N by dollar volume, not a
fixed list, so the universe moves as the market does.

The honest caveat this leaves: entering a liquidity-ranked universe is itself
correlated with recent momentum, so universe composition and a momentum signal
are not independent. Worth stating, not worth pretending away.

## 2. Timing convention, fixed once

    factor at close D  ->  position at close D  ->  return D to D+1

Written down once and enforced in the harness rather than re-derived per
analysis. This is where lookahead creeps in: a rolling statistic that is not
shifted, a regime variable read on the day it is used, a forward return aligned
one step off. Every rolling estimate in the leverage layer is shifted by a day
for the same reason.

## 3. Signal quality before portfolio construction

The information coefficient, daily cross-sectional rank correlation between
the factor and its forward return, measures the signal without any sizing
choice mixed in. A good Sharpe from a bad IC usually means the portfolio
construction is doing the work, which is worth knowing before you attribute
skill to the idea.

Reported as mean IC, annualised ICIR, and a t-stat. Daily IC near zero is not
automatically fatal: some genuine edges live entirely in the tails and show a
flat day-to-day IC with a strongly positive 7-day one.

## 4. Portfolio: quintiles, rank-weighted, inverse-vol

Two reference constructions, both gross 1:

- **LS**: rank-weighted long/short, weights = quantile − mean.
- **LSiv**: the same, times inverse 20-day volatility, legs normalised 50/50.

Inverse-vol sizing is not decoration. Without it, a handful of high-vol
small-caps dominate the risk and the "factor" becomes a bet on those names.

Two diagnostics matter more than the headline Sharpe:

- **Monotonicity** across quantiles (`binned_forward`). If only Q5 works, you
  have one bucket, not a factor, far more fragile out of sample.
- **Up/down beta** (`updown_beta`). A strongly negative down-beta with a flat
  up-beta means the strategy is short volatility: it sells crash insurance and
  collects a premium that looks like alpha until it doesn't.

## 5. Robustness before optimisation

A single tuned parameter pair proves nothing. The test is a grid
(`sharpe_grid`): are *all* the neighbours positive, and does performance vary
smoothly? A lone bright cell in a noisy grid is an overfit, no matter how good
that cell looks.

For the worked example in the README, 28 of 28 parameter pairs are positive
with a median Sharpe of 1.21. That flat plateau is the actual result, the peak
cell is not.

## 6. Costs, which is where most of it dies

Turnover is free in a research harness and expensive in production. Two numbers
close the loop:

- **Mean daily turnover**, as a fraction of gross.
- **Breakeven round-trip cost in bps**: the cost at which net return hits
  zero. Compare it to the spread plus fees you actually pay. A 1.5 Sharpe with
  a 2 bps breakeven is not a strategy, it is a measurement of the fee schedule.

One accounting subtlety worth being precise about: on perpetuals, funding
accrued by the book is a *component of the return*, not a fee. A long pays it,
a short earns it, exactly like a dividend. So it belongs in both gross and net,
leaving the gross-net gap as pure fee drag. Booking it as a cost double-counts
a carry edge.

## 7. Combination: correlation decides

A new factor earns its place by being *additive*, not by having a good
standalone Sharpe. Correlation of return streams is what decides whether it is
a genuinely different bet or an expensive re-parameterisation of one you
already trade, a factor correlated 0.85 to an existing one usually replaces
it rather than joining it.

Composites use gaussianised ranks so factors with different distributions are
comparable, and equal weighting as the default. Optimised weights over a
handful of correlated factors overfit readily; equal weighting is a
deliberately hard baseline.

## What is deliberately not here

Sizing and regime work built on funding, open interest and order-book data, and
the factor definitions themselves beyond the standard price/volume ones. The
methodology is the transferable part; the signals are not.
