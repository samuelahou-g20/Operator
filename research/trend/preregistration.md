# Pre-Registration — Group I (Trend / Momentum)

**Registered:** 2026-06-22 · **Author:** quant-researcher · **Status:** committed before predictive testing

This document fixes the hypotheses, the search space, the evaluation design, and the
go/no-go thresholds **before** any out-of-sample / predictive result is examined. It is the
cheapest, most effective defence against the garden-of-forking-paths problem (brief §4.6,
§5). Deviations from this document must be logged with a reason in the research report; an
unlogged deviation invalidates the deflation statistics.

## 0. The firewall (brief §0.1, §4.1)

Every analysis is tagged **descriptive** (does the feature *measure* a regime — offline,
look-ahead labels, no tradeability claim) or **predictive** (does the strictly-past feature
forecast forward returns — the tradeable claim). A descriptive result is never cited as
evidence of edge. The bridge analysis (does `F_t` lead `R_{t+h}`?) is tagged separately.

## 1. Scope

- **Instruments:** CME futures continuous front strips `es1`, `nq1` (ratio-adjusted close);
  Binance perps `btcusdt`, `ethusdt` (perps bucketed separately; returns include funding).
- **Period:** full backfilled history once the delegated backfill lands (CME bars ~7yr;
  crypto ~2019→now). Reserve the most recent ~20% as a final hold-out, untouched until the
  go/no-go.
- **Frequencies:** daily is primary. Intraday (5m/15m/1h) is secondary and is **conditioned
  on session bucket** — never pooled across time-of-day (brief §4.9).

## 2. Signal families and the FIXED ladder (trial count declared in advance)

Families (from `src.signals.trend`): EWMAC (Baz-normalized + response), Donchian/Keltner/
Carver channels, TSMOM (continuous and discrete), regression-slope t-stat.

The geometric EWMAC ladder is **fixed** at `(2,8),(4,16),(8,32),(16,64),(32,128),(64,256)`
(brief §1.3). TSMOM lookbacks fixed at `{21, 63, 126, 252}` (daily). Carver lookbacks
`{32, 64, 128}`. Donchian `{20, 55}`. Slope windows `{21, 63, 126}`. No grid expansion after
seeing results.

**Declared trial count** for the Deflated Sharpe / PBO (brief §4.6):
`N_trials = (number of distinct signal configurations above) × (number of instruments)`.
This number is recorded in the report and fed to `eval.deflation` — not reverse-engineered.

## 3. Normalization (fixed)

- EWMAC: `vol_window=63`, `scale_window=252`, Baz response **on**.
- TSMOM: `vol_span=32`.
- Cross-instrument pooling: vol-normalize returns first (brief §4.9); high-vol instruments
  must not dominate any pooled statistic.

## 4. Forward-return horizons (decay curve computed ONCE)

Daily horizons `h ∈ {1, 2, 3, 5, 10, 21, 42, 63}`. The full IC-decay curve is computed once
per signal (`eval.ic.ic_decay`); we do **not** cherry-pick the best `h`. The headline horizon
for the CV/deflation pass is pre-committed to `h = 21` (≈1 month) for daily.

## 5. Regime gate (conditional analysis)

Efficiency ratio (`regimes.efficiency_ratio`, window 64) with trend threshold `0.3`; the
predictive pass is reported both unconditional and conditional on the trending state. The
variance-ratio / Hurst diagnostics are reported as descriptive context.

## 6. Evaluation design (fixed)

- **CV:** purged k-fold, `k = 6`, `label_horizon = h`, `embargo = h` (brief §4.3,
  `eval.cv.purged_kfold_indices`).
- **Metric:** rank IC, IC-IR over monthly sub-periods, IC-decay curve (`eval.ic`).
- **Inference:** Newey–West HAC t-stat (auto lag) on mean IC / mean strategy return, and a
  stationary block bootstrap (mean block ≈ `h`) for CIs (`eval.inference`). Report the
  **effective sample size**, not nominal N.
- **Reduction:** within-family PCA / hierarchical clustering for description; PLS for
  prediction (`eval.reduce`); fit on the training fold, applied OOS only.

## 7. Go / no-go thresholds (committed)

A signal is promoted to "validated" only if **all** hold:
1. HAC t-stat of mean IC clears the **Harvey–Liu–Zhu `t > 3`** bar (brief §4.6) — not 2.
2. **Deflated Sharpe > 0.95** at the declared trial count (`eval.deflation`).
3. **PBO < 0.5** via CSCV (`eval.deflation.pbo_cscv`).
4. Parameter surface is a **plateau, not a spike** — IC stable under ±1 ladder step.
5. **Sign stability**: IC sign does not flip across time subsamples / instrument buckets.
6. **Net-of-cost** IC/Sharpe remains positive after the cost assumptions below.

Failing any one ⇒ "investigate further" or "reject"; we do not relax thresholds post hoc.

## 8. Cost assumptions (for economic significance, brief §4.5)

Round-trip cost (commission + half-spread + slippage), pre-committed per bucket:
`es1`/`nq1` ≈ 0.75 bps; crypto perps ≈ 2 bps + realized funding. Capacity/turnover reported
alongside; a signal whose edge is smaller than costs is rejected regardless of gross IC.

## 9. Hold-out protocol

The most recent ~20% of each series is sealed until steps 1–8 pass on the development sample.
It is evaluated **once**; a failure there is reported as-is and is not iterated on.
