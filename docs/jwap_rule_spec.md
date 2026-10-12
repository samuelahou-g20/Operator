# JWAP — the gate, for implementation

JWAP is a TWAP whose children fire only when a JASON condition holds. Three
parameters: the **side** (buy or sell), the **level multiple k** (which JSON level)
and the **minimum pivot age d** in days. Nothing else.

## Inputs

1. **OHLC candles** of the instrument (1h in our runs; 1-minute for more
   precision, see below), and the JASON frame computed from them with the
   production calculator (lookback 7 days, 5 pivots, asset class crypto). One
   row per bar, indexed by the bar's **open** time. Columns used:
   - buy side:  `high_jsonk` (the JSON level at k on the high side), `high_days`
   - sell side: `low_jsonk`  (the JSON level at k on the low side),  `low_days`
2. **1-minute price** of the instrument: the 1-minute kline **open** is the mid
   proxy at the child's minute.
3. The TWAP child schedule: entry time `k0`, horizon `n` hours, `c` children per
   hour; child `t, j` fires at minute `k0 + 60·t + round(60·j / c)`, each with the
   same base quantity `child_q = q_base / (n·c)`.

## Candle size: 1h is what we ran, 1-minute also works

The bar size is a parameter of the calculator, not of the rule. Our published
runs use **1h** bars. We also ran the same rule on **1-minute** bars
(`jwap_1min_2025`), which gives a finer level and age and reacts within the
minute. If you go to 1-minute, four things change:

1. **Causality shift becomes one minute**: a row stamped T is usable from T+1min.
   The same `knowable_on_minutes` with `bar = 1min`.
2. **Pin the vol blend to wall-clock hours.** The engine's tanh blend time
   constant is set per candle size (`TAU_HOURS_BY_FREQ`, 12 h at 1h). With no
   1-minute entry it falls back to one trading day = 1,440 candles. We kept the
   1h spec's 12 wall-clock hours, i.e. `tau = 720` one-minute candles.
3. **The threshold is not frequency-invariant.** The Garman-Klass vol reads
   ~11 % lower on 1-minute bars, so the same k is more sigmas: k = 1.5 fired
   about twice as often at 1-minute as at 1h on BTC/ETH/SOL. Re-choose k (or d)
   at 1-minute rather than carrying the 1h values over.
4. **Cost.** A 7-day window is 10,080 one-minute bars; the engine's per-bar loop
   ran ~4,300 bars/s for us (~4 min per perp per year of history, once; then
   incremental).

## The JSON level at k (to add to the calculator next to json2)

Inside the per-window kernel, where `json2` accumulates
`pivot_px · exp(−2·sd) · weight`, accumulate the same at k:

```
high_jsonk = Σ_i w_i · P_i · exp(−k · sd_i) / Σ_i w_i      (below the weighted highs)
low_jsonk  = Σ_i w_i · P_i · exp(+k · sd_i) / Σ_i w_i      (above the weighted lows)
sd_i       = pivot_vol_i · sqrt(t_i)                       same blended vol and age json2 uses
```

Check: at k = 2 the new columns must equal `high_json2` / `low_json2` exactly.
Do **not** derive it as `pivot · (json2/pivot)^(k/2)`: that is exact for one
pivot only and is off by bps for five. The per-pivot accumulation lives in
`src/signals/jason.py`.

## Causality (the contract)

A JASON row stamped `T` is computed from bar `T`'s **close**, so it exists only
at `T + 1h`. Before comparing with any minute price:

```
level_known(t)  = level of the last bar whose open ≤ t − 1h     (shift the series forward
days_known(t)   = same for the pivot age                         one bar, then forward-fill
                                                                 onto the minute grid)
```

Every minute of hour `T+1h .. T+2h` sees the values of bar `T`. `src/signals/jwap.py`
does this in `knowable_on_minutes`.

## The rule

At each scheduled child minute `t`, with `mid(t)` = that minute's 1m open:

```
buy :  fire  iff  mid(t) ≤ high_jsonk_known(t)   and   high_days_known(t) ≥ d
sell:  fire  iff  mid(t) ≥ low_jsonk_known(t)    and   low_days_known(t)  ≥ d
```

- A child that does not fire is skipped. **No catch-up**: skipped quantity is
  never bought or sold later. The parent ends short by the share of children
  the gate was closed for.
- A child that fires trades the same `child_q` as the TWAP would have, at the
  market, at that minute.
- `k` and `d` are fixed for the whole parent.

## Parameters used so far (sell side, 1h, L7_N5)

| instrument | k | d | share of children that fired |
|---|---:|---:|---:|
| BTC / ETH / SOL (2023–2026) | 1.5 | 1.5 | ~16 % |
| NEAR (2024-09 .. 2026-08) | 1.5 | 2.5 | ~10 % |
| ICP  (2024-09 .. 2026-08) | 1.5 | 2.0 | ~10 % |
| CGPT (2025-03 .. 2026-08) | 1.5 | 2.5 | ~7 % |

Buy side mirrors it with `high_jsonk` / `high_days` (the buy study used
k = 1.5, d = 1.5).

## Not in scope of this hand-over

Scaled clips (clip = TWAP clip / trailing gate rate), the bar-close control
(`z ≥ k` instead of the price level), trend-conditioned thresholds, and all of
the cost / impact modelling. Only the gate above.
