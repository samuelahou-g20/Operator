# Funding-carry — research track

One subfolder per experiment/version, one reproducible artifact each. **Rule:** every step
(subversion, veto, sizing idea) gets a notebook built by its `_build_*.py` script, with the heavy
logic in the shared **`lab.py`** (which reproduces the production `engine.run` to 1e-6). Notebooks
are executed *outputs* — never hand-edit them; change the `_build` script or `lab.py` and re-run.

## Structure

```
research/funding_carry/
  lab.py                    shared harness — every experiment imports this
  v0_baseline/              v0 edge check (preregistration + result, one README)
  v1_layer_d/               v1 = Layer-D hysteresis: rationale + knob validation
  universe/                 survivorship audit of the tradeable universe
  v2_selection/             v2 cross-sectional selection + basis-blowout veto (v2.1)
  v3_sizing/                v3 risk-aware sizing (inverse basis-vol) + Layer G factor null
  estimation_hygiene/       Layer H (shrinkage / winner's curse) — null
  v4_drawdown_overlay/      v4 "shadow" drawdown overlay (portfolio works; per-asset fails)
  v5_capital_efficiency/    v5 margin model: returns per $ of equity + PM safety (uniMMR)
  multiple_testing/         DSR + PBO honesty pass over the whole version ladder (raw + excess-of-rf bases)
  capacity/                 capacity/AUM study: 5%-ADV participation cap, decay curves
  v6_selection_refinements/ v6 = min-age veto + CVaR-blend sizing (vol-regime scaling = null)
  v7_cross_venue/           v7 = multi-venue breadth/full book (venue-selection = null)
  v8_cash_yield/            v8 = idle-cash yield at the T-bill rate + excess-of-rf Sharpe reporting
  final_report/             THE final report — the shipped product (v8.0) in one presentable document
  flat_years/               was there carry to earn in 2022/2023/2025? (oracle attainability study)
  rf_gates/                 rf-indexed gates (v9 candidate) — null
  leverage_mechanics.md     cited Binance margin/liquidation mechanics reference (for v5)
```

## Map: experiment → question → result

| folder | question | artifact | headline |
|---|---|---|---|
| `v0_baseline/` | Net edge after cost on 5 majors? | `README.md` | **Fails gate** — net +1.6% / Sh 0.84 @1×, −8.7% @2×; churn 54× is the killer |
| `v1_layer_d/` | Does an EWMA-mean *exit* cut the churn, and are the knobs overfit? | `mean_exit_rule.ipynb` (rationale) · `holdout_validation.ipynb` (validation) | Hysteresis collapses turnover; knobs robust in time (corr 0.83); majors Sh 6.4 → ~1.5 on the broad universe |
| `universe/` | What is the clean universe; delisted-name bias? | `survivorship_audit.ipynb` | 466 tradeable pairs; delisted names ran hotter funding (median 8.1% vs 0.8%); funding is two-sided |
| `v2_selection/` | Can cross-sectional selection beat the diversified band? | `v2_selection.ipynb` | Top-K-by-funding **fails** (basis concentration); **v2.1 basis veto** Sharpe 1.5 → **5.4**, DD −10→−3% |
| `v3_sizing/` | Does risk-aware weighting beat equal weight? (v3.0 sizing + v3.1 band) | `v3_sizing.ipynb` | **v3.0** inverse-basis-vol sizing: Sharpe 5.4 → **5.75**; **v3.1** adds a no-trade band (0.5) → best version: Sharpe **6.32**, 2×-cost **3.62**, DD −3%, turnover back to ~20×; robust OOS |
| `v3_sizing/` (Layer G) | Does factor-aware (covariance) carry sizing beat v3.1's diagonal? | `factor_sizing.ipynb` | **Null — not shipped.** Carry IS a concentrated factor bet (held-book N_eff ≈ 4), but tilt / one-factor min-var / Σ⁻¹a all lose to v3.1 (lower Sharpe/net, no tail gain) — the factor's downside is already vetoed upstream. Keep N_eff as a diagnostic |
| `estimation_hygiene/` (Layer H) | Does shrinking the noisy funding estimate improve selection? | `shrinkage.ipynb` | **Null — not shipped.** Winner's curse is real (forward funding ≈ 0.5× estimate) but it's a *level* bias, not a *ranking* one — shrinkage leaves rank-IC unchanged (0.516), backtest slightly worse; θ_keep exit + vetoes already handle it. `lab.shrink` kept |
| `v4_drawdown_overlay/` | Does a shadow drawdown overlay help; portfolio vs per-asset? | `drawdown_overlay.ipynb` | **Portfolio: works** — same net, maxDD −2.8%→−0.9% (**Calmar 3.5→11**), sits out 2022, beats placebo, point-in-time; caveat n≈1 episode. **Per-asset: fails** (net 9.9%→3.6%, maxDD doubles) — name-level DD is idiosyncratic basis noise; signal only tradeable at the aggregate |
| `v5_capital_efficiency/` | What does each margin setup earn per $ of equity, and is the setup safe? | `capital_efficiency.ipynb` | Deploy d = k/(k+1) isolated, **1.0 under Portfolio Margin** (= what v0–v4 assumed; engine now books per-equity via the `margin` config). **PM validated**: daily uniMMR ≥ ~11–25 vs liquidation 1.05, even harsh alt haircuts. Isolated dominated (less capital + ~2.1k intraday liq events at k=4). Borrowed >1× ≈ not worth it (borrow ≈ carry) |
| `multiple_testing/` | Does the ladder survive correcting for the ~143 configs tried (DSR + PBO), on the raw AND the excess-of-rf basis? | `dsr_pbo.ipynb` | **Raw (historical record):** 1× undeflatable (blind to cost); at 2× v6.1/v4.0 survive, v3.1 doesn't; PBO 0.10/0.002. **Excess-of-rf (the current headline metric; rf = lake fred/tb3ms):** at 1× DSR = 1.000 for every book against a *real* ~2.5 expected-max — **v8.0's 4.99 (1×) is deflation-robust**. At 2× the rf haircut halves the Sharpes while grid dispersion barely shrinks: every book sits below the ~3.98 expected-max (v8.0 best, DSR 0.05) — **the 2× figure is strictly cost-stress**. Excess PBO 0.034/0.031, IS-best config unchanged raw↔excess: a *level* shortfall, not selection-overfit. (Canonical figures under the final (E[r]−E[rf])/vol(r) definition: 5.07/3.48 — differences ~2%, verdicts unchanged.) Caveat: the excess edge is front-loaded (2021 +31.5%, 2024 +8.6%; 2022/2023/2025 ≈ flat) — DSR's full-T treatment doesn't price that regime concentration |
| `capacity/` | How much AUM can v4 absorb before the edge decays? | `capacity.ipynb` | **Ceiling ~$20–30M.** Carry bleeds slowly ($1M→$100M net 10.3→8.0%, majors are deep) but the drawdown protection breaks first: the overlay's all-out 2022 exit lags 0→6 days under the 5%-ADV cap → maxDD −0.9→−7.6%, Calmar 11→1.0. √-impact is 2nd-order; execution latency is the bind (n≈1 episode caveat) |
| `v7_cross_venue/` | Does adding Bybit `venue:pair` assets pay (venue selection / breadth / capacity)? | `cross_venue.ipynb` | **Mixed → shipped.** Venue-selection **null** (spread half-life ~0.7d, concentrated in veto-dropped names; best-venue switching doubles turnover, halves 2×-Sharpe). **Breadth ships** (113 Bybit-only names: OOS 2×-Sharpe 5.36→6.05, no regression). **Capacity uplift** via the full book (summed ADV: ceiling ~$20–30M→~$40M+, $30M Calmar 1.18→2.17). **Shipped as v7.0** (breadth; engine `sources`/venue:pair extension; full-window net 10.23%, Sh 7.00, **2× 5.40** — the headline) **and v7.1** (full book, 2× 5.18, the ≥$30M capacity config). Engine↔study parity exact on the joint window; v6.1 rerun byte-identical; never venue-switch |
| `v8_cash_yield/` | Does crediting idle cash (overlay-out) at the T-bill rate justify an accounting change? | `cash_yield.ipynb` | **Modest ship — accounting, not alpha.** Idle cash is binary, all in 2022/early-23 (the 0→5% ramp); credit adds **+0.5%/yr**, 2022 flips ~0→+2%, v7.1's maxDD −1.11→−0.90%. **Shipped as v8.0/v8.1** (engine `cash` block: rf from the lake, fred/tb3ms actuals via the FRED API, capture 0.75): **v8.0 net 10.68%, Sharpe 5.07 (1×) / 3.48 (2×)** — Sharpe is (E[r]−E[rf])/vol(r) everywhere in the product surface; ratios ignoring rf are not reported |
| `final_report/` | The shipped product (v8.0), one presentable document | `final_report.ipynb` | **The claim sheet**: net 10.68%/yr incl. cash, **Sharpe 5.07** (1×, deflation-robust) / 3.48 (2×, stress-only) — Sharpe = (E[r]−E[rf])/vol(r) everywhere, rf = fred/tb3ms — maxDD −0.88%, turnover 11.8×, ~146 names, capacity ~$20–30M → ~$40M+ (v8.1). Every run number recomputed live + parity-asserted vs `metrics.json`; every study number cited to its folder. Run the product: `python -m src.strategies.funding_carry.v8.backtest` (writes results + `report.html`) |
| `flat_years/` | Was there harvestable carry in the flat years (2022/2023/2025), and was it reachable ex-ante? | `flat_years.ipynb` | Three answers: **2022 correctly declined** (supply thin, ~+2% in-family required bearing the tail; overlay cash was right). **2023 in-family miss ~+1.7%**: funding was harvested but the fixed 8%/2% gates held sub-rf-hurdle names while T-bills paid 5% — best knobs ≈ oracle (+3.4% vs shipped +1.7%) → **rf-indexed gates** are the mechanism-backed Phase-2 lever. **2025 regime**: oracle +3.4% but best trailing-signal config +0.6% — funding unpredictability, no realistic lever. Kills: daily funding timing (O-day oracle **−11%** net of costs), funding-prediction models, short-spot negative-carry (borrow unverifiable) |
| `rf_gates/` | Do rf-indexed gates (θ = rf + spread) recover the flat-year excess? (v9 candidate) | `rf_gates.ipynb` | **NULL — all three preregistered variants rejected** (8th null). The construction is sound (nests the incumbent bit-for-bit at rf≈0) but the economics backfire: flat-year funding supply compresses exactly when rf rises, so rf+8% sits above the entire 2023 signal distribution (variant A: 99% idle) while the raised keep-gate evicts still-profitable 2024 names (−4.3pp). BASE wins on excess Sharpe at both costs (4.79 vs 2.3–4.1). Reconciliation: flat_years' 2023 attainable +3.4% was reachable via signal smoothness/breadth, not rf-indexing — chasing that per-year knob surface = mining; not pursued |
| `v6_selection_refinements/` | Do a min-age veto, CVaR sizing, or vol-regime scaling improve v3.1? | `min_age_veto.ipynb` · `cvar_sizing.ipynb` · `vol_regime_scaling.ipynb` | **Min-age veto X≈15d: modest ship** — OOS Sh 7.5→8.0, 2× 4.3→5.1, lower turnover; the young<30d cohort is toxic (standalone Sh 0.93, maxDD −11.7%). **CVaR-60d sizing (50/50 blend w/ inv-vol): modest ship** — 2× 3.62→3.97, turn 20.5→18.6; winsor clip load-bearing. **Vol-regime scaling: null** — ρ=0.07 / zero overlap with the v4 overlay; carry's drawdown is funding compression, not price vol. **Shipped as v6.0/v6.1** (on v4+v5): 2× Sharpe 4.85→**5.15/5.18**, turnover 13→11.8× |

## Reproduce

```bash
uv run python research/funding_carry/lab.py                                        # harness self-test (engine parity)
uv run python research/funding_carry/v2_selection/_build_v2_selection.py           # rebuild a notebook (executes + writes in place)
uv run python research/funding_carry/<folder>/_build_<name>.py                      # any other step
```

Backtests behind the notebooks (write `results/funding_carry/<version>/<run>/`):

```bash
uv run python -m src.strategies.funding_carry.v0.backtest --config strategies/funding_carry/v0.yaml
uv run python -m src.strategies.funding_carry.v1.backtest --config strategies/funding_carry/v1.yaml
uv run python -m src.strategies.funding_carry.v2.backtest --config strategies/funding_carry/v2.yaml     # v2.0 breadth
uv run python -m src.strategies.funding_carry.v2.backtest --config strategies/funding_carry/v2.1.yaml   # v2.1 basis veto
uv run python -m src.strategies.funding_carry.v3.backtest --config strategies/funding_carry/v3.yaml     # v3.0 inv-vol sizing
uv run python -m src.strategies.funding_carry.v3.backtest --config strategies/funding_carry/v3.1.yaml   # v3.1 + no-trade band
uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v4.yaml     # v4.0 drawdown overlay
uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v5.yaml     # v5.0 margin model (config-only, v4 code)
uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v6.yaml     # v6.0 min-age veto (config-only, v4 code)
uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v6.1.yaml   # v6.1 + CVaR-blend sizing
uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v7.yaml     # v7.0 multi-venue breadth (binance + bybit-only)
uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v7.1.yaml   # v7.1 multi-venue full book (capacity config)
uv run python -m scripts.backfill --source fred && uv run python -m scripts.normalize --source fred     # rf series (needs FRED_API_KEY in env/.env)
uv run python -m src.strategies.funding_carry.v8.backtest --config strategies/funding_carry/v8.yaml     # v8.0 THE PRODUCT (+ HTML report in the run dir)
uv run python -m src.strategies.funding_carry.v8.backtest --config strategies/funding_carry/v8.1.yaml   # v8.1 full book (capacity config)
```

## `lab.py` (shared harness)

`load_panels`/`universe` · `signal_apr`/`hysteresis_weights` (v1) ·
`feature_panels`/`veto_mask`/`held_matrix`/`weights_from_held`/`book` + `evaluate_select` (v2
selection) · `evaluate`/`sweep`/`select` · `validate_engine` (production parity).
