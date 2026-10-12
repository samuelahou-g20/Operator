# Crypto regime atlas

Companion to `research/funding_carry/`. The funding-carry book (v8.0, delta-neutral, Binance+Bybit)
earns its excess-of-rf return almost entirely in leverage-euphoria years (2021 **+28.9%**, 2024
**+8.6%**) and is flat in the deleveraging / low-carry years (2022 **−0.5%**, 2023 **+1.7%**, 2025
**−0.2%**). This track asks, rigorously: *what kind of markets were those, was anything else
systematically harvestable in the carry-dead years, and does a complement sleeve + an allocation rule
beat carry alone?*

**Ground rules (stated in every notebook).** Everything is judged in **excess of rf** (FRED `tb3ms`,
daily = annual/365; the `excess = net − rf` convention reproduces the v8.0 annual excess to the
decimal). All data is read only through `src.data.query` (via `atlas_lab.py`). The Phase-2 sleeve menu
is **preregistered and frozen** in `_build_sleeve_audit.py` — no tuning, no variants beyond the six
declared; this is a regime AUDIT with standard instruments, not an edge search. Every sleeve and every
allocation rule is enumerated in a per-notebook **trial ledger** (7 sleeve families + 4 allocation
rules = **11 trials**) for the repo's DSR accounting. Cost table is `config/strategies/funding_carry/
v3.1.yaml` (perp single-leg one-way 7bps, spot single-leg 12bps, carry-pair one-way 19bps).

Shared harness: **`atlas_lab.py`** (lake loaders, rf, liquid-universe construction, stats, and the
canonical `sleeve_*` builders). Each phase is a `_build_*.py` script that writes and executes its
`.ipynb` (never hand-edit the notebooks; SVG figures, time-series panels).

Build/run: `uv run python research/regime_atlas/_build_atlas.py` (and `_build_sleeve_audit.py`,
`_build_allocation.py`).

---

## Phase 1 — `atlas.ipynb` (descriptive regime atlas)

**Question.** What kind of market was each year 2021-2026 on the dimensions that matter for carry and
its complements — leverage demand, price persistence, breadth, liquidity, macro? (No strategies, no
tuning — pure description.)

**Artifact.** `_build_atlas.py` → `atlas.ipynb`. BTC/ETH trend·vol·drawdown; an EW liquid-alt index
(point-in-time top-30 by 60d dollar ADV); breadth (share of the $5M+ ADV pool above its 200d MA);
aggregate dollar volume; listings/delistings per quarter; **carry supply** (cross-sectional median
30d funding APR) vs the rf hurdle; rates + SP500 macro context (FRED). One regime table + labeled
timeline.

**Headline.** Carry is a **regime-gated bet on leverage demand**: median liquid-universe funding APR
ran **~40% (2021) → ~0%/below-rf (2022-23) → ~16% (2024) → ~3% (2025)**. Year labels: **2021 leverage
euphoria**, **2022 deleveraging bear** (LUNA/FTX, breadth ~11%), **2023 low-carry recovery grind**
(BTC +154% on ~0% funding — the archetypal "market up, carry dead" year), **2024 leverage re-rating**,
**2025 carry-dead chop** (BTC −7%, alt index −71% — dispersion, not leverage). The complement question
is precise: in 2022/2023/2025, was trend or dispersion harvestable with standard instruments?

## Phase 2 — `sleeve_audit.ipynb` ("what paid when")

**Question.** Was anything else systematically harvestable in the carry-dead years, and does the
carry-vs-trend complement hold with proper (perp, funding-inclusive, cost-aware) implementations?

**Artifact.** `_build_sleeve_audit.py` → `sleeve_audit.ipynb`. Six frozen sleeve families
(TS-TREND-LF, TS-TREND-LS, XS-MOM, REV, BETA-BTC/EW, CARRY), family×year excess & Sharpe heatmaps,
monthly correlation matrix, equity panels, and the 2022 short-trend callout.

**Headline.** Standard crypto trend/mom/reversal is **real but low-quality** — the L/S perp sleeves run
**50-70% vol at 70-150x turnover for Sharpe ~0.3-0.5** after costs; REV is a net loser. The one clean
risky sleeve is **TS-TREND-LF** (majors long/flat, ~6x turnover, Sharpe ~0.8). No hidden Sharpe-5
alternative exists; carry (Sharpe ~5) is in a different league. **Carry is near-orthogonal to all of
them** (monthly |corr| < 0.35). The complement hypothesis **holds but modestly**: in 2022 the crude
BTC long/flat proxy bled **−21.3%** while the frozen **TS-TREND-LS printed +6.3%** (it can short the
downtrend) — directionally as hypothesized, small once implemented cost-aware. Trend is a *diversifier,
not a rescue* (it also lost −55% in 2024). Different sleeves own different years.

## Phase 3 — `allocation.ipynb` (portfolio view)

**Question.** Does a complement sleeve + an allocation rule beat carry alone — statically, or with one
mechanism-first dynamic rotation? Does static diversification capture most of the benefit?

**Artifact.** `_build_allocation.py` → `allocation.ipynb`. Candidate set = {TS-TREND-LF, TS-TREND-LS,
XS-MOM} (corr < 0.30 to carry AND positive excess, by rule). Static mixes (native 50/50, equal-risk,
vol-matched 50/50) + one preregistered dynamic rule (opportunity-weighted carry-vs-trend, cash
fallback). Compared at a common 10%-vol footing.

**Headline.** **Nothing beats carry alone on aggregate risk-adjusted return** — at 10% vol, Sharpe
ranks **carry 4.99 > static vol-match 4.05 > dynamic 3.88** (Calmar agrees); any blend dilutes the
carry-rich 2021/2024. Equal-risk collapses to ~93% carry (still Sharpe 3.00); naive native 50/50 is a
trap (Sharpe 0.94, ~all trend risk). The complement earns its keep **only on the flat-year path**:
2025 carry −1.6% → static **+8.9%**, 2022 softened, maxDD −11.7% → −7.3% — but it **hurts 2023**, so
it is a targeted smoother, not a general upgrade. **Static beats the dynamic rule** (higher Sharpe,
half the drawdown), confirming the classic prior. **Recommendation: carry alone for aggregate return;
a static vol-matched carry+trend blend only if de-risking the carry-dead path is the explicit
objective; do not build the dynamic rotation.**
