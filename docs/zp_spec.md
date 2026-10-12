# ZP (Zeyi Process) — quoting review: what it is and how the numbers are made

Extracted 2026-09-10 from the MM2 desk's `mmanalytics` repo (Streamlit dashboard + report Telegram
bot) so The Operator can produce the report on request. Frozen source copies live in
`docs/references/zp_source/mmanalytics_extract/` (reference only, not importable); the example output
is `docs/references/zp_source/NEAR_ZP_9h_quoting_review.pdf`.

## 1. What a ZP is

A **9-hour (or any window) review of one token's market making**, in a fixed order:

1. **Frame**: token, quoter mode, UTC window and session, period OHLC and % move.
2. **Headline**: total P&L split into **MM (passive quoting) P&L** and **hedging P&L**; MM volume and
   net inventory; flow-level capture (sell avg vs buy avg in bps); vol-weighted **markouts** at
   60/120/180/300/600 s; **capture** (our share of venue volume); **2% depth compliance** and uptime.
   Each with a one-line read, then a bottom line.
3. **Venue scorecard**: per venue volume, P&L, buy/sell avg, 180 s markout, capture, action.
4. **Findings** in a fixed order: hedger vs quoter attribution → markouts/adverse selection → tick
   size vs min-spread scaling → reference vol vs realized (was dynamic mode floored?) → scale
   asymmetry compressing the ladder in high vol → depth compliance as calibration artefact and
   mandate risk → capture as consequence of ladder shape.
5. **Recommendations** ranked High/Med/Low with a rationale tied to a finding.
6. **Open questions**: what the data could not settle (hedger setup, fees, which vol drives the
   multipliers, KPI definitions, unmonitored venues).

Steps 1–3 are produced by code in `mmanalytics` (`trade_analysis_core.compute_zp` + `zp_html`).
**Steps 4–6 are written by a human** from that HTML. That narrative layer is what The Operator
adds; the numbers stay deterministic.

## 2. Data sources (all external to both repos)

| Input | Source | In our repo today |
|---|---|---|
| Fills per venue/account | trade-streaming `GET http://trade-streaming.ext.g20api.com:8767/trades/` (params `inst`, `min/max_exch_ts` µs, `location_id`, paginated by `clock_id`; no auth) | ✅ `src/adapters/trade_service` (by location ids; add `inst` filter) |
| Account tagging MM / GAMMA / HEDGERV2 | reference service `GET /ref-svc/location` → `exchange_uid`, `tags.strategy[0]` per `location_id`. `account = f"{exchange} {acct_type}"`; **MM = `' MM'` in account, everything else = "Gamma/other"** | ✅ `src/adapters/refsvc` (`locations()`, `tags`) |
| Program (desk) of a coin | `POST /ref-svc/token` (password grant) → `GET /ref-svc/programs`; first of `MM2, MM3, MM4, PERP` containing the coin = `product_area` | ⚠ add `programs()` to our RefSvc client |
| Reference market klines (markouts, OHLC, P&L marking) | ccxt `fetch_ohlcv` on the token's `md_pair` @ `md_exchange` (usually `X-USDT` @ binance) | ✅ ccxt already used (`twap_bot`, `hedge_bot`) |
| Market volume per venue/pair (capture) | ccxt klines per configured exchange×pair, 1 m (1 h for BITVAVO, LCX); notional = `volume × (O+H+L+C)/4`, FX to USD | ✅ ccxt |
| FX for non-USD quotes | derived from binance klines (`CCY/USDT`, inverted, or via BTC), 1 m, cached 7 d | ⚠ port `fx_cache.py` (small) or reuse `desk_trades.usd_rate` |
| **Orderbook snapshots** (widths, 2% depth compliance, uptime) | **Postgres `OrderbookAnalysis`** on RDS (`ZP_OB_DB_URL`, psycopg): `exch_order_books(id, ccypair, exchange, product_area, timestamp_micros)` ⟕ `exch_order_book_entries(order_book_id, price, size, type∈{BID,ASK})`, filtered by `ccypair IN (pairs)`, `product_area`, window. ~87M snapshots, programs `mm1..mm5`, `perp`, `algodesk` | ✅ connected 2026-09-10 (the MM2 repo's secrets said MSSQL; it is Postgres, so no new dependency) |
| Dynamic-mode settings & vol history | gamma-dash `GET /api/price-bias?token=T` and `/api/price-bias/history?token&base&exchange&hours` (`http://127.0.0.1:8618` then `https://gamma-dash.ext.g20api.com`; VPN, no auth) | ❌ reachable from this host (HTTP 302), needs a client |
| Tick sizes | ccxt `load_markets` precision → tick, `tick_bps = tick/mid·1e4`, `min_ticks = base_min_spread_bps / tick_bps` | ✅ ccxt |
| Token → venues × quote pairs, `md_pair`, `md_exchange`, CoinRoutes client code | `config.yaml` (110 tokens, both desks) — copied to `docs/references/zp_source/mmanalytics_extract/config.yaml` | ⚠ decide: import as `config/zp/tokens.yaml` or derive venues from CoinRoutes strategies |

## 3. Formulas (exact)

**Summary per (account, acct_type, inst)** (`fetch_summary`): buy/sell quantity and USD notional,
`net = buy − sell`, `buy_avg = buy_notional/buy_qty`, `sell_avg = sell_notional/sell_qty`,
`last_px` = mean price of the last-timestamp trades in the window,
`pnl_$ = (last_px − buy_avg)·buy_qty + (sell_avg − last_px)·sell_qty` (mark-to-last-trade, not
FIFO). TOTAL rows recompute weighted averages. Books: **MM** (`' MM'` in account), **Gamma/other**
(complement), **Total**.

**Delta & P&L curve** (`_compute_pnl`): running average-cost position over all accounts; realised on
reductions `(avg − price)·closing_qty`, unrealised `(price − avg)·position`, re-marked to the 1-minute
mid `(O+C)/2` of the reference market; `cum_delta = position·price`.

**Markouts** (`fetch_markouts`, MM fills only): reference = **open of the kline bar ceil-aligned to
`ts + sec`** on `md_pair@md_exchange` (1 s bars on binance, 10 s gate, else 1 m); windows
`[60,120,180,300,600]` s.
`pnl_bps = 1e4·(ref/price − 1)·(+1 buy / −1 sell)`; `pnl_$ = (ref − price)·qty·sign`.
Per venue: **qty-weighted** mean per window, `volume_$ = Σ price·qty`. Cross-venue headline
(`format_report`): **notional-weighted** mean. Headline horizon 180 s.

**Capture** (`fetch_captures`): market key `(exchange, is_perp, quote)` so venue aliases (AIGENSYN,
BCHSV) collapse; `Capture = G20 $ / Market $`; TOTAL = Σ/Σ; coloured/headlined on **MM-active
venues only** (`ob_active` = orderbook collected with uptime ≥ 5%). Chart: 5-minute cumulative
`cumsum(ours)/cumsum(market)`.

**Orderbook widths** (`_ob_widths_from_raw`): snapshots rounded to 1 minute; per venue×pair
`TOB = 1e4·(min ask − max bid)/min ask`, `BOB = 1e4·(max ask − min bid)/max ask`; smoothed with a
15-minute rolling median; summary = median [min–max] of the raw series.

**2% depth compliance** (`_ob_compliance_from_raw`), per snapshot `book_id`:
`two_sided = has bid & has ask`, `mid = (best_bid+best_ask)/2`,
`bob_bps = 1e4·(worst_ask − worst_bid)/mid`, **`compliant = two_sided & bob_bps < 400`**.
Per venue over `n` snapshots (empty books included): `uptime = two_sided/n`,
`compliance = compliant/n`, `wide = (two_sided − compliant)/n`, `one_sided = (in_snap − two_sided)/n`,
`empty = (n − in_snap)/n`; `nodata` when `uptime < 0.05`. Headline = **flat mean** across venues
with data. Per-minute buckets (`snaps, in_snap, two_cnt, comp_cnt, bob_sum, bob_cnt, lo, hi`) are
kept so the page can re-cut any sub-window.

**Period OHLC** (`fetch_period_ohlc`): timeframe 5 m (≤6 h), 1 h (≤72 h), else 1 d;
`pct = (close/open − 1)·100`.

**Hedger execution cost** (dashboard side, `data.fetch_hedger_pnl`): per hedge
`pnl_bps = 1e4·sign·(hedge_avg/trade_net_price − 1)`, `sign = +1` if the hedged trade was a buy;
`pnl_usd = sign·(hedge_avg − trade_net_price)·hedged_qty`. In the ZP the hedger P&L is simply the
GAMMA/HEDGER accounts' `pnl_$` from the summary (HEDGERV2, GAMMA tags).

## 4. Thresholds and bands (the judgement scale)

| Metric | Bands (first match, descending) |
|---|---|
| Compliance % and uptime % | ≥95 green · ≥90 amber · ≥75 orange · else red |
| Capture % | ≥10 green · ≥8 amber · ≥5 orange · else red |
| Markout bps (180 s) and P&L bps | ≥+5 dark green · ≥+1 green · ≥−1 neutral · ≥−5 amber · else red |
| Vol / ref vol ratio | ≥1.6 red · ≥1.3 orange · ≥1.12 amber · ≥0.9 neutral · ≥0.8 light blue · else blue |
| Tick size bps | ≥8 red · 4–8 orange · 2–4 amber · 0.8–2 light · <0.8 fine |
| Min spread in ticks | ≥5 fine · 2–5 ok · 1–2 "floor" · <1 tick red (setting unachievable) |

**Dynamic-mode hard rules** (`dmccheck._HARD`): `top_level_weight > 0.2`; `levels < 2`;
`base_min_spread_bps ≤ 0` or `> base_max_spread_bps`; `base_max_spread_bps < 170 or > 200`;
`total_value_usd < 0`; `reference_annualized_volatility_pct < 0 or > 250`; any of the three
`volatility_*_scaling_factor < 0.25`; empty `volatility_halflife_seconds_list`. Peer-outlier
backstop: ratio to desk median outside `[0.33, 3]` (`[0.2, 5]` for size and ref vol) and rare
(<8% of desk values within ±25%).

**Dynamic-mode fields** (gamma-dash `ob_dynamic`): `levels`, `base_min_spread_bps`,
`base_max_spread_bps`, `total_value_usd`, `top_level_weight`,
`volatility_min_spread_scaling_factor`, `volatility_max_spread_scaling_factor`,
`volatility_size_scaling_factor`, `reference_annualized_volatility_pct`, `volatility_interval_ms`,
`volatility_halflife_seconds_list`. History series: `vt_total_value_usd`, `volatility_ann_pct`,
`vt_min_spread_bps`, `vt_max_spread_bps`, `timestamp`.

## 5. The rules the human applied in the PDF (to encode as deterministic checks)

- **Attribution**: hedging P&L vs MM P&L; a hedger round trip (buy avg − sell avg on the hedger
  account) is "trend chasing" when negative in bps; report `|hedge P&L| / MM P&L`.
- **Adverse selection**: markouts negative at 60–300 s and recovering at 600 s = momentum
  pick-off, not informed flow; compare to flow-level capture `(sell_avg − buy_avg)/mid` in bps.
- **Tick dead zone**: `min_spread_bps / tick_bps < 1` → scaling is cosmetic; effective spread is
  one tick; quotes pinned at the tick, first in queue.
- **Floored dynamic mode**: `ref_vol` vs realized (`volatility_ann_pct` history) — if realized
  < ref most of the window, min spread sits at base and size at max: dynamic mode effectively off.
- **Scale asymmetry**: `vol_min_scale > vol_max_scale` means the TOB (`min_spread`) widens faster
  with vol than the BOB (`max_spread`) — the BOB expands at a smaller scale than the TOB. Both
  multipliers are floored at 1, so **neither side ever tightens** and the ladder does NOT compress
  toward the inside (the PDF's reading, and our first port of it, said it did; a desk trader
  corrected it on 2026-09-17). Only a defect where the TOB would actually catch the BOB, at
  `(base_max/base_min) ** (1/(vol_min_scale − vol_max_scale))` × reference vol — severity is set
  from that ratio, not from the asymmetry alone.
- **Capture gap**: total capture against the desk's standing **≥10%** target; the shortfall is
  sized in volume (`market_volume × 0.10 − our volume`) and attributed to the venues holding the
  most missing volume. The parameters named as levers are **that venue's**, and the 180 s markout
  is the guardrail (positive → room to tighten or size up; negative → take the volume at the back
  of the ladder instead). How much capture a parameter change buys is **not modelled** — that
  response is nowhere in this data.
- **Compliance on the boundary**: BOB medians within a few bps of 400 → compliance is a coin flip
  driven by vol scaling; recommend base max spread ~165–170 bps/side or cap the max multiplier.
- **Barbell capture**: capture low despite volume when L0 carries ~1% of notional and the next level
  sits ~50 bps out; add a level in the 10–25 bps zone if the mandate has volume/presence KPIs.
- **Per-venue actions**: worst markouts + negative P&L + low volume → widen or cut L0; best
  markouts on large volume → increase weight; tiny venue with high capture → modest increase only.

## 5a. Dynamic-mode formulas (hardcoded, from MM3_Aggregator "Dynamic mode")

Hardcoded in `src/the_operator/zp/dynamic.py` on the desk's instruction: the formulas change
rarely, the **parameters change daily** and are what the desk actually tunes, so a review that
cannot say what a parameter does is not actionable.

```
vol        = max(ema_hl300(returns), ema_hl1800(returns))      annualised %
ref_vol    = Garman-Klass over the reference bars              annualised %
min_spread = base_min_spread * max(1, (vol / ref_vol) ** vol_min_scale)
max_spread = base_max_spread * max(1, (vol / ref_vol) ** vol_max_scale)
size       = base_size       * min(1, (ref_vol / vol) ** vol_size_scale)
tob_size   = base_size * top_level_weight / levels
```

The parameters arrive from the quoting boxes (`sources.fetch_strategy_params`), whose `ob_dynamic`
field names map 1:1 onto the sheet's: `total_value_usd` is `base_size`, `top_level_weight` is
`top_lvl_wt`, and `volatility_halflife_seconds_list` is the `[300, 1800]` pair above.

`tests/the_operator/zp/test_dynamic.py` reproduces the sheet's own OUTPUT column
(24.76333333 / 229.3156485 / 45512.42317 / 500) exactly. Note `tob_size` takes **`base_size`, not
the vol-scaled `size`** — the sheet's formula text said `size`, which the desk confirmed as a typo
on 2026-09-17 and fixed. The consequence is real: the ToB slice is fixed, so rising vol thins the
ladder behind it while the top of book stays put.

## 5b. Known uncertainty in the source report

The PDF's headline says "Flow-level capture +1.3 bps" beside "Sell avg 2.30527 vs buy avg
2.30234". Those averages differ by 12.7 bps, not 1.3. Our port reproduces the source's MM TOTAL
averages exactly (2.3024 / 2.3053, checked against its own generated HTML) and so reads
+12.6 bps, which means the arithmetic is right and **the PDF's headline figure is off by about a
factor of ten** — the source code has no `flow_capture` function, so that line was computed by
hand. Worth raising with the author. Either way, carry the report's own caveat: in a trending
session average prices conflate spread capture with inventory drift, so the markouts are the
cleaner toxicity read.

**Period OHLC differs by design.** The source picks the timeframe from the window span (5m ≤6h,
1h ≤72h, else 1d), so a 9h window's "open" is the open of the hour containing the start — 54
minutes before it, 2.311 on the NEAR run. We take OHLC from the 1-minute series we already fetch,
so it is exact to the requested window (2.308, −1.47% against their −1.64%). Ours answers "what
did price do over this window"; theirs is bar-aligned.

**Kraken market volume is missing** where the source had it (capture 5.3%), so that venue's
capture reads n/a. Worth chasing: likely the ccxt symbol for NEAR-USD on Kraken.

## 6. Port plan (proposed)

**Built** (2026-09-10) as `src/the_operator/zp/`: `config.py` (token registry
`config/zp/tokens.yaml`), `sources.py` (trades + tagging through our trade-service and
reference-service adapters, ccxt klines, the orderbook query behind `ZP_OB_DB_URL`, a gamma-dash
client), `compute.py` (the §3 formulas, `ZPResult`), `checks.py` (the §5 rules as deterministic
findings with evidence), `run.py` (orchestration; every section degrades to a warning),
`report.py` (Telegram card, model packet, HTML in the house `src/reporting` style). The Operator
tool is `zp(token, hours, refresh)` in `tools/quoting.py`: the card and HTML go to the chat as-is
and the model writes findings, recommendations and open questions from the packet.

Differences from the reference, deliberate: `captures` takes pre-fetched market volume so
`compute` stays I/O-free; the chart-only gap-break step is dropped; tick sizes are not fetched yet
so `tick_dead_zone` stays dormant until a caller supplies them; LCX needs the custom ccxt-style
connector we did not port, so its market volume is a warning.

The orderbook DB is connected (`ZP_OB_DB_URL` in `.env`). A coin is quoted by ONE program, which
is not always the desk that asked, so `fetch_ob_snapshots` tries the caller's desk, then falls
back to whichever program actually collected those pairs in the window and says which it used.

**gamma-dash is not needed.** It is behind Entra SSO from a server (`.ext.` redirects to a
Microsoft login; the `.prod.` host is a dead load balancer, 503 with no healthy targets). But it
only aggregates the quoting boxes, and those serve the same `ob_dynamic` block unauthenticated at
`GET http://{ip}:{port}/rest/calculators/params`, keyed `LM2Strategy:{FLOW}:{PAIR}:{EXCHANGE}`.
`sources.fetch_strategy_params` scans them (own program first, then the others, saying which it
used) and returns the gamma-dash shape, so `run_zp` simply falls back to it. The box list is
`config/zp/strategy_boxes.yaml`, generated from the AWS SSM parameter
`/configs/deployments/deployments.yaml` by `uv run python -m scripts.zp_boxes --write` — a
generated registry like `config/universe/`, so boto3 and zstd stay off the request path.

What the boxes do not carry: the vol/spread history and the live ladder. Instead the realised vol
comes from the reference klines (`compute.realized_vol_ann_pct`, independent of the strategy's own
tracker) and the ladder geometry from the settings (`top_level_weight`, and the L0→L1 gap as
`(base_max − base_min)/(levels − 1)`). Tick sizes come from ccxt (`sources.fetch_ticksize`), so
`tick_dead_zone` is live too. Six of the eight checks fire on a typical run.

Still degraded: LCX (market volume and tick size) needs the custom ccxt-style connector we did not
port, and a couple of quoting boxes time out on any given scan — both are warnings, not failures.
