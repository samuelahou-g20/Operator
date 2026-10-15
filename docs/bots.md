# Telegram bots — reference

The `scripts/*_bot.py` family: on-demand Telegram bots for the MM3/MM2 market-making desks. Each is
a self-contained long-poller that renders a house-style PNG (or places orders) and never crashes on
a bad command. **Read this before touching any bot.** Module docstrings are the per-bot source of
truth; this file is the cross-cutting map.

## Roster

| Bot | Command | Desks | Kind | BotFather token env | Other key env |
|---|---|---|---|---|---|
| `agg_bot` | `/agg`, `/mm2gamma`, `/analysis <symbol> <window>` (treasury only) | mm3 + mm2 + mm4 + treasury + options + mm5 | read | `TELEGRAM_AGG_BOT_TOKEN` | `GOOGLE_AGG_SA_KEY` (→ `.gsa_key.json`); `PLUTO_API_KEY`/`_SECRET` (treasury **and options** = Pluto portfolio, not a sheet) |
| `orders_bot` | `/orders [<asset>]`, `/orders hedge`, `/orders <asset> fills [N]` | mm3 + mm2 + mm4 + treasury | read | `TELEGRAM_ORDERS_BOT_TOKEN` | `CR_API_TOKEN_{MM3,MM2,G20JPY}` (per desk), `PLUTO_API_KEY`/`PLUTO_API_SECRET` |
| `hedge_bot` | `/hedge` | mm3 + mm2 | read | `TELEGRAM_BOT_TOKEN` (shared "PnL bot") | `CR_API_TOKEN_MM3` / `CR_API_TOKEN_MM2` (per desk) |
| `trade_bot` | `/trade …`, `/trade cancel <id>`, `/trade today [cat]`, `/trade <token> today [cat]`, `/trade <token> plot [horizon=30d]`, `/scalper …`, `/jwap …` | mm3 + mm2 + mm4 + treasury + otc | **WRITE (live money)** + two read-only reports | `TELEGRAM_TRADE_BOT_TOKEN` | `CR_API_TOKEN_{MM3,MM2,G20JPY}` (per desk), `TELEGRAM_TRADE_TRADERS[_MM2/_TREASURY]` |
| `jason_bot` | `/jason [universe]` | mm3 | read | `TELEGRAM_JASON_BOT_TOKEN` | `PLUTO_API_KEY`/`PLUTO_API_SECRET` |
| `ls_bot` | `/ls` | mm3 | read | `TELEGRAM_LS_BOT_TOKEN` | lake / vendored trend score |
| `vol_bot` | `/vol [report]` + RV>IV breach push | mm3 | read | `TELEGRAM_VOL_BOT_TOKEN` | `PLUTO_API_KEY`/`PLUTO_API_SECRET` |
| `macro_bot` | `/macro` | mm3 | read | `TELEGRAM_MACRO_BOT_TOKEN` | `ANTHROPIC_API_KEY` |
| `sentiment_bot` | `/sentiment [topic]` | mm3 | read | `TELEGRAM_SENTIMENT_BOT_TOKEN` | `ANTHROPIC_API_KEY`, `X_BEARER_TOKEN` |
| `twap_bot` | `/twap <token> start= end= [candle=] [exchange=] [type=]` | mm3 (market-data allowlist) | read | `TELEGRAM_TWAP_BOT_TOKEN` | none — ccxt public candles only |
| `botstatus_bot` | `/botstatus`, `/botstatus <bot> [tabs\|<tab>]` | mm3 + mm2 + treasury | read | `TELEGRAM_BOTSTATUS_BOT_TOKEN` | `PLUTO_API_KEY`/`PLUTO_API_SECRET` |
| `flow_alert_bot` | *(push daemon, no command)* | mm3 + mm2 | push-only | `TELEGRAM_BOT_TOKEN` (send only, reused) | RDS + trade service; `TELEGRAM_MM3_CHAT_ID`/`_MM2_`; `FLOW_ALERT_THRESHOLD_USD[_<DESK>]`/`_WINDOW_MIN`/`_POLL_SECONDS` |
| `move_alert_bot` | *(**DISABLED** — see below)* | — | — | — | — |
| `mm2_trade_log_bot` | *(daemon, no command)* | mm2 | write-sheet | *(none — not a Telegram bot)* | `GOOGLE_MM2_SA_KEY` (→ `.mm2_trades_gsa_key.json`), `CR_API_TOKEN_MM2` |

`mm2_trade_log_bot` is the canonical MM2 token→strategy map (`MM2_STRATEGIES`) — other bots import
it rather than re-listing MM2 tokens. Because they import *from* it, it must never import them back:
its live-price table calls `CoinRoutesClient.mid_on`/`strategy_exchanges` directly rather than
reusing `orders_bot.spot_refs`, which would be a cycle.

It writes **two tabs** of the MM2 Trades Log: `Sheet1` is the fill log, and `Sheet2` carries a live
price table (Token / Mid / 24h %) for the 18 mm2 tokens, refreshed on the same 60s poll and anchored
at **H1** to stay clear of that tab's pivot in A–G. Mids are venue-scoped to each token's own gamma
strategy (an account-wide book can carry a venue the strategy cannot trade — a stale HUOBI CC-USDT
quote once pulled the mid 1.8% off). The 24h column comes from the daemon's own samples in
`data/state/mm2_trade_log.db`, so it is **blank for the first 24h after a fresh start**. Resolving
the venue sets costs ~80s on the first poll (one full `/api/strategies/` page-through per token,
then cached); every later refresh is ~3.7s. `trade_bot` is the canonical MM3 map (`STRATEGY`,
`DEFAULT_QTY`); `orders_bot` imports both.

`scripts/trade_plot.py` is **not a bot** — it is the market-data/PnL/render module behind
`/trade <token> plot`. It lives outside `trade_bot` because trade_bot's contract is "command
parsing + safety + Telegram plumbing only, no market-data or PnL logic"; trade_bot imports
`handle_plot` and nothing else. Its styling comes from repo-root `palette.json`, loaded **lazily**
so a malformed palette surfaces as a chat error instead of crash-looping the money bot at import.

`agg_bot` serves **six** desks, three of them sheet-backed and three Pluto-backed:

| Chat var | Desk | Source |
|---|---|---|
| `TELEGRAM_MM3_CHAT_ID` | MM3 | Google Sheet (Dashboard tab) |
| `TELEGRAM_MM2_CHAT_ID` | MM2 | Google Sheet (Summary tab) |
| `TELEGRAM_MM4_CHAT_ID` | MM4 | Google Sheet (Dashboard tab, MM4_Aggregator_V2) |
| `TELEGRAM_TREASURY_CHAT_ID` | Treasury | Pluto portfolio `TREASURYPROPTOTAL` |
| `TELEGRAM_OPTIONS_CHAT_ID` | Options book | Pluto portfolio `TOTAL_OPTIONS` |
| `TELEGRAM_MM5_CHAT_ID` | MM5 | Pluto portfolio `QUANTMM_CLEAN` |

MM4's Dashboard carries every column MM3's view renders, under the same header names and on the
same header row, so `_MM4` reuses MM3's column spec verbatim — only `sheet`/`gid`/`label` differ.
It has no `ALL` totals row; the reader stops at the first blank Coin, so `/agg` shows the coin
rows and no total line.

The Pluto desks (`source="pluto"`) render per-asset greeks — spot, Δ$, Γ$, vega, theta + total PV —
instead of a sheet, and both send `RiskReport="STICKY STRIKE FITTED"` so the numbers tie out to the
UI. All six chat vars must be **disjoint** (checked pairwise at startup via `combinations`, so
adding a fifth desk needs no new comparison code). ⚠️ The options chat is pinned for `agg_bot`
**only** — `orders_bot`/`trade_bot` don't read `TELEGRAM_OPTIONS_CHAT_ID`, so that chat is refused
there, which is what you want for a read-only book.

**`/mm2gamma`** is the one command that is *not* desk-routed. It always reads the same workbook —
**MM2GAMMA-MM**, `Dashboard` tab (`11Ne3nFRAg6I90qHpyWbxcKwgu7uo_8NLB5ragX7f9gU`, gid
`1242308581`) — so it is served only in the **MM2 and MM3 chats**; anywhere else it is refused
rather than showing one desk another desk's book. Columns are sheet B/E/K/L/M/P/Q/R → Coin, Price,
Start Bal, Curr Bal, Opt MTM, Pluto Δ, Net Δ coins, Net Δ $; that book has no market-share column
and no ALL row (the table just ends at a blank Coin).

Two sheet gotchas this bot now handles, both worth knowing before adding a column:
- **Header text is matched verbatim, line breaks included.** MM2's theta column is literally
  `"Time-Decay\nD+1"` and the gamma book has `"Starting\nBalance"` / `"Net Delta\n(Coin)"`. A
  space-joined name silently fails the `missing` check and the command errors out.
- **The header row is per-table** (`header_row`, default 2). The aggregators put headers on row 3;
  the gamma book puts them on row 5.

### `trade_bot`'s two read-only reports

Both are read-and-render only — they place nothing, and both reply as a house-style PNG.

- **`/trade today [all|mm|hedge|gamma|bot]`** — what the desk actually traded today, from the
  **trade service** (not `client_orders`), aggregated per token/category/side with quantity,
  notional-weighted VWAP, notional and **MtM**. Default is **`gamma`**. Categories come from the
  RDS `strategy` tag on each AssetLocation via `src.adapters.refsvc`.
  **MtM** is what the day's trades are worth against the current mark: a buy is
  `mark*qty - paid`, a sell is `received - mark*qty` (what it would cost to buy back). Summed over
  a token's rows this is exactly `mark*net_qty - net_usd` — the same identity `flow_alert_bot`
  marks its residual with, verified live on MM2 ICP to the cent on both the spot and perp leg.
  ⚠️ Marks are CoinRoutes `cost_calculator` mids on the **USDT leg** (`.PERP` for perp rows),
  fetched concurrently — 37 keys in ~2.1s against a 25.5s trade-service fetch, so the column is
  not what makes the command slow. A perp is marked on the perp book: it is a different
  instrument with a different mark.
  ⚠️ A key with **no consolidated book is omitted, never zeroed** — a zero mark would print the
  whole position as a total loss. Those rows show `—` and stay out of the total, and the caption
  says how many are unmarked so a short MtM total does not read as a small one. Four of ~64 keys
  across both desks have none today: `BCHSV`, `TGBP`, `CCN`, `CGLD`.
  ⚠️ A non-USD-quoted row is **still marked in dollars** (an `ICP/EUR` trade is the same asset), but
  a row with no USD rate at all gets no MtM — it already prints `? EUR` in the USD column and must
  not silently become a dollar P&L.
  ⚠️ Ambiguous tickers are pinned to their own venues (`_MARK_MARKETS`), duplicating
  `trade_bot.TOKEN_VENUES` because trade_bot imports this module; a test keeps them in step.
  Unpinned, AI's mid takes a bid from Gensyn and an ask from the unrelated AI.
  ⚠️ Two reasons it is not built on `client_orders`: an **open TWAP** has no `finished_at`, so a
  date taken from `updated_at` drops a days-old order into today's list carrying its *cumulative*
  `executed_qty`; and `client_orders` barely sees the MM book at all — measured on one mm3 day,
  **116** client orders desk-wide against **1,704** fills from just 60 of its 521 MM/GAMMA
  accounts. Filtering `api_user_gamma` was never what hid the MM flow; the data source was.
  ⚠️ `hedge` maps to **both** `HEDGING` and `HEDGERV2`: MM2 carries 1 and 130 accounts of each, so
  a single-tag mapping would report 1/131 of that desk's hedge book. `CUSTODY` is excluded from
  every category including `all`. A location with no category tag surfaces as `?` rather than
  vanishing. Rows are per token/side, not per fill — `all` on a busy desk is >13k fills.
  ⚠️ **`notional` is in the pair's QUOTE currency, not USD.** MM2 trades `POL-IDR`, `*-EUR`,
  `*-JPY`, `*-BRL`, `*-BNB` and `*-BTC`; summing rupiah notionals as dollars reported one MM2 day
  as "$298,921,656 bought" against a true ~$2.8m. Non-USD notionals are **converted** via
  `usd_rate()`, which probes `CCY/USDT` then the inverse `USDT/CCY` across Binance then Kraken —
  venues quote fiat both ways round (Binance lists `EUR/USDT` but `USDT/IDR`), and JPY is on
  Kraken only. The caption prints every rate applied. A currency none of them price shows as
  `? CCY` and stays out of the totals rather than being converted at a guess.
  The **VWAP stays in the pair's own quote currency** and the row is labelled `ICP/BTC` — a
  converted VWAP would print a dollar price that was never quoted anywhere.
  ⚠️ Rates are **spot at reply time**, not at each fill, so a day's converted notional is an
  approximation — fine for "what did we trade", not for P&L attribution.
  Caption counts describe the ROWS SHOWN, not the raw fetch — a token-scoped reply used to report
  the whole desk's fill/account count beside one token's money.
  `bot` keeps the older, different view: the orders **this bot** placed, one line per ORDER.
  Bot orders are identified by `trader_name` == the desk key's own CoinRoutes user
  (`bot_mm3user` / `bot_mm2user`), resolved at startup via `CoinRoutesClient.username()`
  (`GET /api/user`) rather than hardcoded, so it self-corrects if a key is rotated. That is what
  separates them from the `api_user_gamma` quoting engine and from humans on the CoinRoutes UI.
  ⚠️ Treasury runs on the personal `CR_API_TOKEN_G20JPY` (identity `samuel`), so a treasury
  listing also includes manual UI orders — mm3/mm2 are exact.
- **`/trade <token> today [category]`** — the same, scoped to one token (`/trade icp today mm`). The `bot` category still answers the older question: the
  **desk's** discretionary flow on that token today, whoever placed it (bot or a human on the UI),
  across every account in `desk_strategies` — so the gamma spot book **and** the desk perp book,
  whichever the token trades on. An MM3 `/trade icp today` returns ICP **perps** on
  `mm3-futurestrading-ha-j` even though ICP's gamma account belongs to MM2; the `Token` column
  marks perp rows with a `.P` suffix. Only `api_user_gamma` is excluded. Unlike the bare form it
  does not need the bot's identity resolved, so it still works if `GET /api/user` failed.
- **`/trade <token> plot [horizon=30d]`** — candlestick chart of the desk's **discretionary** fills
  on one token (the `api_user_gamma` quoting engine is excluded; it places hundreds of orders a day
  and would bury the flow), with buy/sell triangles whose **area** is proportional to notional. The
  numbers go in the **Telegram caption**, not the image — a monospace block with avg buy/avg sell,
  bought/sold/turnover, net qty and both TWAP legs in bps (the caption is sent with
  `parse_mode=Markdown` and falls back to plain text if Telegram rejects it, so the chart always
  lands). Candle bodies wear the palette's near-greyscale **up/down polarity ramps**, never the
  saturated accents: `buy_green`/`sell_red` are reserved for order flow, which keeps the fill
  triangles the only strong colour on the chart.
  It reports the average buy/sell price and the quantities traded — **no position and no P&L**,
  deliberately. Neither is derivable from a windowed order list: the window starts mid-book, and
  `client_orders` is not a complete record of the book anyway (over BTW's entire life the spot leg
  nets −17.5m tokens while the desk holds +51.7k, because a market-making mandate also receives
  inventory off-book — issuer deposit/loan, OTC, migration). An earlier version walked position from
  zero and read BTW's covered short as an accumulated long into a 10x rally, printing +$1.97m where
  the answer was about −$140k. Position and P&L belong to the tools that see the whole book: the
  `/hedge` risk panel and `scripts/mm3_perf_report`. Do not add them back here.
  `horizon=` is **days only** (`30d`, default 30, capped
  at 730) — deliberately not months, because `trade_bot._num` reads a trailing `m` as *millions*
  for `size=` and a month unit would collide inside the same command grammar.
  **Spot and perp legs are netted on purpose** — a token's gamma spot position and its perp hedge
  are one book, so the netted position and PnL is the number that matters. Which accounts get read
  comes from `trade_bot.desk_strategies(desk)` = the desk's gamma strategies **plus its own perp
  book**. ⚠️ That scoping is load-bearing, not cosmetic: `client_orders(currency=X)` is **not**
  desk-scoped and every desk key can see all ~219 strategies, so without it an MM3 chart of ETH
  silently included MM2's 24 orders on `futurestrading_ha_j` alongside MM3's 18 on
  `mm3-futurestrading-ha-j`. The two desks' account sets are disjoint and a test pins that.
  Markers sit at **fill time** (`finished_at`, falling back to `updated_at`/`created_at`) and the
  read is bounded on `updated_at`, **not** `created_at` — these are passive `smart post` orders that
  rest before filling (one DODO sell was created 27 Jul and filled 5 Aug), so a created_at key put
  markers at the right price on the wrong date, far off the candles, and a created_at window dropped
  orders that actually traded inside it.
  *Known approximations, all inherent to one marker per order:* an order filled in pieces has a
  single `avg_price` but no single timestamp, so it is stamped at completion (for a
  partially-filled-then-**cancelled** order that is the cancel time, which can sit well after the
  fill); and the SOR routes across venues while the candles come from one, so a fill can
  legitimately print outside that venue's range. Together these leave ~5 of 59 DODO markers outside
  the candle range — expected, not a bug. Exact per-fill timestamps would need the child
  `sor_orders`/`sor_trades` records (one extra call per order).
  **Instrument follows the book**, spot vs perp: if most orders are on the `.PERP` pair, candles are
  fetched as the ccxt linear swap `TOKEN/USDT:USDT` and `mark` from `TOKEN-USDT.PERP`; otherwise
  spot `TOKEN/USDT` and `TOKEN-USDT`. Both symbol forms are always tried since coverage differs per
  venue. ⚠️ This is not cosmetic — tokenized equities/indices (`QQQ`, `SPY`, `XAU`) exist **only**
  as perps, so requesting spot returns `BadSymbol` on every venue and the chart came back with no
  candles at all, while `mid("QQQ-USDT")` 404s.

`twap_bot` answers one question: the time-weighted average price of a token over an **explicit**
window. It reduces each candle to a single price (`type=`, default `(O+H+L+C)/4`) and averages those
with **equal weight per candle** — that equal weighting is what makes it a TWAP; volume weighting
would be a VWAP and a different tool. It reports the candle count and flags gaps, so a venue with
missing bars is visible rather than silently biasing the mean.

Two things it gets right that are easy to get wrong:
- **Paged fetches.** ccxt caps a call at ~1000 candles, so a 4-day 1-minute window (5,760) would
  silently truncate and bias the average. It pages until the window is covered, capped at
  `MAX_CANDLES` (20,000) with a clear refusal beyond that.
- **The ambiguous-ticker guard.** It reuses `trade_plot._TOKEN_VENUES`, so `/twap ai … exchange=binance`
  is **refused** — Binance's `AI-USDT` is an unrelated token, and a TWAP taken there would be a
  plausible-looking number for the wrong coin. Windows are always interpreted as **UTC**.

It needs no API credentials at all (public ccxt candles), and it is gated on `TELEGRAM_CHAT_ID`
rather than a desk pin — it is desk-agnostic market data, not a desk report.

### `agg_bot /analysis` — what treasury did on one symbol, and how it went

`/analysis btc 7d`, `/analysis hype 24h`, `/analysis eth 1m` (`h`/`d`/`m`, `m` = 30 days, 1h–90d;
either order; symbol required). Treasury chat only. Logic in `scripts/treasury_analysis.py`;
`agg_bot` only routes. Replies with a written summary, a chart, and an options table when the
symbol has options.

- **Chart**: perp candles (15m up to 2d, 1h beyond, drawn at 4h past 300 bars) with our fills as
  ▲/▼ — one per bar and side, at that bar's VWAP, area ∝ notional; avg buy/sell lines; a dotted
  line per option trade. Below, the symbol's **delta $ from Pluto**: perps, options, and net
  (perps + options + coin holdings). Two panels, never a dual axis.
- **Summary sections**: what we traded (qty, $, avg prices, orders/fills, maker share, fees);
  **P&L split** — `holding = pos0·(S1−S0)` (the starting position riding the move) and
  `trading = Σ q·(S1−p)` (our fills marked at the end price: what the trades added versus just
  holding), plus fees; **execution** vs the window TWAP; **timing** — markouts 1h and 24h after
  each fill, in bp, + = the price moved our way; **worst episodes**; **hedge** — net delta range,
  the costliest cube-to-cube stretch (≈ net Δ$ × return) and the window total; **options** —
  every line opened / closed / expired / resized, bracketed by the two cubes it happened between.
- **Episodes, not orders.** The desk works size through algos, so an order is a 10-HYPE slice and
  the "worst five orders" were five identical −$25 slices of one decision. An episode is a run of
  same-side fills with no pause over 30 min.
- **Sources**: Pluto's **historic cube** (`load_historic_cube`, see `docs/pluto_portfolio_api.md`)
  for positions at both ends, the delta path and option trades; the **trade service** for perp
  fills on the `client=N009-TREASURY_PROP` accounts; public **ccxt perp candles** (venues ordered
  by `trade_plot.venue_order`, which keeps the ambiguous-ticker pins) for prices. Not
  `trade_plot.fetch_candles`: it starts exactly N days back and goes 4h past 7 days, which left no
  bar before the first cube and made a 1h markout meaningless.
- ⚠️ **Reconciliation.** The P&L split is only complete when the fills explain the whole position
  change. A venue whose Pluto position moved differently from its net traded quantity is named
  (`⚠ DERIBIT: position moved 5,310 but fills show 0`) — those trades are missing from the numbers.
  **Deribit fills never reach the trade service**, and some Binance majors (BTC/ETH/HYPE/BNB/PAXG)
  trade outside the tagged accounts (measured 23–30 Sep: 58 of 66 venue legs reconciled).
- ⚠️ **Option PV change is not option P&L** once anything opened, closed or expired — premiums and
  settlements land in cash. ❓ Deribit options (`F1C`/`F2C`/`F1P`/`F2P`) show a *negative* PV on a
  long call, which suggests Pluto marks them net of premium like perps; unconfirmed.
- Prices at a time are the last **closed** bar's close (no look-ahead); fills too recent for a
  markout horizon are left out of it and the count says so. BNB fees are converted at today's rate.
- **Sampling**: a cube per hour exists but each is ~13 MB / 1–14 s, so cubes are sampled hourly up
  to 2d, every 6h up to 14d, daily beyond (4 in flight) — option trades are timed to that bracket.
- ⚠️ **Fills are fetched one day per request.** The trade-service client treats a dropped
  connection as the last page, so one 7-day request returned exactly 10,000 of 19,311 fills with
  only a log warning.
- Runs synchronously inside the poll loop (~1 min for `7d`), after a "⏳" ack — other `agg_bot`
  commands wait until it finishes.

### `botstatus_bot` — are the desk's Pluto trading bots up, and on what parameters?

Desk-aware by chat pin (`TELEGRAM_MM3_CHAT_ID` / `_MM2_` / `_TREASURY_CHAT_ID`, disjoint, no
default — an unpinned chat is refused and told its own chat id). Four views, all format-B PNGs:

| Command | Answers |
|---|---|
| `/botstatus` | the desk roster: Bot / Type / Status / Operational / Conn (Status green `ACTIVE`, red otherwise), with each bot's `lastError` in the **caption** |
| `/botstatus <bot>` | that bot's `Strategy<N>` tabs, one row each: Biprod, Active, Hedge Instr, LP, Max Exec, Resting Δ, Deadband — "what is this leg actually doing" |
| `/botstatus <bot> tabs` | every config tab + its field count (the index for the next command) |
| `/botstatus <bot> <tab>` | that tab's fields: Field / Value / Type |

- ⚠️ **Strictly read-only, deliberately.** Only the two BotManager **Read** endpoints
  `GET /api/BotManager/status` and `GET /api/BotManager/{botName}/config` are wired. The write side
  (`start`/`stop`/`operations`, config writes) is **not implemented, called or imported**, and a
  test scans the source for every URL template built off the BotManager base — a third one fails
  the test — plus asserts no `requests.post/put/delete/patch`. Halting or re-tuning a live trading
  bot from a group chat is a `trade_bot`-class action, so control would need a trader allowlist and
  the owner's sign-off first.
- ⚠️ **The config payload is nested one level deeper than it looks.** The response is
  `{success, botName, secretsRedacted, config}` and **`config` is a mapping**
  (`type_of_bot`/`name`/`class_name`/`tabs`) — the tab list is `config["tabs"]`, each tab
  `{name, fields:[{name, value, type}]}`. Reading `config` as the list yields a `TypeError` deep in
  the renderer; a test pins the real shape.
- **Never render a whole config.** `*TREASURY_PROP_JASONDH` is **58 tabs / 862 fields**
  (`*MM3_JASONDH` 55/845, `*MM3_REGIME_TRADING_CANIO` 57/914, `*TREASURYPROP NOTIFIER` 35/332).
  Every view is a slice, capped at **60 rows**, and a non-zero drop is **always stated in the
  caption** — a silently short table reads as a complete one. Tables over 30 rows fold into
  side-by-side column pairs (Telegram downscales the long side of a tall image).
- `type` is either a scalar type (`Double`, `String`, `Boolean`, `UInt32`) **or the comma-separated
  list of allowed values** for an enum — up to **156 chars** live
  (`STICKYSTRIKE,STICKYSTRIKE2,…`). Values and enum types are truncated per cell.
- ⚠️ **Strategy tabs must be sorted numerically.** The API returns tabs lexicographically, so
  `Strategy10` lands between `Strategy1` and `Strategy2`. Not every bot has them: the treasury
  NOTIFIER and Complex-futures bots have **none**, and those answer with a pointer to `tabs`
  rather than an empty picture.
- ⚠️ **`secretsRedacted` is reported, not assumed.** If a response comes back `false` the caption
  says so — observed `true` everywhere so far, which is exactly when a silent assumption rots.
- **`<bot>` resolves inside the calling desk's roster only** — case-insensitive substring on the
  name with the leading `*` stripped, so `/botstatus jasondh` works and `/botstatus complex` finds
  `*TREASURYPROP Complex futures`. A treasury chat naming an MM2 bot gets a refusal listing its own
  roster. Several matches → the candidates are listed and nothing is fetched; guessing between two
  live trading bots is not this bot's call. An **exact** name wins outright, or
  `*MM2_FAKE_GAMMA_HEDGER` would be unaddressable (its full name is a substring of
  `*MM2_FAKE_GAMMA_HEDGER_OKX`).
- Tab match is case-insensitive exact, then a **unique** substring; an ambiguous or missing tab
  replies with the closest few names (substring hits, then `difflib`), never a 58-line dump.
- Bot names are URL-encoded with `quote(name, safe="")` — every name starts with a literal `*` and
  two contain spaces.
- **The roster is `config/desks.yaml` → `<desk>.bots`**, not a name glob and not a list in the
  script: only listed bots are shown. A rostered bot the API does **not** return still gets a row
  marked `MISSING` — dropping it would hide exactly the failure this bot exists to catch.
- Names are the API's exact `name` values, **leading `*` included** (that star is a real character
  marking a Firm-owned bot, not a wildcard; it is stripped for display only). Some names contain
  **spaces** (`*TREASURYPROP Complex futures`) — never assume otherwise when building a query.
- **One request per desk, not per bot.** The `name=` glob is derived from the roster's common
  prefix + `*` (`*MM3_*`, `*TREASURY*`), then the reply is filtered against the exact roster. ⚠️ A
  repeated `name` param does **not** work — the API honours only the first value (verified: sending
  `*MM3*` and `*MM2*` returned MM3's two bots only). A roster with no shared prefix degrades to
  `*` (~172 firm-wide bots, ~0.5s), which is wider than needed but never wrong.
- `include_inactive=true` is sent explicitly even though it is the server default — a default flip
  would hide precisely the dead bots the table is for.
- `lastError` is free text and routinely a **multi-line stack trace**, so caption lines are
  whitespace-collapsed and capped at 200 chars each, with the whole caption capped at Telegram's
  1024 — an over-long caption is rejected and the photo would arrive with none at all.
- **Every** fetch (status and config alike) is dispatched on a daemon thread and wrapped in
  `run_with_deadline` (60s), the `vol_bot` shape: a stalled Pluto call must never make the poll
  loop go deaf. Heartbeat every 5 min.

### `log_bot` — a trader logs a manual trade into a sheet

`/log <buy|sell> <token> <qty> <price> [trader] [trade time]` appends one row to `LOG_SHEET_ID`. It
exists because the desk already types these trades into chat as prose ("sold 100 ETH @ 1905.2"); the
same keystrokes prefixed with `/log` become a row instead of a message someone transcribes later.

```
/log buy eth 100 1905.2                        ->  100 ETH, traded now, trader = you
/log buy eth 100 1905.2 sam                    ->  the same, logged on sam's behalf
/log sell eth 10000usd 1905.2                  ->  $10,000 of ETH  (= 5.2488 ETH)
/log sell eth 100 1905.2 09:35                 ->  traded 09:35 UTC today
/log sell eth 100 1905.2 sam 2026-09-06-09:35  ->  backdated, for sam
```

- **Trade time and log time are separate columns** (`trade_ts_utc` / `logged_ts_utc`). A trader
  logging an hour late must not produce a row that looks like an hour-late trade; the optional
  trailing time says when it actually happened, and defaults to now. A bare time in the FUTURE is
  refused (at 10:30, `23:00` means last night) — say it with a date instead.

- **Grammar is strictly positional**, but the **trader defaults to the sender's Telegram username**
  — logging your own trade costs one word less. Naming someone else still works and is how "logged
  on behalf of" stays visible; both columns are always stored (`trader` and `logged_by`), and when
  the trader is defaulted they are equal by construction. A sender with **no** Telegram username
  has nothing to fall back on, so the command is refused rather than writing an ownerless row.
- **Both trailing fields are optional, and a lone one is disambiguated by SHAPE** — a colon or a
  leading `Y-M-D` makes it a time, anything else a trader. Deliberately not "try to parse it as a
  time, else treat it as a name": `parse_trade_time` rejects a bare FUTURE time, so that fallback
  would quietly file `23:00` as a trader called `23:00`. Shape-testing keeps the refusal.
- **`qty` is TOKENS unless suffixed `usd`.** Only the suffix decides; magnitude never does, because
  `10000` is a plausible token count for a cheap coin and a plausible dollar amount for an
  expensive one, and the two differ by the price. The row stores `qty` (tokens),
  `notional_usd` **and** `entered_as`, so it can never be misread later.
- The reply echoes what was **stored**, not what was typed, so a mis-parse is visible immediately.
- Gated on `TELEGRAM_CHAT_ID` (the shared market-data allowlist). Each row records the desk the
  chat maps to, or the raw chat id when it maps to none.
- Auth is the **shared editor SA** (`.mm2_trades_gsa_key.json`,
  `mm2-trades-agg@lucky-starlight-505208-u0.iam.gserviceaccount.com`). A new sheet needs *sharing*
  with that address, not a new key. The header row is created if the tab is empty.
- The sheet write runs on a thread under a 60s `run_with_deadline` ceiling; a write that fails or
  times out says **NOT logged** rather than leaving the trader unsure.
- It only ever **appends** — it never edits or deletes an existing row.

## Run & operate

- **The bots run as `systemd --user` units: `desk-bot@<name>.service`** (name ∈ agg, orders, hedge,
  trade, jason, vol, macro, sentiment, ls, flow_alert). They have `Restart=always`, so **do NOT
  `kill` them** —
  systemd respawns them, and a manual `nohup` launch creates a *duplicate poller* (→ Telegram 409).
  Use systemd:
  - Restart on new code / env: `systemctl --user restart desk-bot@<name>` (units run `uv run python -m
    scripts.<name>_bot` from the repo, so they pick up the current working tree — no reinstall needed).
  - Status / logs: `systemctl --user status desk-bot@<name>`, `journalctl --user -u desk-bot@<name>`,
    or the app log `logs/<bot>/<bot>.log`.
  - After a crash-loop (e.g. a broken `.venv`), clear it with `systemctl --user reset-failed
    desk-bot@<name>` then `restart`.
- **Manual run (dev only):** `uv run python -m scripts.<bot>` from repo root (module mode — script
  mode breaks `import src`). Stop the systemd unit first or you get two pollers.
- **One poller per BotFather token** (409 on a duplicate). `trade`/`jason`/`vol` also hold a
  single-instance `flock` (`/tmp/mm3_*.lock`) that refuses a second copy — the other bots rely on the
  one-systemd-unit-per-bot invariant, so never hand-launch them alongside systemd.
- **`mm2_trade_log_bot`** (mm2 fill→sheet daemon) is **not** a systemd unit here — run/monitor it
  separately.

### Supervision (optional)

Nothing below is installed or enabled by default — a crashed bot still needs a manual restart
unless you opt in. `deploy/systemd/desk-bot@.service` is a templated systemd **user** unit; `%i`
is the bot name, so one file covers all 9 (`desk-bot@jason.service`, `desk-bot@trade.service`, ...).

- **Install/enable one bot** (example: `jason`):
  ```bash
  cp deploy/systemd/desk-bot@.service ~/.config/systemd/user/
  # edit ~/.config/systemd/user/desk-bot@.service: set WorkingDirectory to this repo's checkout path
  systemctl --user daemon-reload
  systemctl --user enable --now desk-bot@jason.service
  ```
- **`Restart=on-failure`, deliberately not `Restart=always` — the one trap.** `trade_bot`'s
  single-instance-`flock` conflict and Telegram's 409-on-duplicate-poller both make the process
  **exit 0** (a clean exit, not a crash). With `on-failure`, systemd correctly leaves the unit
  stopped in that case. With `always`, systemd would restart-loop against the flock (or
  hammer Telegram with duplicate-poller requests) forever. Don't flip this without understanding
  why it's `on-failure`.
- **Interaction with the manual-restart flow:** if a bot is under systemd, restart it with
  `systemctl --user restart desk-bot@<name>.service` instead of the manual
  kill-then-`uv run` dance above. If it isn't under systemd, the manual flow still applies
  unchanged — systemd supervision is per-bot opt-in, not all-or-nothing.
- **Status/logs:** `systemctl --user status desk-bot@<name>.service` and
  `journalctl --user -u desk-bot@<name>.service -f`. Bots still log to `logs/<bot>/<bot>.log`
  regardless of how they were launched — that file doesn't depend on systemd.

## Conventions every bot shares

- **Long-poll skeleton:** shared plumbing lives in `src/common/telegram.py`
  (`drain_backlog(api)` on startup so a restart doesn't replay old commands, then
  `poll_updates(api, offset)` per loop iteration — both handle the `RequestException` backoff
  internally); match the first token case-insensitively and strip `@botname`
  (`text.split()[0].split("@")[0].lower()`); reply with `telegram.send` / `telegram.send_photo`.
  Every handler is wrapped so a failure is reported over chat and the loop never dies.
  `macro_bot`/`sentiment_bot` are the deliberate exception: they keep their own local `_send`
  (HTML `parse_mode` via `_md_to_html`, `TG_MSG_LIMIT` truncation, plain-text fallback on a
  non-200) because `telegram.send` doesn't implement HTML or truncation — this is intentional,
  not an oversight; only their poll loop is on the shared helpers. `vol_bot` also keeps its own
  local `_send`/`_send_photo`/`_send_doc` for the same reason (its breach-alert broadcasts use
  real `parse_mode="HTML"` formatting, e.g. `<b>`/`&gt;`) — only its poll loop was migrated.
  `trade_bot` is untouched by this migration (money-moving; out of scope) — it keeps its own
  `_send`/`_send_photo` and its own hand-rolled poll loop. That loop mirrors `poll_updates`'
  two guards deliberately: it logs an `ok:false` body (a 409 duplicate poller makes the bot go
  silently deaf) and swallows a non-JSON body (a Telegram 5xx HTML page would otherwise crash
  the loop and exit the process).
- **Chat allowlist:** `TELEGRAM_CHAT_ID` is a **shared, comma-separated** list of MM3 chat ids
  (all bots read it). A chat not in the allowlist is refused.
- **Chat-aware desk pattern** (agg / orders / hedge / trade): the desk is chosen by *which chat*
  sent the command, from **explicit per-desk chat pins — no defaulting**.
  ```python
  mm3 = {c for c in require_env("TELEGRAM_MM3_CHAT_ID").split(",") if c}
  mm2 = {c for c in (os.environ.get("TELEGRAM_MM2_CHAT_ID") or "").split(",") if c}
  if mm3 & mm2: raise RuntimeError("a chat is pinned to two desks — fix .env")
  desk_by_chat = {**{c: MM3 for c in mm3}, **{c: MM2 for c in mm2}}
  desk = desk_by_chat.get(chat_id)   # None -> refuse. Never defaults to a desk.
  ```
  A chat is a desk **only** if pinned in that desk's own var (`TELEGRAM_MM3_CHAT_ID` /
  `TELEGRAM_MM2_CHAT_ID` / `TELEGRAM_MM4_CHAT_ID` / …), and the vars must be **disjoint** (checked
  at startup with `itertools.combinations`, so a further desk needs no new comparison code). This is the safety
  boundary: an unpinned chat is refused, so putting a chat in the market-data allowlist can never arm
  `trade` for it. `TELEGRAM_CHAT_ID` is now **only** the market-data-bot allowlist (jason/ls/vol/
  macro/sentiment) — it no longer signals a trading desk.
- **MM4 is spot gamma only.** Its `config/desks.yaml` block has **no `perp_strategy`**, because the
  desk does not hedge in a futures book. That None is load-bearing: `trade_bot.resolve` refuses an
  off-roster token (and the `perp` keyword) instead of falling through to another desk's hedge
  account, `orders_bot` refuses `/orders hedge` in the MM4 chat, and `desk_strategies` drops it from
  the account scope. MM4 also rides **MM3's** CoinRoutes key and trader allowlist — `bot_mm3user` is
  already in all 15 strategies' `users` lists, so no CoinRoutes change was needed. Its orders
  therefore carry `bot_mm3user` as `trader_name`, same as MM3's; desk separation comes from the
  strategy set, not the key.
- **House "format-B" table PNG** (all rendered bots): matplotlib `Agg`, Liberation Sans; ruled style
  = horizontal rules only, gray body, **color the text not the cells** (sell red / buy green, dark
  bold for the key column, a top rule + bold on a totals row). `bbox=[0,0,1,1]` to fill the axes;
  split tall tables into 2 columns (Telegram downscales the long side of a tall image, shrinking text).
- **Secrets:** configs name the env var only; `require_env` auto-loads repo-root `.env` (gitignored).
  Never hardcode a token/key; never commit `.env` or a `*.gsa_key.json`. **`.env.example` (repo
  root) is the enumeration and sourcing authority for every env var** — the roster table above is
  only the bot→var *routing* map; a new var goes into `.env.example` first, then gets a row here.
- **Read-only vs write:** every bot except `trade_bot` is read-and-render only (and
  `mm2_trade_log_bot` writes only its sheet + state db). Only `trade_bot` places orders — and
  only via `/trade <token> <buy|sell> …`; its `/trade today` and `/trade <token> plot`
  subcommands are read-and-render, dispatched before the order path is ever reached.

## External data sources & the gotchas that cost real time

- **CoinRoutes SOR API** (`src/adapters/coinroutes/`, host `g20jpy.coinroutes.io`; **one key per
  desk** — `CR_API_TOKEN_MM3` / `CR_API_TOKEN_MM2`, treasury on `CR_API_TOKEN_G20JPY`; the adapters
  take an explicit `api_token`, and each desk-aware bot passes its desk's key):
  - `client.py` is **read-only** (positions, balances, trades, orders, `cost_calculator`);
    `trade.py::CoinRoutesTrader` is the **only** write path (create/cancel client orders).
  - **Create-order matches the strategy by its exact `name` (case-sensitive), NOT the slug.** Names
    are inconsistent (mostly UPPERCASE, some mixed/lowercase). Hardcode exact names — see
    `trade_bot.STRATEGY` and `mm2_trade_log_bot.MM2_STRATEGIES`.
  - **Order-book / mid / touch price:** use `cost_calculator/{symbol}` (consolidated book across all
    venues) — **not Binance** (many gamma micro-caps don't trade there). Helpers on the client:
    `mid`, `touch_price`, `top_of_book_size`, `book`, `cost`.
  - `sor_trades` has **no strategy filter** — fetch a time window and filter client-side.
  - **`client_orders` is paginated and unbounded** — a bare `currency=X` query walks the token's
    *entire* history and times out (60s/request) for high-volume tokens. Always bound reads with
    `updated_at_after` / `created_at_after` + `ordering`. (This is why `orders_bot.recent_fills`
    uses a 30-day window.)
  - `pct_executed`/`avg_price`/`executed_qty` (base-token) / `net_consideration` (USD) live on the
    client-order record; a partially-filled order that was later cancelled still carries its fill.
    Because these are already the order's fills aggregated, **one row per order is the fill-level
    view collapsed** — no need to walk `sor_orders`/`sor_trades` to report what an order did.
  - **`trader_name` on an order is the identity that placed it** — `api_user_gamma` for the MM
    quoting engine, a person's name for a CoinRoutes-UI order, `bot_mm3user`/`bot_mm2user` for the
    desk bot keys. This is the only reliable way to separate bot flow from engine flow from human
    flow: they all share the desk key and the same strategies. `client_tags` is **read-only** (not
    settable on POST) and `client_order_id` is a bare uuid4, so neither can be used to tag orders.
  - **There is NO historical-candle endpoint.** `/api/market_data/properties` is the only
    market-data path; `cost_calculator` is a live book snapshot, not history. Price history for a
    chart must come from elsewhere — see the venue note below.
- **One asset, several currency codes** (`trade_plot.TOKEN_CURRENCIES` / `canonical_token`): when a
  venue's own ticker collides, CoinRoutes gives our asset a different symbol there. Gensyn is
  `AI-USDT` on okex/kucoin/bitget and **`AIGENSYN-USDT` on Binance** — same asset, same gamma
  account (`mm-c420_ai-gamma-j`), two currency codes. Every read that filters `currency=` must
  query the whole group or a venue's leg silently disappears: `/orders ai` showed 2 of 3 open
  orders and missed the Binance leg, which held **3.3m tokens vs 1.2m** across the other three
  venues. `TOKEN_CURRENCIES` drives `orders_bot.read_orders`/`recent_fills`,
  `trade_bot.today_orders` and `trade_plot.fetch_orders` (deduped on `uid`), and
  `canonical_token` folds the alias back onto the desk token so the whole-book `/orders` view
  groups both legs under `AI`. `token_currencies` resolves through the canonical token, so asking
  for either symbol reports the entire asset.
  Both legs are separately tradeable and **pinned to opposite venues** — `AI` to
  okex/kucoin/bitget, `AIGENSYN` to binance — with their own `DEFAULT_QTY` (200k / 300k).
- **Ambiguous tickers** (`trade_plot._TOKEN_VENUES`): some symbols are used by **unrelated tokens
  on different exchanges**, so a price from the wrong venue is a wrong *asset*, not a stale quote.
  `AI` is **Gensyn**, traded on okx/kucoin/bitget/mexc (~0.021); binance/gate/htx list a different
  `AI` (~0.018) whose 4h returns correlate only **0.28** with ours. `_TOKEN_VENUES` is a hard
  allow-list that overrides both the routing hints and the venue cache, so an AI candle can never
  come from Binance. Add a token there the moment its ticker is found to be ambiguous, and record
  which asset is ours in `config/desks.yaml`.
  ⚠️ **This is a live money rail, not cosmetics.** The AI gamma strategy on CoinRoutes has
  **BINANCE and HUOBI configured alongside okex/kucoin/bitget**, and on those two `AI-USDT` is the
  *other* token (ours on Binance is `AIGENSYN-USDT`). Since `exchanges` and
  `market_data_exchanges` both default to "all exchanges defined by the strategy", an unpinned
  order could **execute against the wrong coin**. `trade_bot.TOKEN_VENUES` pins ambiguous tickers
  on three legs:
  - `exchanges` — where the order may execute
  - `market_data_exchanges` — where smart-post sources its book
  - `cost_calculator?markets=OKEX;KUCOIN;BITGET` (UPPERCASE, semicolon-separated) — so the default
    limit and the token-quantity cap check are computed off the right coin
  Measured on a live `/trade ai buy`: unrestricted touch 0.0181 -> limit 0.0183 (13% below the real
  market, and the cap valued 200k AI at $3,620); pinned touch 0.020823 -> limit 0.021031 and $4,165.
  The durable fix is **CoinRoutes-side** — remove BINANCE/HUOBI from that strategy's book sources in
  the UI; the rail is defence in depth either way.
- **Historical candles for gamma micro-caps** (`scripts/trade_plot.py`): only **8 of 31** MM3 gamma
  tokens trade on Binance, and Pluto's `TimeSeries` is Binance-backed so it covers the same 8. The
  other 22 are reachable via **ccxt** (MEXC 18, Gate 16, KuCoin 14, Bitget 10, Bybit 9) — 30 of 31
  total; only TGBP has no venue anywhere. Two rules that make this usable from a bot:
  - **Never call `load_markets()`** — it is the multi-second part. Construct `TOKEN/USDT` directly
    and let `fetch_ohlcv` raise if the venue doesn't list it, then try the next venue (1.7–7.5s/token).
  - **Prefer the venue the orders actually routed to** (the order record's `exchanges` list;
    `exchange_summary_stats` is usually empty), else a preference chain. Plotting MEXC candles for a
    Gate fill leaves the markers floating outside the bar range. Map CoinRoutes names to ccxt ids
    (`gateio`→`gate`, `huobi`→`htx`).
- **Pluto hosts and the fallback.** Every Pluto GET goes through `src.common.pluto.get`, never
  `requests.get`: it calls the primary (`pluto-lon.g20digital.io:667`), and on a timeout,
  connection error or 5xx retries once on `pluto.g20digital.io:667` with `plutoService=local`
  added, then keeps using the fallback for 10 min. The primary gets at most 30s; the fallback gets
  the caller's own timeout. A 4xx is returned as-is. Why: on 1–2 Oct the primary timed out 23 times
  overnight (vol_bot, agg_bot) and 500'd `jason_levels_v2` once, and each of those calls just
  failed. A fallback is logged as `Pluto primary failed on <path> (<reason>)`.
  ⚠️ `/api/jsonrpc` (historic cube, `/analysis`) answers **401 on the fallback**, so it posts to the
  primary only. ⚠️ The fallback is a separate instance: values match closely but not exactly
  (JASON vol 35.14% vs 35.03% on the same pair, same minute).
- **Pluto JASON levels API** (`PLUTO_API_KEY`/`PLUTO_API_SECRET`, `alt_format=true`, `ccypairs`):
  ±2σ targets `targetPricUp2StdevsFromLowPivot` / `targetPriceDown2StdevsFromHighPivot`, pivots,
  z-scores, vols. **Chunk requests at ~45 pairs** (the API stalls on large batches). Coverage is a
  **subset** — many small tokens return nothing (degrade to `—`, never fail the table).
- **Pluto BotManager API** (`https://pluto.g20digital.io/api/BotManager`, same
  `PLUTO_API_KEY`/`PLUTO_API_SECRET`): `GET /status` returns `{success, count, items[]}` with
  `name, runtimeName, ownerType, type, status, operationalStatus, connectionStatus, lastError,
  hasCube`. `status` is `ACTIVE`/`NOT_LOADED`; `operationalStatus` is free text
  (`PV:-80k@12:43 0 ACTIVE`, `IDLE; last=…`, `N/A`, or empty) — parse it at your peril, print it.
  Query params: `name`/`type` (globs, `*`/`?`), `include_inactive`, `force_refresh`. Per-bot reads
  `/{botName}/last-error`, `/connection-status` and `/config` also exist — ⚠️ `/config`'s tab list
  is nested at `config["tabs"]`, and `config` itself is a mapping (see `botstatus_bot` above).
  Bot names need `quote(name, safe="")`: they start with a literal `*` and some contain spaces.
  **The write endpoints are off-limits to the bot family** — see `botstatus_bot` above.
- **Google Sheets** (`gspread` service account): one SA JSON key per purpose, shared as Viewer
  (read) or Editor (write). `agg_bot` uses a readonly key; `mm2_trade_log_bot` an editor key. A
  single SA reads/writes any sheet shared with its email — no new SA per sheet.
- **Anthropic API** (`macro_bot`, `sentiment_bot`): `ANTHROPIC_API_KEY`; default model
  `claude-opus-4-8`.

### The two alert daemons

Both are **push-only**: no `getUpdates`, no BotFather token of their own, reusing
`TELEGRAM_BOT_TOKEN` to `sendMessage`. Single-instance flock each. Both alert in **one line** —
they are read on a phone lock screen, so wrapping defeats the point.

#### `flow_alert_bot` — market-making flow accumulating delta

```
🟢 MM3 LAPTOP +$9,860 @ 0.949446
10,385 LAPTOP · buy 0.838652 / sell 0.815407 · ref 0.8266 · MtM -$1,276 · last 15m
mm -$32,128
gamma +$41,994
day +$45,384
```

Headline carries the net and the price the residual went on at; then one line per book; then the day. A single line
held the same figures but ran them together and was hard to scan. Every figure is in full
dollars — the `k` abbreviations only existed to fit one line.

- Alerts on the **NET** of MM and hedge legs: MM2 auto-hedges its own flow, so MM-only figures
  alerted on delta the hedger had already taken off. Measured on POL over 15m — net −$8,905 was
  MM −$63,692 against a hedger returning +$54,787.
- The **mm/gamma/hedge decomposition** is shown on **every** alert, including when one book is
  the whole net. A line that appears only sometimes cannot be told apart from one that failed to
  render, and "all of it is mm" is itself worth knowing. Only sub-dollar dust is dropped.
  The category is the RDS `strategy` tag on the ACCOUNT the fill landed in, **not** the trader's
  intent — see the roster notes on `desk_trades`.
- ⚠️ **Detection is ~3s** (was ~40s). The interval was never the bottleneck: `fetch_fills` ran
  its 16 chunked reads serially at ~0.9s each — 14.6s per cycle for both desks. They now run
  concurrently (2.6s), which is what made a 5s poll meaningful. Same request count, same load
  per cycle, but **5x the request rate** on the trade service (0.64 → 3.2 req/s), so
  `FLOW_ALERT_POLL_SECONDS` exists to back it off without a deploy.
- The quantity stays (a hedge is sized in tokens) and `ref` — the most recent fill's USD price in
  the window — is the level the flow is happening at, which is what sizes a hedge.
- The **headline price** is the net one (`net_usd / net_qty`) — the break-even of the residual
  position, which is exactly what `MtM` on the second line is measured against
  (`mtm = ref * net_qty - net_usd`, the equivalent exact form).
- ⚠️ **That headline price is LEVERED by the gross-to-net ratio and is not a level anyone traded
  at.** The example above is the real case: bought 59,883 @ 0.83865, sold 49,498 @ 0.81541, net
  10,385 — **9.5% of the 109,381 gross**. That ~10x netting amplified the 2.8% gap between the
  side VWAPs into **0.949446**, against a 0.83 market and a true VWAP of 0.82813. Not a bug —
  that is genuinely where a small residual out of large two-way flow breaks even — but it reads
  like a market price and has been queried as one.
  Guarding it by size cannot work: `abs(net_qty) < 1e-9` rejects only a net of *zero*, never one
  merely small against the gross. What makes it safe to show is the **side VWAPs beside it**,
  which are real traded averages whatever the netting does. The two are meant to be read together.
- `MtM` has no such failure mode — it is exact however the netting falls out.
- The gap between the two sides is the spread being captured, which is worth seeing in itself.
- Only the fill count stays log-line-only now.
- ⚠️ Prices go through `_px`, never a bare `:g`: that printed a sub-cent token as `1.23e-05` and
  anything over a million as `1.23457e+06`. Same no-scientific rule as `_qty` and trade_bot's
  `_qty_str`.

#### `move_alert_bot` — DISABLED (2026-09-07)

⚠️ **Not running.** `systemctl --user disable --now desk-bot@move_alert` — the traders did not
want it. The code and tests are kept because the findings below were expensive to establish and
would have to be rediscovered; re-enable with
`systemctl --user enable --now desk-bot@move_alert`.

Before re-enabling, note the two calibration problems it hit: the first threshold could never
fire, and the first arming logic sent four alerts for one move. Both are described below and
both are fixed in the code — but neither was validated over more than an hour of live traffic,
so the noise level is still unproven.

What it did: alerted when a token moved while the desk carried delta in it.

```
🔴 MM3 NYM -3.2%/5m · Δ +$146.9k · MtM -$4.7k · 0.03012 → 0.029156
```

- Alerts on **dollar impact** (`delta_$ × move_%`), not on percentage — a 5% move on a flat book
  stays silent, a 1% move on $2m of delta does not.
- ⚠️ The arrow follows the **impact, not the price**. A fall is good news on a short book; the
  alert says "this is costing you", not "this went down".
- **Delta** comes from the desk's aggregator sheet (`Net Delta ($)`, read by header name so an
  inserted column cannot shift it), re-read every 2 min — it moves slowly. **Price** is
  CoinRoutes `cost_calculator` mid, fetched concurrently (12 mids in 0.99s) and **only for
  tokens whose delta clears `MIN_DELTA_USD`** — 4-9 symbols a poll, not the 35-48 coin sheet.
  That floor is what makes the CoinRoutes cost affordable.
- The move is measured against the oldest sample still inside the window, from the daemon's
  **own** price history — no dependence on anyone's session-open or previous-close mark.
- ⚠️ **Both thresholds are sized against the real books.** Largest deltas are ~$147k (MM3) and
  ~$438k (MM2), so a 1% move is worth only $1.5k / $4.4k: a $10,000 threshold would have needed
  a 7% move on MM3's biggest position and **never fired**. Default is $2,500, plus a
  `MIN_PCT` floor of 0.5% — the impact test alone scales the wrong way, since a bigger book
  trips on a smaller move and would alert on noise.
- ⚠️ **One move, one message.** After alerting, a token stays quiet for `COOLDOWN` (15m, longer
  than the window) unless the move **extends** to `ESCALATION` x the impact already reported;
  state resets only when the move is genuinely over (back under `MIN_PCT`). The first attempt
  re-armed whenever the impact decayed to half the threshold and re-alerted on any sign flip —
  both wrong for a ROLLING-window measure, where the reference advances every poll so the impact
  oscillates and `pct` crosses zero mechanically. Observed cost: NYM chopping ±2–3% sent
  **four alerts in eight minutes** (−4,469 / −3,099 / +2,968 / +3,310). Sign is now ignored
  entirely; only magnitude and time matter.
- ⚠️ Several sheet rows have **no `-USDT` book**: fiat legs (`CAD`/`AUD`/`CHF`/`JPY`) and
  non-instrument rows (`EXCHANGE`, `MM2GAMMA`). They 404 once, are logged, and then skipped for
  the process lifetime — retrying each poll would burn a call forever to learn the same thing.
  `EUR` *does* price, and MM2 genuinely runs FX delta, so it stays in.
- Ambiguous tickers are restricted to their own venues (`TOKEN_MARKETS`, mirroring `trade_bot`) —
  unrestricted, `AI-USDT`'s mid mixes two different assets.

### `trade_bot` — `quote=` picks the spot pair's quote currency

`/trade <token> <side> size=5k quote=usd`, and on the ladder
`/trade <token> 0.26x5k 0.22x5k quote=usd`, and `/scalper <token> 5k 2 quote=usd`. Default is
USDT, so nothing existing changes.

The same asset can trade against several quote currencies **on different venues**: LAPTOP-USDT
prices on KuCoin only, LAPTOP-USD on Kraken only. They are separate books with their own touch,
mid and resting orders, so every price and cap lookup uses the resolved pair rather than assuming
USDT. Verified live — the two produce different default limits from different venues.

- ⚠️ **Only dollar-equivalents are accepted** (`QUOTE_CURRENCIES = USDT/USD/USDC`). Every cap in
  the module (`max_usd`, `LADDER_MAX_USD`) compares a raw quantity against a dollar figure, so a
  EUR quote would be capped ~16% loose and a BTC quote would wave through 50,000 BTC. A token that
  genuinely needs a non-dollar quote needs the **caps converted first**, not this set widened.
- ⚠️ `quantity_currency` follows the pair. Sending USDT against a `-USD` pair is a wrong order,
  not a cosmetic mismatch — and the USD cap keyed on `qcur == "USDT"`, so a USD-quoted order would
  have skipped it entirely. Both are pinned by tests.
- ⚠️ `quote=` is **refused on perps** — the desk futures book is USDT-margined, and silently
  ignoring it would place the wrong pair.
- On the ladder, `quote=` is stripped before `parse_levels` (which would reject it as a malformed
  level) and the whole ladder goes to ONE pair: the two books have different mids, and sides are
  inferred from mid, so a ladder spanning both would be split against the wrong reference.
- The confirmation names a non-default pair (`buy $5000 [LAPTOP-USD] …`); otherwise the same
  ladder placed on both books produces identical replies.

### `trade_bot` — `/jwap`, a JASON-gated TWAP

`/jwap <token> <buy|sell> size=<50k|250000tok> over=<90m|6h|2d> [every=5m] [sigma=1.5] [age=1.5d]
[perp] [staged] [quote=usd]`, plus `/jwap status` and `/jwap stop <#id|token|all>` (`cancel` is
an alias; both scoped to the chat's desk). Each parent gets a `#id`, never reused (the next id
comes from the record folder, so a restart does not reset it), so one of several parents on the same token can be stopped alone; a bare `/jwap stop` lists the
running parents instead of guessing. Same chat → desk and trader-allowlist gate as `/trade`. Rule and study:
`docs/jwap_rule_spec.md`; pure logic: `src/signals/jwap.py`.

- **Children:** `over / every` of them (every = whole minutes, over an exact multiple), at
  `entry + i·every`, entry = the next whole minute. Each is `target / n`, in the target's unit.
- **The gate:** a buy fires iff `mid ≤ jsonk_dn` and `days_dn ≥ age`; a sell iff `mid ≥ jsonk_up`
  and `days_up ≥ age`, with level/age from the last **closed** 1h bar (`knowable_on_minutes`). A
  closed-gate child is **skipped, never caught up** — `size=` is a ceiling; the study's sell side
  filled ~7–16% of children. A child due **more than 60s late** is skipped too.
- **Order path:** the full target is capped once at start (`resolve` + `build_order`: 3× per-token,
  desk `jwap_max_usd`; a token target also at its touch notional) and refused whole if it breaches.
  `jwap_max_usd` is $50k on every desk except **OTC ($1m)**; a USD child over the desk's per-order
  `max_usd` ($50k) is refused at start, since `prepare_order` would refuse every slice. Each
  child is an internal `/trade <token> <side> size=<child> aggressive [perp] [staged] [quote=]`
  through `prepare_order` (every /trade rail), then placed under **our own** `client_order_id`
  `jwap-<id>-<entry>-<slice>`. `aggressive` (limit 1% through the touch) stands in for "market".
  A child **still working 5 minutes after placement is cancelled**; what it did not fill is left
  over, never re-sent.
- **Prices:** the gate mid is CoinRoutes `mid_on` on the strategy's own venues (`price_venues`).
  **Only the JASON candles come from Binance** — USD-M perp 1h klines for `<TOKEN>USDT`, ~9 days,
  forming bar dropped — recomputed once per new hourly bar. No Binance perp / no klines at start →
  refused. A mid or kline failure mid-run (after retries) → that child's gate is **closed**.
  ⚠️ A token pinned by `TOKEN_VENUES` to non-Binance venues (`AI`) is refused: Binance's `AIUSDT`
  is a different coin, so its levels would be wrong.
- **Timing:** no threads. `poll_jwaps` runs next to `poll_pending`, and while any parent is active
  the getUpdates long-poll drops from 25s to **5s**. ⚠️ The first `/jwap` per process spends ~5s
  compiling JASON's numba kernel, during which the bot is unresponsive.
- **Messages:** a multi-line start message (`#id`, target and slice size, the level and age
  condition, price now → trading/waiting, how to stop), **one summary per hour** (fired/elapsed, filled so
  far from the children's last CoinRoutes records), and a final `✅ JWAP #id SOL done · filled … of …
  (N%) · k of n slices · avg …` once every child is final. The first child failure is sent once;
  later ones are counted in the summaries. Children are **not** in the per-order fill tracker.
- **Stop:** no new slices; working children are cancelled at once; the final message follows when
  they are final (`🛑 … stopping` until then).
- **CoinRoutes retries:** reads (mid, `prepare_order`'s prices, `get_order`) and cancels retry a
  timeout, connection error, 404, 408, 425, 429 or 5xx after 1, 2 and 4s. A **place is never
  retried blind**: after a transient error CoinRoutes is asked for our `client_order_id`, and the
  order is re-sent (same id) only if it is not there. A 400 is a real rejection: recorded, not
  retried. Retries block the poll loop for at most a few seconds each.

#### The record: `data/state/jwap/<id>.jsonl` — the truth after any crash

One append-only JSON line per event, fsync'd **before** the bot acts on it (`scripts/jwap_record.py`).
A parent whose record cannot be written is refused at start. Events: `created` (command, who,
parameters, arrival mid / level / gate), `jason` (every hourly row the gate reads, both sides),
`slice` (**every** scheduled slice: due time, mid, level, pivot age, result `fired` /
`gate_closed` / `no_price` / `jason_failed` / `late` / `bot_down`), `intent` (written before the
order is sent, with its full parameters), `placed` / `failed`, `retry`, `fill` (the full
CoinRoutes order record each time its status or executed quantity changes), `cancel`, `stopped`,
`resumed`, `done` (filled, average price, left over).

- **Restart:** at start-up every parent without `done` is rebuilt from its record. An `intent` with
  no answer is settled by asking CoinRoutes for its id (`placed` with `recovered: true`, or
  `failed` at stage `restart`), so a slice is never placed twice or lost. Slices that fell due while
  the bot was down are recorded as `bot_down` and not caught up. The chat gets `♻️ JWAP #id … resumed`.
- ⚠️ **Never leave test or hand-made files in `data/state/jwap/`**: the bot resumes every
  unfinished record there as a live parent. Tests are sandboxed by `tests/conftest.py`.

### Desk shapes — three, not one

`config/desks.yaml` carries three different kinds of desk, and which keys a block has is what
decides how `trade_bot.resolve` routes a token:

| Shape | Desks | Keys | A token resolves to |
|---|---|---|---|
| Gamma roster | mm3, mm2, mm4 | `strategy` (+ `perp_strategy`) | its own gamma-j book, spot |
| Single perp book | treasury | `perp_strategy`, `strategy: {}` | `<TOKEN>-USDT.PERP` on that one strategy |
| Single spot book | **otc** | `spot_strategy`, `strategy: {}` | `<TOKEN>-<quote>` on that one strategy |

- ⚠️ **`spot_strategy` is what makes OTC spot.** Without it, an unrostered token falls through to
  the perp branch and would place `BTC-USDT.PERP` on `OTC-C04.2-HA-J`, which is a **spot** account
  (`is_margin: false`). A test pins that the OTC pair never ends in `.PERP`.
- ⚠️ The OTC book executes on **BINANCE and KUCOIN only**. A token listed on neither has no book on
  the allowed venues and is refused — never silently routed elsewhere.
- OTC has no `perp_strategy`, so `/trade <token> perp` is refused there.
- **Off-roster perps on a gamma-roster desk (mm3, mm2) default to Binance.** When
  `BINANCEFUTURES` carries `<TOKEN>-USDT.PERP`, the order is pinned there for execution and pricing
  (`exchanges` + `market_data_exchanges`); otherwise it keeps the futures strategy's full venue set.
  A roster token sent with `perp`, and the treasury book, are not affected.
- **OTC carries its own trader allowlist** (`TELEGRAM_TRADE_TRADERS_OTC`). Pinning the chat does
  **not** arm the MM3 traders on it — the two lists are deliberately separate, because an OTC book
  reaches any ticker on 11 venues with no roster to constrain it.
- It rides `CR_API_TOKEN_MM3` (`bot_mm3user` is already in the strategy's `users` list).

### `trade_bot` — the typed-`limit=` fat-finger band

`_LIMIT_BAND = 0.05` refuses a **typed** `limit=` more than 5% **through mid**, in the aggressive
direction only (a buy above mid, a sell below it). Added after a live sell went out with one extra
0 in the decimals and crossed the book far below the intended price.

- **Reference is mid, computed from BOTH touches on the strategy's own venues** — never
  `client.mid()`, which prices the unrestricted book and would read the wrong coin on a
  `TOKEN_VENUES` ticker (`AI`/`AIGENSYN`). That is one extra `cost_calculator` call, taken **only**
  when a limit was typed, so a plain `/trade` keeps its current latency.
- **One-sided**, same reasoning as the ladder band below: a bid below the market just rests.
- **5% clears every real book.** Measured live across the MM3/MM4 gamma tokens, the widest spread
  was DNX at 2.23% (half-spread 1.11%) and most sit under 0.5% — so an order marketable at the
  touch is never close to the band.
- It cannot fire on a **default** limit: that is priced off the touch with at most the 1%
  `aggressive` buffer. Only a trader-typed `limit=` is checked, so `/scalper` and `/jwap` (which
  type no limit) are unaffected; `/scalper`'s own ladder band still applies.
- There is **no override keyword**. A genuinely intended 5%+ sweep has to be retyped as a limit
  inside the band, or worked with `/jwap`.

### `trade_bot` ladder — the fat-finger band is ONE-SIDED

`_LADDER_BAND = 0.30` refuses a quote more than 30% **through** the touch — in the aggressive
direction only. `touch` is the best **ask** for a bid ladder and the best **bid** for an offer
ladder, so:

| | passive direction | aggressive direction |
|---|---|---|
| bid | below the ask — **allowed at any distance** | above the ask — refused past 30% |
| offer | above the bid — **allowed at any distance** | below the bid — refused past 30% |

- A bid below the ask just **rests**; if it fills, the desk bought cheap. There is nothing there to
  protect against. Only the aggressive direction crosses the book and fills instantly at a bad
  price, which is what a slipped decimal actually costs.
- ⚠️ It used to be symmetric, and that refused a legitimate `/scalper aurora 5k 2 vol=400%`: at
  σ 20.94%/day the 2σ levels sit **42%** out. Worse, `check_ladder` validates a side **atomically**,
  so the whole side was dropped — including the 1σ levels that were inside the band. The reply read
  `0 order(s) working · 2 FAILED`.
- The **crossed check is unaffected** and still guards the desk's own resting quotes, as does
  `_MIN_MID_BPS` on `/trade`'s typed ladders (a quote too CLOSE to mid fills on contact — that one
  is a real hazard and stays).
- ⚠️ AURORA has **no JASON vol series** (`no JASON volatility for AURORAUSDT`), so `/scalper` on it
  only runs with an explicit `vol=` override. The levels are then only as good as that number.

### `trade_bot` ladder — orders it cannot price

⚠️ **A CoinRoutes `spread` order carries `limit_price = None`.** It is an algo with no static
limit, and the whole ladder is priced off `limit_price`: `plan_quotes` ranks on it, diffs on it and
reports it; `check_ladder` crosses against it. `/trade aurora clear` died on `float(None)` and
replied *"that command hit an internal error"* — the dispatch catch-all, with the real `TypeError`
only in the journal.

- `split_manageable` now separates them **before** either consumer sees them. Fixing only
  `check_ladder` would have moved the crash: `plan_quotes` makes the same assumption in three more
  places.
- They are **excluded, not coerced**. A spread order has no price to reprice *to*, and treating a
  null as `0` would rank it best-of-book and cancel someone else's algo.
- ⚠️ **NaN is rejected too, and it is reachable** — when every open order on a pair is a spread
  order the column is all-null and pandas types it `float64`, turning `None` into `NaN`, which
  `float()` accepts and `plan_quotes` would then sort unpredictably.
- ⚠️ The reply **names what it left alone** (`! left alone (no limit price): spread 500,000
  AURORA`). A `clear` that silently leaves a 500k sell resting reads as a flat book.
- Measured 2026-09-22: 2 of 146 open MM3 orders, both `spread`, both placed in the UI —
  **AURORA** and **TNSR**. `twap` orders are unaffected; they carry a real limit.

### `trade_bot` order tracking — two CoinRoutes fields that mislead

⚠️ **`order_status="error"` is TRANSIENT, not final.** The venue reports it while the execution
algo is still working. Over 1,711 client orders in 7 days exactly one carried it — 77% executed,
algo still running. A live `ZEC-USDT.PERP` order reported `error` at 14:44:30 and closed **100%
filled** at 14:44:57. It is therefore **not** in `TERMINAL`; an order showing it keeps being
polled until it reaches `closed`/`cancelled`/`rejected`. Treating it as final both sent a false
❌ and dropped the order from tracking, so the real fill was never reported.

⚠️ **`error_code` is not an error.** It is the algo's working-state annotation, and the single
most common value in the whole population is **`dust` on a fully-filled order** (1,193 of 1,711).
Others seen: `active` (= actively working), `throttled`, `paused`, `target_wait`,
`limit_not_marketable`, `send_failed` — each appearing on `closed`, `cancelled` **and** `open`
orders alike. Never render it as a failure reason; only `error_text` carries one. Rendering it
produced the notorious `❌ order … error: active` on an order that had filled.

A terminal non-fill now reports **what filled first** (`❌ order … rejected — 77% filled`), and
the stop-tracking timeout names the actual status rather than calling everything "still working".

### `vol_bot` — when the RV>IV breach alert actually fires

The rule is **`RV(2W) > (1 + ALPHA) x IV(1W ATM)`** with **`ALPHA = 0.10`** — *not* `RV > IV`.
A name merely above implied stays silent; it has to clear it by 10%. At the previous `ALPHA = 0.20`
both BTC (x1.02) and ETH (x1.18) sat above IV unannounced, which is what prompted the change.

- A name is announced **on crossing in**, again **every 4h** (`REMIND_SECONDS`) while it stays
  there, and once on recovery. Edge-triggered alone was too quiet — XRP sat in breach an entire
  morning on a single 07:27 message while every 15-minute poll logged it.
- The reminder clock runs from the **last message about that name**, not from when it entered
  breach, so the cadence stays a clean 4h either way. A name crossing in is announced
  immediately — the reminder never delays a new breach.
- ⚠️ **Only 4 names** (`UNIVERSE`: BTC/ETH/SOL/XRP). VOLPUB returns a *bucketed fallback IV* for
  the rest — groups of unrelated names sharing one identical series — so comparing a real RV to a
  proxy IV fires constant false breaches. Widening it needs the uniqueness re-probe described in
  the module docstring, not a blind append.
- State is in memory, so a restart re-announces whatever is currently in breach. That is
  deliberate: after a deploy the chat should say where things stand.

## Scheduled jobs (not bots)

### `icp_snapshot` — nightly ICP-MM state stamp

`scripts/icp_snapshot.py`, run by the `icp-snapshot.timer` systemd user unit at **23:31 UTC**.
Not a Telegram bot: no token, no chat, no poller. It appends one JSON object per night to
`data/state/icp_snapshot.jsonl` (`data/` is a symlink outside the repo, so the series is never
committed).

**Why it exists.** The ICP book's daily P&L needs *yesterday's* delta, spot, vols and option
book, and the `ICP-MM` workbook keeps none of that — every cell is live. Only its `FX P&L` tab
persists anything (balances and rates, hard-stamped nightly by the desk's looper on
`g20-rapidengine-dev-02`). This job stamps the rest at the same cutoff, so the day-over-day
terms become computable **from the first snapshot forward** — a missed night is a hole that
cannot be backfilled.

- Sheet `1NcgtFY4RcaASVrJGftTcbEvIz-7IQKvQcLQg1thEedk`, read with the **same read-only SA as
  `agg_bot`** (`GOOGLE_AGG_SA_KEY` / `.gsa_key.json`). One batched `values_batch_get` with
  `UNFORMATTED_VALUE` — formatted values arrive as `'2,949,935'` and `'60%'`, strings that
  poison arithmetic silently.
- ⚠️ **`Dashboard!E74` already includes option delta** (`= E73 + GammaSlide!Q15 +
  GammaSlide!AD36`). A P&L anchored on it must add only **gamma/vega/theta** for the options —
  revaluing them in full double-counts option-delta × the day's move, which looks plausible.
  `delta_spot_coins` (`E73`) is captured separately so the two can never be confused.
- ⚠️ Timer is `Persistent=true` and `write_snapshot` **replaces** a same-date row, so a late or
  repeated run corrects the day instead of duplicating it. Re-running by hand is safe.
- Each option line is tagged with the **vol cell that prices it** (`vol_cell`). The sheet wires
  this POSITIONALLY — `GammaSlide` rows 4-12 read `$F$5`, row 13 reads `$H$5` — so a line that
  moves row silently reprices at the other vol. Tagging at snapshot time makes that visible.
- ⚠️ **The vol marks (`F5` 60% / `H5` 65%) are entered by hand**, so they go *stale* rather than
  wrong: a mark left untouched while IV drifts reads clean every day, then takes the whole
  correction at once. One vol point is worth **~$2,700** on this book against a **~$200/day**
  trading P&L — a 5-point re-mark is ~70× the day's market-making. `vol_history()` reports the
  mark in force and days since it last changed, logged every night; until the series has seen a
  change it says "at least N days", never a confident zero. Any report quoting an option number
  must carry that age, and a re-mark must be **its own labelled line**, never anonymous inside
  the total.
- `--dry-run` prints the snapshot and writes nothing.
- Check it: `systemctl --user list-timers icp-snapshot.timer`,
  `journalctl --user -u icp-snapshot.service -n 20`.
- ⚠️ `sheet_update_ts` is the looper's heartbeat. If it lags the run by more than a few minutes
  the workbook was stale and the row is suspect — the looper had died, not the timer.

### `pluto_prc_push` — Pluto risk into the MM2GAMMA-MM sheet

`scripts/pluto_prc_push.py`, run by `pluto-prc.timer` every **5 minutes** (`OnCalendar=*:0/5`).
Pushes the `MM2GAMMA_TOTAL` risk view into the `MM2GAMMA-MM` workbook's `[Pluto-PRC-Summary]`
tab, one row per biprod. **Two Pluto calls, one sheet write.** Full API reference:
`docs/pluto_portfolio_api.md`.

⚠️ `detailed=positions` is still fetched even though no position-level tab is written: Pluto's
`pv` is a **single portfolio-wide scalar**, so per-biprod PV exists only by summing the positions.
Dropping that call to save a request would silently zero the PV column rather than fail.

- Layout copies the sheet's own convention for machine-written `[...]` tabs (`[Positions]`,
  `[Adjustments]`): headers row 1, data row 2+, a label/value metadata block to the right with
  `Requested At`, `Portfolio`, `RiskReport` and `Portfolio PV`.
- The Dashboard `QUERY`s `$A:$I` by biprod: column **C (PV) -> "Live Options MTM"** and
  **D (Delta (Coin)) -> "Options Delta (Coin)"**. Reordering or inserting a Summary column breaks
  those formulas silently — `SELECT C`/`SELECT D` are positional.
- **`PV` is summed per biprod over every leg** — options, forwards, cash and the intra-desk
  loans alike, no filter. It has to be summed because Pluto's `pv` is a single portfolio-wide
  scalar with no per-biprod split. **Every other column is Pluto's own per-biprod aggregate**,
  i.e. exactly what the Pluto summary tab shows, which is what the desk reconciles against.
- The metadata carries **`Summed PV`** beside **`Portfolio PV (Pluto)`** so the invariant stays
  visible: the per-biprod column must add up to Pluto's own total, give or take the book moving
  between the two calls (~$20).
- ⚠️ With the loans in, a biprod can show a large negative PV against a long call book — NEAR
  reads about **−329,000** off a −200,000 `CASH1` at −374,400. That is intended, not a bug.
  Two earlier filters (by position code, then by counterparty) were tried and removed; if one is
  ever wanted again, note that **code is the wrong axis** — POL's hedge is a `CASH1` exactly like
  its loan is, so only `counterparty` separates them.
- ⚠️ **Delta must be the aggregate, not a filtered sum.** `CASH1` legs are spot/loan coin balances
  and their delta *is* the notional — real exposure, not funding noise. **POL's hedge is booked
  `CASH1`, not `FX`** (net −2,765,199), so filtering left POL reading **+2.37m against Pluto's
  −398,407**, while CC — whose hedge really is an `FX` — matched. Gamma/vega/theta are unaffected
  either way: cash and forwards carry none.
- `detailed=positions` is therefore needed for **PV only**; everything else comes from the
  `greeks` call. Both calls stay. (Gamma/vega/theta happen to be clean either
  way, since forwards and cash carry none; delta and PV are the contaminated ones.)
- ⚠️ **Do not assume the `CASH1` legs cancel.** They were ±478,000 and netting to zero; they are
  now −485,000 / +368,025. They resize daily, which is another reason the filter is by code
  rather than by hoping offsetting legs wash out.
- A biprod with no options keeps a **zero row** rather than vanishing: the Dashboard's `QUERY`
  returns 0 either way, and a visible zero says "no options" instead of "not checked".
- ⚠️ **No total row.** These tabs get `QUERY`'d by the Dashboard; a `TOTAL` row would be
  double-counted by any `SUM` over the range. The portfolio total lives in the metadata block,
  separated from the data by a blank column so a range query can't reach it.
- ⚠️ **Cleared and rewritten atomically.** The book **rolls daily** (`was_rolled`, expiries
  stepping a day) and changes shape — POL appeared mid-session — so a shrinking book must not
  leave stale rows. `spreadsheets.batchUpdate` → `updateCells` with an **unbounded `range`**
  (`startRowIndex`/`startColumnIndex` 0, *no* end indices) clears every cell the payload does
  not cover, in the same request. So: no separate `clear` call, no padding to guess at, and
  **no instant where the tab is empty** for a Dashboard `QUERY` to read — which a
  clear-then-write would expose. Verified with sentinels planted at `A40`/`Z60`; both were gone
  after one push. A test asserts the range stays unbounded, since adding an end index would
  silently resurrect the stale-row bug.
- ⚠️ **`riskreport` is always sent.** Omitting it silently applies Pluto's `DEFAULT` model and
  moves delta ~10%; PV is unaffected. The value used is written into the metadata block.
- ⚠️ **Typed `userEnteredValue`, the equivalent of `RAW`** — numbers land as real numbers for
  the Dashboard's `QUERY`/`SUM`, and nothing re-parses the biprod strings the way
  `USER_ENTERED` would. `bool` is checked before `int` (it is a subclass), or `TRUE` writes as 1.
- Writing needs the **`spreadsheets`** scope; `agg_bot` and `icp_snapshot` use
  `spreadsheets.readonly`. Same `.gsa_key.json` SA, which has edit rights on this workbook.
- ⚠️ The two ICP `CASH1` legs are **+/-476,000** and cancel within a biprod. A per-biprod PV is
  correct; anything filtering by counterparty pulls half a million of phantom PV.
- ⚠️ **A failed cycle is not fatal.** Any error — Sheets or Pluto — is counted, not raised: the
  write is one atomic request so there is no half-written state, the tabs keep their last good
  data, and the next tick retries a minute later. The metadata block's `Requested At` is how
  staleness is spotted. The unit exits 1 so the failure is visible in `systemctl status`, which
  does not affect the timer.
- ⚠️ **Sustained-outage alert.** Consecutive failures are counted in
  `data/state/pluto_prc_failures.json` (each tick is its own process, so the streak must live on
  disk). Once the streak passes `ALERT_AFTER` (5, so the **6th** consecutive failure) the MM2
  desk chat gets one message naming the streak, when it started, and the last error. **Once per
  outage, not once a minute** — at 60s cadence re-alerting each tick would be 60 messages an
  hour. A success resets both the streak and the alerted flag, so a later outage alerts again.
  A single success mid-outage restarts the count: "in a row" means in a row.
- Alerting reuses `TELEGRAM_BOT_TOKEN` send-only to `TELEGRAM_MM2_CHAT_ID`, the same pattern as
  `flow_alert_bot` — no poller, so no 409 on the shared token.
- `--dry-run` prints both blocks and writes nothing.
- Check it: `systemctl --user list-timers pluto-prc.timer`,
  `journalctl --user -u pluto-prc.service -n 20`,
  `cat data/state/pluto_prc_failures.json`.

### `icp_pnl` — ICP-only daily P&L

`scripts/icp_pnl.py`, a CLI over two `icp_snapshot` rows. Rebuilds the ICP part of a book that
also holds BTC/ETH/EUR/CHF, so the reported number stops moving when those move against USD:

```
delta (spot)     E73 * (S1-S0)          the term the sheet never had -- it prices today's
delta (options)  Q15 * (S1-S0)          fills but never marks yesterday's inventory
gamma            0.5 * G0 * (S1-S0)^2
vega             v0 * dsigma            zero unless someone re-marked by hand
theta            AN20, prorated
trading          M67                    matched spread + residual marked to spot
fees             W2
```

- ⚠️ **`delta_total_coins` already includes option delta**, so options add only gamma/vega/theta.
  The split into spot/options is presentation; `delta_check` asserts the halves still sum to
  `E74`, and the render shouts if they don't.
- The FX legs never enter — the workbook's `FX P&L` tab owns them, and excluding them is the
  whole point. A test asserts no term is keyed on a foreign currency.
- **Reconciled, not asserted**: the option block's Taylor total is printed against its actual PV
  change. Live on a 1.4% move it agreed to **$4 on an $18k option move**, which is what validates
  the greeks, the vol/tau wiring and the decomposition all at once. A large move or a line near
  expiry shows up as a residual instead of hiding in the total.
- ⚠️ Theta prorates on **fractional** days. Flooring to whole days charged a 15-minute window a
  full day of decay (~$1,840) and dumped it into the residual.
- ⚠️ `trading`/`fees` are the sheet's `"Today (UTC)"` counters (`Config!A3`) — they reset at
  **00:00 UTC**, which is why the snapshot timer is 23:58 and not the `FX P&L` looper's 23:31.
  On a sub-daily window they cover more than the window, and the render says so.
- `--live` closes the window at a fresh sheet read instead of the last stamped night; `--date`
  picks the day to close.

## When adding a new bot

1. Its **own** BotFather token in `.env` (one poller per token).
2. Use `src.common.telegram` for the long-poll skeleton (`drain_backlog`/`poll_updates`/`send`/
   `send_photo`) — don't copy-paste it. Only reach for a bot-local `_send` if you genuinely need
   HTML `parse_mode` or message truncation (see `macro_bot`/`sentiment_bot`/`vol_bot` above).
   `trade_bot` (money-moving) is the one exception that stays fully self-contained.
3. Reuse `src.common.config.require_env`, `src.common.logging.get_logger`, the house table renderer
   pattern, and the canonical token maps — don't re-list strategies.
4. Gate on `TELEGRAM_CHAT_ID` (mm3) and add the chat-aware desk pattern if it serves mm2 too.
5. Module docstring stating its responsibility **and what it must NOT do** (repo convention).
6. `uv run ruff check scripts/<bot>.py` and a dry-run render before wiring it into a chat.
