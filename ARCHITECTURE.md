# Operator — Architecture

The **design source of truth**: the repository's structure, the responsibilities of each
module, and the interfaces between them — and, more importantly, the *rationale* behind
them (why a single read path, why no shared engine, why chunked writes). Code shows *what*;
this document exists for the *why* that code can't express.

**What the repo is for.** Tooling for G-20's market-making desks. The product is the **desk
layer**: The Operator (`src/the_operator/`, a Telegram agent the desks ask in plain language), the
desk bots (`scripts/*_bot.py`), the ZP quoting review and a few scheduled jobs, all run as
`systemd --user` units. The data lake, signals, OMS and research described in §2–§13 are the
**supporting layer** underneath. The desk layer's own design lives in `docs/operator.md` and
`docs/bots.md`; its load-bearing rules are summarized in §15. Everything in §0 applies to both
layers.

Deviating from this design (new top-level dir, new `src/` module, new dependency, an
alternative roll rule, an ABC, async) needs the owner's approval — ask first. Day-to-day
agent operating rules live in `CLAUDE.md` and `.claude/rules/`, not here.

---

## 0. Principles (read first, these override everything)

1. **Simplicity first.** Implement the minimal version that works. No abstract base class hierarchies, no plugin registries, no dependency injection frameworks, no premature generalization. If a function does the job, do not write a class. We iterate later if needed.
2. **No strategies here.** This repo serves the desks; strategy backtests and live runtimes are out of scope (§11).
3. **Single read path.** Any code that reads from the data lake goes through `src/data/query`. No module opens parquet/csv files from `data/` directly. (The only exceptions: the fetchers/ETL that *write* the data, and `query` itself.)
4. **Raw is immutable.** `data/raw/` is append-only, written only by `src/data/fetching/`. Nothing else ever writes or modifies files there. All transformations write to `data/processed/`.
5. **Flat files only.** Parquet (partitioned) for large data, CSV for small data. No database for market data. The only durable non-flat-file store is the OMS state (SQLite, see §10).
6. **Config over code.** Instruments, strategy parameters, fetcher settings live in YAML under `config/`. Code reads config; it does not hardcode symbols, dates, or credentials.
7. **No new top-level directories or `src/` modules** without explicit approval from the owner.

---

## 1. Directory tree

```
repo/
├── pyproject.toml            # uv: name, requires-python, dependency list
├── uv.lock                   # uv: hash-pinned lockfile (committed)
├── README.md
├── ARCHITECTURE.md           # this file
│
├── data/
│   ├── raw/                  # immutable, written only by fetchers
│   │   └── {asset_type}/{source}/{symbol}/{data_type}/
│   ├── processed/            # derived data, written only by ETL
│   │   └── {asset_type}/{source}/{symbol}/{data_type}/
│   └── state/                # OMS durable state (sqlite), live strategy state
│
├── logs/                     # runtime logs, one subfolder per process/strategy
├── results/                  # generated reports and figures
│
├── config/
│   ├── desks.yaml            # desk roster: tokens, hedge strategy, bots per desk
│   ├── operator/             # per-desk Operator playbooks (appended to its system prompt)
│   ├── zp/                   # ZP review token and strategy-box configs
│   ├── universe/             # one yaml per source: instruments + characteristics
│   └── data/                 # fetcher/streamer/backfill configs
│
├── scripts/                  # desk bots (*_bot.py), scheduled jobs, thin CLIs
├── deploy/systemd/           # unit files for the Operator and the bots (make install-units)
├── docs/                     # Operator, bots, ZP and JWAP references; vendor docs in references/
├── research/                 # notebooks & scratch work, NOT importable by src/
├── tests/                    # mirrors src/ layout; bot and script tests at the top level
│
└── src/
    ├── the_operator/         # the Operator: gateway, agent, policy gate, tools, ZP review
    ├── signals/              # JASON, JWAP, trend and mean-reversion signals (pure)
    ├── monitor/              # live terminal monitor (scripts.dashboard)
    ├── common/               # paths, logging, config loading, universe loading
    ├── data/
    │   ├── fetching/{source}/    # backfill.py + fetcher.py per source
    │   ├── streaming/{source}/   # streamer.py per source, + aggregator.py, consumer.py
    │   ├── etl/                  # contracts, stitch, resample, per-asset-type ETL
    │   ├── schemas.py            # canonical schemas for every data_type
    │   └── query.py              # the single read API
    ├── adapters/{source}/        # live exchange APIs (orders, positions, balances)
    ├── oms/                      # orders/positions/trades tracking + reconciliation
    └── reporting/                # metrics, figures and the G-20 report template
```

---

## 2. Data lake conventions

> **Physical location.** `data/` is a symlink to a shared canonical lake *outside* the repo
> (contents never committed). All code writes to `data/…` via `src/common/paths.py`; the
> symlink redirects to the shared copy. Setup and the fresh-clone step are in README → *Data
> lake location*. The conventions below describe the layout *inside* that lake.

### 2.1 Path template

```
data/{stage}/{asset_type}/{source}/{symbol}/{data_type}/...
```

- `stage`: `raw` | `processed`
- `asset_type`: `futures` | `options` | `perps` | `spot` | `metadata` (extend only with approval)
- `source`: exchange or vendor, lowercase (`cme`, `eurex`, `ice`, `binance`, `hyperliquid`, ...)
- `symbol`: lowercase (`es`, `ng`, `btcusdt`, ...). For futures this is the family symbol (`es`), never an individual contract (`esm6`) — individual contracts appear only at the `contract=` partition level inside the symbol folder.
- `data_type`: `bars`, `trades`, `l1`, `l2`, `funding_rate`, `open_interest`, `contracts`, ... Bars get one extra path level for the timeframe: `bars/{timeframe}/` (e.g. `bars/1m/`, `bars/1d/`).

### 2.2 File format & partitioning

- **Large datasets** (intraday bars, trades, book data): parquet, hive-partitioned by date:
  `.../{data_type}/date=YYYY-MM-DD/part.parquet`
  The `date=` / `contract=` prefixes are intentional (hive-style partitioning): pyarrow/duckdb recognize `key=value` folders as partition columns automatically, enabling partition pruning and exposing `date`/`contract` as queryable columns for free. Plain folder names (`2024-01-05/`) would lose that.
  For futures data — in **both** `raw/` and `processed/` — partition additionally by contract: `.../contract=ESM6/date=.../part.parquet`. `raw/` holds the data as fetched from the provider (vendor format, possibly compressed); `processed/` holds the cleaned, schema-conforming per-contract data produced by ETL from it.
- **Small datasets** (daily bars, funding rates, contracts tables): single parquet or CSV file per symbol: `.../{data_type}/{symbol}.parquet`.
- The decision big-vs-small is made once per `(source, data_type)` in the fetcher config (`config/data/`), not ad hoc in code.

### 2.3 Processed data naming

`processed/` uses the same layout down to `{symbol}/`. Inside, derived datasets follow these conventions:

- **Continuous futures series**: `processed/futures/{source}/{symbol}/continuous/{strip}/{timeframe}/`
  where `strip` is the continuous strip name: `es1` (front), `es2` (second), etc. There is **one** stitching algorithm (the owner's, implemented in `etl/futures/stitch.py`, §7) — do not add alternative roll rules or variants.
- **Resampled bars**: same `bars/{timeframe}/` convention as raw, e.g. `processed/.../bars/5m/`.
- Anything else: pick a descriptive `data_type` folder name and document it in the ETL module that produces it.

### 2.4 Universal column conventions

- Timestamps: UTC, column name `ts`, nanosecond-precision `datetime64[ns]` (or int64 ns epoch in parquet). Event time, not receive time. If receive time is needed, it is `ts_recv`.
- All column names lowercase snake_case.
- No index magic: data is stored with `ts` as a regular column; `query` returns DataFrames indexed by `ts`.

---

## 3. `config/`

- **`config/universe/{source}.yaml`** — one file per source. A `sessions:` block at the top defines named trading-hour templates (hours are product-specific, not exchange-wide — see below — but many products share one, so we template them); each instrument references one by name. Per instrument:
  ```yaml
  # top of file: shared session templates for this exchange
  sessions:
    equity_index: "Sun 17:00 → Fri 16:00 America/Chicago, daily halt 16:00–17:00"
    # add more as products require

  es:
    asset_type: futures
    exchange: cme
    currency: USD          # denomination — matters for multi-currency books
    tick_size: 0.25         # min price increment
    multiplier: 50          # point value (tick_value = tick_size × multiplier, derived not stored)
    rolling_cycle: [H, M, U, Z]   # active contract months
    strips: 2               # number of continuous strips to build (es1, es2)
    sector: equity
    subsector: us_large_cap
    session: equity_index   # references a template above
    settlement: cash        # cash | physical — drives roll urgency & delivery risk
  ```
  Exact field set is finalized when implementing `src/common/universe`; keep it minimal, extend when a real consumer needs a field. Note `data_types` is **not** a universe field — what data exists is discovered from the lake via `query.available()` (§8) and what to fetch is declared in `config/data/{source}.yaml`.
- **`config/data/{source}.yaml`** — fetcher/streamer settings: endpoints, rate limits, data_types to fetch, big-vs-small storage decision, backfill date ranges.
- **Credentials are NOT in config files.** They live in environment variables; config references the env var name only.

---

## 4. `src/common/`

| module | responsibility | interface |
|---|---|---|
| `paths.py` | single source of truth for all filesystem paths | `data_path(stage, asset_type, source, symbol, data_type) -> Path`, plus `LOGS_DIR`, `RESULTS_DIR`, `CONFIG_DIR`, `STATE_DIR` constants |
| `logging.py` | uniform logging setup | `get_logger(name) -> Logger` — console + rotating file in `logs/{name}/` |
| `config.py` | load & validate yaml configs | `load_config(relative_path) -> dict` |
| `universe.py` | load universe files | `load_universe(source) -> dict[symbol, dict]`, `get_instrument(source, symbol) -> dict` |
| `telegram.py` | Telegram long-poll plumbing shared by `scripts/*_bot.py` | `send`/`send_photo`/`drain_backlog`/`poll_updates` |

No other module hardcodes a path, configures logging itself, or parses yaml directly.

---

## 5. `src/data/schemas.py`

Canonical schema for **every** data_type, defined once, as plain dicts of `{column: dtype}` plus a short docstring. Fetchers normalize vendor data **into** these schemas before writing; `query` guarantees output **conforms** to them.

Initial set (extend here only, never define schemas elsewhere):

| data_type | columns |
|---|---|
| `bars` (per timeframe) | `ts, open, high, low, close, volume` (+ `contract` for raw futures) — timeframe lives in the path (`bars/{timeframe}/`), not the schema |
| `trades` | `ts, price, size, side` (+ `contract`) |
| `l1` | `ts, bid_px, bid_sz, ask_px, ask_sz` |
| `l2` | `ts, level, side, px, sz, action` (snapshot/update convention documented in module) |
| `funding_rate` | `ts, rate, next_funding_ts` |
| `open_interest` | `ts, oi` (+ `contract`) |
| `contracts` | `contract, symbol, expiry, first_trade, last_trade, fnd, ltd, tick_size, multiplier, currency` |

A tiny helper `validate(df, data_type) -> df` checks column presence and dtypes. Used by fetchers on write and available to tests. Nothing fancier (no pandera, no pydantic models for dataframes).

---

## 6. `src/data/fetching/` and `src/data/streaming/`

### 6.1 Fetching

Per source folder with two scripts. **No shared abstract base class** — each source implements two plain entry points with the same *informal* contract:

- `backfill.py` — `run_backfill(config: dict) -> None`. Bulk-fetches historical data in the most efficient way the vendor allows (batch downloads, bulk endpoints). Writes raw partitions **idempotently**: re-running for a date range overwrites those partitions cleanly, never duplicates. **Must process and write in chunks** (per day / per contract / per month depending on the vendor) — never accumulate a full backfill in memory. This is a hard requirement: previous projects died of OOM here.
- `fetcher.py` — `fetch_daily(config: dict, date: date) -> None`. Small incremental fetch (e.g. yesterday's data), run by cron via `scripts/`. Runs the lightweight quality checks (§12) after writing. Backfills do **not** run quality checks.

### 6.2 Streaming

- `streaming/{source}/streamer.py` — owns the websocket/API connections for live data, normalizes messages to the canonical schemas, publishes to Redis. Handles reconnects. One process per source.
- **Redis channel convention**: `live:{source}:{symbol}:{data_type}` (e.g. `live:binance:btcusdt:trades`).
- **Message format**: JSON, fields exactly matching the canonical schema for that data_type, plus `ts` as int64 epoch-ns. One message per event; no envelopes, no batching (revisit only if throughput demands it).
- `streaming/aggregator.py` — subscribes to raw channels, builds derived live streams (time bars, vwap, ...), republishes to `live:{source}:{symbol}:bars:{timeframe}` (e.g. `live:binance:btcusdt:bars:1m`). Bar messages are published on bar close.
- `streaming/consumer.py` — the helper any live code uses to subscribe:
  `Consumer(channels: list[str]).run(callback)` and/or a generator interface `for msg in Consumer(channels)`. Strategies and OMS use this; nobody talks to Redis directly except streamer/aggregator/consumer.

---

## 7. `src/data/etl/`

- `etl/futures/normalize.py` — cleans raw vendor contract data (decompress, parse vendor format, validate against schemas) into schema-conforming per-contract parquet under `processed/`, same `contract=`/`date=` partitioning as raw. This is the raw → processed step everything downstream (stitching, query) relies on.
- `etl/futures/contracts.py` — builds and maintains the contracts tables (`contracts` schema) per symbol from vendor metadata. Output: `raw/metadata/...` or `processed/futures/{source}/{symbol}/contracts/`.
- `etl/futures/stitch.py` — implements the **single** stitching algorithm (specified by the owner; implementation details provided as a separate task) that rolls individual contracts into continuous strips (`es1`, `es2`, ...). Reads per-contract processed data + contracts table, writes per §2.3. Agents must not introduce alternative roll rules, back-adjustment variants, or configuration knobs beyond what the owner's algorithm requires.
- `etl/resample.py` — generic OHLCV resampling between bar timeframes.
- `etl/perps/normalize.py` — perps have no real transformation: this just rewrites raw vendor data to schema-conforming processed parquet (decompress/parse → validate → write). No stitching, no contracts table. Keep it to a single normalize function.
- Other asset-type ETL folders are added only when a concrete need exists; until then, raw→processed for an asset type is just a normalize step like perps.
- All ETL reads via `query`, writes to `processed/` via `paths`.

---

## 8. `src/data/query.py`

The single read API for the data lake. Initial interface (keep it this small):

```python
def get(
    asset_type: str, source: str, symbol: str, data_type: str,
    timeframe: str | None = None,   # required when data_type == "bars"
    start=None, end=None, columns=None, stage: str = "auto",
) -> pd.DataFrame: ...

def get_continuous(
    source: str, symbol: str, strip: int = 1,   # strip 1 = "es1" (front), 2 = "es2", ...
    timeframe: str = "1d",
    start=None, end=None,
) -> pd.DataFrame: ...

def available(asset_type=None, source=None, symbol=None) -> pd.DataFrame:
    """Inventory of what exists in the lake (walks the folder tree)."""
```

- `stage="auto"`: prefer `processed/`, fall back to `raw/`. Can be forced to `"raw"`/`"processed"`.
- Implementation: pyarrow/duckdb-style partition-pruned reads so date filters never load the whole dataset. Returns `ts`-indexed DataFrames conforming to the canonical schema.
- `query` never mutates files.

---

## 9. `src/adapters/{source}/`

One module per exchange wrapping its live API. Plain class, no shared ABC; by convention every adapter exposes (matching names across sources):

```python
class Adapter:
    # market/account state
    def get_positions(self) -> pd.DataFrame: ...
    def get_balances(self) -> pd.DataFrame: ...
    def get_open_orders(self) -> pd.DataFrame: ...
    def get_fills(self, since: datetime) -> pd.DataFrame: ...
    # actions
    def place_order(self, symbol, side, qty, order_type, price=None, **kw) -> str:  # -> order_id
    def cancel_order(self, order_id) -> bool: ...
    def cancel_all(self, symbol=None) -> None: ...
```

- Adapters are **thin**: translate to/from the exchange API, normalize symbols and units, raise on errors. No business logic, no retry policies beyond basic transient-error retry, no position math.
- Credentials from environment variables, names declared in `config/data/{source}.yaml`.

## 10. `src/oms/`

Tracks and reconciles everything execution-related, on top of adapters.

- **Responsibilities:** local books of orders, fills/trades, and positions per `(source, account, strategy)`; reconciliation loops that compare local state against `adapter.get_positions()/get_open_orders()/get_fills()` and flag/repair drift; PnL bookkeeping from fills.
- **Persistence:** one SQLite file `data/state/oms.db` (tables: `orders`, `fills`, `positions`). SQLite is the simplicity-compatible choice: durable across restarts, zero infrastructure, queryable. No ORM — plain `sqlite3` and SQL.
- **Interface (initial):**
  ```python
  class OMS:
      def submit(self, source, strategy, symbol, side, qty, order_type, price=None) -> str
      def cancel(self, order_id) -> bool
      def positions(self, strategy=None) -> pd.DataFrame
      def orders(self, strategy=None, status=None) -> pd.DataFrame
      def reconcile(self, source) -> ReconcileReport   # plain dataclass of diffs
  ```
- Orders on the Binance / Hyperliquid adapters go **through the OMS**, not through the adapters directly, so that activity is recorded and reconcilable in one place. Desk orders take the other path: CoinRoutes, behind `trade_bot`'s rails (§15).

---

## 11. Strategies (out of scope)

Strategy code and its research were removed: this repo serves the desks. Adding
`src/strategies/` back needs the owner's approval (§0.7).

---

## 12. Data quality (lightweight, daily fetches only)

Run by `fetcher.py` after each daily fetch — **never** during big backfills (cost/OOM unrealistic there):

- timestamp monotonicity & duplicates,
- gap check vs expected sessions (from universe trading hours),
- basic sanity: non-negative volume, OHLC consistency (`low <= open,close <= high`).

Failures are logged as warnings to `logs/quality/` — they never block the write. A tiny `scripts/quality_report.py` can summarize recent warnings. Nothing more for now.

---

## 13. `scripts/`, `tests/`, `research/`, `reporting/`, `logs/`, `results/`

- **`scripts/`** — thin CLI entry points (argparse, call into `src`, exit): `backfill.py --source cme --symbols es,ng`, `daily_fetch.py`, `start_streamer.py --source binance`, `stitch.py`. The exception is the desk bots: each `*_bot.py` is a self-contained long-poller that owns its fetch and render (§15, `docs/bots.md`).
- **`tests/`** — pytest, mirrors `src/` layout (`tests/data/test_query.py`, ...). Priority order for coverage: schemas/validate, stitch roll logic, query, OMS bookkeeping. No CI for now; run locally with `pytest`.
- **`research/`** — analysis notebooks (today: JASON pivots). May import `src`; `src` must **never** import from `research/`.
- **`reporting/`** — generic metrics, figures, HTML layout, and the G-20 docx template fill used by the sentiment report.
- **`logs/`** — `logs/{process}/...` rotating files via `common.logging`.
- **`results/`** — generated reports (e.g. `results/mm3_perf/`).

---

## 14. Environment & execution

- **uv only.** A minimal `pyproject.toml` at repo root (name, `requires-python`, dependency list — nothing more) plus the committed `uv.lock` define the environment. No conda, no poetry, no pip-tools, no build backend, no `[project.scripts]` or other packaging ceremony.
- **Reproducibility**: `uv sync` recreates the exact environment (hash-pinned transitive graph from `uv.lock`) on any machine — research box and servers resolve identically. Commit `uv.lock`; never edit it by hand. Add deps with `uv add <pkg>`, not by hand-editing `pyproject.toml`.
- **Execution convention**: run everything from repo root in module mode (e.g. `uv run python -m scripts.backfill --source cme`); script mode breaks `import src`. `src` is imported as a package (`from src.data.query import get`). No separate install step, no manual `PYTHONPATH`.
- **Supervision**: the Operator and the desk bots run as `systemd --user` units from
  `deploy/systemd/` (`operator.service`, `desk-bot@<name>.service`), installed with `make
  install-units` and driven through the `Makefile` (`make status`, `make operator`, `make
  restart-<bot>`). The `pluto-prc` and `icp-snapshot` timers live there too and install the same
  way.

---

## 15. The desk layer (summary; details in `docs/operator.md`, `docs/bots.md`)

- **The Operator never holds a credential or executes anything itself.** The model only proposes
  tool calls; `src/the_operator/policy.py` is a pure gate that returns a mode per call
  (`execute`, `confirm`, `dry_run`, `denied`), and every verdict and result is journaled under
  `logs/operator/`.
- **Money moves only through `trade_bot`'s rails.** Operator money tools build orders through
  `trade_bot.prepare_order` / `prepare_scalper`, the same path a typed `/trade` takes, and need a
  human confirm. `trade_bot` is the only money-moving bot and stays behind its allowlist and caps.
- **One long-poller per BotFather token.** A second poller on the same token gets Telegram 409;
  never start a bot by hand while its unit runs.
- **Chat-aware desks.** Bots route by chat id (`TELEGRAM_CHAT_ID`, `TELEGRAM_MM2_CHAT_ID`, ...);
  desk facts (tokens, hedge strategy, bot roster) come from `config/desks.yaml`.
- **Order-book and price data come from CoinRoutes** (`cost_calculator`), not from Binance.
