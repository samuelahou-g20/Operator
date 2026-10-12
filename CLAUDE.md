# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A **quantitative trading system**, implemented and running: a multi-venue crypto data lake, a
funding-rate carry strategy shipped through eight validated versions (v8.0 is the product), a
reproducible research track behind every claim, and CME futures groundwork. **`ARCHITECTURE.md`
is the design source of truth** (read the relevant section before implementing); this file plus
the code are the operational truth. If a task seems to require new top-level structure, a new `src/`
module family, a new dependency, or an alternative to a single-algorithm rule below, **stop and
ask the owner** — do not invent structure or generalize.

## State of the system (orientation, not authority — verify against code)

- **Data lake**: Binance (468 spot+perp pairs incl. delisted, 2021+), Bybit (343, ~mid-2021+),
  OKX (forward-accumulating; REST funding retention ~93 days), Hyperliquid (hourly funding 2023+,
  1h bars are a rolling ~5000-bar window — must be accumulated), FRED rates (`fred/tb3ms`, needs
  `FRED_API_KEY` in env/`.env`). `scripts/daily_task.sh` runs the daily fetch+normalize for all
  sources.
- **Funding carry**: `config/strategies/funding_carry/v8.yaml` is the shipped product
  (multi-venue breadth + drawdown overlay + portfolio-margin accounting + idle-cash yield);
  `v8.1.yaml` is the full-book capacity configuration. Entry point:
  `python -m src.strategies.funding_carry.v8.backtest` (runs + renders the HTML report).
  `research/funding_carry/README.md` indexes every research arc (8 shipped versions, 8 documented
  nulls, honesty passes) — read it before touching the strategy.
- **Sharpe convention (owner directive)**: Sharpe = (E[r] − E[rf]) / vol(r) everywhere; ratios
  that ignore rf are not Sharpe ratios and are not reported. The engine persists the daily rf
  series in each run's `returns.parquet`; reports must not read the lake.

## Commands

This is a **uv-only** project run from repo root; `src` is imported as a package. Neither `uv`
nor `python` may be on PATH — see the owner's environment notes if invocation fails.

```bash
uv sync                                              # recreate exact env from uv.lock
uv add <pkg>                                         # add a dependency (never hand-edit pyproject.toml / uv.lock)
uv run python -m scripts.backfill --source bybit --universe   # sources: binance|bybit|okx|hyperliquid|cme|fred
uv run python -m scripts.daily_fetch --source okx
uv run python -m scripts.normalize --source bybit --universe
uv run python -m scripts.discover_universe --source binance --write   # regenerate the universe registry
uv run python -m src.strategies.funding_carry.v8.backtest --config strategies/funding_carry/v8.yaml
uv run python -m scripts.report --strategy funding_carry --version v8.0
uv run pytest                                        # all tests (no CI; run locally)
uv run pytest tests/data/test_query.py::test_name    # single test
uv run ruff check src tests scripts                  # lint (notebooks and _build_*.py are excluded)
```

Scripts run in **module mode** (`python -m scripts.x`), not script mode — script mode breaks
`import src`.

## Architecture: the rules that are easy to get wrong

- **Single read path.** Everything that *reads* the lake goes through `src/data/query.py`
  (`get`, `get_continuous`, `available`, `coverage`). Only fetchers/ETL that *write* the lake
  (and `query` itself) may open parquet/csv under `data/` directly.
- **Centralized infra, no exceptions.** Paths from `src/common/paths.py`; logging from
  `src/common/logging.py`; YAML only via `src/common/config.py` / `universe.py`. Credentials:
  configs name the env var only; `require_env` auto-loads repo-root `.env` (gitignored).
- **Raw is immutable.** `data/raw/` is append-only, written only by `src/data/fetching/`.
  Sanctioned exceptions are documented in-module (e.g. FRED full-series refresh; `--overwrite`
  re-backfills).
- **The universe registry is the instrument record.** `config/universe/{source}.yaml` holds every
  instrument any strategy uses (generated via `discover_universe --write`; hand-set fields survive
  regeneration). Strategies resolve `universe: all` FROM the registry; the lake only gates data
  availability. Never resolve a universe from the lake alone.
- **Strategy autonomy.** No shared backtest engine across strategies and no shared live
  orchestrator; each `strategies/{name}/` owns its lifecycle. Within funding_carry, versions share
  `engine.py` and supply **pure** `signals.py` (no I/O) — backtest and live must import identical
  logic. Version modules are thin wrappers; logic lives in engine + signals.
- **All live orders go through the OMS**, never directly through adapters. Adapters are thin API
  translation — no business logic, no position math.
- **Schemas defined once** in `src/data/schemas.py`. Fetchers normalize vendor data on write;
  `query` guarantees output conforms. Funding is stored one row per settlement with
  `next_funding_ts` from settlement spacing — venue intervals are dynamic; **never annualize
  ×3/day; always sum per day**.
- **One stitching algorithm** for futures (`etl/futures/stitch.py`). No alternative roll rules.
- **Chunked, idempotent backfills are mandatory.** Process per day/contract/symbol; never
  accumulate unbounded data in memory; re-running a range overwrites cleanly, never duplicates.
- **Reports read only a completed run's artifacts** — never the lake. Anything a report needs
  (e.g. the rf series) is persisted into the run dir by the engine.

## Data lake & config conventions

- **Path template:** `data/{stage}/{asset_type}/{source}/{symbol}/{data_type}/...` — `stage` ∈
  `raw|processed`; lowercase source/symbol; bars add `bars/{timeframe}/`. Big data → hive
  `date=YYYY-MM-DD/part.parquet`; small → one file per symbol; the choice lives per
  `(source, data_type)` in `config/data/{source}.yaml`.
- **Columns:** UTC tz-naive `ts` (event time; bar open for bars, settlement for funding/rates),
  lowercase snake_case. `query` returns ts-indexed frames.
- **Symbols are venue-native lowercase** (`btcusdt`, `shib1000usdt`, `ubtc`); cross-venue joins go
  through canonical base+multiplier mapping (`_canon_base`), never raw strings. Multi-venue
  configs key columns `venue:pair`.
- **Streaming:** Redis channels `live:{source}:{symbol}:{data_type}`; only
  streamer/aggregator/consumer touch Redis directly.

## Telegram bots (`scripts/*_bot.py`)

A family of on-demand Telegram bots for the MM3/MM2 market-making desks (aggregator, open
orders/fills, hedge PnL, live trading, JASON levels, long/short, vol, macro, sentiment, plus the
mm2 fill-logging daemon). **`docs/bots.md` is the reference** — read it before building or fixing a
bot. Key rules: one long-poller per BotFather token (409 on a duplicate); the **chat-aware desk
pattern** routes mm3 vs mm2 by chat id (`TELEGRAM_CHAT_ID` / `TELEGRAM_MM2_CHAT_ID`, mm2 wins the
overlap); order-book/price data comes from CoinRoutes `cost_calculator` (never Binance); `trade_bot`
is the **only** money-moving bot and stays behind its allowlist + cap rails. Prefer the **`bot-dev`
subagent** (`.claude/agents/bot-dev.md`) for bot work — it starts pre-loaded with these conventions.

## Research conventions

`research/{track}/` = one folder per experiment/version (version-prefixed once shipped), each a
`_build_*.py` that programmatically builds AND executes its `.ipynb` (SVG figures, time-series
panels; notebooks are outputs — never hand-edit). Shared harness `research/funding_carry/lab.py`
reproduces the engine to 1e-6 — validate parity before trusting any number. Preregister designs,
pay real costs, judge in excess of rf, count every trial (the DSR ledger), and document nulls —
a clean null is a shipped result. `research/` may import `src`; `src` must never import
`research/`.

## Coding conventions

- **Simplicity first.** Plain functions; classes only where state requires (adapters, OMS,
  consumer, streamer). No ABCs, plugin registries, DI frameworks, dataframe-validation
  frameworks, or premature generalization. No async unless the task requires it.
- Touch only the modules named in the task; do not refactor neighbors.
- Every module gets a short docstring stating its responsibility **and what it must NOT do**.
- `tests/` mirrors `src/`; all fetcher tests are offline/mocked; live API checks are documented
  manual commands, not pytest.
- After a full backfill, **audit coverage** (join funding-symbols vs bars-symbols) — isolated
  smoke tests passing does not mean the universe-scale run is correct.
