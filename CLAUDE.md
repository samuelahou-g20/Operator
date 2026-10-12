# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

**Tooling for G-20's market-making desks.** The product is the desk layer: **The Operator**
(`src/the_operator/`, a Telegram agent traders ask in plain language; money and write tools need a
human confirm), the **desk bots** (`scripts/*_bot.py`), the **ZP quoting review**, and the
`pluto-prc` / `icp-snapshot` jobs, all running as `systemd --user` units on the desk host. The
multi-venue data lake, signals, OMS and JASON research are the **supporting layer**: they exist
to feed and justify the desk tools. Strategy code is out of scope here. **`ARCHITECTURE.md`
is the design source of truth** (read the relevant section before implementing); this file plus
the code are the operational truth. If a task seems to require new top-level structure, a new `src/`
module family, a new dependency, or an alternative to a single-algorithm rule below, **stop and
ask the owner** — do not invent structure or generalize.

## State of the system (orientation, not authority — verify against code)

- **Desk layer (live)**: `operator.service` plus one `desk-bot@<name>.service` per bot, installed
  from `deploy/systemd/` and driven through the `Makefile`. A code change to a bot or the Operator
  is not live until its unit is restarted (`make operator`, `make restart-<bot>`); run
  `make check` first. `docs/operator.md` and `docs/bots.md` are the references.
- **Data lake**: Binance (468 spot+perp pairs incl. delisted, 2021+), Bybit (343, ~mid-2021+),
  OKX (forward-accumulating; REST funding retention ~93 days), Hyperliquid (hourly funding 2023+,
  1h bars are a rolling ~5000-bar window — must be accumulated), FRED rates (`fred/tb3ms`, needs
  `FRED_API_KEY` in env/`.env`). `scripts/daily_task.sh` runs the daily fetch+normalize for all
  sources.
- **Sharpe convention (owner directive)**: Sharpe = (E[r] − E[rf]) / vol(r) everywhere; ratios
  that ignore rf are not Sharpe ratios and are not reported.

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
  instrument the lake fetches (generated via `discover_universe --write`; hand-set fields survive
  regeneration); the lake only gates data availability. Never resolve a universe from the lake
  alone.
- **All live orders go through the OMS**, never directly through adapters. Adapters are thin API
  translation — no business logic, no position math.
- **Schemas defined once** in `src/data/schemas.py`. Fetchers normalize vendor data on write;
  `query` guarantees output conforms. Funding is stored one row per settlement with
  `next_funding_ts` from settlement spacing — venue intervals are dynamic; **never annualize
  ×3/day; always sum per day**.
- **One stitching algorithm** for futures (`etl/futures/stitch.py`). No alternative roll rules.
- **Chunked, idempotent backfills are mandatory.** Process per day/contract/symbol; never
  accumulate unbounded data in memory; re-running a range overwrites cleanly, never duplicates.

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

## The Operator (`src/the_operator/`)

**`docs/operator.md` is the reference** — read it before changing the Operator. Key rules: the
model only proposes tool calls and never holds a credential; `policy.py` is a pure gate returning
a mode per call (`execute` / `confirm` / `dry_run` / `denied`), and the phase lives in its tables;
money tools build through `trade_bot.prepare_order` / `prepare_scalper` and never place an order
themselves (a test enforces that only `execution.py` touches `CoinRoutesTrader`); every verdict is
journaled under `logs/operator/`. Per-desk playbooks are `config/operator/{desk}.md`.

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

`research/{track}/` = one folder per analysis (today: `jason/`), each a `_build_*.py` that
programmatically builds AND executes its `.ipynb` (SVG figures, time-series panels; notebooks are
outputs — never hand-edit). Judge results in excess of rf and document nulls. `research/` may
import `src`; `src` must never import `research/`.

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
