# Operator — Trading System

A quantitative trading system. **[ARCHITECTURE.md](ARCHITECTURE.md) is the design source of
truth** for structure, module responsibilities, and interfaces — read it before changing
anything. Agent operating rules live in [CLAUDE.md](CLAUDE.md) and `.claude/rules/`.

**New to this repo?** Start with [ONBOARDING.md](ONBOARDING.md) — setup, credential handoff, and
what's safe to change.

## What's implemented

- **Data layer** — fetch → normalize → query, OOM-safe and chunked:
  - **Binance** USD-M perps: historical backfill (data.binance.vision dumps + REST) and
    live streaming (trades, L1) over Redis.
  - **CME** futures via Databento: historical, live, and **continuous stitching** (roll
    + back-adjustment into es1/es2 strips).
- **Execution layer**:
  - Thin adapters for **Binance** (USD-M) and **Hyperliquid** (perps).
  - **OMS** (`src/oms/`): orders/fills/positions in SQLite, avg-cost PnL, reconciliation,
    and **live fill ingestion** from the Binance user-data stream.

## Setup

`uv`-only, run from the repo root (see ARCHITECTURE §14):

```bash
uv sync                # create/refresh the venv from uv.lock
uv run pytest          # run the test suite
uv run ruff check .    # lint
```

`src` is a package (`from src.data.query import get`); run entry points as modules:
`uv run python -m scripts.<name>`.

**Credentials** go in a git-ignored `.env` at the repo root (only the env-var *names*
appear in `config/`). **[.env.example](.env.example) is the complete, sourced list** of every
credential this repo reads — copy it (`cp .env.example .env`) and fill in real values.

**Live streaming needs Redis** (`localhost:6379`): `docker run -d --name operator-redis -p 6379:6379 redis:7-alpine`.

### Data lake location

The lake lives **outside the repo**: `data/` is a symlink to a shared canonical lake (this
machine: `/home/sam.ahou/projects/operator-data`, ~GBs), so the repo stays small and every
worktree shares one copy. Code always writes to `data/…` via `src/common/paths.py`
(`DATA_DIR = REPO_ROOT / "data"`) — the symlink does the redirect, so backfills and fetches
"just work" with no extra flags. **Run them from the repo root** (`uv run python -m
scripts.backfill …`). Don't run two backfills on the same `(source, data_type)` at once —
they write the same physical files.

- **This worktree is already wired** — nothing to do.
- **New worktrees** are wired automatically by the `.git/hooks/post-checkout` hook on
  `git worktree add` (idempotent; it refuses to touch a `data/` that holds real files).
- **A fresh `git clone`** has neither the symlink nor the hook (both live under `.git/`, so
  they aren't cloned). Wire it manually before any fetch/backfill:
  ```bash
  git update-index --skip-worktree data/raw/.gitkeep data/processed/.gitkeep data/state/.gitkeep
  rm -r data                                   # remove the tracked empty scaffold
  echo '/data' >> .git/info/exclude            # ignore the symlink locally
  ln -s /path/to/operator-data data               # point data/ at the shared lake
  ```

## Common tasks

```bash
# Historical data -> raw -> processed
uv run python -m scripts.backfill  --source binance --symbols btcusdt
uv run python -m scripts.normalize --source binance --symbols btcusdt
uv run python -m scripts.resample  --source binance --symbols btcusdt --src 1m --dst 5m
uv run python -m scripts.daily_fetch --source binance              # yesterday + quality checks

# CME futures (Databento) + continuous strips
uv run python -m scripts.backfill --source cme --symbols es
uv run python -m scripts.normalize --source cme --symbols es
uv run python -m scripts.stitch   --source cme --symbols es

# Live (needs Redis)
uv run python -m scripts.start_streamer --source binance                      # market data
uv run python -m scripts.start_streamer --source binance --component aggregator   # derived bars
uv run python -m scripts.start_streamer --source binance --component user --testnet  # fills -> OMS
```

## Reading data & placing orders (from code)

```python
from src.data import query
query.get("perps", "binance", "btcusdt", "bars", timeframe="1m")  # ts-indexed DataFrame
query.get_continuous("cme", "es", strip=1, timeframe="1d")        # continuous futures strip
query.available()                                                  # what's in the lake

from src.oms.oms import OMS
oms = OMS()
oms.submit("binance", "my_strategy", "btcusdt", "buy", 0.01, "limit", price=59000)
oms.reconcile("binance")                                           # local books vs venue
```

All lake reads go through `src/data/query.py`; all live orders go through the OMS — never
the adapters directly (ARCHITECTURE §3, §10).

## Repository layout

`src/common/` (paths, logging, config, universe) · `src/data/` (fetching, streaming, etl,
schemas, query, lake_io) · `src/adapters/` · `src/oms/` · `config/` (universe, data) ·
`scripts/` (thin CLIs) · `tests/` (mirrors `src/`). Data lake under `data/` (git-ignored).