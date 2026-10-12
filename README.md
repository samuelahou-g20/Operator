# Operator

Tooling for G-20's market-making desks (MM3, MM2, MM4, treasury, options, OTC). At its center is
**The Operator**, a Telegram agent: a trader types `/operator <plain language>` in a desk chat and
it runs the desk's standard lookups, or prepares an order that a human must confirm. Around it sit
the **desk bots** (one Telegram command each), the **ZP quoting review**, and a few **scheduled
jobs**. A data lake and a research track support all of this; they are not the product.

**New here?** Read [ONBOARDING.md](ONBOARDING.md) first: setup, credential handoff, and what is safe
to change. Agent rules live in [CLAUDE.md](CLAUDE.md) and `.claude/rules/`.

## What runs

| Piece | What it does | Reference |
|---|---|---|
| **The Operator** (`src/the_operator/`) | Plain-language desk agent. Read tools run directly; every money or write tool needs a human confirm. Also speaks first when quoting books stop or market share leaves its band. | [docs/operator.md](docs/operator.md) |
| **Desk bots** (`scripts/*_bot.py`) | `/agg`, `/orders`, `/hedge`, `/trade`, `/jason`, `/ls`, `/vol`, `/macro`, `/sentiment`, `/twap`, `/botstatus`, plus push alerts and the MM2 fill logger. `trade_bot` is the only one that moves money, behind its allowlist and caps. | [docs/bots.md](docs/bots.md) |
| **ZP quoting review** (`src/the_operator/zp/`) | Per-token quoting report: markouts, captures, widths, compliance, P&L. | [docs/zp_spec.md](docs/zp_spec.md) |
| **Scheduled jobs** | `pluto-prc` refreshes the Pluto risk tabs in the MM2 sheet; `icp-snapshot` stamps the ICP book's state at the daily cutoff. | module docstrings |

Everything runs as `systemd --user` units on the desk host: `operator.service`, one
`desk-bot@<name>.service` per bot, and the two timers.

## Run it

Use the Makefile, from the repo root (`make help` lists everything):

```bash
make status              # one line per unit: what is up, what is not
make operator            # restart the Operator (the usual one after a change)
make restart-trade       # restart one bot; also start-, stop-, logs-<name>
make list                # the bot names those targets accept
make restart             # restart everything, e.g. after pulling new code
make check               # lint + tests: run before restarting anything
make install-units       # first time on a host: install deploy/systemd/* for this checkout
```

## Setup

`uv` only, run from the repo root:

```bash
uv sync                  # create the env from uv.lock
cp .env.example .env     # then fill in real values
uv run pytest            # confirm the suite passes
```

**Credentials** go in the git-ignored `.env`. [.env.example](.env.example) is the complete list of
every variable this repo reads, with where to get each one. `docs/bots.md` maps which bot reads
which variable.

Run entry points as modules (`uv run python -m scripts.<name>`): `src` is imported as a package.

### Data lake

`data/` is a symlink to a shared lake outside the repo (this machine:
`/home/sam.ahou/projects/operator-data`), so every worktree shares one copy. New worktrees are
wired by the `.git/hooks/post-checkout` hook. A fresh `git clone` has neither the symlink nor the
hook, so wire it by hand before any fetch:

```bash
git update-index --skip-worktree data/raw/.gitkeep data/processed/.gitkeep data/state/.gitkeep
rm -r data                                   # remove the tracked empty scaffold
echo '/data' >> .git/info/exclude            # ignore the symlink locally
ln -s /path/to/operator-data data            # point data/ at the shared lake
```

Live streaming needs Redis on `localhost:6379`:
`docker run -d --name operator-redis -p 6379:6379 redis:7-alpine`.

## Supporting pieces

- **Data lake** (`src/data/`): fetch → normalize → query for Binance, Bybit, OKX, Hyperliquid,
  MEXC, CME and FRED. Every read goes through `src/data/query.py`; `scripts/daily_task.sh` runs
  the daily fetch.
- **Signals** (`src/signals/`): JASON levels, the JWAP gate, trend and mean-reversion scores that
  the bots and the Operator use.
- **Research** (`research/`): JASON pivot analysis, each folder with a builder that writes its
  notebook.
- **OMS** (`src/oms/`): order/fill/position records and reconciliation for the venue adapters.
- **Live monitor** (`uv run python -m scripts.dashboard`): a terminal view of live prices and
  features for the symbols in `config/dashboard.yaml`.

[ARCHITECTURE.md](ARCHITECTURE.md) has the design rules for the data and research layers.

## Layout

```
src/the_operator/   the Operator agent, its tools, policy gate and the ZP review
scripts/            desk bots (*_bot.py), scheduled jobs and thin CLIs
src/adapters/       thin API clients: CoinRoutes, trade service, refsvc, Binance, Hyperliquid, Derive
src/signals/        JASON, JWAP and other signals
src/data/           data lake: fetching, streaming, ETL, query
src/oms/            order management
config/             desks, Operator playbooks, ZP, universe and data configs
deploy/systemd/     unit files for the Operator and the bots
docs/               bots, Operator, ZP and JWAP references; vendor docs in docs/references/
research/           experiments and their notebooks
tests/              mirrors src/; bot and script tests sit at the top level
```
