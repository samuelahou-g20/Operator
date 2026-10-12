# Onboarding — taking over or joining this repo

This is the handoff doc: what a new engineer or a new owner needs to get running, what's safe to
change, and what has to be transferred out-of-band (nothing here duplicates `README.md`,
`ARCHITECTURE.md`, or `docs/bots.md` — it points at them).

## 1. Prerequisites

- `uv` and Python 3.12 (see `ARCHITECTURE.md` §14 — no conda/poetry/pip-tools).
- `rg` (ripgrep) for searching the codebase.
- Redis, only if you'll run live streaming (`README.md` has the docker command).

## 2. Get the environment running

```bash
uv sync                 # recreate the exact env from uv.lock
cp .env.example .env    # then fill in real values — .env.example is the complete, sourced list
uv run pytest           # confirm the test suite passes before changing anything
```

`.env.example` is the single source of truth for every environment variable this repo reads,
grouped by purpose with a note on where to obtain each value. `docs/bots.md` additionally maps
*which bot reads which var* — if you add a new var, add it to `.env.example` first.

## 3. Data lake

See `README.md` → "Data lake location" for the symlink wiring (fresh clone vs. existing worktree).
Don't duplicate those steps here — that section is kept current there.

## 4. Running and restarting a bot

See `docs/bots.md` → "Run & operate" for the long-poll skeleton, the one-poller-per-token rule,
and the manual restart procedure (find with `ps`, kill, relaunch). If opt-in systemd supervision
has been set up (`deploy/systemd/`, see `docs/bots.md` → "Supervision"), prefer
`systemctl --user restart desk-bot@<name>.service` instead of a manual kill/relaunch.

Logs: `logs/<bot>/<bot>.log` (rotating, 10MB × 5 backups).

## 5. Out-of-band handoff — things git cannot carry

A `git clone` gets you the code. It does **not** get you:

- The real `.env` values (credentials — never committed).
- The two Google service-account key files: `.gsa_key.json` (agg_bot, readonly) and
  `.mm2_trades_gsa_key.json` (mm2_trade_log_bot, editor) — both git-ignored. Get these transferred
  by whoever administers the GCP project, or create new service-account keys and re-share the
  target Sheets with the new key's email (one SA can read/write any sheet shared with it — no new
  SA per sheet needed).
- BotFather token ownership — Telegram bot tokens are tied to whoever holds the BotFather
  conversation for that bot. Get `/token` from the current owner via BotFather, or have them
  transfer bot ownership.
- CoinRoutes account access (`CR_API_TOKEN_G20JPY`) and Pluto API credentials
  (`PLUTO_API_KEY`/`_SECRET`) — these come from the respective vendor accounts, not from this repo.

## 6. Desk-specific values a new owner must change in code

Most configuration lives in `.env` (per-deployment) or `config/*.yaml` (per-desk data). A few
things are intentionally **not** environment- or YAML-driven and require editing source directly
if you're standing this up for a different desk or account:

- `scripts/trade_bot.py`'s safety constants — `MAX_MULT`, `DEFAULT_MAX_POST_USD`, per-desk
  `max_usd`, `DEFAULT_QTY`, `PLACE_STATUS`. These are deliberately kept as Python constants, not
  config: they're safety invariants, not desk data — see the module's own docstring
  ("Config (hardcoded per the owner)") and `.claude/agents/bot-dev.md`'s rail: "never widen a cap
  unless told to." Keeping them in source means a widened cap shows up in a code diff a reviewer
  reads, not a silent config-file edit.
- `src/adapters/coinroutes/client.py`'s `DEFAULT_BASE_URL`/`TOKEN_ENV` constants — account-named
  (`g20jpy.coinroutes.io` / `CR_API_TOKEN_G20JPY`). Both are overridable via constructor args, but
  every current call site relies on the default, so a different CoinRoutes account means either
  passing overrides at each call site or changing these constants.
- Desk-specific data that *is* externalized — CoinRoutes strategy-name maps and Google Sheet
  IDs/gids — lives in `config/desks.yaml`. Edit that file, not the bot source, when onboarding a
  new desk/token roster.

## 7. Known limitations

- **Crash recovery is opt-in, not automatic by default.** Without installing the systemd units in
  `deploy/systemd/` (see `docs/bots.md` → "Supervision"), a crashed bot process stays down until a
  human notices and restarts it manually.
- **`trade_bot`'s trader allowlist gates who can place orders**, not who can read the desk chat —
  read the module docstring before assuming a chat member can or can't trade.

## 8. Who to ask

Start with the repo owner (`samuel.ahou@g20strategies.com`) for anything not covered above,
especially vendor account transfers (CoinRoutes, Pluto, BotFather, GCP).
