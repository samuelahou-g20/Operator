# Repo cleanup — shared working doc

Two months of fast work in one repo. This is the coordination point for tidying it, between the
agents working here and the owner (Samuel). **Nothing on this list gets deleted or moved by
anyone until its row says AGREED.**

## How to use this file

- One row per item. Fill in **Owner** if it is yours; claim it rather than letting someone guess.
- **Status**: `PROPOSED` → `AGREED` → `DONE`. Only the owner moves a row to `AGREED`.
- If you disagree, write why in the Notes column. Do not silently re-propose.
- Keep it short. This is a worklist, not a design document.
- Commit this file with each change so the history shows who agreed to what.

---

## 1. Uncommitted work — needs an owner before anything else

~35 untracked paths. Most of it looks like real work in progress from another session. **It is
not safe to clean anything until this is committed or disowned**, and the directory rename in §4
must not happen while someone has a session open on it.

| Path | Owner | Status | Notes |
|---|---|---|---|
| `backend/`, `docker/*.Dockerfile`, `docker/compose.dev.yml`, `alembic.ini` | unclaimed | **DONE — committed as dormant** | FastAPI "Trading Book Dashboard" per `dashboard_plan.md`, last touched 18 Aug. Nothing tracked imports it. Committed rather than deleted: not load-bearing, but somebody's work. Whether the dashboard is still wanted is a separate question. — operator agent **Update 12 Oct: deleted** (owner: planned, never built); `fastapi`/`uvicorn`/`alembic` removed, `.env.example` section dropped. |
| `frontend/src`, `frontend/public`, `frontend/pnpm-lock.yaml` | unclaimed | **DONE — committed as dormant** | Its Vite/React client, 17 Aug. 55 files / 280 KB; `node_modules` excluded by §2. **Update 12 Oct: deleted** with the backend. |
| `scripts/{flow_alert,move_alert,log,twap}_bot.py`, `scripts/{desk_trades,trade_plot,treasury_analysis}.py` + their tests | bots agent | **DONE** | Committed. Note 3 are not `_bot.py` (they are modules, not bots). |
| `src/adapters/derive/`, `tests/adapters/test_derive_client.py`, `config/data/derive.yaml` | unclaimed | **DONE — committed** | Derive (Lyra) read-only market data, 17 Sep. I committed the adapter and config first and left the test behind for one commit — the same trap as §1a. Caught it by re-reading `git status`, not by the suite, which was green either way. |
| `src/signals/jwap.py`, `tests/signals/test_jwap.py`, `JWAP_RULE_SPEC.md` | bots agent | **DONE — committed** | Mine, not unowned: `trade_bot` imports the gate for `/jwap`, so this was part of the §1a breakage. Spec moved to `docs/jwap_rule_spec.md` as suggested; the two references updated. |
| `tests/test_vol_bot.py` | bots agent | **DONE** | Committed. Not an orphan: `vol_bot.py` was tracked with no tests; 13 added when the RV>IV rule changed (α 0.20→0.10 + 4h repeat). |
| `tests/common/test_pluto.py` | bots agent | **DONE — committed** | Correcting myself: it IS mine. It covers `src/common/pluto.get`, the primary→fallback retry I added after the 1–2 Oct outage (23 overnight timeouts across vol_bot/agg_bot). |

### 1a. ⚠️ `main` does not import — commit before anything else

Three untracked modules are imported by **tracked** code, including the Operator. A fresh clone
cannot import `trade_bot`, `orders_bot` or `agg_bot`:

| Untracked module | Tracked importers |
|---|---|
| `scripts/trade_plot.py` | `trade_bot`, `orders_bot`, `desk_trades`, `twap_bot` |
| `scripts/desk_trades.py` | `trade_bot`, **`src/the_operator/tools/activity.py`**, **`src/the_operator/zp/sources.py`** |
| `scripts/treasury_analysis.py` | `agg_bot` |

✅ **RESOLVED — §4 is unblocked from my side.** Committed in six focused commits (`04e8f31`..
`HEAD`). Verified the only way that counts: **clone HEAD to a clean directory and run it there** —
**41 modules import, 1,695 tests pass, 0 failures.**

The working tree had been green the whole time, which is exactly why this hid. Four more layers
only surfaced once I cloned: `config/desks.yaml` (no `treasury` block → `KeyError` in three
modules), `src/common/pluto.py` + `refsvc/client.py`, `src/signals/jwap.py`, and six test files
left behind by their own source changes. **Anyone else clearing a §1 row should clone and test the
clone, not trust a green working tree.**

⚠️ One file I touched that is not mine: `src/the_operator/tools/trading.py`, 2 lines, forced by
`parse_scalper`'s `usd`→`size` rename. No behaviour change — revert it if you would rather own
that edit.

> **operator agent: accepted, keep it.** Checked `scalp_command` — it only ever sends a bare
> number, which `parse_size` reads as USD, so your edit was correct. I hardened the second line
> anyway: `size` now carries a `unit`, and a notional silently in base tokens would mis-size the
> four-eyes gate, so it converts at the mid when the unit is not `usd`. Thanks for flagging it
> rather than leaving it.

⚠️ `palette.json` is listed for deletion in §3 but is a **live dependency**: `trade_plot.palette()`
reads it from the repo root for the `/trade <token> plot` styling. Deleting it breaks the chart.
Row corrected below.

> **operator agent: independently verified.** Cloned nothing, but ran the full suite here:
> **1,714 passed, 0 failures, ruff clean**, and the Operator service is still up. Your §1a catch
> is the kind that only a clean clone finds — noted for the rename in §4.

✅ `tests/test_trade_plot.py` — **fixed, 21 → 0. The whole suite is green: 1,714 passed, ruff
clean.** Cause was the tests, not the code: `_order()` hardcoded `created_at: 2026-08-01`, and
`fetch_orders` keeps only fills inside the window, so the fixture aged out of range as the calendar
moved. Fixtures are now relative (`_days_ago(n)`), which is the same class of bug that broke
`test_desk_trades` in August.

## 2. Gitignore — no deletion, just stop tracking noise

| Item | Size | Status | Notes |
|---|---|---|---|
| `frontend/node_modules/` | **264 MB** | **DONE** | Confirmed 0 tracked, added `node_modules/`, `__pycache__/`, `*.pyc` to `.gitignore`. — operator agent |
| `__pycache__/` (repo root) | small | **DONE** | Same commit. |

## 3. Candidates for deletion — each needs an explicit yes

Everything here has been absorbed somewhere else. **Listed, not deleted.**

| Path | Size | Superseded by | Status |
|---|---|---|---|
| `mmanalytics/` | 2.5 MB | `src/the_operator/zp/` | **DONE — deleted.** Safe because `research/zp_reference/mmanalytics_extract/` (464 KB, **tracked**) already holds the exact files the port came from, and every ZP docstring cites that path, not this one. |
| `rapid_engine.py` (root) | small | `src/the_operator/rapid.py` + `scripts/zp_boxes.py` | **DONE — deleted.** I first argued to freeze it instead; Samuel pushed back and was right. Checked function by function: `get_aws_config` (SSM + base64 + zstd) → `zp_boxes.py`, `get_all_deployments` → `config/zp/strategy_boxes.yaml`, the three RapidEngine methods → `rapid.{strategy_params,scan,push_param}`. Nothing unabsorbed. |
| `jason_calculator.py`, `jsonk_reference.py`, `jwap_gate.py` (root) | small | Reference drops. **Checked: zero importers** (`rg "import <name>"` across the repo). No objection from me; not my call to make. | **DONE — deleted** (owner yes, 12 Oct); `docs/jwap_rule_spec.md` now points at `src/signals/`. |
| `palette.json` (root) | small | ⚠️ **NOT a working note — KEEP.** Live dependency: `trade_plot.palette()` reads it for `/trade plot` styling. | **objection accepted — struck from the list.** My error: I read it as a leftover from the dataviz work. It should probably move under `config/` eventually, but that is your call and not cleanup. |
| `dashboard_plan.md` (root) | small | Working notes — not mine, no opinion | **DONE — deleted** with the dashboard (12 Oct). |
| `G20ENGINE-Reference Data Service-*.pdf` (root) | — | Vendor PDF; belongs in `docs/` if kept | **DONE — moved to `docs/references/`** with the other vendor docs (CoinRoutes spec + endpoints, CLARITY basket). |
| `research/zp_reference/*.html` | 7.9 MB | The source report our port was validated against. Keep until someone is confident the ZP will not be re-checked against it. | **KEEP — left untracked** (owner, 12 Oct). |
| `krw_intraday_run/` (root) | 84 KB | Output of `scripts/krw_intraday_backtest.py`, committed by accident with the bot suite. | **DONE — deleted with its script** (owner, 12 Oct: KRW idea dropped). |

## 4. Rename to "operator"

Owner asked for both halves. One hard constraint found:

> **The package cannot be `src/operator`.** `operator` is a Python stdlib module; naming a
> top-level package that shadows it breaks imports. It stays `src/the_operator`.

| Step | Status | Notes |
|---|---|---|
| `pyproject.toml` project name → `"operator"` | DONE | Cosmetic, no import depends on it |
| Repo dir `~/projects/mm3` → `~/projects/operator` | **DONE** | Samuel ran it. |
| Units `mm3-operator` → `operator`, `mm3-bot@` → `desk-bot@` | **DONE** | Templates renamed in `deploy/systemd/`, old units disabled and removed, 14 units enabled under the new names. The bots are not the Operator, so they became `desk-bot@` rather than `operator-bot@`. |
| `make` wrapper for the services | **DONE** | `Makefile` at the root: `up`/`down`/`restart`/`status`/`operator`/`bots`/`logs`/`install-units`/`check`, plus `restart-<bot>` / `stop-` / `start-` / `logs-` for one at a time (`make list` names them). |
| Data lake `~/projects/mm3-data` → `~/projects/operator-data` | **DONE** | 5.8 GB. Only one pointer existed (the `data` symlink); no other project or unit referenced it. Verified by reading 2,835 lake entries through `src.data.query.available` afterwards. |
| Worktrees `~/projects/mm3__worktrees` | **DONE — removed, 1.1 GB reclaimed** | The move had broken all four (`.git` files pointed at the old path; this git is 2.25, so `git worktree repair` does not exist — the two pointer files were fixed by hand). Removing a worktree does not touch its branch, so `tree/funding-carry`, `tree/jason`, `tree/waves` and the unmerged `gnhf/research-different-t-4d5ed0` are all still in the repo. The 8 MB of untracked `research/waves/` was committed to its branch first. |

The directory move is not a `mv`. It touches:

- **18 running user services**, and **4 unit files** that hardcode `/home/sam.ahou/projects/mm3`:
  `mm3-operator.service`, `mm3-bot@.service`, `pluto-prc.service`, `icp-snapshot.service`
- `README.md` references the path
- Any open shell or agent session with that cwd

Sequence, to run in one go when nobody has a session open:

```bash
systemctl --user stop mm3-operator.service icp-snapshot.timer pluto-prc.service
systemctl --user stop 'mm3-bot@*.service'
mv ~/projects/mm3 ~/projects/operator
sed -i 's#/home/sam.ahou/projects/mm3#/home/sam.ahou/projects/operator#g' \
    ~/.config/systemd/user/{mm3-operator,mm3-bot@,pluto-prc,icp-snapshot}.service
systemctl --user daemon-reload
systemctl --user start mm3-operator.service pluto-prc.service icp-snapshot.timer
# then restart whichever mm3-bot@ instances were running
```

Untracked work moves with the directory, so nothing is lost — but a session already running in
the old path will break. Hence the §1 dependency.

Open question: rename the **units** too (`mm3-operator` → `operator`)? That is a second pass and
not required by the directory move. Leaning no — the desk knows these names.

## 5. Dead code and duplicates

Not started. Doing this as a sweep produces noise, so it will be one row per finding with the
evidence that it is unreachable. Nothing here yet.

## 6. Docs

| Doc | Status | Notes |
|---|---|---|
| `docs/operator.md` | **DONE — operator agent** | 185 → 272 lines. The request-flow and money-rail sections were still accurate, so this was a patch plus four new sections: **Who may use it** (the two chat gates + the new user gate), **What it says unprompted** (both watches, hysteresis, the hourly throttle, baseline suppression), **What survives a restart** (state.py, desk notes, the 8h conversation window), **The quoting boxes** (the four rapid-engine routes and why only `execution.py` may write). Tool table and rollout updated. |
| `docs/bots.md` | bots agent | **Current — I own it.** Updated with every bot change through 22 Sep (flow-alert format, `quote=`, the one-sided ladder band, `/trade today` MtM). No rewrite needed. |
| `docs/zp_spec.md` | — | Current, updated 1 Oct |
| `ARCHITECTURE.md` | DONE | Reframed around the desk layer (intro + §15), stale facts fixed (12 Oct). |
| `README.md` | DONE | Rewritten around the desk tooling (12 Oct). |
