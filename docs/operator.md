# The Operator — reference

`src/the_operator/` is a Telegram-facing agent: traders type `/operator <plain language>` in a
desk chat and it runs the desk's standard lookups with the same code the `scripts/*_bot.py` bots
use, then answers tersely. **Phase 3 (current): read tools execute; every MONEY and WRITE tool
needs a human confirm.** Money tools go out exactly as a typed `/trade` / `/scalper` would
(`trade_bot.PLACE_STATUS`, or `staged` if the requester asked) and live orders are watched for
fills. Four-eyes exists but is switched off. `CONTROL` is still denied outright — nothing starts
or stops an external system.

It is no longer purely request/response: two background watches **speak first** when the quoting
books stop or market share leaves its band (see "What it says unprompted").

## How a request flows

```
Telegram ──/operator …──▶ gateway (auth: served chat ∧ desk pin; admins; kill switch)
                             │ worker thread, one at a time per chat
                             ▼
                           agent (claude-opus-5-5 tool runner; frozen system prompt + desk facts)
                             │ proposes tool calls
                             ▼
                           executor ─▶ policy.check ─▶ handler (a bot's fetch+render)
                             │ journal every verdict/result; divert PNG/CSV to the reply
                             ▼
                           gateway sends text, then attachments
```

The model never executes anything and never holds a credential; it only emits tool calls.
`policy.py` is the gate: a pure function of risk class, desk, the requester's trader status, the
kill switch and the args, returning a **mode** per call (`execute`, `confirm`, `dry_run`, `denied`). The
phase is one table in `policy.MODES`. Caps and rails stay in `scripts/trade_bot.py`: the money
tools build through `trade_bot.prepare_order` / `prepare_scalper`, the exact pre-placement path a
typed `/trade` or `/scalper` takes, so a preview can never skip a rail the live bot enforces.

**Money tools never place.** `tools/trading.py` does not import `CoinRoutesTrader` and a test
scans every module but `execution.py` for that name. A handler returns a preview plus
`ToolResult.proposal`; the executor decides what happens to the proposal from the verdict's mode:

- `dry_run`: stamped "nothing was sent" (no tool uses it in phase 3; `policy.TOOL_MODES` can
  put one back).
- `confirm`: the gateway stores the proposal under a 6-char nonce and delivers the tool's card
  verbatim (no model paraphrase turn), followed by a ✅ Confirm / ❌ Abort button message. A tap
  arrives as a `callback_query` with the tapper's user id, so the gate is identical to the
  `/operator confirm <nonce>` fallback: same chat, within `CONFIRM_TTL_S` (120 s), and
  - the **requester** confirms their own proposal (any size). Four-eyes is built but **off**
    (`FOUR_EYES_USD = None`, the owner's call: traders need to place fast). Setting it to a USD
    notional turns it on above that size: a *different* allowlisted trader must confirm, the
    requester is refused, and the card and button message both say so.
  The model has no confirm tool. Buttons are removed once the nonce is consumed; `/operator halt`
  drops every pending proposal. On confirm, `execution.execute`:
  - `place_order`: re-runs `trade_bot.prepare_order` from the stored command (fresh caps, venues,
    max-post), refuses if the market moved more than `LIMIT_DRIFT_MAX` (5%, a dislocation), and
    otherwise sends **at the limit the human confirmed** (a limit order never fills worse than its
    limit) through `trade_bot.place_prepared`, with the status a typed `/trade` would use
    (`MONEY_STATUS = None`; a phase can force one). A live order is then watched with
    `trade_bot.poll_pending` and its fill / cancel / error announced in the chat.
  - `place_scalp`: re-reads the mid, refuses on the same 5% rule, then applies the **previewed**
    ladder through `trade_bot.apply_ladder` (reconciliation: unlisted resting levels are cancelled).
  - `cancel_order`: `trade_bot.handle_cancel`.

The requester must be on the desk's `TELEGRAM_TRADE_TRADERS*` allowlist (read from
`trade_bot.DESKS`, one source) and every money call carries a `reason`.

**Kill-switch drill** (do it once per desk before enabling the desk chat): `/operator status`
→ request a small order, do not confirm → `/operator halt` from an admin → the button message
loses its buttons and status shows 0 pending → `/operator confirm <code>` is refused → a read
request is refused → `/operator resume` → reads work again. Journal shows `admin halt` with the
dropped count.

## Who may use it

Two independent gates, both from `.env`:

- **`TELEGRAM_OPERATOR_CHAT_ID`** — the chats it answers in at all. A chat must ALSO be pinned to
  a desk by that desk's own `TELEGRAM_<DESK>_CHAT_ID`; a served chat with no pin is refused, never
  guessed.
- **`TELEGRAM_OPERATOR_USERS`** — the Telegram user ids allowed to use it. Checked in the gateway
  **before the model runs**, so an unlisted member costs nothing and is told the id to pass on.
  **Unset means every member of a served chat**, which is the pilot behaviour; startup logs a
  warning when it is unset, and `/operator status` says "OPEN to everyone in chat".

Money tools need more: the requester must also be on that desk's `TELEGRAM_TRADE_TRADERS*`
allowlist, read from `trade_bot.DESKS` so a trader the bot refuses is a trader the Operator
refuses. `OPERATOR_ADMINS` is a third, smaller list — halt/resume only.

## What it says unprompted

`watch.py` runs two background rounds per served desk. Both **announce sustained changes, not
states**, and both are throttled to **at most one message per desk per hour** (`Throttle`); what
fires inside the hour is held and folded into the next message rather than dropped. Scanning
stays frequent so a real outage is still found within minutes — detection cadence and
notification cadence are deliberately different.

- **Quoting watch** (5 min): a strategy must hold a bad state for `STOPPED_ROUNDS` (3) scans
  before it is mentioned, and hold `ok` for as many before the all-clear. A price resting on its
  band edge crosses many times an hour; alerting on each crossing produced hundreds of messages a
  day, which is what the hysteresis exists to stop. A strategy missing from a scan (its box did
  not answer) keeps its streak untouched — a box outage neither raises an alarm nor clears one.
- **Volume watch** (15 min): market share against the desk's 5%-50% rule, triaged from the
  aggregator sheet and diagnosed per venue only for the coins actually in breach. A coin the desk
  deliberately keeps low is dismissed in chat (`volume_snooze`) and skipped for the rest of the
  UTC day — the exceptions change daily, so there is no list in config.

**Whatever is already broken at the first scan after a restart is marked as known and never
replayed** (`watch.baseline`); only its recovery is reported. Without that, every restart pasted
the standing outage into the chat.

## What survives a restart

- **`state.py`** — pending money proposals and the live-order fill watchlist, one JSON file
  written atomically and whole. The watchlist is the one that matters: an order the Operator
  placed used to stop being watched mid-flight, so its fill was never announced. Proposals past
  `CONFIRM_TTL_S` are dropped on load rather than resurrected. Restoring happens in `main()`, not
  the constructor, so a test never inherits the running service's state.
- **`config/operator/{desk}.md`** — the desk playbook, read into every prompt for that desk. Above
  the `<!-- operator-notes -->` marker is human-curated prose; below it `remember` appends dated,
  attributed notes and **only ever appends**. Notes rank below the curated text and are labelled
  as what somebody said, not desk policy. They are in git, so a note the bot wrote shows up as an
  uncommitted change — that is how you see what the desk taught it.
- **Conversation memory** is in-process and lost on restart: 8-hour idle TTL, last 10 exchanges
  (`gateway.History`). 30 minutes was short enough that a follow-up after a coffee found the
  Operator with no memory of its own answer.

## The quoting boxes (rapid-engine)

`rapid.py` is the client for the LM2 strategy deployments listed in `config/zp/strategy_boxes.yaml`
(generated from AWS SSM by `scripts/zp_boxes.py`). Four routes, **all unauthenticated**:

| Route | Used for |
|---|---|
| `GET /rest/status/full` | strategy status, implied mid, the market's touch, 24h executed notional |
| `GET /rest/calculators/params` | every strategy's full parameter set |
| `GET /rest/calculators/overrides` | what is hand-set on top of the deployed config |
| `POST /rest/calculators/overrides/modify` | **the only write** — replaces a whole attribute group |

Because that POST takes no authentication, our own call sites are the only rail: `push_param`
lives in `rapid.py` and **only `execution.py` may call it**, after a human confirm. A test asserts
no other module even names it. Both writers re-read the live group first (the box replaces the
whole group, so a partial write silently drops the fields it omits) and refuse if the value moved
between the preview and the confirm.

The quoting formulas themselves are hardcoded in `zp/dynamic.py` from the desk's "Dynamic mode"
sheet, on the desk's instruction: the formulas change rarely, the parameters change daily.

## Commands

| Command | Who | Effect |
|---|---|---|
| `/operator <request>` | anyone on `TELEGRAM_OPERATOR_USERS` in a served chat | run the request |
| `/operator help` | anyone | usage |
| `/operator status` | anyone | phase, armed/halted, desk, model, tools offered on this desk |
| `/operator why <request id>` | anyone | replay the journal trail of one request (ids are in the "Working on it… (req …)" ack) |
| ✅ Confirm / ❌ Abort buttons (or `/operator confirm <code>` / `abort <code>`) | the requester of that preview only | release (re-priced, sent `staged`) or drop a pending money proposal |
| `/operator reset` | anyone | clear this chat's conversation memory (8-hour idle TTL otherwise) |
| `/operator halt` / `resume` | `OPERATOR_ADMINS` only | kill switch: every tool call is denied while halted |

## Tools

| Tool | Wraps | Desks |
|---|---|---|
| **money** | | |
| `place_order` (confirm) | `trade_bot.prepare_order` (parse → caps → venues → touch → limit/max-post), then `place_prepared`; fills via `poll_pending` | mm3 mm2 treasury |
| `place_scalp` (confirm) | `trade_bot.prepare_scalper` (mid, JASON sigma, 1..n σ levels), then `apply_ladder` | mm3 mm2 treasury |
| `cancel_order` (confirm) | `trade_bot.handle_cancel` | mm3 mm2 treasury |
| `mm_rebound` (confirm) | re-centres a token's `price_boundaries` across every venue (`rapid.push_param`) | any |
| `mm_set_param` (confirm) | one field of one quoting parameter group on the live strategies | any |
| **write** | | |
| `remember` (confirm) | appends a dated, attributed desk note to `config/operator/{desk}.md` (`notes.py`) | any |
| `volume_snooze` (confirm) | silences volume alerts for one coin for the rest of the UTC day | any |
| **read** | | |
| `aggregator` | `agg_bot` sheet / Pluto portfolio | mm3 mm2 treasury options mm5 |
| `jason_levels` | `jason_bot` | any |
| `twap` | `twap_bot.handle_twap` | any |
| `vol_surface` | `vol_bot` Deribit ATM surface | any |
| `ls_ranking` | `ls_bot.build_ranking` | any |
| `macro_report` | `macro_bot` (Claude web search); cached 4h, `refresh` to force | any |
| `sentiment` | `sentiment_bot` (paid X API); cached 4h, `refresh` to force | any |
| **advisory / awareness** | | |
| `desk_view` | evidence packet: macro → sentiment → positions (`hedge_bot`) + gamma deltas (`agg_bot`) → LS ranking (`ls_bot`) → rule flags (`tools/advisory.py`) | mm3 mm2 treasury |
| `buzz` | X posts from `config/operator/following.yaml` + CoinGecko trending → themes, tickers, evidence, tradability, mention velocity (`buzz.py`); cached 4h | any |
| `recent_chat` | the room's own conversation, last ≤8h (`chatlog.py`, in memory, never journaled) | any |
| `open_orders` / `recent_fills` | `orders_bot` | mm3 mm2 treasury |
| `fills_today` | `desk_trades.handle` | mm3 mm2 treasury |
| `hedge_pnl` / `hedge_risk` | `hedge_bot` | mm3 mm2 |
| `bot_status` / `bot_config` | `botstatus_bot` (Read endpoints only) | mm3 mm2 treasury |
| `zp` | the Zeyi Process quoting review (`zp/`): card + interactive HTML report + deterministic checks | mm3 mm2 |
| `mm_status` | which LM2 strategies are quoting, stopped in error, or deliberately disabled (`rapid.scan`) | any |
| `mm_overrides` | parameters hand-set on top of the deployed config | any |
| `volume_check` | market share vs the desk's 5-50% rule, per venue, with the cause and the parameter to change (`volume.py`) | any |

Adding a tool: write a `ToolSpec` (`contract.py`) in the right `tools/*.py`, reusing the bot's
fetch/render functions; strict-shaped schema (`additionalProperties: false`, every property in
`required`, nullable types for optional fields; no `minimum`/`maximum`, no `enum` next to a union
type — use `anyOf`). Only MONEY tools are *sent* with `strict: true`: with all 17 strict the API
rejects the request as "Schema is too complex". Declare `risk` honestly — it is what the policy
gates on. Changing what a risk class may do is a change to `policy.MODES`, reviewed as such; do
not "temporarily" widen it, and never give a money handler an execution branch — execution belongs
to the executor's confirm path (phase 2).

## Advisory mode

Only when a trader asks for ideas, a view, or whether the book is positioned right. The prompt
makes the model call `desk_view` first and reason in the owner's order: macro → sentiment →
risk-on/off and sectors → JASON stretch (`z_up ≥ 2` trim/sell, `z_dn ≥ 2` buy/cover, read
against our position) → every position versus its LT trend → every size versus the vol-scaled
resting-state delta (VS-RSD from `ls_bot`). The packet's section 5 applies those rules
deterministically (`advisory.rules`, tested), so the model narrates and justifies; it does not do
the arithmetic. Output is an IDEAS card ending "A view, not an order. Ask me to place if you
want." An idea becomes an order only through a separate explicit request and the confirm button.

`buzz` is the narrative leg: what the followed accounts and CoinGecko trending are talking
about, extracted to forward-looking themes (early / building / already moved; thesis, catalysts,
laggards, risk) with verbatim evidence and one web-research pass per top theme, checked for
tradability against the universe registry and the desk rosters, with mention velocity versus
previous scans (`data/state/operator/buzz/`). `card=true` sends the card to the chat as-is (the
"what's trending" ask); `card=false` makes it research input for a specific question, and
`topic` adds a keyword pull on X and focuses extraction and research on that subject. Tweets are
untrusted content; the extraction prompt says so.

A read tool can mark its result `verbatim` (`ToolResult.verbatim`, used by `buzz`): the gateway
sends that card to the chat exactly as built, before the model's reply, and the model is told to
add only its own read. Different from money cards, which end the request without a model turn.

`recent_chat` exists because the bot's privacy mode is off: the gateway records what traders say
in served chats (3h, memory only, never journaled) and the model reads it on demand rather than
on every request, which keeps the cached prompt prefix stable.

Paid or slow inputs (macro, sentiment, buzz) are cached 4h in-process (`cache.py`); the LS
ranking 30 min. Every cached line carries "(cached 2h13m ago)" so staleness is visible. A
background thread (`warm.py`, started in `main`) keeps macro and the LS ranking warm, because a
cold macro run is two web-search calls and can take three minutes; sentiment is never prefetched
(it is paid). With warm caches a `desk_view` takes about as long as the positions leg (~45 s).

## Journal

`logs/operator/journal/YYYY-MM-DD.jsonl`, append-only, one line per event: `request`, `turn`
(model stop reason, usage, API request id), `tool_call` (args + verdict), `tool_result` (timing,
mode, attachment digests, text, and for money tools the full `proposal` — the exact
`place_kwargs` or ladder that would have gone out), `pending` (nonce issued), `confirm` (who,
verdict), `execute` (proposal + CoinRoutes reply), `abort`, `reply`, `admin`, `error`.
Attachment bytes are never written.
`/operator why <id>` reads it back.

## Env (`.env.example` is the authority)

`TELEGRAM_OPERATOR_BOT_TOKEN` (own BotFather bot, one long-poller), `TELEGRAM_OPERATOR_CHAT_ID`
(served chats; each must also be pinned to a desk via `TELEGRAM_<DESK>_CHAT_ID`, startup fails
otherwise), **`TELEGRAM_OPERATOR_USERS`** (who may use it; unset = everyone in a served chat),
`OPERATOR_ADMINS`, `ANTHROPIC_API_KEY`, plus the read keys the tools use
(CoinRoutes per desk, Pluto, refsvc, Google SA, X).

## Run

```bash
uv run python -m src.the_operator                                  # dev, from repo root
```

On the desk host it runs as the `systemd --user` unit `operator.service`
(`deploy/systemd/operator.service`, same pattern as `desk-bot@`). **Use the Makefile**: `make
operator` restarts it, `make status` shows every desk unit, `make up` / `make down` the lot,
`make restart-<bot>` / `stop-` / `start-` / `logs-` one bot at a time (`make list` names them),
`make install-units` re-installs the unit templates with this checkout's path substituted in —
the templates carry a `/path/to/operator` placeholder and copying one in raw installs a unit that
cannot chdir. Logs via `journalctl --user -u operator -f` or
`logs/operator/operator.log`. Never hand-launch it alongside the unit: the single-instance flock
`/tmp/mm3_operator.lock` makes the second copy exit 0 (which is why the unit is
`Restart=on-failure`, never `always`).

## Rollout

0. read-only tools, pilot chat, journal. 1. money tools in dry-run. 2. `place_order` /
`cancel_order` behind the requester's confirm, sent `staged`. 3. **now**: orders as `/trade`
would send them, scalps live, fill watching, halt drops pending; four-eyes built but off. Since
then, still phase 3: the quoting levers (`mm_rebound`, `mm_set_param`) on the same confirm rails,
two background watches, durable state, desk notes, and a user allowlist.

**Before a real desk chat**: set `TELEGRAM_OPERATOR_USERS`, run the kill-switch drill above, and
expect the watches to start posting in that chat the moment it is served.

Not built: Pluto BotManager writes. The API exists (`POST /{bot}/start|stop`,
`/operations/start|stop`, `PUT /config/value`) and our key reaches it, but nothing calls it and
`policy` denies `CONTROL` outright; `botstatus_bot` is read-only by design and stays so.
