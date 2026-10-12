# Pluto `/api/portfolio` — verified reference

Everything here was probed live against `MM2GAMMA_TOTAL` on 2026-08-26. Where a claim could not
be established, it says so — this endpoint has a **silent-fallback** habit that makes guessing
dangerous.

## Endpoint

```
GET https://pluto-lon.g20digital.io/api/portfolio
```

Vendor's own example (from the error body) names `https://derivalytics.com/api/portfolio` — the
G20 host above is the one in use.

- **Fallback host**: `https://pluto.g20digital.io:667/...?plutoService=local` serves the same GET
  endpoints (`portfolio`, `jason_levels_v2`, `TimeSeries/query`, `BotManager`) — verified
  2026-10-02 — but not `/api/jsonrpc` (401). Callers use `src.common.pluto.get`, which falls back
  automatically; see docs/bots.md.
- **Auth**: headers `X-Api-Key` + `X-Api-Secret` (`PLUTO_API_KEY` / `PLUTO_API_SECRET` in `.env`).
  Missing → `401 {"error":"Missing API key credentials."}`; wrong → `401 "Invalid API key
  credentials."`
- ⚠️ **Self-signed cert** — requests need `verify=False` (and
  `urllib3.disable_warnings(InsecureRequestWarning)`), as `agg_bot` already does.

## Parameters

Omitting `portfolio` makes the API document itself:

```json
{"Params": "portfolio (required), riskreport (default: DEFAULT), market (default: VOLPUB),
            horizon (default: NOW), user_id (default: G20),
            detailed (optional: true|detailed|greeks|positions|cube)"}
```

Param **names are case-insensitive** (`riskreport` and `RiskReport` both work).

| Param | Validated? | Effect | Notes |
|---|---|---|---|
| `portfolio` | ✅ 400 on bad name | required | `TREASURYPROPTOTAL`, `TOTAL_OPTIONS`, `MM2GAMMA_TOTAL`, `*OPTIONSMM` |
| `riskreport` | ✅ 400 on bad name | **large, on greeks only** | see below |
| `user_id` | ✅ 400 on bad name | resolves object ownership | default `G20`; portfolios are per-user |
| `detailed` | ❌ **silent fallback** | selects response shape | see below |
| `market` | ❌ not validated | **none observed** | `BOGUSMKT` == `VOLPUB`, same numbers |
| `horizon` | ❌ not validated | **none observed** | `NOW`/`YESTERDAY`/`T+30` all identical |

### `riskreport` — the one that matters

Names a saved `RiskReport` object; Pluto resolves it via `GetUserObject("G20","RiskReport",name)`,
so the value must exist. **Case-insensitive**: `sticky strike fitted` returns numbers identical to
`STICKY STRIKE FITTED` (proven by interleaved calls at one spot).

⚠️ **Omitting it silently applies `DEFAULT`, a different model.** Reproduced three times at two
spots — it moves the **greeks**, not the PV:

| | `STICKY STRIKE FITTED` | omitted (`DEFAULT`) |
|---|---|---|
| pv | 207,567.27 | 207,567.20 (unchanged) |
| **delta** | **34,367.98** | **37,784.81  (+3,417, ~10%)** |
| vega | 2,730.30 | 2,715.43 |

So a PV-only consumer is insensitive to it; **anything reading delta must send it.**

### `detailed` — unrecognised values do NOT error

Only `greeks`, `positions` and `cube` give distinct shapes. `true` and `detailed` return the
greeks shape. **Anything else — including a typo — returns `200` and a
`{counterparty: pv}` dict**, not an error. A misspelt `detailed` therefore looks like success and
silently yields the wrong shape.

## Response shapes

**`detailed=greeks`** (also `true`, `detailed`) — portfolio aggregate, keyed by `biprod`:

```json
{"pv": 207567.27,
 "deltas": {"ICPUSD": 34367.98},  "delta_usds": {"ICPUSD": 82246.6},
 "gammas": {"ICPUSD": 14144.25},  "gamma_usds": {"ICPUSD": 33592.6},
 "vegas":  {"ICPUSD": 2730.30},   "thetas":     {"ICPUSD": -1698.73},
 "spotrefs": {"ICPUSD": 2.393}}
```

⚠️ `pv` is a **single portfolio-wide scalar, not per-biprod**. To split PV by currency you must
use `detailed=positions`.

**Units** (all pinned by reconciliation, not assumed):

| Field | Unit | Check |
|---|---|---|
| `deltas` | coins | sums to per-position `_values[2]` |
| `delta_usds` | USD | `= deltas x spotref` exactly |
| `gammas` | coins of delta **per 1% spot move** | predicted Δdelta 2,362 vs actual 2,252 over a 0.167% move |
| `gamma_usds` | USD | `= gammas x spotref` exactly |
| `vegas` | **USD per 1 vol point** | 2,730 vs 2,734 computed by hand from strikes |
| `thetas` | USD per day | |

**`detailed=positions`** — a list of `{"posn": {...}, "greek": {...}}`.

`posn` carries `code` (`FX` = forward, `C` = call, `P` = put), `biprod`, `ntl_ccy`, `ntl_amount`
(signed), `K`, `expiry_date`, `counterparty`, `reval_vol`, `official_vol`, `contract_size`,
`settle_style`. ⚠️ `official_pv` and `total_profit_loss` were **0.0 on every position** — do not
use them for PV.

`greek` is `{"_values": [40 floats], "pv_ccy": "USD", "biprod": "ICPUSD"}`. ⚠️ **`_values` has no
labels in the payload.** Indices established by reconciliation:

| Index | Meaning | How verified |
|---|---|---|
| `[1]` | **PV** | sums to aggregate `pv`; forward reproduced by hand: `-491,000 x (2.377 - 2.37) = -3,437` |
| `[2]`, `[3]` | delta (coins) | sums to aggregate `deltas` |
| `[4]` | delta (USD) | `= [2] x spotref` |

Only these four are trusted; the rest of the 40 are unmapped.

**`detailed=cube`** — the positions list wrapped as `{"positions": [...]}`.

## ⚠️ The book revalues live — fingerprinting a param does not work

Consecutive calls drift on their own: `pv` creeps ~-0.08 and `delta` ~+0.05 per call from the
clock advancing, and delta steps with spot via gamma. Two calls with different params, taken
seconds apart, differ **whether or not the param does anything**. This cost real time twice.

To test any parameter, **interleave** it (A, B, A) and require B to differ from *both*
neighbouring A's at an unchanged `spotref`. That is how `horizon`/`market` were shown to be inert
and `riskreport` shown to be real.

## Historic cube — `load_historic_cube` on `/api/jsonrpc`

Probed live against `TREASURYPROPTOTAL` on 2026-09-30. This is the only way found to see a
**past** book — `/api/portfolio`'s `horizon`/`asof`/`date` params all return the live book.

```
POST https://pluto-lon.g20digital.io:667/api/jsonrpc     (same X-Api-Key / X-Api-Secret)
{"jsonrpc":"2.0","id":1,"method":"load_historic_cube","params":{"UserId":"G20",
 "portfolio":"TREASURYPROPTOTAL","riskreport":"STICKY STRIKE FITTED",
 "start_date":"2026-09-23","start_time":"08:00"}}
```

- ⚠️ **Send compact JSON** (`separators=(",", ":")`). With the default `", "` / `": "` spacing
  the server answers `200 {"error":{"code":-32603,"message":"Json Error"}}` — for ANY method,
  including a made-up one, so the error says nothing about the payload.
- Needed a server-side enablement for our key (it was `401`, empty body, before).
- **One stored snapshot per hour**, times in UTC. It returns the stored cube nearest the ask —
  `08:00` came back as `07:00:49`, `14:05` as `14:00:35` — so use `result.cube.timestamp`
  (epoch ms), never the requested time. History reaches back at least to 2026-01-01.
- **~13 MB, 1–14 s per call.** `result.cube` holds `positions` and `greeks` (parallel lists, same
  shape as `detailed=positions`: `_values[1]` PV, `[4]` delta USD), plus `surfs` (~9.5 MB of vol
  surfaces) and `data` (~4.5 MB) that nothing here reads. `result.Msg` is a one-line
  `PV:$158,739 @ 2026-09-23 07:00:49 (128 trades)` + spots; `MsgType` is `SUCCESS` on success.
- The summed position PVs equal the PV in `Msg` exactly.
- `cube.utils` carries `EODBotMTDPnL` / `EODBotYTDPnL` — unverified, unused.
