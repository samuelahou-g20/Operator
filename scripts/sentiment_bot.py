"""Telegram bot: on-demand X/Twitter sentiment for a topic (read-only research tool).

Long-polls Telegram. On bare ``/sentiment`` it pulls recent (last 48h) tweets across several
market THEMES (market, crypto, alts, tech/AI stocks, trending stocks, macro/Fed, earnings) from
the X API v2 recent-search endpoint (sorted by relevancy/engagement) and returns a macro-style
sentiment DIGEST scored by Claude — one bullet per theme with a **-1..+1 sentiment score**, an
overall net score, and trending names. Tweets from a curated FOLLOWING list are pulled as a
high-signal stream, weighted, and their contributors @-tagged. ``/sentiment <topic>`` gives a
single-topic read instead. ``/sentiment report`` generates the one-page G-20 PDF brief (frozen
template) and sends it as a document. Spam/giveaway/retweet-excluded at query time.

Cost is pay-per-use on the X API ($0.005/post read); every query is HARD-CAPPED at ``MAX_TWEETS``
so one ``/sentiment`` costs ~$MAX_TWEETS*0.005 at most (≈$1.50 at 300). No lake, no order path.

Env (loaded from repo-root .env):
  TELEGRAM_SENTIMENT_BOT_TOKEN  its OWN BotFather bot (one long-poller per token).
  TELEGRAM_CHAT_ID              authorized chat/group id(s); comma-separated to serve several.
  X_BEARER_TOKEN                X API v2 bearer token (app with pay-per-use credits).
  ANTHROPIC_API_KEY             Claude API key (scoring).

Run: ``uv run python -m scripts.sentiment_bot``

MUST NOT: place/modify orders, touch the data lake, or write anything but chat replies.
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timedelta, timezone

import anthropic
import requests

from src.common.config import load_config, require_env
from src.common.logging import get_logger
from src.common.telegram import drain_backlog, poll_updates

_log = get_logger("sentiment_bot")

MODEL = "claude-sonnet-5"
X_SEARCH = "https://api.x.com/2/tweets/search/recent"
MAX_TWEETS = 300           # single-topic cost cap: ~$MAX_TWEETS * $0.005 = ~$1.50
MAX_PER_THEME = 20         # digest: per-theme cap = what the digest sends Claude (8 * 20 * $0.005 = ~$0.80)
_PER_PAGE = 100            # X recent-search max_results ceiling
WINDOW_HOURS = 48          # look-back window
SORT_ORDER = "relevancy"   # X ranks by engagement/quality — ~90x more signal than "recency" (bots)
CACHE_TTL_MIN = 15         # reuse a query's tweets for this long (avoid re-fetch / re-pay)
TG_MSG_LIMIT = 4096

# In-process tweet cache: {f"{cap}:{query}": (fetched_at_epoch, tweets)}. Cleared on restart.
_CACHE: dict[str, tuple[float, list[dict]]] = {}

# Excluded at query time to cut the crowd noise Claude would otherwise have to discount.
_SPAM_EXCL = "-is:retweet -is:reply -giveaway -airdrop lang:en"

# Digest themes: (label, recent-search keyword group). One bullet per theme, in this order.
THEMES = [
    ("Market", "(\"stock market\" OR \"S&P 500\" OR SPX OR equities OR \"risk on\" OR \"risk off\")"),
    ("Crypto majors", "(bitcoin OR $BTC OR ethereum OR $ETH)"),
    ("Alts", "(altcoins OR \"alt season\" OR $SOL OR $XRP OR $AVAX OR $LINK)"),
    ("Tech stocks", "($NVDA OR $AAPL OR $MSFT OR $TSLA OR Nasdaq OR semiconductors)"),
    ("AI", "(\"AI stocks\" OR \"AI bubble\" OR \"AI capex\" OR datacenter OR \"AI trade\")"),
    ("Trending stocks", "(\"trending stocks\" OR \"stocks to watch\" OR \"most active\" OR \"unusual options\")"),
    ("Macro / Fed", "(Fed OR FOMC OR CPI OR inflation OR \"rate cut\" OR \"rate hike\" OR yields)"),
    ("Earnings", "(earnings OR \"after the bell\" OR guidance OR \"earnings beat\" OR \"earnings miss\")"),
]

# Single-topic (`/sentiment <topic>`) keyword groups; unknown topic falls back to the raw words.
TOPICS = {
    "btc": "(BTC OR bitcoin OR $BTC)",
    "eth": "(ETH OR ethereum OR $ETH)",
    "sol": "(SOL OR solana OR $SOL)",
    "crypto": "(crypto OR bitcoin OR ethereum OR altcoin)",
    "hype": "(HYPE OR hyperliquid OR $HYPE)",
    "fed": "(Fed OR FOMC OR Powell OR \"rate cut\" OR \"rate hike\")",
    "macro": "(macro OR CPI OR inflation OR recession OR yields)",
}

# Handles the desk follows (WITHOUT @), shared with the Operator's `buzz` scan. Their recent
# tweets are pulled into the digest as a high-signal stream (weighted + tagged by Claude); also
# used by `/sentiment <topic> infl` to scope a single-topic pull. Read at import: restart to pick
# up edits.
FOLLOWING: list[str] = load_config("operator/following.yaml").get("x_accounts") or []
FOLLOWING_CAP = 60          # max followed-account tweets pulled for the digest (all sent to Claude)
# Handles per query, to stay under recent-search's 512-char limit (fewer when a keyword rides along).
_FROM_CHUNK = 25
_FROM_CHUNK_KW = 20


# --------------------------------------------------------------------------- #
# X API pull (cost-capped)
# --------------------------------------------------------------------------- #
def _from_chunks(size: int) -> list[str]:
    return ["(" + " OR ".join(f"from:{h}" for h in FOLLOWING[i:i + size]) + ")"
            for i in range(0, len(FOLLOWING), size)]


def _build_query(topic: str, scope: str, handles: str = "") -> str:
    kw = TOPICS.get(topic.lower(), f"({topic})")
    parts = [kw, "-is:retweet", "lang:en"]
    if scope == "infl" and handles:
        parts.insert(1, handles)
    return " ".join(parts)


def _x_get(params: dict, bearer: str) -> dict:
    headers = {"Authorization": f"Bearer {bearer}", "User-Agent": "operator-sentiment"}
    for attempt in range(4):
        try:
            r = requests.get(X_SEARCH, headers=headers, params=params, timeout=30)
        except requests.exceptions.RequestException:
            if attempt == 3:
                raise
            continue
        if r.status_code == 429 or r.status_code >= 500:  # rate limit / transient — back off
            if attempt < 3:
                continue
        if r.status_code != 200:
            raise RuntimeError(f"X API {r.status_code}: {r.text[:200]}")
        return r.json()
    raise RuntimeError("X API failed after retries")


def _start_time() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pull(query: str, cap: int, bearer: str, start: str, sort: str = SORT_ORDER) -> list[dict]:
    """Pull up to `cap` recent tweets for `query`. Returns {user, text, likes, impressions, ts}.

    Cached in-process for CACHE_TTL_MIN minutes (keyed by query+cap+sort): repeated requests in a
    short window reuse the tweets instead of re-fetching — no extra X API reads, no extra cost."""
    key = f"{cap}:{sort}:{query}"
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_MIN * 60:
        _log.info("cache hit (%.0fs old, $0): %s", time.time() - hit[0], query[:60])
        return hit[1]
    params = {
        "query": query, "start_time": start, "max_results": _PER_PAGE, "sort_order": sort,
        "tweet.fields": "created_at,public_metrics,lang",
        "expansions": "author_id", "user.fields": "username",
    }
    out: list[dict] = []
    while len(out) < cap:
        data = _x_get(params, bearer)
        users = {u["id"]: u["username"] for u in data.get("includes", {}).get("users", [])}
        for t in data.get("data", []):
            m = t.get("public_metrics", {})
            out.append({"user": users.get(t["author_id"], "?"), "text": t["text"],
                        "likes": m.get("like_count", 0), "impressions": m.get("impression_count", 0),
                        "ts": t.get("created_at", "")})
        nxt = data.get("meta", {}).get("next_token")
        if not nxt or not data.get("data"):
            break
        params["next_token"] = nxt
    result = out[:cap]
    _CACHE[key] = (time.time(), result)
    return result


def fetch_tweets(topic: str, scope: str) -> list[dict]:
    """Single-topic pull (`/sentiment <topic>`), hard-capped at MAX_TWEETS in total."""
    bearer, start = require_env("X_BEARER_TOKEN"), _start_time()
    if scope != "infl" or not FOLLOWING:
        return _pull(_build_query(topic, scope), MAX_TWEETS, bearer, start)
    out: list[dict] = []
    for handles in _from_chunks(_FROM_CHUNK_KW):
        if len(out) >= MAX_TWEETS:
            break
        out.extend(_pull(_build_query(topic, scope, handles), MAX_TWEETS - len(out), bearer, start))
    return out


def fetch_themes() -> list[tuple[str, list[dict]]]:
    """Digest pull: MAX_PER_THEME tweets for each THEME (spam-excluded)."""
    bearer, start = require_env("X_BEARER_TOKEN"), _start_time()
    return [(label, _pull(f"{q} {_SPAM_EXCL}", MAX_PER_THEME, bearer, start)) for label, q in THEMES]


def fetch_following() -> list[dict]:
    """Recent tweets from the followed accounts (their latest takes, by recency). Empty if
    FOLLOWING is unset. Scoped to the accounts, not to keywords — Claude routes them to themes."""
    if not FOLLOWING:
        return []
    bearer, start = require_env("X_BEARER_TOKEN"), _start_time()
    out: list[dict] = []
    for handles in _from_chunks(_FROM_CHUNK):
        # cap split by handle count so a small trailing chunk doesn't get half the budget
        cap = max(1, FOLLOWING_CAP * handles.count("from:") // len(FOLLOWING))
        out.extend(_pull(f"{handles} -is:retweet lang:en", cap, bearer, start, sort="recency"))
    return out


# --------------------------------------------------------------------------- #
# Claude scoring
# --------------------------------------------------------------------------- #
_SYSTEM = """You are a markets sentiment analyst reading X/Twitter for a trading desk. Given \
recent tweets about a topic, produce a concise sentiment read. Weight by engagement/reach; \
discount spam, bots, and obvious coordinated shilling; note if the sample is thin. Output ONLY \
the report (no preamble, no "here is"):

**SENTIMENT — {TOPIC} · {stamp}**
**Net:** <bullish|neutral|bearish> (<-100..+100>) · <N> tweets analysed
- **Bull:** <1 short line — the strongest bullish threads/claims>
- **Bear:** <1 short line — the strongest bearish threads/claims>
**Notable:** 2-3 lines "– @user: <gist> (<likes> likes)" — the highest-signal posts.

Keep under ~1100 characters. Bold via **double asterisks**."""


def score_sentiment(client: anthropic.Anthropic, topic: str, tweets: list[dict], stamp: str) -> str:
    corpus = "\n".join(
        f"[@{t['user']} | {t['likes']}L {t['impressions']}imp] {t['text'][:240]}" for t in tweets
    )
    user = (f"Topic: {topic}. Timestamp: {stamp}. {len(tweets)} tweets from the last "
            f"{WINDOW_HOURS}h below. Write the sentiment report now.\n\n{corpus}")
    resp = client.messages.create(model=MODEL, max_tokens=3000, system=_SYSTEM,
                                  messages=[{"role": "user", "content": user}])  # headroom for adaptive thinking
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    m = re.search(r"\*{0,2}SENTIMENT\b", text)   # drop any preamble before the title
    return text[m.start():] if m else text


_DIGEST_SYSTEM = """You are a markets sentiment analyst reading X/Twitter for a trading desk. \
Given recent (last 48h) tweets grouped by THEME — plus a FOLLOWED ACCOUNTS block (handles the \
desk trusts) — write a concise sentiment digest. Weight by engagement/reach and weight FOLLOWED \
accounts heavily; discount spam, bots, giveaways and coordinated shilling; flag a theme whose \
sample is thin or mostly noise.

Attach a SENTIMENT SCORE in [-1.00, +1.00] to each theme and overall: engagement-weighted net \
mood where -1 = strongly bearish/panic, 0 = mixed / no consensus, +1 = strongly bullish/euphoric. \
Be calibrated — reserve |score| > 0.6 for clearly one-sided conviction; most themes sit within ±0.4.
Tag the contributors driving each theme with their @handle (especially FOLLOWED accounts).

Output ONLY the report (no preamble, no "here is"):

**X SENTIMENT — {stamp}**
- **<Theme>** <+0.NN|-0.NN> (<bullish|neutral|bearish>) — one short line: what's being said + why, \
tagging the key @handles. (ALWAYS one bullet for EVERY theme, in the order given — never omit one; \
if a theme's sample is thin or mostly noise, still score it ~0.00 and say so)
**→ Net:** <+0.NN|-0.NN> — one-line overall market mood.
**Trending:** 3-6 tickers / names / narratives most mentioned right now.

Keep under ~1600 characters. Bold via **double asterisks**. No clause pile-ups."""


def build_digest(client: anthropic.Anthropic, themed: list[tuple[str, list[dict]]],
                 followed: list[dict], stamp: str) -> str:
    def _top(tweets: list[dict], n: int) -> str:
        top = sorted(tweets, key=lambda t: t["likes"] + t["impressions"], reverse=True)[:n]
        return "\n".join(f"[@{t['user']} {t['likes']}L] {t['text'][:200]}" for t in top)

    blocks = [f"=== {label} ({len(tw)} tweets) ===\n{_top(tw, MAX_PER_THEME) or '(none)'}" for label, tw in themed]
    if followed:
        blocks.append(f"=== FOLLOWED ACCOUNTS ({len(followed)} tweets — weight heavily, tag them) "
                      f"===\n{_top(followed, len(followed))}")
    user = (f"Timestamp: {stamp}. Themes and their top tweets below. Write the digest now.\n\n"
            + "\n\n".join(blocks))
    resp = client.messages.create(model=MODEL, max_tokens=4000, system=_DIGEST_SYSTEM,
                                  messages=[{"role": "user", "content": user}])  # headroom for adaptive thinking
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    m = re.search(r"\*{0,2}X SENTIMENT\b", text)
    return text[m.start():] if m else text


_STRUCT_SYSTEM = """You are a markets sentiment analyst reading X/Twitter for a trading desk.
Given recent (last 48h) tweets grouped by THEME plus a FOLLOWED ACCOUNTS block (handles the desk
trusts), return STRICT JSON only (no prose, no markdown fence) with keys:
  "abstract": 2 SHORT sentences (<=42 words total) on the day's cross-market mood (mention the date),
  "net_score": number in [-1,1],
  "net_line": one short sentence (<=28 words), overall market mood,
  "themes": array with EXACTLY one object per theme in the order given, each
     {"name": <theme>, "score": number in [-1,1], "line": "<=15 words, what's said + why, tag @handles",
      "contributors": "comma-separated @handles or —"},
  "trending": "4-5 tickers/names/narratives, ' · ' separated",
  "voices": "exactly 2 short takes from FOLLOWED accounts (<=14 words each), '; ' separated, tag @handle",
  "takeaway": "Positioning implication: <one line, <=24 words>".
Engagement-weighted; weight FOLLOWED accounts heavily; discount spam/bots/giveaways; calibrated
(reserve |score|>0.6 for clearly one-sided conviction). Score EVERY theme even if thin (~0.00)."""


def structured_digest(client: anthropic.Anthropic, themed: list[tuple[str, list[dict]]],
                      followed: list[dict], stamp: str) -> dict:
    """Same inputs as build_digest but returns a structured dict for the PDF/DOCX report."""
    def _top(tweets: list[dict], n: int) -> str:
        top = sorted(tweets, key=lambda t: t["likes"] + t["impressions"], reverse=True)[:n]
        return "\n".join(f"[@{t['user']} {t['likes']}L] {t['text'][:200]}" for t in top)

    blocks = [f"=== {label} ({len(tw)} tweets) ===\n{_top(tw, MAX_PER_THEME) or '(none)'}" for label, tw in themed]
    if followed:
        blocks.append(f"=== FOLLOWED ACCOUNTS ({len(followed)}) ===\n{_top(followed, len(followed))}")
    order = ", ".join(label for label, _ in themed)
    user = (f"Date: {stamp}. Themes in order: {order}. Tweets below. Return the JSON now.\n\n"
            + "\n\n".join(blocks))
    resp = client.messages.create(model=MODEL, max_tokens=4000, system=_STRUCT_SYSTEM,
                                  messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("structured_digest: no JSON in model output")
    return json.loads(match.group(0))


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def _md_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text), flags=re.S)


def _send(api: str, chat_id: str, text: str, parse_mode: str | None = None) -> None:
    payload = {"chat_id": chat_id, "text": text[:TG_MSG_LIMIT]}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(f"{api}/sendMessage", data=payload, timeout=30)
        if r.status_code != 200 and parse_mode:
            requests.post(f"{api}/sendMessage",
                          data={"chat_id": chat_id, "text": text[:TG_MSG_LIMIT]}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram send failed: %s", e)


def _send_document(api: str, chat_id: str, path: str, caption: str | None = None) -> None:
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    try:
        with open(path, "rb") as fh:
            requests.post(f"{api}/sendDocument", data=data,
                          files={"document": (path.rsplit("/", 1)[-1], fh)}, timeout=120)
    except (requests.exceptions.RequestException, OSError) as e:
        _log.warning("telegram sendDocument failed: %s", e)


def _generate_report_pdf() -> str:
    """Build the daily one-page sentiment PDF (frozen G-20 template) and return its path.
    Imported lazily to keep docx/weasyprint out of the bot's import path unless requested."""
    from scripts.daily_sentiment_report import render_one_page
    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y")
    data = structured_digest(client, fetch_themes(), fetch_following(), stamp)
    return render_one_page(data, ".")


def main() -> None:
    token = require_env("TELEGRAM_SENTIMENT_BOT_TOKEN")
    authorized = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("sentiment_bot started; serving /sentiment to chats %s", sorted(authorized))
    while True:
        updates, offset = poll_updates(api, offset)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            parts = text.split()
            cmd = parts[0].split("@")[0].lower() if parts else ""
            if cmd != "/sentiment":
                continue
            if chat_id not in authorized:
                _send(api, chat_id, "Not authorized.")
                _log.warning("rejected /sentiment from unauthorized chat %s", chat_id)
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            try:
                if len(parts) >= 2 and parts[1].lower() == "report":  # /sentiment report -> PDF
                    _send(api, chat_id, "Generating the sentiment PDF report…")
                    pdf = _generate_report_pdf()
                    _send_document(api, chat_id, pdf, "Daily X sentiment brief")
                    continue
                if len(parts) < 2:                         # bare /sentiment -> market digest
                    _send(api, chat_id, "Reading X market sentiment digest…")
                    themed = fetch_themes()
                    followed = fetch_following()
                    total = sum(len(t) for _, t in themed) + len(followed)
                    _log.info("digest: %d theme + %d followed tweets / %d themes — est $%.2f",
                              total - len(followed), len(followed), len(themed), total * 0.005)
                    if not total:
                        _send(api, chat_id, "No recent tweets found.")
                        continue
                    report = build_digest(client, themed, followed, stamp)
                else:                                      # /sentiment <topic> [crowd|infl]
                    topic = parts[1]
                    scope = parts[2].lower() if len(parts) > 2 else "crowd"
                    _send(api, chat_id, f"Reading X sentiment for {topic} ({scope})…")
                    tweets = fetch_tweets(topic, scope)
                    _log.info("single: %d tweets for %s (%s) — est $%.2f", len(tweets), topic,
                              scope, len(tweets) * 0.005)
                    if not tweets:
                        _send(api, chat_id, f"No recent tweets found for '{topic}'.")
                        continue
                    report = score_sentiment(client, topic, tweets, stamp)
                _send(api, chat_id, _md_to_html(report), "HTML")
            except Exception as exc:  # noqa: BLE001 - report over a chat, never crash the bot
                _log.exception("sentiment failed")
                _send(api, chat_id, f"Sentiment failed: {exc}")


if __name__ == "__main__":
    main()
