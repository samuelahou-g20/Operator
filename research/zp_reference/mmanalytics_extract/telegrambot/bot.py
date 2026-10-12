"""Telegram front-end for the report suite (ZP, compliance, captures). Long-running; reuses
the shared core (trade_analysis_core) and renders the interactive HTML reports.

    python -m telegrambot.bot

Token and allowlist come from .streamlit/secrets.toml:

    [telegram]
    bot_token = "123456:ABC-DEF..."          # from @BotFather
    allowed_chats = [11111111, 22222222]      # chat/user IDs allowed to run the commands

(TELEGRAM_BOT_TOKEN env var overrides bot_token if set.) Unknown chats are refused and
told their chat id so an admin can add it.
"""

import asyncio
import math
import os
import shutil
import tempfile
import time
import tomllib
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import trade_analysis_core
from . import bootstrap as zp_bootstrap
from . import zp_html
from .period import parse_period
from . import programs as ref_programs
from . import compliance as compliance_html
from . import captures as captures_html
from . import markouts as markouts_html
from . import aiozp as aiozp_html
from . import dmc as dmc_mod
from . import volgrid
from . import ticksize as ticksize_html
from . import summary_table
from . import dmccheck

_SECRETS_PATH = Path(__file__).resolve().parent.parent / '.streamlit' / 'secrets.toml'
_zp_lock = asyncio.Lock()  # serialise ZP runs (shared core state + don't hammer services)
_zp_jobs = []              # ordered in-flight heavy jobs (index 0 = the one running); each: {msg,label}


async def _refresh_queue():
    """Update each WAITING job's status message with its live position (jobs ahead). The running job
    (index 0) drives its own message, so it's skipped. Best-effort — edit failures are ignored."""
    for i, rec in list(enumerate(_zp_jobs)):
        if i == 0 or not rec.get('msg'):
            continue
        noun = 'request' if i == 1 else 'requests'
        try:
            await rec['msg'].edit_text(f'⏳ Queued: {rec["label"]} — {i} {noun} ahead of you.')
        except Exception:
            pass


@asynccontextmanager
async def _zp_queue(update, label):
    """Serialise heavy commands on _zp_lock behind a live, labelled status message: a counting-down
    'queued' line while it waits, then 'running', removed when done. Yields the status Message so the
    handler can keep editing it (e.g. aiozp's per-coin progress). Compute stays one-at-a-time (only
    the status messages are concurrent, which needs Application concurrent_updates). The job list is
    mutated synchronously on the single-threaded loop, and asyncio.Lock is FIFO, so positions hold."""
    rec = {'msg': None, 'label': label}
    ahead = len(_zp_jobs)                 # jobs already in flight = jobs ahead of me
    _zp_jobs.append(rec)
    try:
        msg_api = getattr(update, 'message', None)
        if msg_api:
            if ahead:
                noun = 'request' if ahead == 1 else 'requests'
                text = f'⏳ Queued: {label} — {ahead} {noun} ahead of you.'
            else:
                text = f'▶️ Running: {label}…'
            try:
                rec['msg'] = await msg_api.reply_text(text)
            except Exception:
                rec['msg'] = None
        async with _zp_lock:
            if rec['msg'] is not None and ahead:      # was queued -> now our turn
                try:
                    await rec['msg'].edit_text(f'▶️ Running: {label}…')
                except Exception:
                    pass
            yield rec['msg']
    finally:
        try:
            _zp_jobs.remove(rec)
        except ValueError:
            pass
        if rec.get('msg') is not None:                # clear our status line; the result follows
            try:
                await rec['msg'].delete()
            except Exception:
                pass
        await _refresh_queue()


_AIOZP_MAX_MB = 49  # Telegram bot document limit is ~50 MB — the hard per-file ceiling
_AIOZP_PACK_MB = 45  # per-part budget when splitting a too-big desk (margin under the ceiling)
_DMC_MAX_HOURS = 336  # cap how far back to pull gamma-dash history for a backdated window (2 weeks)

HELP = (
    'ZP bot\n'
    'Usage: /zp TICKER PERIOD [flags]\n'
    'e.g. /zp MAGMA 4h   |   /zp ICP yesterday\n'
    'periods: Nh, Nd, today, yesterday, "last 24h"\n'
    'flags: --usd-only --no-markouts --no-captures --no-widths --no-perp-capture --capture-threshold 0.08\n'
    '\n'
    '/compliance PROGRAM [PERIOD] — 2% depth compliance grid, e.g. /compliance MM2 8h\n'
    '/captures PROGRAM [PERIOD] — capture % grid (slow), e.g. /captures MM2 8h\n'
    '/markouts PROGRAM [PERIOD] — markout (bps) grid, horizon toggle (slow), e.g. /markouts MM2 8h\n'
    '/vol PROGRAM [PERIOD] — realized vol vs dynamic-mode setting grid, metric toggle, e.g. /vol MM2 8h\n'
    '/ticksize PROGRAM — tick size (bps) per venue + min-spread-in-ticks, snapshot, e.g. /ticksize MM2\n'
    '/dynamicsettings PROGRAM — scan dynamic-mode settings, flag suspect/outlier values, e.g. /dynamicsettings MM3\n'
    '/aiozp PROGRAM [PERIOD] — all-in-one desk report: Summary grids + a ZP per coin in one file '
    '(dropdown), very slow, e.g. /aiozp MM2 8h\n'
    '/dmc TOKEN [PERIOD] — dynamic mode settings + volatility/spread charts, e.g. /dmc FET 4h'
)


def _load_telegram_conf():
    with open(_SECRETS_PATH, 'rb') as f:
        secrets = tomllib.load(f)
    tg = secrets.get('telegram', {})
    token = os.environ.get('TELEGRAM_BOT_TOKEN') or tg.get('bot_token')
    if not token:
        raise SystemExit('No bot token. Add [telegram] bot_token to .streamlit/secrets.toml '
                         'or set TELEGRAM_BOT_TOKEN.')
    allowed = set(tg.get('allowed_chats', []))
    open_access = bool(tg.get('open_access', False))
    return token, allowed, open_access


def _parse_flags(args):
    opts = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--usd-only':
            opts['usd_only'] = True
        elif a == '--no-markouts':
            opts['no_markouts'] = True
        elif a == '--no-captures':
            opts['no_captures'] = True
        elif a == '--no-perp-capture':
            opts['no_perp_capture'] = True
        elif a == '--no-widths':
            opts['no_widths'] = True
        elif a == '--capture-threshold' and i + 1 < len(args):
            opts['capture_threshold'] = float(args[i + 1])
            i += 1
        i += 1
    return opts


def _resolve_product_area(ticker):
    """Which program's orderbook to scope to (mm2/mm3/mm4/perp), or None if unresolved."""
    try:
        prog = ref_programs.program_for_coin(ref_programs.fetch_programs(), ticker)
        return prog.lower() if prog else None
    except Exception as e:
        print(f'[!] program lookup failed for {ticker}: {e}', flush=True)
        return None


def _dmc_tokens(token):
    """A coin's gamma-dash names — every distinct base-name in its configured pairs (config.yaml),
    e.g. AI -> ['AI', 'AIGENSYN'] since it trades as AIGENSYN on binance. A perp is keyed under its
    OWN prefixed token in gamma-dash (P.PARTI-USDT's dynamic strategy lives under 'P.PARTI', while
    spot 'PARTI' has none), so we keep BOTH the prefixed and stripped names. Falls back to [token]
    when it isn't a config coin (or config isn't loaded)."""
    try:
        bases = set()
        for p in trade_analysis_core.get_pairs(token):
            b = p.split('-', 1)[0]
            bases.add(b)                     # e.g. 'P.PARTI' — the perp's own gamma-dash token
            bases.add(b.removeprefix('P.'))  # e.g. 'PARTI'  — the spot name
        return sorted(bases) or [token]
    except Exception:
        return [token]


def _fetch_dmc(token, start, end):
    """gamma-dash settings + vol/spread history trimmed to the ZP window [start, end], across all
    of the coin's gamma-dash names. The history API only takes 'hours from now', so fetch back far
    enough to reach `start` (capped) and let fetch_dmc trim to the window — so a backdated period
    lines up with the other grids. Non-fatal — returns None on any failure so the report just skips
    the section."""
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        hours = max(1, min(_DMC_MAX_HOURS, math.ceil((now - start).total_seconds() / 3600)))
        return dmc_mod.fetch_dmc(_dmc_tokens(token), hours=hours, start=start, end=end)
    except Exception as e:
        print(f'[!] DMC fetch failed for {token}: {e}', flush=True)
        return None


def _run_zp(ticker, start, end, opts):
    """Blocking: compute + render the HTML report. Returns (html_path_or_None, tmpdir_or_None)."""
    threshold = opts.get('capture_threshold', 0.10)
    label = opts.get('period_label')
    result = trade_analysis_core.compute_zp(
        ticker, start, end,
        want_markouts=not opts.get('no_markouts'),
        want_captures=not opts.get('no_captures'),
        want_widths=not opts.get('no_widths'),
        no_perp_capture=opts.get('no_perp_capture', False),  # perps included by default (they live in spot programs too)
        usd_only=opts.get('usd_only', False),
        product_area=_resolve_product_area(ticker),
    )
    dmc_data = _fetch_dmc(ticker, start, end)
    tick_html = ''
    try:  # per-venue tick sizes (ccxt) for this coin
        tk = ticker.upper()
        tick_html = ticksize_html.coin_table(
            trade_analysis_core.fetch_ticksize([tk], no_perp=opts.get('no_perp_capture', False)),
            ticksize_html.dmc_min_spreads({tk: dmc_data}), tk)
    except Exception as e:
        print(f'[!] tick sizes failed for {ticker}: {e}', flush=True)
    has_trades = result.trades is not None and not result.trades.empty
    has_ob = ((result.ob_widths_ts is not None and not result.ob_widths_ts.empty)
              or (result.ob_compliance is not None and not result.ob_compliance.empty))
    has_dmc = dmc_data is not None and bool(dmc_data.rows)
    if not has_trades and not has_ob and not has_dmc:  # nothing to show at all
        return None, None
    tmpdir = tempfile.mkdtemp(prefix='zp_')
    html_path = os.path.join(tmpdir, 'report.html')
    zp_html.write_html_report(result, html_path, capture_threshold=threshold, period_label=label,
                              dmc_data=dmc_data, tick_html=tick_html)
    return html_path, tmpdir


async def zp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return

    args = context.args
    if len(args) < 2:
        await update.message.reply_text('Usage: /zp TICKER PERIOD\n\n' + HELP)
        return

    ticker = args[0].upper()
    period = args[1]
    config = context.bot_data['config']
    if ticker not in config:
        await update.message.reply_text(f'Unknown coin {ticker!r} (not in config).')
        return
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    opts = _parse_flags(args[2:])
    opts['period_label'] = period

    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'ZP {ticker} {period}'):
        try:
            html_path, tmpdir = await asyncio.to_thread(_run_zp, ticker, start, end, opts)
        except Exception as e:
            await update.message.reply_text(f'ZP failed: {e}')
            return
    try:
        if html_path:
            period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
            fname = f'{ticker}_ZP_{period_tag}_{end:%Y%m%d-%H%M}Z.html'  # window end ("as of"), Z = UTC
            with open(html_path, 'rb') as f:
                await update.message.reply_document(document=f, filename=fname)
        else:
            await update.message.reply_text('No data in this period (no trades or orderbook data).')
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_compliance(program, start, end, period):
    """Blocking: resolve the program's coins, compute the compliance grid, write the HTML.
    Returns (html_path, tmpdir)."""
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    cells, ignored, tokens = trade_analysis_core.fetch_compliance_matrix(
        coins, start, end, product_area=program.lower())
    tmpdir = tempfile.mkdtemp(prefix='cmp_')
    path = os.path.join(tmpdir, 'compliance.html')
    compliance_html.write_compliance_report(program, cells, ignored, tokens, start, end, path,
                                            period_label=period)
    return path, tmpdir


async def compliance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    args = context.args
    if not args:
        await update.message.reply_text('Usage: /compliance PROGRAM [PERIOD]\n'
                                        'e.g. /compliance MM2 8h   (period defaults to 8h)')
        return
    program = args[0].upper()
    period = args[1] if len(args) > 1 else '8h'
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'compliance {program} {period}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_compliance, program, start, end, period)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Compliance failed: {e}')
            return
    try:
        period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
        fname = f'{program}_compliance_{period_tag}_{end:%Y%m%d-%H%M}Z.html'  # window end ("as of"), Z = UTC
        with open(path, 'rb') as f:
            await update.message.reply_document(document=f, filename=fname)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_captures(program, start, end, period):
    """Blocking: resolve the program's coins, compute the captures grid, write HTML.
    Returns (html_path, tmpdir)."""
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    cells, ignored, tokens = trade_analysis_core.fetch_captures_matrix(
        coins, start, end, no_perp_capture=False)  # include perps (they live in spot programs too)
    tmpdir = tempfile.mkdtemp(prefix='cap_')
    path = os.path.join(tmpdir, 'captures.html')
    captures_html.write_captures_report(program, cells, ignored, tokens, start, end, path,
                                        period_label=period)
    return path, tmpdir


async def captures_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    args = context.args
    if not args:
        await update.message.reply_text('Usage: /captures PROGRAM [PERIOD]\n'
                                        'e.g. /captures MM2 8h   (period defaults to 8h)')
        return
    program = args[0].upper()
    period = args[1] if len(args) > 1 else '8h'
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'captures {program} {period}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_captures, program, start, end, period)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Captures failed: {e}')
            return
    try:
        period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
        fname = f'{program}_captures_{period_tag}_{end:%Y%m%d-%H%M}Z.html'  # window end ("as of"), Z = UTC
        with open(path, 'rb') as f:
            await update.message.reply_document(document=f, filename=fname)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_vol(program, start, end, period):
    """Blocking: resolve the program's coins, fetch each coin's dynamic-mode data (gamma-dash) in
    parallel, and write the desk Vol-vs-setting grid. Returns (html_path, tmpdir)."""
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    tokens = [c for c in coins if c in trade_analysis_core._config_provider()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        dmc_map = dict(pool.map(lambda t: (t, _fetch_dmc(t, start, end)), tokens))
    cells = volgrid.compute_vol_cells(dmc_map)
    tmpdir = tempfile.mkdtemp(prefix='vol_')
    path = os.path.join(tmpdir, 'vol.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(volgrid.build_vol_report(program, cells, start, end, period_label=period))
    return path, tmpdir


async def vol_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    args = context.args
    if not args:
        await update.message.reply_text('Usage: /vol PROGRAM [PERIOD]\n'
                                        'e.g. /vol MM2 8h   (period defaults to 8h)')
        return
    program = args[0].upper()
    period = args[1] if len(args) > 1 else '8h'
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'vol {program} {period}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_vol, program, start, end, period)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Vol grid failed: {e}')
            return
    try:
        period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
        fname = f'{program}_vol_{period_tag}_{end:%Y%m%d-%H%M}Z.html'  # window end ("as of"), Z = UTC
        with open(path, 'rb') as f:
            await update.message.reply_document(document=f, filename=fname)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_ticksize(program):
    """Blocking: tick size per configured exchange/pair (ccxt) for the program's coins, plus min-
    spread-in-ticks for the dynamic-mode venues. A snapshot (no time window). Returns (path, tmpdir)."""
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    tick_cells = trade_analysis_core.fetch_ticksize(coins, no_perp=False)  # include perp venues
    tokens = [c for c in coins if c in trade_analysis_core._config_provider()]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ThreadPoolExecutor(max_workers=4) as pool:  # DMC settings only (1h window is plenty)
        dmc_map = dict(pool.map(lambda t: (t, _fetch_dmc(t, now - timedelta(hours=1), now)), tokens))
    dmc_min = ticksize_html.dmc_min_spreads(dmc_map)
    tmpdir = tempfile.mkdtemp(prefix='tick_')
    path = os.path.join(tmpdir, 'ticksize.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(ticksize_html.build_ticksize_report(program, tick_cells, dmc_min))
    return path, tmpdir


async def ticksize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    if not context.args:
        await update.message.reply_text('Usage: /ticksize PROGRAM\ne.g. /ticksize MM2   (a snapshot — no period)')
        return
    program = context.args[0].upper()
    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'ticksize {program}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_ticksize, program)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Tick size grid failed: {e}')
            return
    try:
        fname = f'{program}_ticksize_{datetime.now(timezone.utc):%Y%m%d-%H%M}Z.html'
        with open(path, 'rb') as f:
            await update.message.reply_document(document=f, filename=fname)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_dmccheck(program):
    """Blocking: scan the DESK's dynamic-mode settings (one settings-only gamma-dash call per name),
    group strategies by their gamma-dash business_area, and report `program`'s coins — flagging hard
    rules + peer outliers judged against the whole desk. A snapshot. Returns (path, tmpdir)."""
    cfg = trade_analysis_core._config_provider()
    revmap, tokens = {}, set()             # gamma token -> config coin, and the set to query
    for coin in cfg:
        for t in _dmc_tokens(coin):
            tokens.add(t)
            revmap.setdefault(t, coin)
    strategies = dmc_mod.fetch_settings(sorted(tokens))
    for s in strategies:
        s['coin'] = revmap.get(s['token'], s['token'])
        s['venue'] = f'{s["exchange"].split(":", 1)[0]}{".P" if s["perp"] else ""} {s["base"]}'
    areas = {s['business_area'] for s in strategies if s['business_area']}
    pl = program.lower()
    if pl not in areas:
        raise KeyError(f'Unknown program {program!r}. Known: {sorted(a.upper() for a in areas)}')
    report = [s for s in strategies if s['business_area'] == pl]
    tmpdir = tempfile.mkdtemp(prefix='dmccheck_')
    path = os.path.join(tmpdir, 'dmccheck.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dmccheck.build_dmccheck_report(program, report, ref_strategies=strategies))
    return path, tmpdir


async def dmccheck_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    if not context.args:
        await update.message.reply_text('Usage: /dynamicsettings PROGRAM\ne.g. /dynamicsettings MM3   (a snapshot — no period)')
        return
    program = context.args[0].upper()
    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'dynamicsettings {program}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_dmccheck, program)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Dynamic settings check failed: {e}')
            return
    try:
        fname = f'{program}_dynamicsettings_{datetime.now(timezone.utc):%Y%m%d-%H%M}Z.html'
        with open(path, 'rb') as f:
            await update.message.reply_document(document=f, filename=fname)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_refvols(program):
    """Blocking: quick per-coin reference-vol list (throwaway). One value per coin (they're uniform
    across a coin's venues; flagged if not). Returns the text body."""
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    tokens = [c for c in coins if c in trade_analysis_core._config_provider()]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ThreadPoolExecutor(max_workers=4) as pool:  # DMC settings only
        dmc_map = dict(pool.map(lambda t: (t, _fetch_dmc(t, now - timedelta(hours=1), now)), tokens))
    lines = []
    for coin in sorted(dmc_map):
        data = dmc_map[coin]
        if not data:
            continue
        refs = sorted({dyn.get('reference_annualized_volatility_pct') for _b, _e, dyn, *_ in data.rows
                       if dyn.get('reference_annualized_volatility_pct') is not None})
        if not refs:
            continue
        lines.append(f'{coin} {refs[0]:g}%' if len(refs) == 1
                     else f'{coin} {min(refs):g}–{max(refs):g}% (varies)')
    return 'Reference Vols\n\n' + ('\n'.join(lines) if lines else '(none)')


async def refvols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    if not context.args:
        await update.message.reply_text('Usage: /refvols PROGRAM')
        return
    program = context.args[0].upper()
    await update.effective_chat.send_action('typing')
    async with _zp_queue(update, f'refvols {program}'):
        try:
            text = await asyncio.to_thread(_run_refvols, program)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Ref vols failed: {e}')
            return
    await update.message.reply_text(text)


def _run_markouts(program, start, end, period):
    """Blocking: resolve the program's coins, compute the markouts grid, write HTML.
    Returns (html_path, tmpdir)."""
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    cells, ignored, tokens, horizons = trade_analysis_core.fetch_markouts_matrix(
        coins, start, end)
    tmpdir = tempfile.mkdtemp(prefix='mo_')
    path = os.path.join(tmpdir, 'markouts.html')
    markouts_html.write_markouts_report(program, cells, ignored, tokens, horizons, start, end, path,
                                        period_label=period)
    return path, tmpdir


async def markouts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    args = context.args
    if not args:
        await update.message.reply_text('Usage: /markouts PROGRAM [PERIOD]\n'
                                        'e.g. /markouts MM2 8h   (period defaults to 8h)')
        return
    program = args[0].upper()
    period = args[1] if len(args) > 1 else '8h'
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'markouts {program} {period}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_markouts, program, start, end, period)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'Markouts failed: {e}')
            return
    try:
        period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
        fname = f'{program}_markouts_{period_tag}_{end:%Y%m%d-%H%M}Z.html'  # window end ("as of"), Z = UTC
        with open(path, 'rb') as f:
            await update.message.reply_document(document=f, filename=fname)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _aiozp_error_doc(token, err):
    return ('<!doctype html><html><head><meta charset="utf-8"></head>'
            '<body style="background:#0f1216;color:#e6e9ef;font-family:sans-serif;padding:20px">'
            f'<h1>{token}</h1><p>Report failed: {err}</p></body></html>')


def _run_aiozp(program, start, end, period, progress=None):
    """Blocking: single-pass desk compute, then assemble the all-in-one HTML (Summary grids +
    a ZP per coin, dropdown). Returns (parts, tmpdir, caption) where parts = [(path, suffix), …]:
    one file when the desk fits a Telegram document, otherwise several — each the Summary + an
    alphabetical slice of coins (suffix = 'FIRST-LAST' for the filename). caption (first doc only)
    explains a split, else None."""
    t0 = time.perf_counter()
    programs = ref_programs.fetch_programs()
    coins = ref_programs.program_coins(programs, program)  # KeyError -> unknown program
    t_fetch0 = time.perf_counter()
    bundle = trade_analysis_core.fetch_desk_bundle(
        coins, start, end, product_area=program.lower(),
        no_perp_capture=False, progress=progress)  # include perps (they live in spot programs too)
    t_fetch = time.perf_counter() - t_fetch0  # per-coin compute (I/O-bound: trades/OB/klines)

    # Prefetch each coin's gamma-dash data in parallel (I/O-bound) — feeds both the per-coin reports
    # and the desk Vol-vs-setting summary grid, so do it before the summary is built.
    t_dmc0 = time.perf_counter()
    computed = [t for t in bundle.tokens if bundle.results.get(t) is not None]
    with ThreadPoolExecutor(max_workers=4) as pool:
        dmc_map = dict(pool.map(lambda t: (t, _fetch_dmc(t, start, end)), computed))
    t_dmc = time.perf_counter() - t_dmc0
    vol_cells = volgrid.compute_vol_cells(dmc_map)
    # Tick sizes (ccxt, snapshot) + min-spread-in-ticks from the DMC settings — for the 5th grid.
    try:
        tick_cells = trade_analysis_core.fetch_ticksize(bundle.tokens, no_perp=False)  # include perp venues
    except Exception as e:
        print(f'[!] aiozp ticksize failed: {e}', flush=True)
        tick_cells = {}
    dmc_min = ticksize_html.dmc_min_spreads(dmc_map)

    # Summary grids — interactive full pages (iframes) + static body_only fragments (no-JS base).
    # The desk 'headline' table leads (net traded, PnL MM/Gamma/Total, compliance, capture, markout).
    t_sum0 = time.perf_counter()
    vr = {}  # per-coin vol stats (from the vol grid cells) for the headline table + its hover
    for (coin, _col), c in vol_cells.items():
        vr.setdefault(coin, []).append(c)
    vol_info = {coin: {
        'ratio': sum(c['ratio'] for c in cs) / len(cs), 'ref': cs[0]['ref'],
        'med': sum(c['med'] for c in cs) / len(cs), 'p90': sum(c['p90'] for c in cs) / len(cs),
        'mx': max(c['mx'] for c in cs)} for coin, cs in vr.items()}
    desk_rows = summary_table.summary_rows(bundle.results, vol_info)
    desk_vol = summary_table._compact(sum(r.get('vol') or 0 for r in desk_rows))  # total gross volume
    desk_pnl = summary_table.desk_pnl_svg(bundle.results)  # inline SVG, shared by both layers
    summary_docs = [
        summary_table.build_summary_table(program, desk_rows, chart_html=desk_pnl),
        dmccheck.build_from_dmc_map(program, dmc_map, outliers=True),  # DMC sanity: hard + peer outliers
        compliance_html.build_compliance_report(program, bundle.compliance_cells, bundle.ignored,
                                                bundle.tokens, start, end, period),
        captures_html.build_captures_report(program, bundle.captures_cells, bundle.ignored,
                                            bundle.tokens, start, end, period),
        markouts_html.build_markouts_report(program, bundle.markouts_cells, bundle.ignored,
                                            bundle.tokens, bundle.horizons, start, end, period),
        volgrid.build_vol_report(program, vol_cells, start, end, period),
        ticksize_html.build_ticksize_report(program, tick_cells, dmc_min),
    ]
    summary_static = (
        summary_table.build_summary_table(program, desk_rows, body_only=True, chart_html=desk_pnl)
        + dmccheck.build_from_dmc_map(program, dmc_map, outliers=True, body_only=True)
        + compliance_html.build_compliance_report(program, bundle.compliance_cells, bundle.ignored,
                                                  bundle.tokens, start, end, period, body_only=True)
        + captures_html.build_captures_report(program, bundle.captures_cells, bundle.ignored,
                                              bundle.tokens, start, end, period, body_only=True)
        + markouts_html.build_markouts_report(program, bundle.markouts_cells, bundle.ignored,
                                              bundle.tokens, bundle.horizons, start, end, period,
                                              body_only=True)
        + volgrid.build_vol_report(program, vol_cells, start, end, period, body_only=True)
        + ticksize_html.build_ticksize_report(program, tick_cells, dmc_min, body_only=True))
    t_summary = time.perf_counter() - t_sum0

    # Build every coin once: its interactive doc (fallback=False → no PNGs in the iframe; the
    # static <details> layer already carries them) + its static (phone) fragment.
    # Timed in two parts: interactive = Plotly HTML (CPU, GIL-bound); static = matplotlib PNGs
    # (CPU, needs processes to parallelise). This split tells us which lever is worth pulling.
    coin_docs, coin_static = {}, {}
    t_interactive = t_static = 0.0
    render_times = {}  # per-coin total render (interactive + static)
    for token in sorted(bundle.tokens):  # alphabetical, plain ticker
        res, dmc = bundle.results.get(token), dmc_map.get(token)
        if res is None:
            coin_docs[token] = _aiozp_error_doc(token, 'compute failed')
            coin_static[token] = '<p class="sub">compute failed</p>'
            continue
        r0 = time.perf_counter()
        try:
            coin_docs[token] = zp_html.build_html_report(
                res, period_label=period, dmc_data=dmc, fallback=False,
                tick_html=ticksize_html.coin_table(tick_cells, dmc_min, token))
        except Exception as e:
            coin_docs[token] = _aiozp_error_doc(token, e)
        r1 = time.perf_counter()
        try:
            coin_static[token] = zp_html.build_static_report(res, dmc_data=dmc, period_label=period)
        except Exception as e:
            coin_static[token] = f'<p class="sub">static render failed: {e}</p>'
        r2 = time.perf_counter()
        t_interactive += r1 - r0
        t_static += r2 - r1
        render_times[token] = r2 - r0

    order = sorted(bundle.tokens)
    tmpdir = tempfile.mkdtemp(prefix='aio_')
    t_assemble0 = time.perf_counter()

    def _write(subset, suffix, idx):
        """Write one aiozp file = the desk Summary + `subset` coins. Returns its path."""
        views = [{'label': 'Summary', 'docs': summary_docs}]
        static = []
        for t in subset:
            views.append({'label': t, 'docs': [coin_docs[t]]})
            static.append((t, coin_static[t]))
        p = os.path.join(tmpdir, f'aiozp_{idx}.html')
        aiozp_html.write_aiozp_report(program, views, summary_static, static, start, end, p,
                                      period_label=period, total_volume=desk_vol)
        return p

    def _finish(parts, caption):
        """Log the phase breakdown (bot stdout → logs) and return the run's result tuple."""
        t_write = time.perf_counter() - t_assemble0  # pack + JSON-encode + write file(s)
        n = len(order)
        slow = ', '.join(f'{t} {d:.1f}s' for t, d in
                         sorted(render_times.items(), key=lambda kv: -kv[1])[:5])
        print(f'[aiozp timing] {program} {period}: {n} coins, {len(parts)} file(s), '
              f'TOTAL {time.perf_counter() - t0:.1f}s | fetch {t_fetch:.1f}s | dmc {t_dmc:.1f}s | '
              f'render {t_interactive + t_static:.1f}s (plotly {t_interactive:.1f}s + '
              f'png {t_static:.1f}s) | summary {t_summary:.1f}s | assemble/write {t_write:.1f}s',
              flush=True)
        print(f'[aiozp timing] slowest coins to render: {slow}', flush=True)
        return parts, tmpdir, caption

    # Estimate the assembled bytes (the interactive docs get JSON-escaped into the payload, ~×1.1;
    # the static fragments go in raw). The Summary is the fixed per-part overhead — it rides in
    # every part. If the whole desk fits one document, ship one file.
    def _b(s):
        return len(s.encode('utf-8'))
    overhead = _b(summary_static) + sum(_b(d) for d in summary_docs) * 1.1 + 60_000  # +CSS/shell
    cost = {t: _b(coin_docs[t]) * 1.1 + _b(coin_static[t]) for t in order}
    budget = _AIOZP_PACK_MB * 1e6

    if overhead + sum(cost.values()) <= budget or len(order) <= 1:
        return _finish([(_write(order, '', 0), '')], None)

    # Too big for one file — greedily pack coins (kept alphabetical) into parts, each = Summary + a
    # contiguous slice of coins that fits the budget. A lone coin bigger than the budget gets its
    # own part regardless (the per-file guard reports it if even that overflows). N falls out of the
    # packing — we split into exactly as many parts as the desk needs.
    groups, cur, cur_sz = [], [], overhead
    for t in order:
        if cur and cur_sz + cost[t] > budget:
            groups.append(cur)
            cur, cur_sz = [], overhead
        cur.append(t)
        cur_sz += cost[t]
    if cur:
        groups.append(cur)

    caption = (f'Desk too large for one file — split into {len(groups)} parts (alphabetical by '
               f'coin); each part includes the Summary.')
    parts = []
    for idx, g in enumerate(groups):
        suffix = g[0] if len(g) == 1 else f'{g[0]}-{g[-1]}'  # alphabetical range → filename
        parts.append((_write(g, suffix, idx), suffix))
    return _finish(parts, caption)


async def aiozp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    args = context.args
    if not args:
        await update.message.reply_text('Usage: /aiozp PROGRAM [PERIOD]\n'
                                        'e.g. /aiozp MM2 8h   (period defaults to 8h)')
        return
    program = args[0].upper()
    period = args[1] if len(args) > 1 else '8h'
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.effective_chat.send_action('upload_document')
    loop = asyncio.get_running_loop()
    tmpdir = None
    parts, caption = [], None
    label = f'aiozp {program} {period}'
    async with _zp_queue(update, label) as status:
        def progress(i, n, token):  # called from the worker thread; marshal the edit to the loop
            try:
                if status:
                    asyncio.run_coroutine_threadsafe(
                        status.edit_text(f'▶️ Running: {label}…\n{i}/{n}: {token}'), loop)
            except Exception:
                pass
        try:
            parts, tmpdir, caption = await asyncio.to_thread(_run_aiozp, program, start, end, period,
                                                             progress)
        except KeyError as e:
            await update.message.reply_text(str(e).strip("'"))
            return
        except Exception as e:
            await update.message.reply_text(f'All-in-one failed: {e}')
            return
    try:
        period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
        for idx, (ppath, suffix) in enumerate(parts):
            size_mb = os.path.getsize(ppath) / 1e6
            sfx = f'_{suffix}' if suffix else ''
            fname = f'{program}_aiozp_{period_tag}_{end:%Y%m%d-%H%M}Z{sfx}.html'  # end = "as of", Z=UTC
            if size_mb > _AIOZP_MAX_MB:  # a single coin bigger than the limit — can't split further
                await update.message.reply_text(
                    f'{fname}: too large to send ({size_mb:.0f} MB > Telegram\'s ~50 MB limit). '
                    f'Try a shorter period.')
                continue
            with open(ppath, 'rb') as f:  # caption (the split note) rides on the first document only
                await update.message.reply_document(document=f, filename=fname,
                                                    caption=caption if idx == 0 else None)
    except Exception as e:
        await update.message.reply_text(f'Failed to send report: {e}')
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_dmc(token, start, end, period):
    """Blocking: fetch gamma-dash settings + vol/spread history and write the standalone report.
    Returns (html_path, tmpdir) or (None, None) if the token has no DMC data. Fetch errors
    propagate (so /dmc reports the real reason instead of a bare 'no data')."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    hours = max(1, min(_DMC_MAX_HOURS, math.ceil((now - start).total_seconds() / 3600)))
    g = dmc_mod.fetch_dmc(_dmc_tokens(token), hours=hours, start=start, end=end)  # raises -> dmc_command
    if g is None or not g.rows:
        return None, None
    tmpdir = tempfile.mkdtemp(prefix='dmc_')
    path = os.path.join(tmpdir, 'dmc.html')
    zp_html.write_dmc_report(g, path, token, period_label=period)
    return path, tmpdir


async def dmc_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.bot_data['open'] and chat_id not in context.bot_data['allowed']:
        await update.message.reply_text(
            f'Not authorised. Your chat id is {chat_id} — add it to allowed_chats in secrets.toml.')
        return
    args = context.args
    if not args:
        await update.message.reply_text('Usage: /dmc TOKEN [PERIOD]\n'
                                        'e.g. /dmc FET 4h   (period defaults to 8h)')
        return
    token = args[0].upper()
    period = args[1] if len(args) > 1 else '8h'
    try:
        start, end = parse_period(period)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return

    await update.effective_chat.send_action('upload_document')
    tmpdir = None
    async with _zp_queue(update, f'DMC {token} {period}'):
        try:
            path, tmpdir = await asyncio.to_thread(_run_dmc, token, start, end, period)
        except Exception as e:
            await update.message.reply_text(f'DMC failed: {e}')
            return
    try:
        if path:
            period_tag = ''.join(c if c.isalnum() else '-' for c in period).strip('-')
            fname = f'{token}_dmc_{period_tag}_{end:%Y%m%d-%H%M}Z.html'  # window end ("as of"), Z = UTC
            with open(path, 'rb') as f:
                await update.message.reply_document(document=f, filename=fname)
        else:
            await update.message.reply_text(
                f'No gamma-dash data for {token} (unknown token or no dynamic strategies).')
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)


def main():
    token, allowed, open_access = _load_telegram_conf()
    print('Loading config and services...')
    config = zp_bootstrap.load_config()
    zp_bootstrap.build_services(config,
                                progress=lambda m: print(f'[.] {m}'),
                                warn=lambda m: print(f'[!] {m}'),
                                background_fx=True)
    # concurrent_updates lets a second user's handler run (and get the queue ack) while another job
    # is still computing; the actual compute stays serialised on _zp_lock inside _zp_queue.
    app = Application.builder().token(token).concurrent_updates(True).build()
    app.bot_data['config'] = config
    app.bot_data['allowed'] = allowed
    app.bot_data['open'] = open_access
    app.add_handler(CommandHandler('zp', zp_command))
    app.add_handler(CommandHandler('compliance', compliance_command))
    app.add_handler(CommandHandler('captures', captures_command))
    app.add_handler(CommandHandler('markouts', markouts_command))
    app.add_handler(CommandHandler('vol', vol_command))
    app.add_handler(CommandHandler('ticksize', ticksize_command))
    app.add_handler(CommandHandler('dynamicsettings', dmccheck_command))
    app.add_handler(CommandHandler('refvols', refvols_command))  # throwaway
    app.add_handler(CommandHandler('aiozp', aiozp_command))
    app.add_handler(CommandHandler('dmc', dmc_command))
    app.add_handler(CommandHandler(['help', 'start'], help_command))
    if open_access:
        print('ZP bot running. OPEN ACCESS — anyone who messages the bot can use it.')
    else:
        print(f'ZP bot running. Allowed chats: {sorted(allowed) or "(none yet — add IDs to secrets.toml)"}')
    app.run_polling()


if __name__ == '__main__':
    main()
