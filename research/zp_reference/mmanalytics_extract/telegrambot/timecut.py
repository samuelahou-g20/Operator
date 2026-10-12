"""Client-side time brush for a ZP report.

A sticky dual-handle range over the report period lets the reader cut the report down to a
sub-window with no server round-trip. Two kinds of section respond:

  * time-series charts (flows, price/delta/PnL, orderbook widths) simply re-range their x-axis;
  * recomputed sections (summary tables, captures, markouts, 2% compliance) are rebuilt from
    per-trade / per-kline / per-snapshot data embedded in the page — the same rows the server
    aggregates, so a full-window selection reproduces the server numbers.

Dynamic-mode settings are a point-in-time snapshot (pulled at the end of the period) and never
move; they carry a note saying so.

This module owns only the brush shell + the JS engine's core (the aggregation appliers for the
recomputed sections are registered by zp_html as each section is emitted). Public surface:

  timecut.CSS                    -> styles, appended after zp_html._CSS
  timecut.brush_html()           -> the sticky brush bar markup
  timecut.engine_js(start, end)  -> the <script> engine (brush wiring + time-series re-rangers)
  timecut.SETTINGS_NOTE          -> the 'reflects end of period' note for the settings section
"""

import json
from datetime import timezone

import numpy as np
import pandas as pd


def _ms(dt):
    """Epoch milliseconds for a report datetime, treated as UTC (report datetimes are naive-UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _msv(series):
    """Vectorised naive-UTC datetimes -> epoch-ms int64."""
    return (pd.to_datetime(series).astype('int64') // 1_000_000)


def _f(x, nd=None):
    """float or None for NaN; optionally rounded (keeps the embedded JSON small)."""
    if x is None or x != x:
        return None
    return round(float(x), nd) if nd is not None else float(x)


CSS = """
/* One sticky header holds the brush + the jump-nav so they stack (both were top:0 before). */
.tc-sticky{position:sticky;top:0;z-index:30;background:#0f1216;padding-top:2px}
.tc-sticky .zp-toc{position:static;top:auto}
.tc-brush{display:flex;align-items:center;gap:10px 14px;
flex-wrap:wrap;padding:9px 12px;margin:0 0 8px;border:1px solid #262b33;border-radius:8px;
background:#161a20}
.tc-brush .tc-title{font-size:12px;font-weight:600;color:#8b94a3;white-space:nowrap}
.tc-slider{position:relative;flex:1 1 300px;min-width:200px;height:26px}
.tc-slider .tc-rail{position:absolute;left:0;right:0;top:11px;height:4px;border-radius:3px;background:#2b3038}
.tc-slider .tc-fill{position:absolute;top:11px;height:4px;border-radius:3px;background:#6ea8ff}
.tc-slider input[type=range]{position:absolute;left:0;top:0;width:100%;height:26px;margin:0;
background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
.tc-slider input[type=range]:focus{outline:none}
.tc-slider input[type=range]::-webkit-slider-runnable-track{background:none;border:none}
.tc-slider input[type=range]::-moz-range-track{background:none;border:none}
.tc-slider input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;pointer-events:auto;
width:16px;height:16px;margin-top:-6px;border-radius:50%;background:#6ea8ff;border:2px solid #0f1216;
cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.5)}
.tc-slider input[type=range]::-moz-range-thumb{pointer-events:auto;width:15px;height:15px;
border-radius:50%;background:#6ea8ff;border:2px solid #0f1216;cursor:pointer}
.tc-read{font-size:12px;color:#e6e9ef;font-variant-numeric:tabular-nums;white-space:nowrap}
.tc-read b{color:#8b94a3;font-weight:600}
.tc-reset{background:#1a1f26;color:#6ea8ff;border:1px solid #3a424e;border-radius:6px;
padding:4px 11px;font-size:12px;cursor:pointer}
.tc-reset:hover{background:#222933}
.tc-reset[disabled]{opacity:.4;cursor:default}
.tc-settings-note{background:#3a2a12;border:1px solid #6b4a1c;border-radius:6px;color:#f0b429;
padding:6px 11px;margin:2px 0 8px;font-size:12px}
"""

SETTINGS_NOTE = ('<p class="tc-settings-note">Dynamic-mode settings are a point-in-time snapshot '
                 'from the <b>end of the full period</b> — they do not change with the time slider '
                 'above.</p>')


def brush_html():
    """The sticky brush bar. Two overlaid range inputs (lo/hi) over a rail; a live read-out of the
    selected window; a reset. Inert with no JS (the report just shows the full period)."""
    return (
        '<div class="tc-brush" id="tc-brush">'
        '<span class="tc-title">Time window</span>'
        '<div class="tc-slider"><div class="tc-rail"></div><div class="tc-fill" id="tc-fill"></div>'
        '<input type="range" id="tc-lo" min="0" max="1000" value="0" step="1" '
        'aria-label="Window start">'
        '<input type="range" id="tc-hi" min="0" max="1000" value="1000" step="1" '
        'aria-label="Window end"></div>'
        '<span class="tc-read" id="tc-read"></span>'
        '<button type="button" class="tc-reset" id="tc-reset" disabled>Reset</button>'
        '</div>')


# The engine. Handles the brush, keeps a registry of appliers (fn(loMs, hiMs)), and registers the
# time-series chart re-rangers. Recomputed sections push their own appliers via window.__ZPCUT.register
# (they run their <script> after this one). `range` is in ms; charts get a UTC 'YYYY-MM-DD HH:MM:SS'
# string so it lines up with the naive-UTC timestamps Plotly plotted.
_ENGINE = """<script>
(function(){
  var START=__START__, END=__END__, SPAN=(END-START)||1, STEPS=1000;
  function toMs(v){ return START + (+v/STEPS)*SPAN; }
  function toStep(ms){ return Math.round((ms-START)/SPAN*STEPS); }
  var C = window.__ZPCUT = { start:START, end:END, appliers:[],
    register:function(fn){ this.appliers.push(fn); }, curLo:START, curHi:END, ready:false };

  var lo=document.getElementById('tc-lo'), hi=document.getElementById('tc-hi');
  var fill=document.getElementById('tc-fill'), read=document.getElementById('tc-read');
  var reset=document.getElementById('tc-reset');
  if(!lo||!hi){ return; }

  function two(n){ return (n<10?'0':'')+n; }
  function fmt(ms){ var d=new Date(ms);
    return d.getUTCFullYear()+'-'+two(d.getUTCMonth()+1)+'-'+two(d.getUTCDate())+' '
         + two(d.getUTCHours())+':'+two(d.getUTCMinutes()); }
  function pstr(ms){ var d=new Date(ms);            // Plotly axis string (naive-UTC, matches server)
    return d.getUTCFullYear()+'-'+two(d.getUTCMonth()+1)+'-'+two(d.getUTCDate())+' '
         + two(d.getUTCHours())+':'+two(d.getUTCMinutes())+':'+two(d.getUTCSeconds()); }
  function durtxt(ms){ var m=Math.round(ms/60000), h=Math.floor(m/60); m=m%60;
    return (h?h+'h ':'')+(m||!h?m+'m':''); }
  C.pstr=pstr;

  function paint(){
    var l=(C.curLo-START)/SPAN*100, r=(END-C.curHi)/SPAN*100;
    fill.style.left=l+'%'; fill.style.right=r+'%';
    read.innerHTML='<b>'+fmt(C.curLo)+'</b> &rarr; <b>'+fmt(C.curHi)+'</b> &nbsp;('+durtxt(C.curHi-C.curLo)+')';
    var full=(C.curLo<=START && C.curHi>=END); if(reset) reset.disabled=full;
  }
  var pending=false;
  function apply(){ pending=false;
    for(var i=0;i<C.appliers.length;i++){ try{ C.appliers[i](C.curLo,C.curHi); }catch(e){} } }
  function schedule(){ if(!pending){ pending=true;
    (window.requestAnimationFrame||function(f){setTimeout(f,16);})(apply); } }
  function onInput(which){
    var a=+lo.value, b=+hi.value;
    if(a>b){ if(which==='lo'){ b=a; hi.value=a; } else { a=b; lo.value=b; } }
    C.curLo=toMs(a); C.curHi=toMs(b); paint(); schedule();
  }
  lo.addEventListener('input',function(){onInput('lo');});
  hi.addEventListener('input',function(){onInput('hi');});
  if(reset) reset.addEventListener('click',function(){
    lo.value=0; hi.value=STEPS; C.curLo=START; C.curHi=END; paint(); schedule(); });

  // Time-series charts: re-range their x-axis (no recompute). No-op if the div/Plotly isn't there.
  function ranger(id, axes){ return function(loMs,hiMs){
    var g=document.getElementById(id); if(!g||!window.Plotly) return;
    var full=(loMs<=START && hiMs>=END), upd={};
    axes.forEach(function(ax){ upd[ax+'.range']= full ? null : [pstr(loMs),pstr(hiMs)];
                               upd[ax+'.autorange']= full ? true : false; });
    try{ Plotly.relayout(g,upd); }catch(e){}
  };}
  C.register(ranger('zpflow',['xaxis','xaxis2']));
  C.register(ranger('zpwidths',['xaxis','xaxis2']));
  // (zppnl is recomputed — rebased to 0 at the window start — by the layer-2 engine, not re-ranged.)

  paint(); C.ready=true;
  // Re-apply once the page (and Plotly) has loaded, so a selection made early still takes.
  window.addEventListener('load', function(){ setTimeout(apply, 900); });
})();
</script>"""


def engine_js(start, end):
    return _ENGINE.replace('__START__', str(_ms(start))).replace('__END__', str(_ms(end)))


# --------------------------------------------------------------------------------------
# Layer 2 — the recomputed sections. `datasets(result)` embeds the per-trade / per-kline /
# per-minute data each section is rebuilt from; the JS below re-aggregates it for [lo,hi] and
# re-renders the tables + restyles the categorical charts on every brush move. A full-window
# selection sums every row, reproducing the server numbers (that's why the tables are safe to
# replace in place).
# --------------------------------------------------------------------------------------
def _mkey(exch, pair):
    p = str(pair)
    return f'{exch}|{1 if p.startswith("P.") else 0}|{p.rsplit("-", 1)[-1].upper()}'


def datasets(result):
    """Compact JSON-able payload for the recompute engine, or {} if there's nothing to re-cut."""
    d = {}
    tr = result.trades
    if tr is not None and not tr.empty:
        px = 'price_usd' if 'price_usd' in tr.columns else 'price'
        acct = tr['account'].astype(str) if 'account' in tr.columns else pd.Series('', index=tr.index)
        atype = tr['acct_type'].astype(str) if 'acct_type' in tr.columns else pd.Series('', index=tr.index)
        mm = acct.str.contains(' MM', na=False)
        d['trades'] = [
            [int(t), 1 if m else 0, a, at, str(e), str(i), 0 if s == 'B' else 1,
             _f(q, 4), _f(n, 2), _f(p, 8)]
            for t, m, a, at, e, i, s, q, n, p in zip(
                _msv(tr['exch_ts']), mm, acct, atype, tr['exchange'], tr['inst'], tr['side'],
                tr['qty'], tr['notional_usd'], tr[px])]

    # Captures: per-market rows (config order), per-kline market volume, MM-venue keys, and the
    # cumulative chart's per-venue trace order (captures_ts groupby order, mapped to a row index).
    cap = result.captures
    if cap is not None and not cap.empty:
        rows, idx_of = [], {}
        for (e, p), r in cap.iterrows():
            if str(e) == 'TOTAL':
                continue
            active = bool(r.get('ob_active', True)) if 'ob_active' in cap.columns else True
            idx_of[(str(e), str(p))] = len(rows)
            rows.append({'e': str(e), 'p': str(p), 'k': _mkey(e, p), 'a': 1 if active else 0})
        mkt, chartv = [], []
        cts = result.captures_ts
        if cts is not None and not cts.empty and rows:
            for t, e, p, v in zip(_msv(cts['timestamp']), cts['exchange'], cts['pair'],
                                  cts['market_volume_usd']):
                i = idx_of.get((str(e), str(p)))
                if i is not None and v == v:
                    mkt.append([int(t), i, _f(v, 1)])
            for (e, p), _g in cts.groupby(['exchange', 'pair']):
                i = idx_of.get((str(e), str(p)))
                if i is not None:
                    chartv.append([str(e), str(p), i])
        mm_keys = sorted({f'{r["e"]}|{r["p"].rsplit("-", 1)[-1].upper()}'
                          for r in rows if r['a']})
        d['cap'] = {'rows': rows, 'mkt': mkt, 'mmKeys': mm_keys, 'chartV': chartv}

    # Markouts: per-MM-trade bps per horizon + the full-period venue order (so the heatmap's venue
    # axis + its <select> stay stable while only the values re-cut).
    mt = result.markout_trades
    if mt is not None and not mt.empty:
        bps = sorted((int(c[4:-5]) for c in mt.columns
                      if c.startswith('pnl_') and c.endswith('s_bps')))
        bcols = [f'pnl_{s}s_bps' for s in bps]
        order = []
        if result.markouts is not None and not result.markouts.empty:
            mk = result.markouts
            mk = mk.sort_values('volume_$', ascending=False) if 'volume_$' in mk.columns else mk
            order = [[str(e), str(i)] for e, i in mk.index]
        cols = [_msv(mt['exch_ts']), mt['exchange'], mt['inst'], mt['qty'], mt['volume_$']] \
            + [mt[c] for c in bcols]
        rows = []
        for vals in zip(*cols):
            rows.append([int(vals[0]), str(vals[1]), str(vals[2]), _f(vals[3], 4), _f(vals[4], 2)]
                        + [_f(v, 3) for v in vals[5:]])
        d['mo'] = {'wins': bps, 'order': order, 'rows': rows}

    # 2% compliance: per venue-minute counts (venues in display order).
    comp = result.ob_compliance
    if comp is not None and not comp.empty:
        venues, vidx = [], {}
        for _, r in comp.iterrows():
            vidx[(str(r['exchange']), str(r['pair']))] = len(venues)
            venues.append({'e': str(r['exchange']), 'p': str(r['pair'])})
        bk = []
        cbt = result.ob_compliance_ts
        if cbt is not None and not cbt.empty:
            for vals in zip(_msv(cbt['timestamp']), cbt['exchange'], cbt['pair'], cbt['snaps'],
                            cbt['in_snap'], cbt['two_cnt'], cbt['comp_cnt'], cbt['bob_sum'],
                            cbt['bob_cnt'], cbt['bob_lo'], cbt['bob_hi']):
                i = vidx.get((str(vals[1]), str(vals[2])))
                if i is None:
                    continue
                bk.append([int(vals[0]), i, int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]),
                           _f(vals[7], 2) or 0.0, int(vals[8]), _f(vals[9], 2), _f(vals[10], 2)])
        d['comp'] = {'venues': venues, 'bk': bk, 'floor': 0.05}

    # Price / Delta / PnL series — re-cut by rebasing the cumulative panels to 0 at the window
    # start (columns: [t, price, cum_delta, cum_realised, cum_unrealised, cum_total]).
    dp = result.delta_pnl_ts
    if dp is not None and not dp.empty:
        d['pnl'] = [[int(t), _f(pr, 8), _f(cd, 2), _f(cr, 2), _f(cu, 2), _f(ct, 2)]
                    for t, pr, cd, cr, cu, ct in zip(
                        _msv(dp['timestamp']), dp['price'], dp['cum_delta'],
                        dp['cum_realised_pnl'], dp['cum_unrealised_pnl'], dp['cum_total_pnl'])]
    return d


_RECOMPUTE = r"""<script>
(function(){
  var C = window.__ZPCUT, D = window.__ZPCUT_DATA;
  if(!C || !D) return;
  var START=C.start, END=C.end;
  function inWin(t,lo,hi){ return t>=lo && t<=hi; }

  // ---- formatters mirroring zp_html._num / _signed ----
  function grp(s){ var n=s[0]==='-'; if(n)s=s.slice(1); var p=s.split('.');
    p[0]=p[0].replace(/\B(?=(\d{3})+(?!\d))/g,','); return (n?'-':'')+p.join('.'); }
  function money(x){ if(x==null||x!==x) return 'n/a'; if(x>-0.5&&x<0.5)x=0;
    return grp(String(Math.round(x))); }
  function sig6(x){ if(x==null||x!==x) return 'n/a'; if(x===0) return '0';
    var s=Number(x).toPrecision(6);
    if(s.indexOf('e')<0 && s.indexOf('.')>=0) s=s.replace(/\.?0+$/,''); return grp(s); }
  function signed(x){ if(x==null||x!==x) return 'n/a'; var v=(x>-0.5&&x<0.5)?0:x;
    var cls=v>0?'pos':(v<0?'neg':''); var t=grp(String(Math.round(v)));
    return cls?('<span class="'+cls+'">'+t+'</span>'):t; }
  function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function set(id,html){ var el=document.getElementById(id); if(el) el.innerHTML=html; }

  // _grid(headers, rows, leftcols): rows = [{cells:[...], total:bool}]
  function grid(headers, rows, left){
    left=left||{}; function lc(i){ return left[i]?' class="l"':''; }
    var th=headers.map(function(h,i){return '<th'+lc(i)+'>'+h+'</th>';}).join('');
    var body=rows.map(function(r){
      return '<tr'+(r.total?' class="total"':'')+'>'
        + r.cells.map(function(c,i){return '<td'+lc(i)+'>'+c+'</td>';}).join('')+'</tr>';
    }).join('');
    return '<div class="tablewrap"><table class="grid"><thead><tr>'+th+'</tr></thead><tbody>'
         + body+'</tbody></table></div>';
  }

  // ================= Summary (MM / Gamma / Total) =================
  function sumAgg(lo,hi){
    var T=D.trades, g={}, lastT=-1, lpxSum=0, lpxN=0;
    for(var i=0;i<T.length;i++){ var r=T[i]; if(!inWin(r[0],lo,hi)) continue;
      if(r[0]>lastT){lastT=r[0];lpxSum=r[9];lpxN=1;} else if(r[0]===lastT){lpxSum+=r[9];lpxN++;} }
    var lpx=lpxN?lpxSum/lpxN:NaN;
    for(i=0;i<T.length;i++){ r=T[i]; if(!inWin(r[0],lo,hi)) continue;
      var k=r[2]+''+r[3]+''+r[5];
      var o=g[k]||(g[k]={acct:r[2],atype:r[3],inst:r[5],mm:r[1],bq:0,sq:0,bn:0,sn:0});
      if(r[6]===0){o.bq+=r[7];o.bn+=r[8];} else {o.sq+=r[7];o.sn+=r[8];} }
    var groups=Object.keys(g).map(function(k){return g[k];});
    groups.sort(function(a,b){ return (a.acct<b.acct?-1:a.acct>b.acct?1:
      a.atype<b.atype?-1:a.atype>b.atype?1: a.inst<b.inst?-1:a.inst>b.inst?1:0); });
    groups.forEach(function(o){ o.buyAvg=o.bn===0?0:o.bn/o.bq; o.sellAvg=o.sn===0?0:o.sn/o.sq;
      o.net=o.bn-o.sn; o.pnl=(lpx-o.buyAvg)*o.bq+(o.sellAvg-lpx)*o.sq; });
    return {groups:groups, lastPx:lpx};
  }
  function summary(lo,hi){
    var groups=sumAgg(lo,hi).groups;
    function totalRow(list, prefix){ var t={bq:0,sq:0,bn:0,sn:0,pnl:0};
      list.forEach(function(o){t.bq+=o.bq;t.sq+=o.sq;t.bn+=o.bn;t.sn+=o.sn;t.pnl+=o.pnl;});
      t.buyAvg=t.bn/t.bq; t.sellAvg=t.sn/t.sq; t.net=t.bn-t.sn; t.acct=prefix+'TOTAL';
      t.atype=''; t.inst=''; return t; }
    function rowCells(o){ return {cells:[esc(o.acct),esc(o.atype||''),esc(o.inst||''),
      money(o.bn),money(o.sn),signed(o.net),sig6(o.buyAvg),sig6(o.sellAvg),signed(o.pnl)],
      total:String(o.acct).slice(-5)==='TOTAL'}; }
    var mmG=groups.filter(function(o){return o.mm;}), otG=groups.filter(function(o){return !o.mm;});
    var heads=['Account','Strategy','Inst','Buy $','Sell $','Net $','Buy avg','Sell avg','PnL $'];
    var left={0:1,1:1,2:1}, out='';
    [['MM',mmG,'MM '],['Gamma',otG,'GAMMA ']].forEach(function(sec){
      var rows=sec[1].map(rowCells);
      rows.push(rowCells(totalRow(sec[1],sec[2])));
      out+='<h3>'+sec[0]+'</h3>'+grid(heads,rows,left);
    });
    var trows=groups.map(rowCells); trows.push(rowCells(totalRow(groups,'')));
    out+='<h3>Total</h3>'+grid(heads,trows,left);
    set('tc-summary', out);
  }
  if(D.trades && document.getElementById('tc-summary')) C.register(summary);

  // ================= Captures table =================
  var CAP_BANDS=[[10,'#2f9e44','#fff'],[8,'#e0b100','#141414'],[5,'#e8833a','#141414'],[0,'#d64545','#fff']];
  function capBand(pct){ for(var i=0;i<CAP_BANDS.length;i++){ if(pct>=CAP_BANDS[i][0]) return CAP_BANDS[i]; }
    return CAP_BANDS[CAP_BANDS.length-1]; }
  function capAgg(lo,hi){
    var rows=D.cap.rows.map(function(cr){return {e:cr.e,p:cr.p,k:cr.k,a:cr.a,g:0,m:0,hasM:false};});
    var byKey={}; rows.forEach(function(r,i){ (byKey[r.k]=byKey[r.k]||[]).push(i); });
    var M=D.cap.mkt,i;
    for(i=0;i<M.length;i++){ if(!inWin(M[i][0],lo,hi)) continue; rows[M[i][1]].m+=M[i][2]; rows[M[i][1]].hasM=true; }
    var T=D.trades||[];
    for(i=0;i<T.length;i++){ var r=T[i]; if(!inWin(r[0],lo,hi)) continue;
      var inst=r[5], perp=inst.indexOf('P.')===0?1:0, q=(inst.split('-')[1]||'').toUpperCase();
      var k=r[4]+'|'+perp+'|'+q, ix=byKey[k]; if(ix) rows[ix[0]].g+=r[8]; }
    return rows;
  }
  function captures(lo,hi){
    var rows=capAgg(lo,hi), body='', tmg=0,tmm=0,tag=0,tam=0;
    rows.forEach(function(r){
      var nod=!r.hasM || r.m<=0, c=nod?NaN:r.g/r.m, cap;
      if(!nod){ tag+=r.g; tam+=r.m; if(r.a){ tmg+=r.g; tmm+=r.m; } }
      if(nod){ cap='<td class="l mut">no market</td>'; }
      else if(r.a){ var b=capBand(c*100); cap='<td style="background:'+b[1]+';color:'+b[2]+';font-weight:700">'+(c*100).toFixed(1)+'%</td>'; }
      else { cap='<td style="color:#8b94a3">'+(c*100).toFixed(1)+'%</td>'; }
      var rc=(r.a&&!nod)?'':' style="opacity:.45"';
      body+='<tr'+rc+'><td class="l">'+esc(r.e)+'</td><td class="l">'+esc(r.p)+'</td><td>'
          + money(r.g)+'</td><td>'+(nod?'n/a':money(r.m))+'</td>'+cap+'</tr>';
    });
    if(tmm){ var mmc=tmg/tmm*100;
      body+='<tr class="total"><td class="l">TOTAL (MM Venues)</td><td></td><td>'+money(tmg)
          +'</td><td>'+money(tmm)+'</td><td style="font-weight:700">'+mmc.toFixed(1)+'%</td></tr>'; }
    if(tam && Math.abs(tag-tmg)>0.5){ var allc=tag/tam*100;
      body+='<tr class="total" style="opacity:.5"><td class="l">TOTAL (all venues)</td><td></td><td>'
          +money(tag)+'</td><td>'+money(tam)+'</td><td>'+allc.toFixed(1)+'%</td></tr>'; }
    var thead='<th class="l">Exchange</th><th class="l">Pair</th><th>G20 $</th><th>Market $</th><th>Capture</th>';
    var note='<p class="sub" style="margin:2px 0 0">Capture coloured only where we run an active '
      +'orderbook (MM); other venues greyed. TOTAL is MM-only (our share on the venues we make).</p>';
    set('tc-captures','<div class="tablewrap"><table class="grid"><thead><tr>'+thead
        +'</tr></thead><tbody>'+body+'</tbody></table></div>'+note);
    captureChart(lo,hi);
  }
  if(D.cap && document.getElementById('tc-captures')) C.register(captures);

  // ================= Captures cumulative chart =================
  function cumCap(mkt, our){  // maps: bucketMs -> volume; returns {x:[str], y:[pct]}
    var keys={},k; for(k in mkt) keys[k]=1; for(k in our) keys[k]=1;
    var ks=Object.keys(keys).map(Number).sort(function(a,b){return a-b;});
    var cm=0,cg=0,x=[],y=[];
    for(var i=0;i<ks.length;i++){ cm+=(mkt[ks[i]]||0); cg+=(our[ks[i]]||0);
      x.push(C.pstr(ks[i])); y.push(cm>0? cg/cm*100 : null); }
    return {x:x,y:y};
  }
  function bucket5(t){ return Math.floor(t/300000)*300000; }
  function captureChart(lo,hi){
    var g=document.getElementById('zpcaptures'); if(!g||!window.Plotly||!D.cap) return;
    var mmSet={}; (D.cap.mmKeys||[]).forEach(function(s){mmSet[s]=1;});
    var rows=D.cap.rows;
    // market series: per row, plus MM/all aggregates; our series from trades
    var M=D.cap.mkt, T=D.trades||[], i, b;
    var mktMM={}, mktAll={}, mktRow={};
    for(i=0;i<M.length;i++){ if(!inWin(M[i][0],lo,hi))continue; b=bucket5(M[i][0]); var ri=M[i][1];
      mktAll[b]=(mktAll[b]||0)+M[i][2]; (mktRow[ri]=mktRow[ri]||{})[b]=(mktRow[ri][b]||0)+M[i][2];
      var rk=rows[ri].e+'|'+rows[ri].p.split('-')[1].toUpperCase();
      if(mmSet[rk]) mktMM[b]=(mktMM[b]||0)+M[i][2]; }
    var ourMM={}, ourAll={}, ourInst={};
    for(i=0;i<T.length;i++){ var r=T[i]; if(!inWin(r[0],lo,hi))continue; b=bucket5(r[0]);
      ourAll[b]=(ourAll[b]||0)+r[8];
      var q=(r[5].split('-')[1]||'').toUpperCase(), rk=r[4]+'|'+q;
      if(mmSet[rk]) ourMM[b]=(ourMM[b]||0)+r[8];
      var ik=r[4]+''+r[5]; (ourInst[ik]=ourInst[ik]||{})[b]=(ourInst[ik][b]||0)+r[8]; }
    var traces=[cumCap(mktMM,ourMM), cumCap(mktAll,ourAll)];
    (D.cap.chartV||[]).forEach(function(cv){ var ri=cv[2];
      var ik=cv[0]+''+cv[1];   // per-venue our-vol keyed by inst==pair (matches server)
      traces.push(cumCap(mktRow[ri]||{}, ourInst[ik]||{})); });
    try{ window.Plotly.restyle(g, {x:traces.map(function(t){return t.x;}),
                                   y:traces.map(function(t){return t.y;})}); }catch(e){}
  }

  // ================= Markouts (table + heatmap) =================
  function moAgg(lo,hi){  // -> {byKey:{ 'ei': {vol, num[], den[]} }}
    var R=D.mo.rows, nH=D.mo.wins.length, by={};
    for(var i=0;i<R.length;i++){ var r=R[i]; if(!inWin(r[0],lo,hi)) continue;
      var k=r[1]+''+r[2], o=by[k]||(by[k]={e:r[1],inst:r[2],vol:0,num:[],den:[]});
      o.vol+=(r[4]||0);
      for(var h=0;h<nH;h++){ var v=r[5+h]; if(v==null||v!==v) continue;
        o.num[h]=(o.num[h]||0)+v*r[3]; o.den[h]=(o.den[h]||0)+r[3]; } }
    return by;
  }
  function moWeighted(o,h){ return (o.den[h]? o.num[h]/o.den[h] : NaN); }
  function markouts(lo,hi){
    var by=moAgg(lo,hi), wins=D.mo.wins;
    // table: sort by windowed volume desc
    var keys=Object.keys(by).sort(function(a,b){return by[b].vol-by[a].vol;});
    var head='<th class="l">Exchange</th><th class="l">Inst</th><th>Vol $</th>'
      + wins.map(function(s){return '<th>'+s+'s</th>';}).join('');
    var body=keys.map(function(k){ var o=by[k];
      var cells='<td class="l">'+esc(o.e)+'</td><td class="l">'+esc(o.inst)+'</td><td>'+money(o.vol)+'</td>';
      for(var h=0;h<wins.length;h++){ var v=moWeighted(o,h);
        cells+='<td>'+(v!==v||v==null?'n/a':(v>=0?'+':'')+v.toFixed(1))+'</td>'; }
      return '<tr>'+cells+'</tr>'; }).join('');
    set('tc-markouts','<div class="tablewrap"><table class="grid"><thead><tr>'+head
        +'</tr></thead><tbody>'+body+'</tbody></table></div>');
    // heatmap: fixed full-period venue order, values re-cut
    var g=document.getElementById('zpmarkouts');
    if(g&&window.Plotly&&D.mo.order&&D.mo.order.length){
      var z=[],text=[];
      D.mo.order.forEach(function(ei){ var o=by[ei[0]+''+ei[1]];
        var zr=[],tr=[];
        for(var h=0;h<wins.length;h++){ var v=o?moWeighted(o,h):NaN;
          zr.push(v!==v?null:v); tr.push(v!==v?'':v.toFixed(1)); }
        z.push(zr); text.push(tr); });
      if(window.__tcMk) window.__tcMk.set(z,text); else { try{ window.Plotly.restyle(g,{z:[z],text:[text]}); }catch(e){} }
    }
  }
  if(D.mo && document.getElementById('tc-markouts')) C.register(markouts);

  // ================= 2% compliance (table + bar) =================
  function compAgg(lo,hi){
    var V=D.comp.venues.map(function(v){return {e:v.e,p:v.p,snaps:0,in_snap:0,two:0,comp:0,
      bobSum:0,bobCnt:0,bobLo:NaN,bobHi:NaN};});
    var B=D.comp.bk;
    for(var i=0;i<B.length;i++){ var b=B[i]; if(!inWin(b[0],lo,hi)) continue; var o=V[b[1]];
      o.snaps+=b[2]; o.in_snap+=b[3]; o.two+=b[4]; o.comp+=b[5]; o.bobSum+=b[6]; o.bobCnt+=b[7];
      if(b[8]!=null){ o.bobLo=(o.bobLo!==o.bobLo)?b[8]:Math.min(o.bobLo,b[8]); }
      if(b[9]!=null){ o.bobHi=(o.bobHi!==o.bobHi)?b[9]:Math.max(o.bobHi,b[9]); } }
    V.forEach(function(o){ o.uptime=o.snaps?o.two/o.snaps:NaN; o.compliance=o.snaps?o.comp/o.snaps:NaN;
      o.wide=o.snaps?(o.two-o.comp)/o.snaps:NaN; o.one=o.snaps?(o.in_snap-o.two)/o.snaps:NaN;
      o.empty=o.snaps?(o.snaps-o.in_snap)/o.snaps:NaN; o.bobMean=o.bobCnt?o.bobSum/o.bobCnt:NaN;
      o.nodata=!(o.uptime>=D.comp.floor); });
    return V;
  }
  function compliance(lo,hi){
    var V=compAgg(lo,hi);
    var rows=V.map(function(o){ var venue=esc(o.e+' '+o.p);
      if(o.nodata) return {cells:['<span class="mut">'+venue+'</span>','—','<span class="mut">no data</span>','—']};
      var bob=(o.bobMean!==o.bobMean)?'n/a':Math.round(o.bobMean).toString();
      return {cells:[venue, Math.round(o.uptime*100)+'%', Math.round(o.compliance*100)+'%', bob]}; });
    set('tc-compliance-tbl', grid(['Venue','Uptime','Compliance','Avg BOB'], rows, {0:1}));
    var g=document.getElementById('zpcompliance');
    if(g&&window.Plotly){
      function seg(f){ return V.map(function(o){ return o.nodata?0:(o[f]*100); }); }
      var xs=[seg('compliance'),seg('wide'),seg('one'),seg('empty'),
              V.map(function(o){return o.nodata?100:0;})];
      try{ window.Plotly.restyle(g,{x:xs}); }catch(e){}
    }
  }
  if(D.comp && document.getElementById('tc-compliance-tbl')) C.register(compliance);

  // ================= Price / Delta / PnL (rebased) =================
  // Cumulative panels (delta + the three PnL lines) are rebased to 0 at the window's left edge, so a
  // sub-window shows what happened *in* the window; price stays absolute. Trace order in
  // _delta_pnl_fig: [price(col1), delta(col2), total(col5), realised(col3), unrealised(col4)].
  function pnlChart(lo,hi){
    var g=document.getElementById('zppnl'); if(!g||!window.Plotly||!D.pnl) return;
    var S=D.pnl, full=(lo<=START && hi>=END), base=null, i;
    if(!full){ for(i=0;i<S.length;i++){ if(S[i][0]<=lo) base=S[i]; else break; }
      if(!base){ for(i=0;i<S.length;i++){ if(S[i][0]>=lo){ base=S[i]; break; } } } }
    function ser(col,rebase){ var b=(rebase&&base)?base[col]:0, x=[],y=[];
      for(var j=0;j<S.length;j++){ var t=S[j][0]; if(t<lo||t>hi) continue;
        x.push(C.pstr(t)); y.push(S[j][col]==null?null:S[j][col]-b); } return {x:x,y:y}; }
    var tr=[ser(1,false),ser(2,true),ser(5,true),ser(3,true),ser(4,true)];
    try{ window.Plotly.restyle(g,{x:tr.map(function(t){return t.x;}),y:tr.map(function(t){return t.y;})});
      var xr=full?null:[C.pstr(lo),C.pstr(hi)];
      window.Plotly.relayout(g,{'xaxis.range':xr,'xaxis.autorange':full,
        'xaxis2.range':xr,'xaxis2.autorange':full,'xaxis3.range':xr,'xaxis3.autorange':full,
        'yaxis.autorange':true,'yaxis2.autorange':true,'yaxis3.autorange':true}); }catch(e){}
  }
  if(D.pnl && document.getElementById('zppnl')) C.register(pnlChart);

  // ================= Headline stat chips + text digest =================
  var GREEN='#3fb950', RED='#d9534f';
  var COMP_BANDS=[[95,'#2f9e44','#fff'],[90,'#e0b100','#141414'],[75,'#e8833a','#141414'],[0,'#d64545','#fff']];
  var CAPB=CAP_BANDS, MO_BANDS=[[5,'#1b6b3a','#d7f5df'],[1,'#3f7a52','#e8f7ee'],[-1,'#33383f','#c7ccd4'],
    [-5,'#9a6327','#ffe9cf'],[-1e9,'#a83232','#ffd9d9']];
  function band(bands,v){ for(var i=0;i<bands.length;i++){ if(v>=bands[i][0]) return bands[i]; }
    return bands[bands.length-1]; }
  function pmoney(x){ if(x==null||x!==x) return '—'; return (x<0?'−':'')+'$'+grp(String(Math.round(Math.abs(x)))); }
  function chip(key,val,color){ var el=document.querySelector('.zp-strip [data-tc="'+key+'"] .v');
    if(!el) return; el.innerHTML=val; el.style.color=color||''; }
  function headline(lo,hi){
    var full=(lo<=START && hi>=END);
    // Total PnL (Total table's grand total)
    if(D.trades){ var groups=sumAgg(lo,hi).groups, tp=0; groups.forEach(function(o){tp+=o.pnl;});
      if(!groups.length) tp=NaN;
      chip('total', pmoney(tp), tp>0?GREEN:(tp<0?RED:'')); }
    // Capture (MM) + Compliance (mean over reporting venues)
    if(D.cap){ var rows=capAgg(lo,hi), tg=0,tm=0;
      rows.forEach(function(r){ if(r.a && r.hasM && r.m>0){ tg+=r.g; tm+=r.m; } });
      if(tm>0){ var c=tg/tm*100; chip('capture', Math.round(c)+'%', band(CAPB,c)[1]); }
      else chip('capture','n/a',''); }
    if(D.comp){ var V=compAgg(lo,hi), cs=0,cn=0;
      V.forEach(function(o){ if(!o.nodata){ cs+=o.compliance*100; cn++; } });
      if(cn){ var cm=cs/cn; chip('compliance', Math.round(cm)+'%', band(COMP_BANDS,cm)[1]); }
      else chip('compliance','n/a',''); }
    // Markout 180s: across-venue volume-weighted of each venue's qty-weighted markout
    if(D.mo){ var h=D.mo.wins.indexOf(180);
      if(h>=0){ var by=moAgg(lo,hi), num=0,den=0;
        Object.keys(by).forEach(function(k){ var o=by[k], mv=moWeighted(o,h);
          if(mv===mv && o.vol>0){ num+=mv*o.vol; den+=o.vol; } });
        if(den){ var mo=num/den; chip('markout',(mo>=0?'+':'')+mo.toFixed(1), band(MO_BANDS,mo)[1]); }
        else chip('markout','n/a',''); } }
    // Text digest is a full-period snapshot — dim it + note when a sub-window is active.
    var ts=document.getElementById('tc-textsum');
    if(ts){ var note=document.getElementById('tc-textsum-note');
      if(!full){ if(!note){ note=document.createElement('p'); note.id='tc-textsum-note';
          note.className='tc-settings-note'; note.textContent='Text summary below reflects the full period.';
          ts.parentNode.insertBefore(note, ts); } note.style.display=''; ts.style.opacity='.5'; }
      else { if(note) note.style.display='none'; ts.style.opacity=''; } }
  }
  if(document.querySelector('.zp-strip [data-tc]') || document.getElementById('tc-textsum'))
    C.register(headline);
})();
</script>"""


def recompute_js(result):
    """The <script> that embeds the recompute datasets + registers the section appliers. '' when
    there's nothing to re-cut (no trades and no orderbook)."""
    d = datasets(result)
    if not d:
        return ''
    payload = json.dumps(d, separators=(',', ':')).replace('</', '<\\/')
    return f'<script>window.__ZPCUT_DATA={payload};</script>{_RECOMPUTE}'
