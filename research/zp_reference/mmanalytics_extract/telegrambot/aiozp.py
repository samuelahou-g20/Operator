"""All-in-one desk report — two layers in one file:

 - `.aio-static` (visible by default, NO JS): the Summary grids + a collapsible `<details>` per coin
   holding that coin's STATIC report (PNGs + tables). This is what a no-JS phone (Telegram's viewer)
   shows — <details> gives tap-to-expand navigation with zero JS.
 - `.aio-interactive` (hidden by default): the dropdown + lazily-created `<iframe srcdoc>` shell. A
   small script hides the static layer and reveals this on desktop (where JS + iframes work),
   restoring the full interactive experience.

Built by bot._run_aiozp: `views` = [{label, docs}] for the interactive iframes; `summary_static`
= the three grid fragments (grid body_only); `coins_static` = [(label, static_fragment), …].
"""

import html as _html
import json

from . import grid
from . import zp_html
from . import logo

_CSS = """
*{box-sizing:border-box}
/* zp_html._CSS (bundled for the static coin fragments) caps body at 1100px — reset it so the
   interactive Summary grids use the full window; the phone layer keeps its own cap below. */
body{margin:0;max-width:none;background:#0f1216;color:#e6e9ef;
font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.aio-static{max-width:1100px;margin:auto;padding:16px}
.aio-static>h1{font-size:20px;margin:0 0 2px}
.aio-static>.sub{color:#8b94a3;font-size:12px;margin:0 0 12px}
.aio-static .gridwrap{max-height:none;margin-top:4px}   /* flatten scrollboxes; tighter for phone */
.aio-static h2{margin:14px 0 6px}
.aio-static h3{margin:14px 0 2px}
.aio-static h3+.sub{display:none}               /* drop the per-grid subtitle (repeats the header) */
.aio-static .headline{margin:6px 0}
.aio-static .legend{margin:6px 0}
.aio-static .note{margin:4px 0 8px}
details.aio-coin{border:1px solid #262b33;border-radius:8px;margin:8px 0;background:#12161c}
details.aio-coin>summary{cursor:pointer;padding:10px 14px;font-weight:600;font-size:15px;
list-style:none}
details.aio-coin>summary::-webkit-details-marker{display:none}
details.aio-coin>summary::before{content:"\\25B8  ";color:#6ea8ff}
details.aio-coin[open]>summary::before{content:"\\25BE  "}
.aio-coin-body{padding:0 14px 12px}
.aio-note{background:#3a2a12;border:1px solid #6b4a1c;border-radius:6px;color:#f0b429;
padding:8px 12px;margin:6px 0 10px;font-size:13px}
.aio-interactive{display:none}
.aio-bar{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:10px 14px;
flex-wrap:wrap;padding:10px 14px;background:#161a20;border-bottom:1px solid #262b33}
.aio-bar h1{margin:0;font-size:17px}
.aio-bar .sub{color:#8b94a3;font-size:12px;margin:0}
select#aio-view{background:#1a1f26;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;
padding:7px 10px;font-size:14px;min-width:150px}
button#aio-summary{background:#1a1f26;color:#6ea8ff;border:1px solid #3a424e;border-radius:6px;
padding:7px 12px;font-size:13px;cursor:pointer}
button#aio-summary:hover{background:#222933}
.aio-vol{color:#e6e9ef;font-size:12.5px;font-weight:600;font-variant-numeric:tabular-nums;
background:#1a1f26;border:1px solid #3a424e;border-radius:6px;padding:5px 10px}
.aio-vol b{color:#8b94a3;font-weight:600}
#aio-host{padding:0}
iframe.aio-frame{width:100%;border:0;display:block;background:#0f1216;min-height:70vh}
"""

# Desktop enhancement: hide the static layer, reveal the interactive shell, then the same
# rebuild-on-switch iframe loader as before. Never runs on a no-JS phone, so the static layer stays.
_LOADER = """<script>
(function(){
  var stat=document.querySelector('.aio-static'); if(stat) stat.style.display='none';
  var itv=document.querySelector('.aio-interactive'); if(itv) itv.style.display='block';
  var VIEWS = __PAYLOAD__;
  var host = document.getElementById('aio-host');
  var sel = document.getElementById('aio-view');
  var sumbtn = document.getElementById('aio-summary');
  function resize(f){
    try{ var d=f.contentDocument; if(!d) return;
      var h=d.documentElement.scrollHeight; if(h>0) f.style.height=(h+24)+'px'; }catch(e){}
  }
  function onload(f){
    try{ var d=f.contentDocument;
      if(d && d.head){ var st=d.createElement('style');
        st.textContent='.gridwrap{max-height:none !important}'; d.head.appendChild(st); }
    }catch(e){}
    resize(f); setTimeout(function(){resize(f);},700);
  }
  function show(i){
    host.textContent='';
    (VIEWS[i]||[]).forEach(function(doc){
      var f=document.createElement('iframe');
      f.className='aio-frame';
      f.addEventListener('load', function(){ onload(f); });
      f.srcdoc=doc; host.appendChild(f);
    });
    if(sumbtn) sumbtn.hidden = (i===0);   // 'Summary' button only while viewing a coin
  }
  sel.addEventListener('change', function(){ show(+sel.value); });
  if(sumbtn) sumbtn.addEventListener('click', function(){
    sel.value='0'; show(0); try{window.scrollTo(0,0);}catch(_){} });
  window.addEventListener('resize', function(){
    var fs=host.querySelectorAll('iframe.aio-frame'); for(var k=0;k<fs.length;k++) resize(fs[k]);
  });
  // The desk summary (a grid iframe) can ask to jump to a coin's view.
  window.addEventListener('message', function(e){
    var d=e.data; if(!d||d.t!=='aiozp-goto') return;
    for(var i=0;i<sel.options.length;i++){ if(sel.options[i].textContent===d.coin){
      sel.value=sel.options[i].value; show(+sel.value); try{window.scrollTo(0,0);}catch(_){} break; } }
  });
  show(0);
})();
</script>"""


def build_aiozp_report(program, views, summary_static, coins_static, start, end, period_label=None,
                       no_png_note=None, total_volume=None):
    period = period_label or f'{start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC'
    subtitle = f'{start:%Y-%m-%d %H:%M} → {end:%H:%M} UTC'
    title = f'{logo.mark()}{_html.escape(program)} — All-in-one'
    vol_bar = (f'<span class="aio-vol"><b>Total vol</b> {_html.escape(str(total_volume))}</span>'
               if total_volume else '')

    # Static layer (no-JS): Summary grids + a <details> per coin holding its static report.
    details = ''.join(
        f'<details class="aio-coin"><summary>{_html.escape(str(lbl))}</summary>'
        f'<div class="aio-coin-body">{body}</div></details>'
        for lbl, body in coins_static)
    note = f'<p class="aio-note">{_html.escape(no_png_note)}</p>' if no_png_note else ''
    vol_static = (f' · Total vol {_html.escape(str(total_volume))}' if total_volume else '')
    static = (f'<div class="aio-static"><h1>{title}</h1>'
              f'<p class="sub">{_html.escape(subtitle)}{vol_static}</p>{note}'
              f'<h2>Summary</h2>{summary_static}{details}</div>')

    # Interactive layer (desktop, hidden by default): the dropdown + lazy iframes.
    options = ''.join(f'<option value="{i}">{_html.escape(str(v["label"]))}</option>'
                      for i, v in enumerate(views))
    # `</` → `<\/` so embedded </script> (Plotly newPlot) can't close our loader early; json.dumps
    # handles the rest of the escaping. In a JS string "<\/" === "</", so srcdoc gets the real HTML.
    payload = json.dumps([v['docs'] for v in views]).replace('</', '<\\/')
    loader = _LOADER.replace('__PAYLOAD__', payload)
    interactive = (f'<div class="aio-interactive"><div class="aio-bar"><h1>{title}</h1>'
                   f'<span class="sub">{_html.escape(subtitle)}</span>{vol_bar}'
                   f'<label class="sub" for="aio-view">View</label>'
                   f'<select id="aio-view">{options}</select>'
                   f'<button id="aio-summary" type="button" hidden>↑ Summary</button></div>'
                   f'<div id="aio-host"></div></div>')

    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'{logo.FAVICON}'
        f'<title>{_html.escape(f"{program} all-in-one {period}")}</title>'
        f'<style>{grid._CSS}{zp_html._CSS}{_CSS}</style></head><body>'
        f'{static}{interactive}{loader}</body></html>')


def write_aiozp_report(program, views, summary_static, coins_static, start, end, path,
                       period_label=None, no_png_note=None, total_volume=None):
    html = build_aiozp_report(program, views, summary_static, coins_static, start, end, period_label,
                              no_png_note=no_png_note, total_volume=total_volume)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path
