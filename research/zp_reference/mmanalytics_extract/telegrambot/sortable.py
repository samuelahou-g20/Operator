"""One reusable client-side sort for the report tables. Included at the end of a full-page report;
makes every `table.cmp` / `table.grid` on the page click-to-sort by any column header. Prefers a
cell's `data-v` numeric attribute, otherwise parses its text (handles $, %, ×, commas and k/M/B
suffixes). Rows whose class contains `total` stay pinned at the bottom. No-JS phones keep the
server-rendered order, so it's pure progressive enhancement.
"""

SORT_JS = """<script>(function(){
function val(td){
  if(!td) return NaN;
  var d=td.getAttribute('data-v');
  if(d!==null){ if(d==='') return NaN; var f=parseFloat(d); return isNaN(f)?NaN:f; }
  var t=(td.textContent||'').trim().replace('\\u2212','-');
  if(t===''||t==='-'||t==='\\u2014'||t==='\\u00b7') return NaN;
  var s=t.replace(/[,$%\\u00d7\\s]/g,''), m=1, c=s.slice(-1);
  if(c==='k'||c==='K'){m=1e3;s=s.slice(0,-1);} else if(c==='m'||c==='M'){m=1e6;s=s.slice(0,-1);}
  else if(c==='b'||c==='B'){m=1e9;s=s.slice(0,-1);}
  var f=parseFloat(s); return isNaN(f)?NaN:f*m;
}
function make(tbl){
  var head=tbl.tHead&&tbl.tHead.rows[0], tb=tbl.tBodies[0];
  if(!head||!tb) return;
  var ths=head.cells, st={col:-1,asc:true};
  for(var i=0;i<ths.length;i++){ ths[i].style.cursor='pointer';
    var a=document.createElement('span'); a.className='__arr'; a.style.color='#8b94a3'; ths[i].appendChild(a); }
  function sort(ci){
    var rows=[].slice.call(tb.rows);
    // Detail/expander rows (data-detail-of) aren't sorted on their own — they're re-glued to
    // their anchor row afterwards, so an expanded coin's breakdown follows it wherever it lands.
    var det=rows.filter(function(r){return r.hasAttribute('data-detail-of');});
    var sortable=rows.filter(function(r){return !r.hasAttribute('data-detail-of');});
    var pin=sortable.filter(function(r){return /\\btotal\\b/.test(r.className);});
    var body=sortable.filter(function(r){return !/\\btotal\\b/.test(r.className);});
    var numeric=false;
    for(var k=0;k<body.length;k++){ if(!isNaN(val(body[k].cells[ci]))){numeric=true;break;} }
    if(ci===st.col){ st.asc=!st.asc; } else { st.col=ci; st.asc=!numeric; }
    var asc=st.asc;
    body.sort(function(a,b){
      var x=val(a.cells[ci]), y=val(b.cells[ci]), nx=isNaN(x), ny=isNaN(y);
      if(nx&&ny){ var sa=a.cells[ci]?a.cells[ci].textContent:'', sb=b.cells[ci]?b.cells[ci].textContent:'';
        return asc?sa.localeCompare(sb):sb.localeCompare(sa); }
      if(nx) return 1; if(ny) return -1; return asc?x-y:y-x;
    });
    body.concat(pin).forEach(function(r){tb.appendChild(r);});
    det.forEach(function(d){ var key=d.getAttribute('data-detail-of');
      var anchor=tb.querySelector('tr[data-coin="'+key+'"]');
      if(anchor) tb.insertBefore(d, anchor.nextSibling); });
    for(var j=0;j<ths.length;j++){ var ar=ths[j].querySelector('.__arr');
      if(ar) ar.textContent=(j===ci?(asc?' \\u25b2':' \\u25bc'):''); }
  }
  for(var i=0;i<ths.length;i++)(function(i){ ths[i].addEventListener('click',function(){sort(i);}); })(i);
}
var t=document.querySelectorAll('table.cmp,table.grid');
for(var i=0;i<t.length;i++) make(t[i]);
})();</script>"""
