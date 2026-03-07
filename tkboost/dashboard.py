"""Localhost dashboard server for visualizing a TKStore."""

from __future__ import annotations

import csv
import io
import json
import re
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _load_store_rows(store_path: str) -> List[Dict[str, Any]]:
    p = Path(store_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_store_rows(store_path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(store_path)
    if not rows:
        existing = _load_store_rows(store_path)
        if existing:
            return
        return
    fieldnames = list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_debug_dir(store_path: str) -> Optional[Path]:
    p = Path(store_path).expanduser().resolve()
    candidate = p.parent / "debug_traces"
    if candidate.is_dir():
        return candidate
    return None


def _load_trace(debug_dir: Path, example_id: str) -> Optional[Dict[str, Any]]:
    candidates = sorted(debug_dir.glob(f"{example_id}_*"), reverse=True)
    if not candidates:
        return None
    trace_dir = candidates[0]
    out: Dict[str, Any] = {"trace_dir": str(trace_dir)}
    for name, key in [
        ("llm_interactions.json", "llm_interactions"),
        ("diff_output.txt", "diff_output"),
        ("rules_output.txt", "rules_output"),
        ("agent_sql_final.sql", "agent_sql"),
        ("gold_sql.sql", "gold_sql"),
        ("parsed_memories.json", "parsed_memories"),
    ]:
        fp = trace_dir / name
        if fp.exists():
            text = fp.read_text(encoding="utf-8")
            if name.endswith(".json"):
                try:
                    out[key] = json.loads(text)
                except Exception:
                    out[key] = text
            else:
                out[key] = text
    return out


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>TK-Boost Knowledge Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f5f6f8;color:#1a1a1a;line-height:1.5}
header{background:#111;color:#fff;padding:14px 28px;font-size:1.1rem;font-weight:600;display:flex;align-items:center;gap:12px}
header span.badge{background:#6c5ce7;border-radius:6px;padding:2px 10px;font-size:.82rem;font-weight:500}
.container{max-width:1440px;margin:0 auto;padding:20px 24px;width:100%}
.intro{background:#fff;border-left:4px solid #6c5ce7;border-radius:0 8px 8px 0;padding:14px 20px;margin-bottom:20px;font-size:.88rem;color:#444;line-height:1.65;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.intro strong{color:#1a1a1a}
.stats{display:flex;gap:16px;margin-bottom:18px;flex-wrap:wrap}
.stat-card{background:#fff;border-radius:10px;padding:14px 22px;box-shadow:0 1px 4px rgba(0,0,0,.06);min-width:140px}
.stat-card .val{font-size:1.5rem;font-weight:700;color:#6c5ce7}
.stat-card .lbl{font-size:.78rem;color:#888;text-transform:uppercase;letter-spacing:.4px}
.filters{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.filters select,.filters input{padding:7px 12px;border:1px solid #ddd;border-radius:6px;font-size:.88rem;background:#fff}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.06);table-layout:auto}
thead{background:#f0f0f4}
th{text-align:left;padding:10px 14px;font-size:.78rem;text-transform:uppercase;letter-spacing:.4px;color:#666;border-bottom:2px solid #e4e4ea;white-space:nowrap}
th.editable-hdr::after{content:' \270E';font-size:.7rem;color:#aaa;margin-left:3px}
td{padding:10px 14px;font-size:.85rem;border-bottom:1px solid #f0f0f4;vertical-align:top}
td:nth-child(1){white-space:nowrap}
td:nth-child(2){white-space:nowrap}
td:nth-child(3){white-space:nowrap}
td:nth-child(4){max-width:120px;word-break:break-word}
td:nth-child(5){max-width:140px;word-break:break-word}
td:nth-child(6){max-width:150px;word-break:break-word}
td:nth-child(7){white-space:nowrap}
tr.clickable{cursor:pointer;transition:background .15s}
tr.clickable:hover{background:#f4f2ff}
tr.active-row{background:#ede8ff!important}
td.knowledge-cell{min-width:320px}
.tag{display:inline-block;background:#eee;border-radius:4px;padding:1px 7px;font-size:.76rem;margin:1px 2px;color:#555}
.tag-db{background:#dfe6fd;color:#3b5bdb;display:none}
.tag-generic{background:#e3faef;color:#20854e;display:none}

/* editable cell indicators */
td.editable{position:relative}
td.editable::after{content:'\270E';position:absolute;top:4px;right:4px;font-size:.62rem;color:#ccc;opacity:0;transition:opacity .15s;pointer-events:none}
td.editable:hover::after{opacity:1}
td.editable:hover{background:#f9f8ff}
td.editing{outline:none;border:1px dashed #6c5ce7!important;background:#faf9ff;border-radius:3px}

/* tag picker popover */
.tag-picker-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:300;background:transparent}
.tag-picker{position:fixed;z-index:301;background:#fff;border:1px solid #ddd;border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,.15);padding:10px;min-width:200px;max-width:320px;max-height:340px;overflow-y:auto}
.tag-picker h5{font-size:.72rem;text-transform:uppercase;letter-spacing:.4px;color:#888;margin-bottom:6px}
.tag-picker .tp-option{display:inline-block;padding:4px 12px;margin:3px;border-radius:5px;font-size:.8rem;cursor:pointer;border:1.5px solid #ddd;color:#555;transition:all .12s;user-select:none}
.tag-picker .tp-option:hover{border-color:#6c5ce7;color:#6c5ce7;background:#f8f6ff}
.tag-picker .tp-option.selected{background:#6c5ce7;color:#fff;border-color:#6c5ce7}
.tag-picker .tp-actions{margin-top:10px;display:flex;gap:8px;justify-content:flex-end;border-top:1px solid #f0f0f0;padding-top:8px}
.tag-picker .tp-btn{padding:4px 14px;border-radius:5px;border:none;font-size:.78rem;cursor:pointer;font-weight:600}
.tag-picker .tp-btn.save{background:#6c5ce7;color:#fff}.tag-picker .tp-btn.cancel{background:#f0f0f0;color:#555}
.tag-picker .tp-btn:hover{opacity:.85}

.save-toast{position:fixed;bottom:24px;right:24px;background:#27ae60;color:#fff;padding:10px 20px;border-radius:8px;font-size:.88rem;box-shadow:0 2px 8px rgba(0,0,0,.15);opacity:0;transition:opacity .3s;pointer-events:none;z-index:400}
.save-toast.show{opacity:1}

.detail-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.35);z-index:100}
.detail-panel{position:fixed;top:0;right:0;width:58%;max-width:900px;height:100%;background:#fff;z-index:101;display:none;overflow-y:auto;box-shadow:-4px 0 20px rgba(0,0,0,.12);padding:0}
.detail-panel .dp-header{background:#111;color:#fff;padding:16px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:2}
.detail-panel .dp-header h3{font-size:1rem;font-weight:600}
.dp-close{cursor:pointer;font-size:1.4rem;opacity:.7;transition:opacity .15s}
.dp-close:hover{opacity:1}
.dp-body{padding:24px}
.dp-section{margin-bottom:24px}
.dp-section h4{font-size:.82rem;text-transform:uppercase;letter-spacing:.5px;color:#6c5ce7;margin-bottom:8px;border-bottom:2px solid #f0f0f4;padding-bottom:4px}
.dp-knowledge{background:#f8f7ff;border-left:4px solid #6c5ce7;padding:12px 16px;border-radius:0 8px 8px 0;font-size:.88rem;line-height:1.6;max-height:240px;overflow-y:auto;overflow-x:hidden}
.kp-line{padding:4px 0;border-bottom:1px dashed #ece9ff;word-break:break-word}
.kp-line:last-child{border-bottom:none}
.dp-summary{background:#fafbff;border:1px solid #e8e8f0;border-radius:8px;padding:14px 18px;font-size:.88rem;line-height:1.6;white-space:pre-wrap}

/* sql diff */
.sql-diff{display:grid;grid-template-columns:1fr 1fr;gap:0}
.sql-pane{padding:14px;overflow-x:auto;font-size:.8rem;font-family:'SF Mono',Menlo,monospace;line-height:1.55;border:1px solid #eee;max-height:420px;overflow-y:auto}
.sql-pane.before{background:#fafafa;border-color:#eee}
.sql-pane.after{background:#fafafa;border-color:#eee}
.sql-label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:6px 14px;background:#f8f8fc;border-bottom:1px solid #eee;color:#666}
.sql-label.before{color:#c0392b}
.sql-label.after{color:#27ae60}
.diff-line{white-space:pre-wrap;min-height:1.55em}
.diff-line.ctx{}
.diff-line.del{background:#ffeef0}
.diff-line.add{background:#e6ffec}
.diff-word-del{background:#fdaeb7;border-radius:2px;padding:0 1px}
.diff-word-add{background:#acf2bd;border-radius:2px;padding:0 1px}
.diff-line-num{display:inline-block;width:28px;text-align:right;margin-right:8px;color:#bbb;font-size:.7rem;user-select:none}

.edit-plan-box{background:#fffbf0;border:1px solid #f0e4c4;border-radius:8px;padding:12px 16px;font-size:.86rem;line-height:1.6;margin-bottom:12px}
.edit-plan-box .ep-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.4px;color:#b8860b;font-weight:700;margin-bottom:4px}
.result-box{background:#f8f8fc;border:1px solid #e8e8f0;border-radius:6px;padding:10px 14px;font-family:'SF Mono',Menlo,monospace;font-size:.78rem;white-space:pre-wrap;margin-top:8px;max-height:180px;overflow-y:auto}
.match-indicator{display:inline-block;background:#6c5ce7;color:#fff;border-radius:4px;padding:2px 8px;font-size:.72rem;font-weight:600;margin-left:8px}
.timeline-step{border-left:3px solid #e0e0e8;padding:0 0 20px 20px;margin-left:10px;position:relative}
.timeline-step:last-child{border-left-color:transparent}
.timeline-step::before{content:'';position:absolute;left:-7px;top:2px;width:11px;height:11px;border-radius:50%;background:#ccc;border:2px solid #fff}
.timeline-step.matched::before{background:#6c5ce7}
.timeline-step.rejected::before{background:#e74c3c}
.timeline-step.executed::before{background:#27ae60}
.step-header{font-size:.82rem;font-weight:600;margin-bottom:6px;color:#333}
.no-trace{color:#999;font-style:italic;font-size:.85rem}
.full-trajectory{margin-top:20px}
</style>
</head>
<body>
<header>TK-Boost Knowledge Dashboard <span class="badge" id="store-badge"></span></header>
<div class="container">
  <div class="intro">
    <strong>Tribal Knowledge Store.</strong>
    Each row is a learned rule extracted from comparing agent SQL drafts against gold-standard queries.
    <strong>Click a row</strong> to inspect the specific edit that produced it.
    Columns marked with &#9998; are editable &mdash; <strong>double-click</strong> to modify. Operations and Data Type use a tag picker; other fields are freeform.
  </div>
  <div class="stats" id="stats-bar"></div>
  <div class="filters">
    <select id="f-db"><option value="">All databases</option></select>
    <select id="f-type" style="display:none"><option value="">All types</option></select>
    <input id="f-search" placeholder="Search knowledge..." style="min-width:220px"/>
  </div>
  <table><thead><tr>
    <th>#</th><th>Example</th><th>Database</th>
    <th class="editable-hdr">Operations</th>
    <th class="editable-hdr">Table</th>
    <th class="editable-hdr">Column</th>
    <th class="editable-hdr">Data Type</th>
    <th class="editable-hdr">Knowledge</th>
  </tr></thead><tbody id="tbody"></tbody></table>
</div>

<div class="save-toast" id="toast">Saved</div>
<div class="tag-picker-overlay" id="tp-overlay"></div>
<div class="tag-picker" id="tp" style="display:none"></div>
<div class="detail-overlay" id="overlay"></div>
<div class="detail-panel" id="panel">
  <div class="dp-header"><h3 id="dp-title">Knowledge Detail</h3><span class="dp-close" id="dp-close">&times;</span></div>
  <div class="dp-body" id="dp-body"></div>
</div>

<script>
let DATA=[], TRACES={};
const $=s=>document.querySelector(s), $$=s=>document.querySelectorAll(s);
let clickTimer=null;

const OP_OPTIONS=['join','filter','aggregation','group_by','order_by','cast','division','round',
  'subquery','window','union','case_when','strftime','limit','having','distinct','insert','update','delete','like'];
const DTYPE_OPTIONS=['numeric','text','date','boolean','json','blob','int','float','str','all'];
const PICKER_FIELDS={sql_operations:{multi:true,options:OP_OPTIONS},data_type:{multi:false,options:DTYPE_OPTIONS}};

async function boot(){
  DATA=await(await fetch('/api/store')).json();
  $('#store-badge').textContent=DATA.length+' rules';
  buildStats(); populateFilters(); render();
}

function buildStats(){
  const bar=$('#stats-bar');
  const dbs=new Set(DATA.map(r=>r.db));
  const examples=new Set(DATA.map(r=>r.instance_id));
  const scopes={db:0,generic:0,question:0};
  DATA.forEach(r=>{const s=(r.scope||'').toLowerCase();if(s in scopes)scopes[s]++});
  bar.innerHTML=[
    {val:DATA.length,lbl:'Total Rules'},{val:examples.size,lbl:'Examples'},
    {val:dbs.size,lbl:'Databases'},
  ].map(c=>`<div class="stat-card"><div class="val">${c.val}</div><div class="lbl">${c.lbl}</div></div>`).join('');
}

function populateFilters(){
  const dbs=[...new Set(DATA.map(r=>r.db))].sort();
  const sel=$('#f-db');
  dbs.forEach(d=>{const o=document.createElement('option');o.value=d;o.textContent=d;sel.appendChild(o)});
  $('#f-db').onchange=$('#f-type').onchange=$('#f-search').oninput=()=>render();
}

function knowledgePreview(rule){
  if(!rule) return '';
  const idx=rule.indexOf('|');
  return idx>0?rule.slice(0,idx).trim():(rule.length>200?rule.slice(0,200)+'\u2026':rule);
}

function render(){
  const fdb=$('#f-db').value, ftype=$('#f-type').value, fsearch=$('#f-search').value.toLowerCase();
  const tbody=$('#tbody');
  const filtered=DATA.filter(r=>{
    const scope=(r.scope||'').toLowerCase();
    if(scope==='question') return false;
    if(fdb && r.db!==fdb) return false;
    if(ftype && scope!==ftype) return false;
    if(fsearch && !(r.rule||'').toLowerCase().includes(fsearch) && !(r.sql_operations||'').toLowerCase().includes(fsearch)) return false;
    return true;
  });
  tbody.innerHTML=filtered.map((r,i)=>{
    const ops=(r.sql_operations||'').split(';').filter(Boolean).map(o=>`<span class="tag">${o.trim()}</span>`).join(' ');
    const di=DATA.indexOf(r);
    return `<tr class="clickable" data-idx="${di}">
      <td>${r.mem_id||i}</td>
      <td>${r.instance_id||''}</td>
      <td>${r.db||''}</td>
      <td class="editable" data-field="sql_operations" data-idx="${di}">${ops||'<span class="tag" style="color:#bbb">none</span>'}</td>
      <td class="editable" data-field="table" data-idx="${di}">${escHtml(r.table||'')}</td>
      <td class="editable" data-field="column" data-idx="${di}">${escHtml(r.column||'')}</td>
      <td class="editable" data-field="data_type" data-idx="${di}"><span class="tag">${escHtml(r.data_type||'')}</span></td>
      <td class="knowledge-cell editable" data-field="rule" data-idx="${di}">${escHtml(knowledgePreview(r.rule))}</td>
    </tr>`;
  }).join('');

  $$('tr.clickable').forEach(tr=>{
    tr.addEventListener('click', ev=>{
      if(clickTimer){clearTimeout(clickTimer);clickTimer=null;return;}
      clickTimer=setTimeout(()=>{clickTimer=null;openDetail(+tr.dataset.idx)},280);
    });
    tr.addEventListener('dblclick', ev=>{
      if(clickTimer){clearTimeout(clickTimer);clickTimer=null;}
      const editCell=ev.target.closest('td.editable');
      if(editCell){ev.stopPropagation();ev.preventDefault();startEdit(editCell)}
    });
  });
}

/* ---- Editing ---- */

function startEdit(td){
  const idx=+td.dataset.idx, field=td.dataset.field;
  const row=DATA[idx]; if(!row) return;
  if(field in PICKER_FIELDS){openTagPicker(td,idx,field);return;}
  const cur=row[field]||'';
  td.textContent=cur;
  td.contentEditable='true';
  td.classList.add('editing');
  td.focus();
  const range=document.createRange();range.selectNodeContents(td);
  const sel=window.getSelection();sel.removeAllRanges();sel.addRange(range);
  const finish=async ()=>{
    td.contentEditable='false';td.classList.remove('editing');
    const newVal=td.textContent.trim();
    if(newVal!==cur){
      row[field]=newVal;
      await persistUpdate(row.mem_id,field,newVal);
    }
    render();
  };
  td.addEventListener('blur',finish,{once:true});
  td.addEventListener('keydown',ev=>{
    if(ev.key==='Enter'){ev.preventDefault();td.blur()}
    if(ev.key==='Escape'){td.textContent=cur;td.contentEditable='false';td.classList.remove('editing');render()}
  });
}

function openTagPicker(td,idx,field){
  const row=DATA[idx];
  const cfg=PICKER_FIELDS[field];
  const current=(row[field]||'').split(';').map(s=>s.trim()).filter(Boolean);
  const selected=new Set(current);
  const rect=td.getBoundingClientRect();
  const tp=$('#tp');
  let html=`<h5>${field.replace(/_/g,' ')} ${cfg.multi?'(multi-select)':'(pick one)'}</h5>`;
  cfg.options.forEach(o=>{
    html+=`<span class="tp-option${selected.has(o)?' selected':''}" data-val="${o}">${o}</span>`;
  });
  html+=`<div class="tp-actions"><button class="tp-btn cancel" id="tp-cancel">Cancel</button><button class="tp-btn save" id="tp-save">Apply</button></div>`;
  tp.innerHTML=html;
  tp.style.display='block';
  tp.style.top=Math.min(rect.bottom+4, window.innerHeight-360)+'px';
  tp.style.left=Math.min(rect.left, window.innerWidth-340)+'px';
  $('#tp-overlay').style.display='block';

  tp.querySelectorAll('.tp-option').forEach(opt=>{
    opt.addEventListener('click',()=>{
      const val=opt.dataset.val;
      if(cfg.multi){opt.classList.toggle('selected')}
      else{tp.querySelectorAll('.tp-option').forEach(o=>o.classList.remove('selected'));opt.classList.add('selected')}
    });
  });

  const close=()=>{tp.style.display='none';$('#tp-overlay').style.display='none'};
  $('#tp-overlay').onclick=close;
  $('#tp-cancel').onclick=close;
  $('#tp-save').onclick=async ()=>{
    const picked=[...tp.querySelectorAll('.tp-option.selected')].map(o=>o.dataset.val);
    const newVal=picked.join(';');
    if(newVal!==(row[field]||'')){
      row[field]=newVal;
      await persistUpdate(row.mem_id,field,newVal);
    }
    close();render();
  };
}

async function persistUpdate(memId,field,value){
  await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mem_id:memId,field,value})});
  showToast('Saved');
}

function showToast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),1800)}

/* ---- Word-level SQL diff ---- */

function diffLines(a,b){
  const al=a.split('\n'), bl=b.split('\n');
  const m=al.length, n=bl.length;
  const dp=Array.from({length:m+1},()=>new Uint16Array(n+1));
  for(let i=1;i<=m;i++) for(let j=1;j<=n;j++)
    dp[i][j]=al[i-1]===bl[j-1]?dp[i-1][j-1]+1:Math.max(dp[i-1][j],dp[i][j-1]);
  const ops=[];
  let i=m,j=n;
  while(i>0||j>0){
    if(i>0&&j>0&&al[i-1]===bl[j-1]){ops.push({type:'eq',a:al[--i],b:bl[--j]});continue}
    if(j>0&&(i===0||dp[i][j-1]>=dp[i-1][j])){ops.push({type:'add',b:bl[--j]});continue}
    ops.push({type:'del',a:al[--i]});
  }
  ops.reverse();
  const merged=[];
  let di=0;
  while(di<ops.length){
    if(ops[di].type==='del'&&di+1<ops.length&&ops[di+1].type==='add'){
      merged.push({type:'chg',a:ops[di].a,b:ops[di+1].b});di+=2;
    } else {merged.push(ops[di]);di++}
  }
  return merged;
}

function diffWords(aLine,bLine){
  const aw=aLine.split(/(\s+)/), bw=bLine.split(/(\s+)/);
  const m=aw.length, n=bw.length;
  const dp=Array.from({length:m+1},()=>new Uint16Array(n+1));
  for(let i=1;i<=m;i++) for(let j=1;j<=n;j++)
    dp[i][j]=aw[i-1]===bw[j-1]?dp[i-1][j-1]+1:Math.max(dp[i-1][j],dp[i][j-1]);
  const ops=[];
  let i=m,j=n;
  while(i>0||j>0){
    if(i>0&&j>0&&aw[i-1]===bw[j-1]){ops.push({type:'eq',w:aw[--i]});j--;continue}
    if(j>0&&(i===0||dp[i][j-1]>=dp[i-1][j])){ops.push({type:'add',w:bw[--j]});continue}
    ops.push({type:'del',w:aw[--i]});
  }
  ops.reverse();
  return ops;
}

function renderWordDiff(aLine,bLine){
  const ops=diffWords(aLine,bLine);
  let beforeHtml='', afterHtml='';
  for(const op of ops){
    const w=escHtml(op.w||'');
    if(op.type==='eq'){beforeHtml+=w;afterHtml+=w}
    else if(op.type==='del'){beforeHtml+=`<span class="diff-word-del">${w}</span>`}
    else if(op.type==='add'){afterHtml+=`<span class="diff-word-add">${w}</span>`}
  }
  return {beforeHtml,afterHtml};
}

function renderSqlDiff(sqlBefore,sqlAfter){
  const ops=diffLines(sqlBefore||'',sqlAfter||'');
  let beforeLines='', afterLines='';
  let ln_a=0, ln_b=0;
  for(const op of ops){
    if(op.type==='eq'){
      ln_a++;ln_b++;
      const t=escHtml(op.a);
      beforeLines+=`<div class="diff-line ctx"><span class="diff-line-num">${ln_a}</span>${t}</div>`;
      afterLines+=`<div class="diff-line ctx"><span class="diff-line-num">${ln_b}</span>${t}</div>`;
    } else if(op.type==='del'){
      ln_a++;
      beforeLines+=`<div class="diff-line del"><span class="diff-line-num">${ln_a}</span>${escHtml(op.a)}</div>`;
      afterLines+=`<div class="diff-line" style="opacity:.3"><span class="diff-line-num">&nbsp;</span></div>`;
    } else if(op.type==='add'){
      ln_b++;
      beforeLines+=`<div class="diff-line" style="opacity:.3"><span class="diff-line-num">&nbsp;</span></div>`;
      afterLines+=`<div class="diff-line add"><span class="diff-line-num">${ln_b}</span>${escHtml(op.b)}</div>`;
    } else if(op.type==='chg'){
      ln_a++;ln_b++;
      const wd=renderWordDiff(op.a,op.b);
      beforeLines+=`<div class="diff-line del"><span class="diff-line-num">${ln_a}</span>${wd.beforeHtml}</div>`;
      afterLines+=`<div class="diff-line add"><span class="diff-line-num">${ln_b}</span>${wd.afterHtml}</div>`;
    }
  }
  return `<div class="sql-diff"><div><div class="sql-label before">Before</div><div class="sql-pane before">${beforeLines}</div></div><div><div class="sql-label after">After</div><div class="sql-pane after">${afterLines}</div></div></div>`;
}

/* ---- Rule-to-edit matching ---- */

function tokenize(text){
  return (text||'').toLowerCase().replace(/[^a-z0-9_]+/g,' ').split(/\s+/).filter(t=>t.length>2);
}

function matchScore(ruleTokens,text){
  const toks=new Set(tokenize(text));
  let hits=0;
  for(const t of ruleTokens) if(toks.has(t)) hits++;
  return ruleTokens.length?hits/ruleTokens.length:0;
}

function buildTimeline(trace){
  const events=(trace.llm_interactions||{}).events||[];
  const initialSQL=trace.agent_sql||'';
  const steps=[];
  let prevSQL=initialSQL;
  for(let i=0;i<events.length;i++){
    const ev=events[i];
    if(ev.event==='sql_attempt_executed'||ev.event==='sql_attempt_rejected'){
      const preceding=events.slice(0,i).reverse().find(e=>e.event==='assistant_response');
      const plans=(preceding||{}).edit_plans||[];
      const planContent=(preceding||{}).content||'';
      steps.push({
        type:ev.event==='sql_attempt_rejected'?'rejected':'executed',
        attempt:ev.attempt,
        editPlans:plans,
        planContent:planContent,
        sqlBefore:prevSQL,
        sqlAfter:ev.sql||'',
        result:ev.sql_result||'',
        agentSim:ev.agent_similarity,
        goldSim:ev.gold_similarity,
      });
      if(ev.event==='sql_attempt_executed') prevSQL=ev.sql||prevSQL;
    }
  }
  return steps;
}

function findBestMatch(ruleText,steps){
  if(!steps.length) return -1;
  const rTokens=tokenize(ruleText);
  let bestIdx=-1, bestScore=0;
  steps.forEach((s,i)=>{
    const planText=s.editPlans.join(' ')+' '+s.planContent;
    const sc=matchScore(rTokens,planText+' '+s.sqlAfter);
    if(sc>bestScore){bestScore=sc;bestIdx=i}
  });
  return bestIdx;
}

function extractMinimalFix(trace){
  const events=(trace.llm_interactions||{}).events||[];
  for(let i=events.length-1;i>=0;i--){
    const ev=events[i];
    if(ev.event==='assistant_response'&&ev.proposed_sql===null){
      const c=ev.content||'';
      const m=c.match(/MINIMAL_FIX:\s*(.+)/i);
      if(m) return m[1].trim();
      const lines=c.split('\n').filter(l=>l.trim()&&!l.startsWith('MATCH_OK'));
      if(lines.length) return lines.join(' ').trim();
    }
  }
  const fo=(trace.llm_interactions||{}).final_output||'';
  const csm=fo.match(/CLEAN_SUMMARY:\s*(.+)/i);
  if(csm) return csm[1].trim();
  const mre=fo.match(/MINIMAL_REQUIRED_EDITS:\s*([\s\S]*?)(?=\n(?:EVIDENCE|CLEAN_SUMMARY)|$)/i);
  if(mre&&mre[1].trim()) return mre[1].trim().split('\n').map(l=>l.replace(/^[-*]\s*/,'')).join('; ');
  return null;
}

/* ---- Detail panel ---- */

async function openDetail(idx){
  const row=DATA[idx]; if(!row) return;
  $$('tr.active-row').forEach(t=>t.classList.remove('active-row'));
  const tr=document.querySelector(`tr[data-idx="${idx}"]`);
  if(tr) tr.classList.add('active-row');

  const eid=row.instance_id||'';
  $('#dp-title').textContent=`${eid} \u2014 ${row.db||''}`;

  let trace=TRACES[eid];
  if(!trace){try{trace=await(await fetch('/api/trace/'+encodeURIComponent(eid))).json();TRACES[eid]=trace}catch(e){trace=null}}

  let html=`<div class="dp-section"><h4>Full Knowledge</h4><div class="dp-knowledge">${formatKnowledge(row.rule||'')}</div></div>`;

  if(!trace||trace.error){
    html+=`<div class="dp-section"><p class="no-trace">No debug trace found for this example. Run generation with <code>debug=True</code> to capture traces.</p></div>`;
    $('#dp-body').innerHTML=html;$('#panel').style.display='block';$('#overlay').style.display='block';return;
  }

  const steps=buildTimeline(trace);
  const matchIdx=findBestMatch(row.rule||'',steps);
  const matched=matchIdx>=0?steps[matchIdx]:null;

  if(matched){
    const plans=matched.editPlans.filter(Boolean);
    if(plans.length){
      html+=`<div class="dp-section"><h4>Source Edit</h4><div class="edit-plan-box"><div class="ep-label">Edit Plan (attempt ${matched.attempt})</div>`;
      plans.forEach(p=>{html+=escHtml(p.replace(/<\/think>/g,'').trim())+'<br/>'});
      html+=`</div></div>`;
    } else {
      let fallbackDesc=extractMinimalFix(trace);
      if(fallbackDesc){
        html+=`<div class="dp-section"><h4>Source Edit</h4><div class="edit-plan-box"><div class="ep-label">Repair Summary (attempt ${matched.attempt})</div>${escHtml(fallbackDesc)}</div></div>`;
      }
    }
    html+=`<div class="dp-section"><h4>SQL Before &rarr; After This Edit (attempt ${matched.attempt})</h4>`;
    html+=renderSqlDiff(matched.sqlBefore,matched.sqlAfter);
    html+=`</div>`;
    if(matched.result){
      html+=`<div class="dp-section"><h4>Execution Result</h4><div class="result-box">${escHtml(matched.result.slice(0,2000))}</div></div>`;
    }
  } else if(steps.length===0){
    const status=(trace.llm_interactions||{}).status||'';
    const summary=(trace.llm_interactions||{}).final_output||'';
    html+=`<div class="dp-section"><h4>Source Edit</h4><div style="background:#f8f8fc;border:1px solid #e8e8f0;border-radius:8px;padding:14px 18px;font-size:.86rem;color:#666;line-height:1.6">
      <strong>No SQL edits were made for this example.</strong> The agent SQL already produced the correct result, so the LLM declared a match without proposing changes.
      This rule is a meta-observation extracted from the comparison, not tied to a specific code fix.
      ${summary?'<br/><br/><em>'+escHtml(summary)+'</em>':''}
    </div></div>`;
  }

  if(trace.gold_sql){
    html+=`<div class="dp-section"><h4>Gold SQL (ground truth)</h4><div class="sql-pane" style="border-radius:8px;background:#f0fff4;border-color:#c3e6cb;padding:14px;font-family:\'SF Mono\',Menlo,monospace;font-size:.8rem;white-space:pre-wrap;line-height:1.55">${escHtml(trace.gold_sql)}</div></div>`;
  }

  if(steps.length>1){
    html+=`<div class="dp-section full-trajectory"><h4>Full Edit Trajectory (${steps.length} steps)</h4>`;
    steps.forEach((s,si)=>{
      const cls=s.type==='rejected'?'rejected':(si===matchIdx?'matched':'executed');
      const label=s.type==='rejected'?`Attempt ${s.attempt} (rejected \u2014 ${s.result&&s.result.includes('minimality')?'minimality violation':'error'})`:`Attempt ${s.attempt}`;
      const matchBadge=si===matchIdx?'<span class="match-indicator">source of this rule</span>':'';
      const plans=s.editPlans.filter(Boolean);
      html+=`<div class="timeline-step ${cls}"><div class="step-header">${label}${matchBadge}</div>`;
      if(plans.length) html+=`<div style="font-size:.82rem;color:#555;margin-bottom:4px">${plans.map(p=>escHtml(p.replace(/<\/think>/g,'').trim())).join('<br/>')}</div>`;
      html+=`<details><summary style="font-size:.76rem;cursor:pointer;color:#6c5ce7">Show SQL diff &amp; result</summary>`;
      html+=renderSqlDiff(s.sqlBefore,s.sqlAfter);
      if(s.result) html+=`<div class="result-box" style="margin-top:6px">${escHtml(s.result.slice(0,1500))}</div>`;
      html+=`</details></div>`;
    });
    html+=`</div>`;
  }

  $('#dp-body').innerHTML=html;$('#panel').style.display='block';$('#overlay').style.display='block';
}

function escHtml(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

function formatKnowledge(rule){
  const text=(rule||'').trim();
  if(!text) return '';
  const parts=text.split('|').map(p=>p.trim()).filter(Boolean);
  if(!parts.length) return escHtml(text);
  return parts.map(p=>`<div class="kp-line">${escHtml(p)}</div>`).join('');
}

$('#dp-close').onclick=$('#overlay').onclick=()=>{
  $('#panel').style.display='none';$('#overlay').style.display='none';
  $$('tr.active-row').forEach(t=>t.classList.remove('active-row'));
};

boot();
</script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    store_path: str = ""
    debug_dir: Optional[Path] = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._respond(200, "text/html", DASHBOARD_HTML.encode())
        elif path == "/api/store":
            rows = _load_store_rows(self.store_path)
            self._respond(200, "application/json", json.dumps(rows, ensure_ascii=False).encode())
        elif path.startswith("/api/trace/"):
            example_id = path.split("/api/trace/", 1)[1]
            if self.debug_dir:
                trace = _load_trace(self.debug_dir, example_id)
            else:
                trace = None
            if trace:
                self._respond(200, "application/json", json.dumps(trace, ensure_ascii=False).encode())
            else:
                self._respond(200, "application/json", json.dumps({"error": "no trace"}).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/update":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                mem_id = str(payload.get("mem_id", ""))
                field = str(payload.get("field", ""))
                value = str(payload.get("value", ""))
                rows = _load_store_rows(self.store_path)
                updated = False
                for row in rows:
                    if str(row.get("mem_id", "")) == mem_id and field in row:
                        row[field] = value
                        updated = True
                        break
                if updated:
                    _write_store_rows(self.store_path, rows)
                self._respond(200, "application/json", json.dumps({"ok": updated}).encode())
            except Exception as e:
                self._respond(400, "application/json", json.dumps({"error": str(e)}).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def serve(store_path: str, port: int = 8501, open_browser: bool = True) -> None:
    _Handler.store_path = store_path
    _Handler.debug_dir = _find_debug_dir(store_path)

    server = HTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"TK-Boost dashboard running at {url}")
    print(f"  store: {store_path}")
    if _Handler.debug_dir:
        print(f"  debug traces: {_Handler.debug_dir}")
    print("Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()
