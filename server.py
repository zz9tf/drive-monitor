#!/usr/bin/env python3
"""Dashboard server - runs on mini2."""

import asyncio
import json
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI()

MAX_HISTORY = 17280          # 24h at 5s intervals
state: dict[str, Any] = {}
history: dict[str, deque] = {}   # hostname -> deque of slim datapoints
state_lock = threading.Lock()
subscribers: list[asyncio.Queue] = []
subs_lock = threading.Lock()


def broadcast(msg: str):
    with subs_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            subscribers.remove(q)


@app.post("/update")
async def update(request: Request):
    data = await request.json()
    hostname = data.get("hostname", "unknown")
    data["_received"] = datetime.now().isoformat()
    with state_lock:
        state[hostname] = data
        if hostname not in history:
            history[hostname] = deque(maxlen=MAX_HISTORY)
        history[hostname].append({
            "t": data["time"],
            "cpu": data.get("cpu_pct", 0),
            "mem": data.get("mem_pct", 0),
            "gpus": [
                {"id": g["id"], "util": g.get("util", 0),
                 "mem_pct": round(g["mem_used_gb"] / g["mem_total_gb"] * 100)
                           if g.get("mem_total_gb") else 0,
                 "temp": g.get("temp", 0)}
                for g in data.get("gpus", []) if g.get("status") == "ok"
            ],
        })
    broadcast(json.dumps({"host": hostname}))
    return {"ok": True}


@app.get("/api/state")
async def api_state():
    with state_lock:
        return dict(state)


@app.get("/api/history/{hostname}")
async def api_history(hostname: str, points: int = 500):
    with state_lock:
        buf = list(history.get(hostname, []))
    # downsample
    n = len(buf)
    if n > points:
        step = n / points
        buf = [buf[int(i * step)] for i in range(points)]
    return buf


@app.get("/events")
async def sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=20)
    with subs_lock:
        subscribers.append(q)

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            with subs_lock:
                if q in subscribers:
                    subscribers.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GPU Monitor</title>
<script>
// Load Chart.js async — page works even if CDN is unreachable
(function(){
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js';
  s.onload = () => { window._chartReady = true; if(window._pendingCharts) window._pendingCharts(); };
  s.onerror = () => { console.warn('Chart.js failed to load, charts disabled'); };
  document.head.appendChild(s);
})();
</script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:ui-sans-serif,system-ui,sans-serif;background:#f0f2f5;color:#1f2937;display:flex;height:100vh;overflow:hidden}

/* Sidebar */
#sidebar{width:220px;background:#fff;border-right:1px solid #e5e7eb;display:flex;flex-direction:column;flex-shrink:0}
#sidebar h1{font-size:.95em;font-weight:700;color:#111;padding:16px;border-bottom:1px solid #e5e7eb;letter-spacing:.02em}
#host-list{flex:1;overflow-y:auto;padding:8px}
.host-item{padding:9px 12px;border-radius:8px;cursor:pointer;margin-bottom:4px;transition:background .15s}
.host-item:hover{background:#f3f4f6}
.host-item.active{background:#eff6ff;border:1px solid #bfdbfe}
.host-name{font-weight:600;font-size:.88em;color:#111}
.host-sub{font-size:.75em;color:#6b7280;margin-top:2px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}
.dot-green{background:#22c55e}.dot-red{background:#ef4444}.dot-gray{background:#9ca3af}

/* Main */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#topbar{background:#fff;border-bottom:1px solid #e5e7eb;padding:10px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0}
#topbar .title{font-weight:700;font-size:1em;color:#111}
#conn-status{font-size:.75em;color:#6b7280;margin-left:auto}

#content{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:16px}

/* Cards row */
.cards-row{display:flex;gap:12px;flex-wrap:wrap}
.stat-card{background:#fff;border-radius:12px;padding:16px;flex:1;min-width:160px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.stat-card .label{font-size:.72em;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.stat-card .value{font-size:1.6em;font-weight:700;color:#111;line-height:1}
.stat-card .sub{font-size:.75em;color:#9ca3af;margin-top:4px}

/* nvitop-style GPU panel — white theme */
.nv-panel{background:transparent;font-family:ui-monospace,'Cascadia Code','JetBrains Mono',monospace}
.nv-grid{display:flex;flex-direction:column;gap:8px}
.nv-card{border:1px solid #e2e8f0;border-radius:10px;padding:11px 16px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.nv-card.nv-dead{border-color:#fca5a5;background:#fff5f5}
.nv-card-inner{display:grid;grid-template-columns:90px 1fr 320px;gap:16px;align-items:center}
.nv-card.nv-dead .nv-card-inner{grid-template-columns:90px 1fr}
.nv-id{color:#1d4ed8;font-weight:700;font-size:.95em;letter-spacing:.02em}
.nv-name{color:#94a3b8;font-size:.72em;margin-top:2px}
.nv-tag{font-size:.68em;padding:1px 6px;border-radius:4px;font-weight:700;letter-spacing:.04em;margin-top:4px;display:inline-block}
.nv-tag-ok{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}
.nv-tag-dead{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
/* bars column */
.nv-bars{display:flex;flex-direction:column;gap:5px}
.nv-row{display:flex;align-items:center;gap:8px}
.nv-rowlabel{width:32px;color:#94a3b8;flex-shrink:0;font-size:.72em;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
.nv-bar-bg{flex:1;height:8px;background:#f1f5f9;border-radius:99px;overflow:hidden}
.nv-bar{height:100%;border-radius:99px;transition:width .5s ease}
.nv-val{width:100px;text-align:right;color:#374151;flex-shrink:0;font-size:.78em}
/* meta */
.nv-meta{display:flex;gap:14px;margin-top:5px;font-size:.72em;color:#94a3b8}
.nv-meta b{color:#374151}
/* process table */
.nv-procs{font-size:.75em}
.nv-proc-hdr{display:grid;grid-template-columns:54px 72px 60px 1fr;gap:6px;color:#cbd5e1;font-weight:600;text-transform:uppercase;letter-spacing:.05em;font-size:.85em;margin-bottom:3px}
.nv-proc-row{display:grid;grid-template-columns:54px 72px 60px 1fr;gap:6px;padding:1px 0;color:#64748b}
.nv-proc-row:hover{background:#f8fafc;border-radius:3px}
.nv-proc-pid{color:#94a3b8}
.nv-proc-user{color:#7c3aed;font-weight:600}
.nv-proc-mem{color:#059669}
.nv-proc-cmd{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#2563eb}
.nv-no-proc{color:#d1d5db;font-size:.85em}
.nv-dead-msg{color:#ef4444;font-size:.84em}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.7em;font-weight:600}
.badge-ok{background:#dcfce7;color:#15803d}
.badge-dead{background:#fee2e2;color:#dc2626}

/* Charts panel */
.chart-panel{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.chart-panel h3{font-size:.8em;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}
.chart-wrap{position:relative;height:180px}

/* Event log */
.event-panel{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.event-panel h3{font-size:.8em;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:.78em}
th{color:#9ca3af;font-weight:600;padding:5px 8px;text-align:left;border-bottom:1px solid #f3f4f6}
td{padding:5px 8px;border-bottom:1px solid #f9fafb;color:#374151}
tr:last-child td{border:none}
.lv-info{color:#2563eb}.lv-warn{color:#d97706}.lv-error{color:#dc2626}

#no-host{display:flex;align-items:center;justify-content:center;flex:1;color:#9ca3af;font-size:.9em}
</style>
</head>
<body>

<div id="sidebar">
  <h1>GPU Monitor</h1>
  <div id="host-list"><p style="font-size:.8em;color:#9ca3af;padding:12px">Waiting for agents...</p></div>
</div>

<div id="main">
  <div id="topbar">
    <span class="title" id="active-title">—</span>
    <span id="conn-status">Connecting...</span>
    <span id="update-rate" style="font-size:.72em;color:#d1d5db;margin-left:8px"></span>
  </div>
  <div id="content">
    <div id="no-host">Select a server from the sidebar</div>
  </div>
</div>

<script>
let activeHost = null;
let allState = {};
let gpuDonuts = {};
let cpuMemChart = null;
let gpuLineChart = null;
let _lastUpdateTime = null;

// ── SSE ──────────────────────────────────────────────
const es = new EventSource('/events');
es.onmessage = async e => {
  const msg = JSON.parse(e.data);
  const now = Date.now();
  if (_lastUpdateTime) {
    const secs = ((now - _lastUpdateTime) / 1000).toFixed(1);
    document.getElementById('update-rate').textContent = `· updated every ~${secs}s`;
  }
  _lastUpdateTime = now;
  await loadState();
  if (activeHost === msg.host) refreshContent();
};
es.onopen = () => setStatus('Live ●');
es.onerror = () => { setStatus('Reconnecting...'); setTimeout(loadState, 3000); };

function setStatus(s) { document.getElementById('conn-status').textContent = s; }

// ── Data ─────────────────────────────────────────────
async function loadState() {
  const r = await fetch('/api/state');
  allState = await r.json();
  renderSidebar();
}

async function loadHistory(host) {
  const r = await fetch(`/api/history/${encodeURIComponent(host)}?points=500`);
  return r.json();
}

// ── Sidebar ───────────────────────────────────────────
function renderSidebar() {
  const hosts = Object.keys(allState);
  const el = document.getElementById('host-list');
  if (!hosts.length) { el.innerHTML = '<p style="font-size:.8em;color:#9ca3af;padding:12px">Waiting for agents...</p>'; return; }

  el.innerHTML = hosts.map(h => {
    const d = allState[h];
    const stale = d._received && (Date.now() - new Date(d._received)) > 30000;
    const gpuCount = (d.gpus||[]).length;
    const deadCount = (d.gpus||[]).filter(g => g.status !== 'ok').length;
    const dotCls = stale ? 'dot-red' : deadCount ? 'dot-red' : 'dot-green';
    const sub = stale ? 'offline' : `${gpuCount} GPU${gpuCount!==1?'s':''}`+(deadCount?` · ${deadCount} dead`:'');
    return `<div class="host-item${activeHost===h?' active':''}" onclick="selectHost('${h}')">
      <div class="host-name"><span class="dot ${dotCls}"></span>${h}</div>
      <div class="host-sub">${sub}</div>
    </div>`;
  }).join('');
}

async function selectHost(host) {
  activeHost = host;
  document.getElementById('active-title').textContent = host;
  renderSidebar();
  gpuDonuts = {};
  cpuMemChart = null;
  gpuLineChart = null;
  document.getElementById('no-host') && (document.getElementById('no-host').style.display = 'none');
  refreshContent();
  const hist = await loadHistory(host);
  renderCharts(hist);
}

// ── Main content ──────────────────────────────────────
function refreshContent() {
  if (!activeHost) return;
  const d = allState[activeHost];
  if (!d) return;
  const stale = d._received && (Date.now() - new Date(d._received)) > 30000;

  let html = '';

  // stat cards
  html += `<div class="cards-row">
    <div class="stat-card">
      <div class="label">CPU</div>
      <div class="value">${d.cpu_pct ?? '—'}<span style="font-size:.5em;color:#9ca3af">%</span></div>
      <div class="sub">utilization</div>
    </div>
    <div class="stat-card">
      <div class="label">Memory</div>
      <div class="value">${d.mem_pct ?? '—'}<span style="font-size:.5em;color:#9ca3af">%</span></div>
      <div class="sub">${d.mem_used_gb ?? '?'} / ${d.mem_total_gb ?? '?'} GB</div>
    </div>
    <div class="stat-card">
      <div class="label">GPUs</div>
      <div class="value">${(d.gpus||[]).filter(g=>g.status==='ok').length}<span style="font-size:.4em;color:#9ca3af"> / ${(d.gpus||[]).length}</span></div>
      <div class="sub">healthy</div>
    </div>
    <div class="stat-card">
      <div class="label">Last Update</div>
      <div class="value" style="font-size:.95em">${d.time?.slice(11) ?? '—'}</div>
      <div class="sub ${stale?'lv-error':''}">${stale ? 'stale >30s' : 'live'}</div>
    </div>
  </div>`;

  // nvitop-style GPU panel
  html += `<div class="nv-panel"><div class="nv-grid">`;
  for (const g of (d.gpus || [])) {
    const dead = g.status !== 'ok';
    const memPct = g.mem_total_gb ? Math.round(g.mem_used_gb / g.mem_total_gb * 100) : 0;
    const utilPct = g.util ?? 0;
    const barColor = pct => pct >= 90 ? '#ef4444' : pct >= 70 ? '#f59e0b' : '#3b82f6';
    const temp = g.temp != null ? `${g.temp}°C` : 'N/A';
    const ecc = g.ecc_errors || 0;

    html += `<div class="nv-card${dead?' nv-dead':''}"><div class="nv-card-inner">`;

    // col 1: ID + tag
    html += `<div>
      <div class="nv-id">GPU ${g.id}</div>
      <div class="nv-name">H100 PCIe</div>
      <span class="nv-tag ${dead?'nv-tag-dead':'nv-tag-ok'}">${dead?'DEAD':'OK'}</span>
    </div>`;

    if (dead) {
      html += `<div class="nv-dead-msg">${g.error || 'Device error'}</div>`;
    } else {
      // col 2: bars + meta
      html += `<div class="nv-bars">
        <div class="nv-row">
          <span class="nv-rowlabel">GPU</span>
          <div class="nv-bar-bg"><div class="nv-bar" style="width:${utilPct}%;background:${barColor(utilPct)}"></div></div>
          <span class="nv-val">${g.util != null ? g.util+'%' : 'N/A'}</span>
        </div>
        <div class="nv-row">
          <span class="nv-rowlabel">MEM</span>
          <div class="nv-bar-bg"><div class="nv-bar" style="width:${memPct}%;background:${barColor(memPct)}"></div></div>
          <span class="nv-val">${g.mem_used_gb} / ${g.mem_total_gb} GB</span>
        </div>
        <div class="nv-meta">
          <span>Temp <b>${temp}</b></span>
          <span>ECC <b ${ecc?'style="color:#ef4444"':''}>${ecc}</b></span>
        </div>
      </div>`;

      // col 3: process table
      if (g.processes?.length) {
        html += `<div class="nv-procs">
          <div class="nv-proc-hdr"><span>PID</span><span>USER</span><span>VRAM</span><span>COMMAND</span></div>`;
        for (const p of g.processes)
          html += `<div class="nv-proc-row">
            <span class="nv-proc-pid">${p.pid}</span>
            <span class="nv-proc-user">${p.user}</span>
            <span class="nv-proc-mem">${p.mem_mb}MB</span>
            <span class="nv-proc-cmd">${p.cmd}</span>
          </div>`;
        html += `</div>`;
      } else {
        html += `<div class="nv-procs"><div class="nv-no-proc">─ no processes ─</div></div>`;
      }
    }

    html += `</div></div>`;
  }
  html += `</div></div>`;

  // chart placeholders
  html += `<div class="chart-panel">
    <h3>CPU & Memory — 24h</h3>
    <div class="chart-wrap"><canvas id="chart-cpumem"></canvas></div>
  </div>
  <div class="chart-panel">
    <h3>GPU Utilization — 24h</h3>
    <div class="chart-wrap"><canvas id="chart-gpu"></canvas></div>
  </div>`;

  // events
  const events = d.events || [];
  html += `<div class="event-panel"><h3>Event Log</h3>`;
  if (events.length) {
    html += `<table><tr><th>Time</th><th>Level</th><th>GPU</th><th>Message</th></tr>`;
    for (const e of events)
      html += `<tr><td>${e.time}</td><td class="lv-${e.level}">${e.level.toUpperCase()}</td><td>${e.gpu_id??'-'}</td><td>${e.msg}</td></tr>`;
    html += `</table>`;
  } else {
    html += `<p style="font-size:.82em;color:#9ca3af;padding:8px 0">No events yet.</p>`;
  }
  html += `</div>`;

  document.getElementById('content').innerHTML = html;

  // restore line charts if history was already loaded
  if (_lastHist) renderCharts(_lastHist);
}

let _lastHist = null;

function renderCharts(hist) {
  _lastHist = hist;
  if (!hist.length) return;
  if (!window._chartReady) { window._pendingCharts = () => renderCharts(hist); return; }

  const labels = hist.map(p => p.t.slice(11, 16));  // HH:MM
  const cpuData = hist.map(p => p.cpu);
  const memData = hist.map(p => p.mem);

  // CPU + Memory line chart
  const cpuCtx = document.getElementById('chart-cpumem');
  if (cpuCtx) {
    if (cpuMemChart) cpuMemChart.destroy();
    cpuMemChart = new Chart(cpuCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'CPU %', data: cpuData, borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,.08)', fill: true, tension: .3, pointRadius: 0, borderWidth: 1.5 },
          { label: 'Memory %', data: memData, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.08)', fill: true, tension: .3, pointRadius: 0, borderWidth: 1.5 },
        ]
      },
      options: lineOpts('Utilization (%)')
    });
  }

  // GPU utilization line chart
  const gpuCtx = document.getElementById('chart-gpu');
  if (!gpuCtx || !hist[0]?.gpus?.length) return;
  const colors = ['#6366f1','#22c55e','#f59e0b','#ef4444','#06b6d4','#ec4899'];
  const gpuIds = [...new Set(hist.flatMap(p => p.gpus.map(g => g.id)))].sort();
  if (gpuLineChart) gpuLineChart.destroy();
  gpuLineChart = new Chart(gpuCtx, {
    type: 'line',
    data: {
      labels,
      datasets: gpuIds.map((id, i) => ({
        label: `GPU ${id}`,
        data: hist.map(p => p.gpus.find(g => g.id === id)?.util ?? null),
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        tension: .3, pointRadius: 0, borderWidth: 1.5,
        spanGaps: false,
      }))
    },
    options: lineOpts('Utilization (%)')
  });
}

function lineOpts(yLabel) {
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: { ticks: { maxTicksLimit: 8, color: '#9ca3af', font: { size: 10 } }, grid: { color: '#f3f4f6' } },
      y: { min: 0, max: 100, ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { color: '#f3f4f6' }, title: { display: false } }
    },
    plugins: { legend: { labels: { font: { size: 11 }, color: '#374151', boxWidth: 12 } } }
  };
}

// ── Init ──────────────────────────────────────────────
loadState();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
