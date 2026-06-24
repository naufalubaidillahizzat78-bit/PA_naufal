"""
STEP 7: DASHBOARD VISUALISASI
- Bubble chart interaktif
- Heatmap cluster
- Klik mahasiswa → tampilkan profil detail
- Rekomendasi dosen berbasis cluster
Output: output/dashboard.html
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')


def build_dashboard(df_labeled_path='output/df_labeled.pkl',
                    princals_path='output/X_princals.pkl',
                    feature_cols_path='output/feature_cols.pkl'):

    df = pd.read_pickle(df_labeled_path)
    X_pc = pd.read_pickle(princals_path)
    feature_cols = pd.read_pickle(feature_cols_path).tolist()

    # Align indices
    min_len = min(len(df), len(X_pc))
    df = df.iloc[:min_len].reset_index(drop=True)
    X_pc = X_pc.iloc[:min_len].reset_index(drop=True)

    # IPS column
    ips_col = 'Rata-Rata IPS' if 'Rata-Rata IPS' in df.columns else feature_cols[0]
    absen_col = 'Rata-Rata Absen Mahasiswa' if 'Rata-Rata Absen Mahasiswa' in df.columns else None

    # Gather course cols (grade columns, no .1 duplicates)
    skip_meta = {'NRP','Nama Mahasiswa','Angkatan Tahun','Prodi','JK','Asal Kab/Kota',
                 'cluster','cluster_label','Prodi_enc','Asal Kab/Kota_enc'}
    skip_ips_absen = {c for c in df.columns if 'IPS' in c.upper() or 'ABSEN' in c.upper()}
    course_cols = [c for c in df.columns
                   if c not in skip_meta and c not in skip_ips_absen
                   and not c.endswith('_enc') and not c.endswith('.1')]

    # Custom light theme palette
    theme_palette = {
        'Sangat Tinggi': '#28C76F',
        'Tinggi': '#00CFE8',
        'Sedang': '#7367F0',
        'Cukup': '#FF9F43',
        'Rendah': '#EA5455',
    }
    labels_unique = sorted(df['cluster_label'].unique(), key=lambda x: (
        0 if 'Sangat Tinggi' in x else 1 if 'Tinggi' in x else
        2 if 'Sedang' in x else 3 if 'Cukup' in x else 4))
    color_map = {lbl: theme_palette.get(lbl, '#7367F0') for lbl in labels_unique}

    # Build student records for JS
    students = []
    for i, row in df.iterrows():
        pc1 = float(X_pc.iloc[i, 0]) if X_pc.shape[1] > 0 else 0
        pc2 = float(X_pc.iloc[i, 1]) if X_pc.shape[1] > 1 else 0
        ips_val = float(row[ips_col]) if ips_col in row.index else 0
        absen_val = float(row[absen_col]) if absen_col and absen_col in row.index else None
        courses = {c: round(float(row[c]), 2) for c in course_cols if c in row.index and not np.isnan(row[c])}
        students.append({
            'id': int(row['NRP']),
            'name': str(row['Nama Mahasiswa']),
            'angkatan': int(row['Angkatan Tahun']) if 'Angkatan Tahun' in row.index else 0,
            'jk': 'Laki-laki' if row.get('JK', 1) == 1 else 'Perempuan',
            'asal': str(row.get('Asal Kab/Kota', '-')),
            'cluster': str(row['cluster_label']),
            'ips': ips_val,
            'absensi': absen_val,
            'pc1': pc1,
            'pc2': pc2,
            'bubble_size': max(8, ips_val * 12),
            'courses': courses
        })

    # Cluster summary for heatmap
    cluster_summary = {}
    for lbl in labels_unique:
        sub = df[df['cluster_label'] == lbl]
        cluster_summary[lbl] = {
            'count': int(len(sub)),
            'avg_ips': round(float(sub[ips_col].mean()), 3) if ips_col in sub.columns else 0,
            'avg_absensi': round(float(sub[absen_col].mean()), 2) if absen_col and absen_col in sub.columns else None,
        }
        for fc in feature_cols[:6]:
            if fc in sub.columns:
                cluster_summary[lbl][fc] = round(float(sub[fc].mean()), 3)

    # Advice per cluster
    advice_map = {
        'Sangat Tinggi': {
            'icon': '🏆',
            'color': '#28C76F',
            'advice': [
                'Berikan tantangan tambahan: proyek riset, kompetisi, atau KKN di instansi bergengsi.',
                'Pertimbangkan jalur fast-track atau beasiswa lanjutan studi S2.',
                'Libatkan sebagai asisten praktikum atau tutor sebaya.',
            ]
        },
        'Tinggi': {
            'icon': '⭐',
            'color': '#00CFE8',
            'advice': [
                'Pertahankan momentum; beri akses ke materi pengayaan atau sertifikasi profesional.',
                'Dorong partisipasi aktif dalam seminar dan konferensi ilmiah.',
                'Monitoring ringan untuk memastikan tidak ada penurunan performa.',
            ]
        },
        'Sedang': {
            'icon': '📘',
            'color': '#7367F0',
            'advice': [
                'Lakukan bimbingan rutin setiap 2 minggu untuk mengidentifikasi hambatan.',
                'Rekomendasikan sumber belajar tambahan (modul, video, teman belajar).',
                'Fokus pada mata kuliah dengan nilai terendah untuk perbaikan terarah.',
            ]
        },
        'Cukup': {
            'icon': '⚠️',
            'color': '#FF9F43',
            'advice': [
                'Jadwalkan konseling akademik individual minimal 1x/bulan.',
                'Identifikasi apakah ada kendala non-akademik (ekonomi, kesehatan, sosial).',
                'Program remedial terstruktur untuk mata kuliah di bawah standar.',
            ]
        },
        'Rendah': {
            'icon': '🚨',
            'color': '#EA5455',
            'advice': [
                'Intervensi segera: konseling akademik intensif dan monitoring kehadiran.',
                'Koordinasi dengan orang tua/wali untuk dukungan di luar kampus.',
                'Pertimbangkan program bimbingan khusus atau cuti semester jika diperlukan.',
            ]
        },
        'Sangat Rendah': {
            'icon': '🆘',
            'color': '#EA5455',
            'advice': [
                'Tindakan darurat: rapat koordinasi dosen, PA, dan BK.',
                'Evaluasi kelayakan melanjutkan semester berjalan.',
                'Tawarkan jalur alternatif: perwalian intensif, remedial total.',
            ]
        },
    }
    # Default
    default_advice = {'icon': '📋', 'color': '#7f8c8d',
                      'advice': ['Pantau perkembangan mahasiswa secara berkala.',
                                 'Berikan dukungan sesuai kebutuhan individual.']}

    students_json = json.dumps(students, ensure_ascii=False)
    summary_json = json.dumps(cluster_summary, ensure_ascii=False)
    color_map_json = json.dumps(color_map, ensure_ascii=False)
    advice_json = json.dumps({k: v for k, v in advice_map.items()
                               if any(k in lbl for lbl in labels_unique)} |
                              {lbl: advice_map.get(lbl, default_advice) for lbl in labels_unique},
                             ensure_ascii=False)
    labels_json = json.dumps(labels_unique, ensure_ascii=False)
    feature_cols_json = json.dumps(feature_cols[:8], ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dashboard Clustering Mahasiswa</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #FFFFFF; color: #566A7F; }}
  header {{ background: linear-gradient(135deg,#7367F0,#5E50EE); color: white; padding: 20px 30px; }}
  header h1 {{ font-size: 1.6rem; }}
  header p {{ opacity: 0.9; font-size: 0.9rem; margin-top: 4px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  .stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 14px; margin-bottom: 20px; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 16px; text-align: center; box-shadow: 0 4px 12px #EAEAEA; border: 1px solid #F0F0F0; }}
  .stat-card .val {{ font-size: 2rem; font-weight: 700; }}
  .stat-card .lbl {{ font-size: 0.78rem; color: #8E9BAE; margin-top: 4px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  @media(max-width:900px){{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px #EAEAEA; border: 1px solid #F0F0F0; }}
  .card h2 {{ font-size: 1rem; font-weight: 700; color: #7367F0; margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #F0F0F0; }}
  canvas {{ width: 100% !important; }}
  #bubble-canvas {{ cursor: crosshair; }}
  .tooltip {{ position: fixed; background: white; color: #566A7F; padding: 10px 14px; border-radius: 8px; font-size: 0.82rem; pointer-events: none; display: none; z-index: 1000; max-width: 220px; line-height: 1.5; border: 1px solid #F0F0F0; box-shadow: 0 4px 12px #EAEAEA; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8rem; background: #F5F3FF; padding: 4px 10px; border-radius: 20px; color: #566A7F; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  /* Modal */
  .modal-overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,.3); display: none; z-index: 2000; align-items: center; justify-content: center; }}
  .modal-overlay.active {{ display: flex; }}
  .modal {{ background: white; border-radius: 14px; padding: 28px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; position: relative; border: 1px solid #F0F0F0; box-shadow: 0 10px 30px #EAEAEA; }}
  .modal-close {{ position: absolute; top: 14px; right: 18px; font-size: 1.4rem; cursor: pointer; color: #8E9BAE; background: none; border: none; }}
  .modal h3 {{ font-size: 1.2rem; color: #7367F0; margin-bottom: 4px; }}
  .modal .cluster-badge {{ display: inline-block; padding: 4px 14px; border-radius: 20px; color: white; font-size: 0.82rem; font-weight: 600; margin-bottom: 14px; }}
  .modal-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }}
  .modal-item {{ background: #F5F3FF; padding: 10px; border-radius: 8px; }}
  .modal-item .mi-lbl {{ font-size: 0.73rem; color: #8E9BAE; }}
  .modal-item .mi-val {{ font-size: 1rem; font-weight: 700; color: #566A7F; }}
  .course-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .course-table th {{ background: #F5F3FF; padding: 6px 10px; text-align: left; font-weight: 600; color: #566A7F; }}
  .course-table td {{ padding: 5px 10px; border-bottom: 1px solid #F0F0F0; }}
  .course-grade {{ font-weight: 700; }}
  .grade-A {{ color: #28C76F; }} .grade-B {{ color: #7367F0; }} .grade-C {{ color: #FF9F43; }} .grade-D {{ color: #EA5455; }}
  /* Heatmap */
  .heatmap-wrap {{ overflow-x: auto; }}
  table.heatmap {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
  table.heatmap th {{ background: #F5F3FF; color: #566A7F; padding: 8px 12px; text-align: center; white-space: nowrap; border: 1px solid #F0F0F0; }}
  table.heatmap td {{ padding: 8px 12px; text-align: center; border: 1px solid #F0F0F0; color: #566A7F; }}
  table.heatmap tr:hover td {{ background: rgba(115,103,240,0.06); }}
  /* Advice */
  .advice-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap: 14px; }}
  .advice-card {{ border-radius: 10px; padding: 16px; border-left: 5px solid; }}
  .advice-card h4 {{ font-size: 0.95rem; font-weight: 700; margin-bottom: 8px; }}
  .advice-card ul {{ padding-left: 16px; font-size: 0.82rem; line-height: 1.7; color: #566A7F; }}
  .tab-bar {{ display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab-btn {{ padding: 7px 18px; border-radius: 20px; border: none; cursor: pointer; font-size: 0.83rem; font-weight: 600; background: #F5F3FF; color: #566A7F; transition: all .2s; }}
  .tab-btn.active {{ background: #7367F0; color: white; }}
</style>
</head>
<body>
<header>
  <h1>📊 Dashboard Clustering Akademik Mahasiswa</h1>
  <p>Analisis berbasis PRINCALS + Clustering &nbsp;|&nbsp; Klik titik pada bubble chart untuk melihat profil mahasiswa</p>
</header>

<div class="container">
  <!-- Stat cards -->
  <div class="stats-row" id="stat-cards"></div>

  <!-- Bubble + Heatmap -->
  <div class="grid2">
    <div class="card">
      <h2>🫧 Bubble Chart Cluster (PC1 vs PC2)</h2>
      <canvas id="bubble-canvas" height="360"></canvas>
      <div class="legend" id="bubble-legend"></div>
    </div>
    <div class="card">
      <h2>🌡️ Heatmap Profil Cluster</h2>
      <div class="heatmap-wrap"><table class="heatmap" id="heatmap-table"></table></div>
    </div>
  </div>

  <!-- Evaluation chart -->
  <div class="card" style="margin-bottom:20px">
    <h2>📈 Distribusi Mahasiswa per Cluster</h2>
    <canvas id="bar-canvas" height="160"></canvas>
  </div>

  <!-- Advice -->
  <div class="card" style="margin-bottom:20px">
    <h2>💡 Rekomendasi untuk Dosen</h2>
    <div class="advice-grid" id="advice-grid"></div>
  </div>

  <!-- Student list -->
  <div class="card">
    <h2>👥 Daftar Mahasiswa</h2>
    <div class="tab-bar" id="tab-bar"></div>
    <input type="text" id="search-input" placeholder="🔍 Cari nama atau NRP..." 
           style="width:100%;padding:9px 14px;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:12px;font-size:0.87rem;">
    <div id="student-list" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;"></div>
  </div>
</div>

<!-- Tooltip -->
<div class="tooltip" id="tooltip"></div>

<!-- Modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal" id="modal-content">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div id="modal-body"></div>
  </div>
</div>

<script>
const STUDENTS = {students_json};
const SUMMARY  = {summary_json};
const COLORS   = {color_map_json};
const ADVICE   = {advice_json};
const LABELS   = {labels_json};
const FEAT_COLS = {feature_cols_json};

// ── Stat cards ──
function buildStats() {{
  const total = STUDENTS.length;
  const avgIPS = (STUDENTS.reduce((s,st)=>s+st.ips,0)/total).toFixed(3);
  const nClusters = LABELS.length;
  const topCluster = LABELS[0] || '-';
  const topCount = STUDENTS.filter(s=>s.cluster===topCluster).length;

  const data = [
    {{val: total, lbl: 'Total Mahasiswa', color: '#7367F0'}},
    {{val: nClusters, lbl: 'Jumlah Cluster', color: '#28C76F'}},
    {{val: avgIPS, lbl: 'Rata-Rata IPS', color: '#FF9F43'}},
    {{val: topCount, lbl: 'Cluster ' + topCluster, color: COLORS[topCluster] || '#7367F0'}},
  ];
  const wrap = document.getElementById('stat-cards');
  data.forEach(d => {{
    wrap.innerHTML += `<div class="stat-card"><div class="val" style="color:${{d.color}}">${{d.val}}</div><div class="lbl">${{d.lbl}}</div></div>`;
  }});
}}

// ── Bubble Chart (pure canvas) ──
let bubbleStudents = STUDENTS;
function drawBubble(filtered) {{
  const canvas = document.getElementById('bubble-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight || 360;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const pad = 40;
  const pts = filtered;
  if (!pts.length) return;

  const x1 = Math.min(...pts.map(s=>s.pc1))-0.3, x2 = Math.max(...pts.map(s=>s.pc1))+0.3;
  const y1 = Math.min(...pts.map(s=>s.pc2))-0.3, y2 = Math.max(...pts.map(s=>s.pc2))+0.3;
  const scX = v => pad + (v-x1)/(x2-x1)*(W-2*pad);
  const scY = v => H-pad - (v-y1)/(y2-y1)*(H-2*pad);

  // Axes
  ctx.strokeStyle='#ccc'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  ctx.fillStyle='#718096'; ctx.font='11px Segoe UI';
  ctx.fillText('PC1', W-pad+4, H-pad+4);
  ctx.fillText('PC2', pad-14, pad-6);

  pts.forEach(s => {{
    const cx = scX(s.pc1), cy = scY(s.pc2);
    const r = s.bubble_size / 2.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.fillStyle = (COLORS[s.cluster] || '#999') + 'cc';
    ctx.strokeStyle = (COLORS[s.cluster] || '#999');
    ctx.lineWidth = 1.5;
    ctx.fill(); ctx.stroke();
  }});

  // Store for hit-test
  canvas._pts = pts.map(s => ({{...s, cx: scX(s.pc1), cy: scY(s.pc2), r: s.bubble_size/2.5}}));
  canvas._scaleInfo = {{W, H, pad, x1, x2, y1, y2, scX, scY, dpr}};
}}

function buildBubbleLegend() {{
  const wrap = document.getElementById('bubble-legend');
  LABELS.forEach(lbl => {{
    wrap.innerHTML += `<div class="legend-item"><div class="legend-dot" style="background:${{COLORS[lbl]||'#999'}}"></div>${{lbl}}</div>`;
  }});
}}

// Bubble hover + click
(function() {{
  const canvas = document.getElementById('bubble-canvas');
  const tip = document.getElementById('tooltip');
  function getPos(e) {{
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }}
  function hitTest(px, py) {{
    if (!canvas._pts) return null;
    return canvas._pts.find(p => Math.hypot(p.cx-px, p.cy-py) <= p.r + 4);
  }}
  canvas.addEventListener('mousemove', e => {{
    const [px,py] = getPos(e);
    const hit = hitTest(px, py);
    if (hit) {{
      canvas.style.cursor='pointer';
      tip.style.display='block';
      tip.style.left=(e.clientX+14)+'px';
      tip.style.top=(e.clientY-10)+'px';
      tip.innerHTML = `<b>${{hit.name}}</b><br>NRP: ${{hit.id}}<br>IPS: ${{hit.ips}}<br>Cluster: ${{hit.cluster}}`;
    }} else {{
      canvas.style.cursor='crosshair';
      tip.style.display='none';
    }}
  }});
  canvas.addEventListener('mouseleave', () => {{ tip.style.display='none'; }});
  canvas.addEventListener('click', e => {{
    const [px,py] = getPos(e);
    const hit = hitTest(px, py);
    if (hit) openModal(hit);
  }});
}})();

// ── Heatmap Table ──
function buildHeatmap() {{
  const feats = Object.keys(Object.values(SUMMARY)[0]).filter(k=>k!=='count');
  const hdr = ['Cluster','Jumlah',...feats.map(f=>f.replace(/_/g,' ').replace('avg ',''))];
  let html = '<thead><tr>'+hdr.map(h=>`<th>${{h}}</th>`).join('')+'</tr></thead><tbody>';
  LABELS.forEach(lbl => {{
    const s = SUMMARY[lbl];
    const color = COLORS[lbl]||'#999';
    html += `<tr><td style="font-weight:700;color:${{color}};border-left:4px solid ${{color}}">${{lbl}}</td><td>${{s.count}}</td>`;
    feats.forEach(f => {{
      const v = s[f];
      html += `<td>${{v!==null&&v!==undefined?v:'-'}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody>';
  document.getElementById('heatmap-table').innerHTML = html;
}}

// ── Bar Chart ──
function buildBar() {{
  const canvas = document.getElementById('bar-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight || 160;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  const counts = LABELS.map(lbl => STUDENTS.filter(s=>s.cluster===lbl).length);
  const max = Math.max(...counts);
  const pad = 40, bGap = 16;
  const bW = Math.min(60, (W - 2*pad - bGap*(LABELS.length-1)) / LABELS.length);

  LABELS.forEach((lbl, i) => {{
    const bH = (counts[i]/max) * (H - pad - 30);
    const bx = pad + i*(bW+bGap);
    const by = H - pad - bH;
    ctx.fillStyle = COLORS[lbl]||'#999';
    ctx.beginPath();
    ctx.roundRect(bx, by, bW, bH, [4,4,0,0]);
    ctx.fill();
    ctx.fillStyle='#2d3748'; ctx.font='bold 13px Segoe UI'; ctx.textAlign='center';
    ctx.fillText(counts[i], bx+bW/2, by-6);
    ctx.fillStyle='#718096'; ctx.font='10px Segoe UI';
    ctx.fillText(lbl.length>8?lbl.slice(0,8)+'…':lbl, bx+bW/2, H-pad+14);
  }});
}}

// ── Advice ──
function buildAdvice() {{
  const wrap = document.getElementById('advice-grid');
  LABELS.forEach(lbl => {{
    const a = ADVICE[lbl] || {{icon:'📋',color:'#999',advice:['Pantau perkembangan mahasiswa.']}};
    const count = STUDENTS.filter(s=>s.cluster===lbl).length;
    wrap.innerHTML += `
      <div class="advice-card" style="background:${{a.color}}18;border-left-color:${{a.color}}">
        <h4 style="color:${{a.color}}">${{a.icon}} ${{lbl}} <span style="font-weight:400;font-size:.8rem">(n=${{count}})</span></h4>
        <ul>${{a.advice.map(x=>`<li>${{x}}</li>`).join('')}}</ul>
      </div>`;
  }});
}}

// ── Student list ──
let activeTab = 'Semua';
function buildTabs() {{
  const wrap = document.getElementById('tab-bar');
  ['Semua', ...LABELS].forEach(lbl => {{
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (lbl === activeTab ? ' active' : '');
    btn.textContent = lbl;
    btn.onclick = () => {{
      activeTab = lbl;
      document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      renderStudents();
    }};
    wrap.appendChild(btn);
  }});
}}

function renderStudents() {{
  const query = document.getElementById('search-input').value.toLowerCase();
  let filtered = STUDENTS;
  if (activeTab !== 'Semua') filtered = filtered.filter(s=>s.cluster===activeTab);
  if (query) filtered = filtered.filter(s=>s.name.toLowerCase().includes(query)||String(s.id).includes(query));

  const wrap = document.getElementById('student-list');
  wrap.innerHTML = '';
  filtered.forEach(s => {{
    const color = COLORS[s.cluster]||'#999';
    const card = document.createElement('div');
    card.style.cssText=`background:white;border-radius:10px;padding:14px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.07);border-top:3px solid ${{color}};transition:transform .15s`;
    card.onmouseenter=()=>card.style.transform='translateY(-3px)';
    card.onmouseleave=()=>card.style.transform='none';
    card.innerHTML=`
      <div style="font-weight:700;font-size:.92rem">${{s.name}}</div>
      <div style="font-size:.78rem;color:#718096">NRP: ${{s.id}} · ${{s.angkatan}}</div>
      <div style="margin-top:8px;display:flex;align-items:center;gap:8px">
        <span style="background:${{color}}22;color:${{color}};padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:700">${{s.cluster}}</span>
        <span style="font-size:.8rem;color:#4a5568">IPS: <b>${{s.ips.toFixed(3)}}</b></span>
      </div>`;
    card.onclick=()=>openModal(s);
    wrap.appendChild(card);
  }});
  if (!filtered.length) wrap.innerHTML='<p style="color:#718096;padding:20px">Tidak ada data.</p>';

  // Redraw bubble with filtered
  const bubFilter = activeTab==='Semua' ? STUDENTS : STUDENTS.filter(s=>s.cluster===activeTab);
  drawBubble(bubFilter.filter(s => !query || s.name.toLowerCase().includes(query)||String(s.id).includes(query)));
}}

document.getElementById('search-input').addEventListener('input', renderStudents);

// ── Modal ──
function openModal(s) {{
  const color = COLORS[s.cluster]||'#999';
  const courseRows = Object.entries(s.courses||{{}}).map(([name,val]) => {{
    const cls = val>=3.5?'grade-A':val>=3?'grade-B':val>=2.5?'grade-C':'grade-D';
    return `<tr><td>${{name}}</td><td class="course-grade ${{cls}}">${{val}}</td></tr>`;
  }}).join('');
  document.getElementById('modal-body').innerHTML = `
    <h3>${{s.name}}</h3>
    <span class="cluster-badge" style="background:${{color}}">${{s.cluster}}</span>
    <div class="modal-grid">
      <div class="modal-item"><div class="mi-lbl">NRP</div><div class="mi-val">${{s.id}}</div></div>
      <div class="modal-item"><div class="mi-lbl">Angkatan</div><div class="mi-val">${{s.angkatan}}</div></div>
      <div class="modal-item"><div class="mi-lbl">Jenis Kelamin</div><div class="mi-val">${{s.jk}}</div></div>
      <div class="modal-item"><div class="mi-lbl">Asal</div><div class="mi-val">${{s.asal}}</div></div>
      <div class="modal-item" style="background:${{color}}18;border:1px solid ${{color}}44">
        <div class="mi-lbl">Rata-Rata IPS</div>
        <div class="mi-val" style="color:${{color}};font-size:1.4rem">${{s.ips.toFixed(3)}}</div>
      </div>
      ${{s.absensi!==null?`<div class="modal-item"><div class="mi-lbl">Rata-Rata Absensi</div><div class="mi-val">${{s.absensi?.toFixed(1)}}%</div></div>`:''}}</div>
    ${{courseRows?`<h4 style="margin:14px 0 8px;font-size:.9rem">Nilai Mata Kuliah</h4><table class="course-table"><thead><tr><th>Mata Kuliah</th><th>Nilai</th></tr></thead><tbody>${{courseRows}}</tbody></table>`:''}}
  `;
  document.getElementById('modal-overlay').classList.add('active');
}}
function closeModal() {{
  document.getElementById('modal-overlay').classList.remove('active');
}}
document.getElementById('modal-overlay').addEventListener('click', e => {{
  if (e.target === document.getElementById('modal-overlay')) closeModal();
}});

// ── Init ──
window.addEventListener('DOMContentLoaded', () => {{
  buildStats();
  setTimeout(()=>{{ drawBubble(STUDENTS); buildBubbleLegend(); }}, 100);
  setTimeout(buildBar, 100);
  buildHeatmap();
  buildAdvice();
  buildTabs();
  renderStudents();
}});
window.addEventListener('resize', () => {{
  setTimeout(()=>{{ drawBubble(STUDENTS); buildBar(); }}, 100);
}});
</script>
</body>
</html>"""

    out_path = 'output/dashboard.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[OK] Step 7 Dashboard complete: {out_path}")
    return out_path


if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/df_labeled.pkl'):
        print("Jalankan step 1-6 terlebih dahulu.")
    else:
        build_dashboard()
