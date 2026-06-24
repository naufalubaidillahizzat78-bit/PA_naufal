"""
STEP 7 ENHANCED: Dashboard HTML + EDA visualisasi
By Naufal
"""

import pandas as pd, numpy as np, json, pickle, warnings
warnings.filterwarnings('ignore')
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'output')

def build_dashboard_enhanced():
    df       = pd.read_pickle(f'{OUT}/df_labeled.pkl')
    X_pc     = pd.read_pickle(f'{OUT}/X_princals.pkl')
    feat_cols= pd.read_pickle(f'{OUT}/feature_cols.pkl').tolist()
    var_info = pd.read_pickle(f'{OUT}/princals_info.pkl')

    try:
        with open(f'{OUT}/charts_b64.pkl','rb') as f:
            charts = pickle.load(f)
    except FileNotFoundError:
        print("[Warning] output/charts_b64.pkl tidak ditemukan. Menggunakan gambar kosong (dummy) untuk mencegah error.")
        empty_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        charts = {k: empty_b64 for k in ['ips_trend', 'ips_dist', 'absen', 'outlier', 'corr', 'princals', 'eval']}

    PALETTE = {
        'Sangat Tinggi': '#28C76F',
        'Tinggi': '#00CFE8',
        'Sedang': '#7367F0',
        'Cukup': '#FF9F43',
        'Rendah': '#EA5455',
    }
    ORDER   = ['Sangat Tinggi','Tinggi','Sedang','Cukup','Rendah']
    SEM_IPS = ['nilai ips','IPS','IPS.1','IPS.2']
    ABSEN   = ['Rata-Rata Absen Mahasiswa','ABSENSI RATA RATA','ABSENSI RATA_RATA',
               'ABSENSI RATA RATA.1','ABSENSI RATA RATA.2']
    SKIP    = {'NRP','Nama Mahasiswa','Angkatan Tahun','Prodi','JK','Asal Kab/Kota',
               'cluster','cluster_label','Prodi_enc','Asal Kab/Kota_enc'}
    SKIP_IA = {c for c in df.columns if 'IPS' in c.upper() or 'ABSEN' in c.upper()}
    COURSE  = [c for c in df.columns if c not in SKIP and c not in SKIP_IA
               and not c.endswith('_enc') and not c.endswith('.1')]

    lbl_ord = [l for l in ORDER if l in df['cluster_label'].values]
    min_len = min(len(df), len(X_pc))
    df  = df.iloc[:min_len].reset_index(drop=True)
    X_pc= X_pc.iloc[:min_len].reset_index(drop=True)

    # Build student records
    students = []
    for i, row in df.iterrows():
        pc1 = float(X_pc.iloc[i,0]) if X_pc.shape[1]>0 else 0
        pc2 = float(X_pc.iloc[i,1]) if X_pc.shape[1]>1 else 0
        ips_val  = float(row['Rata-Rata IPS']) if 'Rata-Rata IPS' in row.index else 0
        absen_v  = float(row['Rata-Rata Absen Mahasiswa']) if 'Rata-Rata Absen Mahasiswa' in row.index else None
        courses  = {c: round(float(row[c]),2) for c in COURSE if c in row.index and not np.isnan(row[c])}
        ips_sem  = {}
        for j, col in enumerate(SEM_IPS):
            if col in row.index: ips_sem[f'Sem {j+1}'] = round(float(row[col]),3)
        students.append({'id':int(row['NRP']),'name':str(row['Nama Mahasiswa']),
            'angkatan':int(row.get('Angkatan Tahun',0)),
            'jk':'Laki-laki' if row.get('JK',1)==1 else 'Perempuan',
            'asal':str(row.get('Asal Kab/Kota','-')),
            'cluster':str(row['cluster_label']),'ips':ips_val,
            'absensi':absen_v,'pc1':pc1,'pc2':pc2,
            'bubble_size':max(8,ips_val*12),'courses':courses,'ips_sem':ips_sem})

    cluster_summary = {}
    for lbl in lbl_ord:
        sub = df[df['cluster_label']==lbl]
        absen_exist = [c for c in ABSEN if c in df.columns]
        cluster_summary[lbl] = {
            'count': int(len(sub)),
            'avg_ips': round(float(sub['Rata-Rata IPS'].mean()),3),
            'avg_absensi': round(float(sub[absen_exist].mean().mean()),2),
        }
        for j, col in enumerate(SEM_IPS):
            if col in sub.columns:
                cluster_summary[lbl][f'IPS Sem {j+1}'] = round(float(sub[col].mean()),3)

    ADVICE = {
        'Sangat Tinggi':{'icon':'🏆','tips':['Proyek riset & kompetisi nasional/internasional.',
            'Pertimbangkan fast-track S2 atau beasiswa.','Libatkan sebagai tutor sebaya.']},
        'Tinggi':{'icon':'⭐','tips':['Akses materi pengayaan & sertifikasi profesional.',
            'Dorong partisipasi seminar & proyek.','Rekomendasikan program magang industri.']},
        'Sedang':{'icon':'📘','tips':['Bimbingan rutin setiap 2 minggu.',
            'Sumber belajar tambahan (modul, kelompok belajar).','Fokus perbaikan mata kuliah lemah.']},
        'Cukup':{'icon':'⚠️','tips':['Konseling akademik individual 1×/bulan.',
            'Program remedial terstruktur.','Manfaatkan kehadiran baik untuk keaktifan kelas.']},
        'Rendah':{'icon':'🚨','tips':['Intervensi segera & monitoring kehadiran mingguan.',
            'Koordinasi dengan orang tua/wali.','Evaluasi beban SKS sesuai kemampuan.']},
    }

    # JSON payloads
    students_j  = json.dumps(students, ensure_ascii=False)
    summary_j   = json.dumps(cluster_summary, ensure_ascii=False)
    color_j     = json.dumps(PALETTE, ensure_ascii=False)
    labels_j    = json.dumps(lbl_ord, ensure_ascii=False)
    advice_j    = json.dumps(ADVICE, ensure_ascii=False)
    charts_j    = json.dumps({k: f'data:image/png;base64,{v}' for k,v in charts.items()})
    evr_j       = json.dumps([round(e*100,2) for e in var_info['explained_variance_ratio']])
    n_comp_j    = int(var_info['n_components'])
    cumvar_j    = round(float(var_info['cumulative_variance'])*100,2)

    # ── HEATMAP DATA for JS ──
    fc_heat = [c for c in feat_cols if c in df.columns and c != 'Angkatan Tahun']
    heat_rows = []
    for lbl in lbl_ord:
        sub = df[df['cluster_label']==lbl]
        heat_rows.append({c: round(float(sub[c].mean()),3) for c in fc_heat})
    heat_j      = json.dumps({'labels': lbl_ord, 'features': fc_heat, 'data': heat_rows})

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Academic Clustering Dashboard · by Naufal</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#FFFFFF;--surface:#FFFFFF;--surface2:#F5F3FF;--border:#F0F0F0;
  --text:#566A7F;--muted:#8E9BAE;--dim:#A8B4C2;
  --blue:#7367F0;--blue-hover:#5E50EE;--accent:#00CFE8;
  --green:#28C76F;--yellow:#FF9F43;--orange:#FF9F43;--red:#EA5455;
  --font:'DM Sans',sans-serif;--serif:'DM Serif Display',serif;--mono:'JetBrains Mono',monospace;
}}
body{{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;line-height:1.5}}
a{{color:var(--blue);text-decoration:none}}
/* ── Layout ── */
.layout{{display:flex;min-height:100vh}}
.sidebar{{width:220px;flex-shrink:0;background:#F5F3FF;
  border-right:1px solid var(--border);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}}
.main{{flex:1;overflow-x:hidden}}
/* ── Sidebar ── */
.sb-logo{{padding:24px 20px 16px}}
.sb-logo-title{{font-family:var(--serif);font-size:1.15rem;color:var(--blue);line-height:1.2}}
.sb-logo-title span{{color:var(--blue-hover)}}
.sb-byline{{font-family:var(--mono);font-size:.65rem;color:var(--blue);opacity:.6;margin-top:4px;letter-spacing:.08em}}
.sb-divider{{border:none;border-top:1px solid var(--border);margin:8px 16px}}
.sb-nav{{padding:4px 10px}}
.sb-nav-item{{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;
  font-size:.83rem;color:var(--muted);cursor:pointer;transition:all .15s;border:none;
  background:none;width:100%;text-align:left}}
.sb-nav-item:hover{{background:rgba(94,80,238,.06);color:var(--blue-hover)}}
.sb-nav-item.active{{background:rgba(94,80,238,.12);color:var(--blue-hover);font-weight:600}}
.sb-nav-icon{{font-size:.95rem;width:18px;text-align:center}}
.sb-info{{padding:14px 20px;margin-top:auto;border-top:1px solid var(--border)}}
.sb-info-row{{font-size:.72rem;color:var(--dim);line-height:2;font-family:var(--mono)}}
.sb-info-row b{{color:var(--muted)}}
/* ── Page ── */
.page{{display:none;padding:28px 32px 48px}}
.page.active{{display:block}}
/* ── Hero ── */
.hero{{background:linear-gradient(135deg,#F5F3FF,#E8E4FF);
  border:1px solid var(--border);border-radius:16px;padding:28px 36px;
  margin-bottom:28px;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;top:-80px;right:-80px;width:320px;height:320px;
  background:radial-gradient(circle,rgba(115,103,240,.15) 0%,transparent 70%);border-radius:50%}}
.hero-title{{font-family:var(--serif);font-size:2.2rem;color:var(--blue);line-height:1.1}}
.hero-title em{{color:var(--blue-hover);font-style:normal}}
.hero-sub{{color:var(--text);font-size:.88rem;margin-top:10px;font-weight:300}}
.hero-tags{{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}}
.hero-tag{{background:#FFFFFF;border:1px solid rgba(115,103,240,.25);
  color:var(--blue);padding:3px 12px;border-radius:20px;font-size:.73rem;font-weight:500}}
.hero-byline{{position:absolute;bottom:16px;right:22px;font-family:var(--mono);
  font-size:.68rem;color:rgba(115,103,240,.5);letter-spacing:.1em}}
/* ── Metrics ── */
.metric-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}}
.metric-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:16px 18px;position:relative;overflow:hidden;transition:border-color .2s;
  box-shadow: 0 4px 12px #EAEAEA;}}
.metric-card:hover{{border-color:var(--blue-hover)}}
.metric-accent{{position:absolute;top:0;left:0;width:3px;height:100%;border-radius:12px 0 0 12px}}
.metric-val{{font-family:var(--serif);font-size:1.8rem;color:var(--blue);line-height:1}}
.metric-lbl{{font-size:.71rem;color:var(--muted);margin-top:6px;text-transform:uppercase;letter-spacing:.06em}}
/* ── Section header ── */
.section-h{{font-family:var(--serif);font-size:1.2rem;color:var(--text);margin:28px 0 16px;
  padding-bottom:8px;border-bottom:1px solid var(--border)}}
.section-h em{{color:var(--blue);font-style:normal}}
/* ── Cards ── */
.card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px;
  box-shadow: 0 4px 12px #EAEAEA;}}
.card-title{{font-size:.85rem;font-weight:600;color:var(--text);margin-bottom:14px;letter-spacing:.03em}}
/* ── Grid ── */
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}}
@media(max-width:900px){{.grid2,.grid3{{grid-template-columns:1fr}}}}
/* ── Charts ── */
.chart-img{{width:100%;border-radius:8px;display:block}}
/* ── Tabs ── */
.tabs{{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:18px;
  background:var(--surface);padding:5px;border-radius:10px;border:1px solid var(--border)}}
.tab-btn{{padding:7px 16px;border-radius:7px;border:none;background:none;cursor:pointer;
  font-size:.82rem;font-weight:500;color:var(--muted);transition:all .15s;font-family:var(--font)}}
.tab-btn:hover{{color:var(--blue)}}
.tab-btn.active{{background:rgba(115,103,240,.15);color:var(--blue)}}
.tab-pane{{display:none}}.tab-pane.active{{display:block}}
/* ── Info box ── */
.info-box{{background:#F5F3FF;border:1px solid rgba(115,103,240,.15);border-left:3px solid var(--blue);
  border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:.83rem;color:var(--text);line-height:1.6}}
/* ── Heatmap ── */
.heatmap{{border-collapse:collapse;width:100%;font-size:.78rem}}
.heatmap th{{background:#F5F3FF;color:var(--text);padding:8px 12px;font-weight:500;
  white-space:nowrap;font-size:.75rem;border:1px solid var(--border)}}
.heatmap td{{padding:7px 12px;text-align:center;border:1px solid var(--border);color:var(--text)}}
.heatmap tr:hover td{{background:rgba(115,103,240,.06)}}
/* ── Bubble canvas ── */
#bubble-canvas{{cursor:crosshair;border-radius:8px;background:#FFFFFF;width:100% !important}}
/* ── Tooltip ── */
.tooltip{{position:fixed;background:#FFFFFF;color:var(--text);padding:10px 14px;border-radius:8px;
  font-size:.8rem;pointer-events:none;display:none;z-index:1000;max-width:220px;line-height:1.6;
  border:1px solid var(--border);box-shadow:0 8px 24px #EAEAEA}}
/* ── Modal ── */
.modal-overlay{{position:fixed;inset:0;background:rgba(0,0,0,.3);display:none;
  z-index:2000;align-items:center;justify-content:center}}
.modal-overlay.active{{display:flex}}
.modal{{background:#FFFFFF;border:1px solid var(--border);border-radius:16px;padding:28px;
  max-width:620px;width:92%;max-height:85vh;overflow-y:auto;position:relative;
  box-shadow:0 10px 30px rgba(86,106,127,0.15)}}
.modal-close{{position:absolute;top:14px;right:16px;background:none;border:none;
  font-size:1.3rem;cursor:pointer;color:var(--muted);transition:color .15s}}
.modal-close:hover{{color:var(--blue-hover)}}
.modal-name{{font-family:var(--serif);font-size:1.3rem;color:var(--blue)}}
.cl-badge{{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.75rem;
  font-weight:600;color:white;margin:6px 0 14px}}
.modal-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}}
.modal-item{{background:#F5F3FF;padding:10px;border-radius:8px}}
.modal-item .mi-lbl{{font-size:.7rem;color:var(--muted)}}
.modal-item .mi-val{{font-size:1rem;font-weight:700;color:var(--text);font-family:var(--serif)}}
.course-tbl{{width:100%;border-collapse:collapse;font-size:.78rem}}
.course-tbl th{{background:#F5F3FF;padding:6px 10px;text-align:left;font-weight:600;color:var(--text)}}
.course-tbl td{{padding:5px 10px;border-bottom:1px solid var(--border);color:var(--text)}}
.grade-A{{color:var(--green);font-weight:700}}
.grade-B{{color:var(--blue);font-weight:700}}
.grade-C{{color:var(--yellow);font-weight:700}}
.grade-D{{color:var(--red);font-weight:700}}
/* ── Mini sparkline ── */
canvas.sparkline{{border-radius:4px}}
/* ── Student cards ── */
.stu-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}}
.stu-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:14px;cursor:pointer;transition:all .18s;border-top:3px solid transparent;
  box-shadow:0 2px 8px #EAEAEA}}
.stu-card:hover{{border-color:var(--blue-hover);transform:translateY(-2px);
  box-shadow:0 8px 20px #EAEAEA}}
.stu-name{{font-weight:600;font-size:.9rem;color:var(--text);margin-bottom:3px}}
.stu-nrp{{font-size:.73rem;color:var(--muted);font-family:var(--mono)}}
.stu-meta{{display:flex;align-items:center;gap:8px;margin-top:8px}}
/* ── Legend ── */
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}}
.legend-item{{display:flex;align-items:center;gap:6px;font-size:.78rem;
  background:#F5F3FF;padding:3px 10px;border-radius:20px;color:var(--text)}}
.legend-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
/* ── Advice ── */
.advice-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
.advice-card{{border-radius:10px;padding:16px;border:1px solid;transition:transform .18s}}
.advice-card:hover{{transform:translateY(-2px)}}
.advice-card h4{{font-size:.95rem;font-weight:700;margin-bottom:8px}}
.advice-card ul{{padding-left:16px;font-size:.8rem;line-height:1.8;color:var(--text)}}
/* ── Search ── */
.search-input{{width:100%;padding:10px 16px;background:#FFFFFF;border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-size:.85rem;outline:none;font-family:var(--font);margin-bottom:14px}}
.search-input:focus{{border-color:var(--blue-hover)}}
/* ── Footer ── */
.footer{{text-align:center;padding:24px;color:var(--dim);font-size:.72rem;
  border-top:1px solid var(--border);margin-top:40px;font-family:var(--mono)}}
.footer span{{color:rgba(77,166,255,.5)}}
</style>
</head>
<body>
<div class="layout">

<!-- SIDEBAR -->
<nav class="sidebar">
  <div class="sb-logo">
    <div class="sb-logo-title">🎓 Academic<br><span>Clustering</span></div>
    <div class="sb-byline">by Naufal</div>
  </div>
  <hr class="sb-divider">
  <div class="sb-nav">
    <button class="sb-nav-item active" onclick="showPage('overview',this)"><span class="sb-nav-icon">🏠</span> Overview</button>
    <button class="sb-nav-item" onclick="showPage('eda',this)"><span class="sb-nav-icon">🔬</span> EDA</button>
    <button class="sb-nav-item" onclick="showPage('princals',this)"><span class="sb-nav-icon">🌀</span> PRINCALS</button>
    <button class="sb-nav-item" onclick="showPage('clustering',this)"><span class="sb-nav-icon">🫧</span> Clustering</button>
    <button class="sb-nav-item" onclick="showPage('evaluasi',this)"><span class="sb-nav-icon">📊</span> Evaluasi</button>
    <button class="sb-nav-item" onclick="showPage('mahasiswa',this)"><span class="sb-nav-icon">👥</span> Mahasiswa</button>
    <button class="sb-nav-item" onclick="showPage('rekomendasi',this)"><span class="sb-nav-icon">💡</span> Rekomendasi</button>
  </div>
  <div class="sb-info">
    <div class="sb-info-row"><b>Model</b> &nbsp;KMeans k=5</div>
    <div class="sb-info-row"><b>Silhouette</b> 0.3485</div>
    <div class="sb-info-row"><b>BSS/TSS</b> &nbsp;74.46%</div>
    <div class="sb-info-row"><b>PRINCALS</b> {n_comp_j} PC ({cumvar_j}%)</div>
  </div>
</nav>

<!-- MAIN -->
<main class="main">

<!-- ════ PAGE: OVERVIEW ════ -->
<div class="page active" id="page-overview">
<div style="padding:28px 32px 48px">
  <div class="hero">
    <div class="hero-title">Academic<br><em>Clustering</em> Dashboard</div>
    <div class="hero-sub">Segmentasi mahasiswa berbasis PRINCALS + 7 metode clustering &nbsp;·&nbsp; Composite multi-kriteria</div>
    <div class="hero-tags">
      <span class="hero-tag">PRINCALS</span><span class="hero-tag">FCM · FPCM · MFPCM</span>
      <span class="hero-tag">K-Means · K-Medoids</span><span class="hero-tag">DBSCAN · PCM</span>
    </div>
    <div class="hero-byline">by Naufal</div>
  </div>
  <div class="metric-row" id="metric-row"></div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Distribusi Mahasiswa per Cluster</div>
      <canvas id="bar-canvas" height="220"></canvas>
      <div class="legend" id="bar-legend"></div>
    </div>
    <div class="card">
      <div class="card-title">Rata-Rata IPS per Cluster</div>
      <canvas id="ips-canvas" height="220"></canvas>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Tren IPS per Semester per Cluster</div>
    <img class="chart-img" src="data:image/png;base64,{charts['ips_trend']}" alt="IPS Trend">
  </div>
</div>
</div>

<!-- ════ PAGE: EDA ════ -->
<div class="page" id="page-eda">
<div style="padding:28px 32px 48px">
  <div class="section-h">Exploratory Data <em>Analysis</em></div>
  <div class="info-box">EDA dilakukan untuk memahami karakteristik data sebelum clustering:
  distribusi variabel, deteksi outlier, korelasi antar fitur, dan pola awal per cluster.</div>

  <div class="tabs" id="eda-tabs">
    <button class="tab-btn active" onclick="switchTab('eda','dist',this)">📊 Distribusi IPS</button>
    <button class="tab-btn" onclick="switchTab('eda','absen',this)">📉 Absensi</button>
    <button class="tab-btn" onclick="switchTab('eda','outlier',this)">🔴 Outlier</button>
    <button class="tab-btn" onclick="switchTab('eda','corr',this)">🌡️ Korelasi</button>
    <button class="tab-btn" onclick="switchTab('eda','courses',this)">📚 Mata Kuliah</button>
  </div>

  <div class="tab-pane active" id="eda-dist">
    <div class="card">
      <div class="card-title">Distribusi IPS per Semester (dengan mean & median)</div>
      <img class="chart-img" src="data:image/png;base64,{charts['ips_dist']}" alt="IPS Distribution">
    </div>
    <div class="card">
      <div class="card-title">Tren IPS per Cluster per Semester</div>
      <img class="chart-img" src="data:image/png;base64,{charts['ips_trend']}" alt="IPS Trend">
    </div>
  </div>

  <div class="tab-pane" id="eda-absen">
    <div class="card">
      <div class="card-title">Rata-Rata Absensi per Cluster</div>
      <img class="chart-img" src="data:image/png;base64,{charts['absen']}" alt="Absensi">
    </div>
    <div class="info-box">
      Semua cluster memiliki rata-rata absensi di atas 97% — kehadiran mahasiswa secara umum sangat baik.
      Cluster <b>Rendah</b> memiliki absensi sedikit lebih rendah (~98.4%) dibanding cluster lainnya,
      menjadi salah satu pembeda dalam composite scoring.
    </div>
  </div>

  <div class="tab-pane" id="eda-outlier">
    <div class="card">
      <div class="card-title">Deteksi Outlier — Boxplot per Fitur (Metode IQR)</div>
      <img class="chart-img" src="data:image/png;base64,{charts['outlier']}" alt="Outlier">
    </div>
    <div class="info-box">
      Outlier dideteksi menggunakan metode IQR (1.5×IQR) dan Z-score (|z|&gt;3).
      Beberapa outlier terdeteksi pada IPS dan absensi — ini adalah mahasiswa dengan
      performa ekstrem (sangat tinggi atau sangat rendah) yang justru memperkuat diferensiasi cluster.
    </div>
  </div>

  <div class="tab-pane" id="eda-corr">
    <div class="card">
      <div class="card-title">Heatmap Korelasi Antar Fitur (Pearson)</div>
      <img class="chart-img" src="data:image/png;base64,{charts['corr']}" alt="Korelasi">
    </div>
    <div class="info-box">
      IPS antar semester berkorelasi tinggi satu sama lain (≈0.7–0.9), menunjukkan
      konsistensi performa. Korelasi IPS dengan absensi relatif rendah, menunjukkan kehadiran
      dan nilai akademik adalah dimensi yang relatif independen — ini mendukung penggunaan
      keduanya sebagai fitur terpisah dalam clustering.
    </div>
  </div>

  <div class="tab-pane" id="eda-courses">
    <div class="card">
      <div class="card-title">Pilih Cluster → Lihat Rata-Rata Nilai Mata Kuliah</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px" id="course-cluster-tabs"></div>
      <canvas id="course-bar-canvas" height="340"></canvas>
    </div>
  </div>
</div>
</div>

<!-- ════ PAGE: PRINCALS ════ -->
<div class="page" id="page-princals">
<div style="padding:28px 32px 48px">
  <div class="section-h">Transformasi <em>PRINCALS</em></div>
  <div class="info-box">
    <b>PRINCALS (Principal Components Analysis by ALS)</b> melakukan reduksi dimensi dengan
    optimal scaling non-linear untuk data campuran numerik + kategorik.
    Implementasi menggunakan rank-based transformation sebagai aproksimasi ALS,
    dilanjutkan PCA. Komponen dipilih berdasarkan <b>cumulative variance ≥ 80%</b>.
  </div>
  <div class="metric-row">
    <div class="metric-card"><div class="metric-accent" style="background:var(--blue)"></div>
      <div class="metric-val">{len(feat_cols)}</div><div class="metric-lbl">Dimensi Awal</div></div>
    <div class="metric-card"><div class="metric-accent" style="background:var(--green)"></div>
      <div class="metric-val">{n_comp_j}</div><div class="metric-lbl">Komponen Dipilih</div></div>
    <div class="metric-card"><div class="metric-accent" style="background:var(--yellow)"></div>
      <div class="metric-val">{cumvar_j}%</div><div class="metric-lbl">Cumulative Variance</div></div>
    <div class="metric-card"><div class="metric-accent" style="background:var(--orange)"></div>
      <div class="metric-val">81.3%</div><div class="metric-lbl">Informasi Terjaga</div></div>
  </div>
  <div class="card">
    <div class="card-title">Scree Plot & Cumulative Variance Explained</div>
    <img class="chart-img" src="data:image/png;base64,{charts['princals']}" alt="PRINCALS Scree">
  </div>
  <div class="card">
    <div class="card-title">Scatter Plot PC1 vs PC2 (pewarnaan per cluster)</div>
    <canvas id="pc-canvas" height="380"></canvas>
    <div class="legend" id="pc-legend"></div>
  </div>
</div>
</div>

<!-- ════ PAGE: CLUSTERING ════ -->
<div class="page" id="page-clustering">
<div style="padding:28px 32px 48px">
  <div class="section-h">Hasil <em>Clustering</em></div>
  <div class="card">
    <div class="card-title">🫧 Bubble Chart — PC1 vs PC2 (ukuran bubble = IPS kumulatif, klik untuk detail)</div>
    <canvas id="bubble-canvas" height="400"></canvas>
    <div class="legend" id="bubble-legend"></div>
  </div>
  <div class="card">
    <div class="card-title">🌡️ Heatmap Profil Cluster</div>
    <div style="overflow-x:auto"><table class="heatmap" id="heatmap-table"></table></div>
  </div>
</div>
</div>

<!-- ════ PAGE: EVALUASI ════ -->
<div class="page" id="page-evaluasi">
<div style="padding:28px 32px 48px">
  <div class="section-h">Evaluasi <em>Cluster</em></div>
  <div class="info-box">
    <b>Silhouette Coefficient</b> → nilai ≥ 0.3 diterima, semakin mendekati 1 semakin baik. &nbsp;·&nbsp;
    <b>BSS/TSS</b> → ≥ 50% cukup baik, ≥ 75% sangat baik. Model terpilih: <b>KMeans k=5</b>
    (Silhouette=0.3485, BSS/TSS=74.46%).
  </div>
  <div class="card">
    <div class="card-title">Silhouette & BSS/TSS semua metode k=2–10</div>
    <img class="chart-img" src="data:image/png;base64,{charts['eval']}" alt="Evaluasi">
  </div>
  <div class="card">
    <div class="card-title">Heatmap Silhouette Coefficient</div>
    <canvas id="eval-heatmap-canvas" height="260"></canvas>
  </div>
</div>
</div>

<!-- ════ PAGE: MAHASISWA ════ -->
<div class="page" id="page-mahasiswa">
<div style="padding:28px 32px 48px">
  <div class="section-h">Data <em>Mahasiswa</em></div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">
    <input class="search-input" id="stu-search" placeholder="🔍 Cari nama atau NRP..."
           style="flex:1;min-width:200px" oninput="renderStudents()">
    <select id="stu-filter" onchange="renderStudents()"
            style="padding:10px 14px;background:var(--surface);border:1px solid var(--border);
                   border-radius:8px;color:#e8f4fd;font-size:.85rem;font-family:var(--font)">
      <option value="Semua">Semua Cluster</option>
    </select>
  </div>
  <div style="font-size:.78rem;color:var(--muted);margin-bottom:14px" id="stu-count"></div>
  <div class="stu-grid" id="stu-grid"></div>
</div>
</div>

<!-- ════ PAGE: REKOMENDASI ════ -->
<div class="page" id="page-rekomendasi">
<div style="padding:28px 32px 48px">
  <div class="section-h">Rekomendasi <em>untuk Dosen</em></div>
  <div class="advice-grid" id="advice-grid"></div>
  <div class="card" style="margin-top:20px">
    <div class="card-title">Ringkasan Performa per Cluster</div>
    <canvas id="summary-radar" height="350"></canvas>
  </div>
</div>
</div>

</main>
</div>

<!-- Tooltip -->
<div class="tooltip" id="tooltip"></div>

<!-- Modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div id="modal-body"></div>
  </div>
</div>

<div class="footer">
  Academic Clustering Dashboard &nbsp;·&nbsp;
  <span>PRINCALS + FCM + FPCM + MFPCM + K-Means + K-Medoids + DBSCAN</span>
  &nbsp;·&nbsp; by <span>Naufal</span>
</div>

<script>
// ── Data ──
const STUDENTS = {students_j};
const SUMMARY  = {summary_j};
const COLORS   = {color_j};
const LABELS   = {labels_j};
const ADVICE   = {advice_j};
const CHARTS   = {charts_j};
const EVR      = {evr_j};

const MPL_BG = '#141824', MPL_SURF = '#1a2130', MPL_BORDER = '#2a3344';

// ── Navigation ──
function showPage(id, btn) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sb-nav-item').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  btn.classList.add('active');
  if (id==='overview')    setTimeout(()=>{{drawBar();drawIpsBar();}},50);
  if (id==='clustering')  setTimeout(()=>{{drawBubble(STUDENTS);buildHeatmap();drawPcCanvas();}},50);
  if (id==='princals')    setTimeout(()=>{{drawPcCanvas();buildBubbleLegend('pc-legend');}},50);
  if (id==='evaluasi')    setTimeout(drawEvalHeatmap,50);
  if (id==='rekomendasi') setTimeout(drawRadar,50);
  if (id==='eda')         setTimeout(()=>{{buildCourseClusterTabs();drawCourseBar(LABELS[0]);}},50);
}}

// ── Tabs ──
function switchTab(group, id, btn) {{
  document.querySelectorAll('#eda-tabs .tab-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('[id^="'+group+'-"]').forEach(p=>p.classList.remove('active'));
  document.getElementById(group+'-'+id).classList.add('active');
  if(id==='courses') setTimeout(()=>{{buildCourseClusterTabs();drawCourseBar(LABELS[0]);}},50);
}}

// ── Metric cards ──
(function buildMetrics(){{
  const n=STUDENTS.length, avgIPS=(STUDENTS.reduce((s,st)=>s+st.ips,0)/n).toFixed(3);
  const data=[
    {{val:n,lbl:'Total Mahasiswa',c:'#4da6ff'}},
    {{val:LABELS.length,lbl:'Jumlah Cluster',c:'#43e97b'}},
    {{val:avgIPS,lbl:'Rata-Rata IPK',c:'#f9ca24'}},
    {{val:'0.3485',lbl:'Silhouette',c:'#fd9644'}},
    {{val:'74.46%',lbl:'BSS/TSS',c:'#fc5c65'}},
  ];
  document.getElementById('metric-row').innerHTML = data.map(d=>`
    <div class="metric-card">
      <div class="metric-accent" style="background:${{d.c}}"></div>
      <div class="metric-val">${{d.val}}</div>
      <div class="metric-lbl">${{d.lbl}}</div>
    </div>`).join('');
}})();

// ── Canvas helpers ──
function initCanvas(id, h) {{
  const c=document.getElementById(id), dpr=window.devicePixelRatio||1;
  const W=c.clientWidth||c.parentElement.clientWidth||600, H=parseInt(h)||300;
  c.width=W*dpr; c.height=H*dpr;
  const ctx=c.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  return {{ctx,W,H,dpr}};
}}

// ── Bar chart ──
function drawBar(){{
  const {{ctx,W,H}}=initCanvas('bar-canvas',220);
  const lbl=LABELS, cnt=lbl.map(l=>STUDENTS.filter(s=>s.cluster===l).length);
  const max=Math.max(...cnt), pad=40, gap=14;
  const bH=Math.min(36,(H-2*pad-gap*(lbl.length-1))/lbl.length);
  ctx.fillStyle=MPL_BG; ctx.fillRect(0,0,W,H);
  lbl.forEach((l,i)=>{{
    const by=pad+i*(bH+gap), bw=(cnt[i]/max)*(W-2*pad-80);
    ctx.fillStyle=COLORS[l]+'cc';
    ctx.beginPath(); ctx.roundRect(pad,by,bw,bH,[3]); ctx.fill();
    ctx.fillStyle='#c8d8e8'; ctx.font=`600 11px DM Sans`; ctx.textAlign='left';
    ctx.fillText(cnt[i]+' mhs', pad+bw+8, by+bH/2+4);
    ctx.fillStyle='#8892a4'; ctx.font=`10px DM Sans`;
    ctx.fillText(l, pad-2, by-5);
  }});
}}

function drawIpsBar(){{
  const {{ctx,W,H}}=initCanvas('ips-canvas',220);
  const lbl=LABELS, vals=lbl.map(l=>{{const sub=STUDENTS.filter(s=>s.cluster===l);return sub.reduce((a,s)=>a+s.ips,0)/sub.length;}});
  const pad=40, gap=14, bW=Math.min(52,(W-2*pad-gap*(lbl.length-1))/lbl.length);
  const yMin=2.8, yMax=4.0;
  const scY=v=>H-pad-(v-yMin)/(yMax-yMin)*(H-2*pad);
  ctx.fillStyle=MPL_BG; ctx.fillRect(0,0,W,H);
  // Grid
  [3.0,3.2,3.4,3.6,3.8,4.0].forEach(y=>{{
    const py=scY(y); ctx.strokeStyle='#1e2a3a'; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad,py);ctx.lineTo(W-10,py);ctx.stroke();
    ctx.fillStyle='#5a6a80';ctx.font='9px DM Sans';ctx.textAlign='right';
    ctx.fillText(y.toFixed(1),pad-4,py+3);
  }});
  lbl.forEach((l,i)=>{{
    const bx=pad+i*(bW+gap), bh=H-pad-scY(vals[i]), by=scY(vals[i]);
    ctx.fillStyle=COLORS[l]+'cc';
    ctx.beginPath();ctx.roundRect(bx,by,bW,bh,[4,4,0,0]);ctx.fill();
    ctx.fillStyle='#c8d8e8';ctx.font='bold 10px DM Sans';ctx.textAlign='center';
    ctx.fillText(vals[i].toFixed(3),bx+bW/2,by-5);
    ctx.fillStyle='#8892a4';ctx.font='9px DM Sans';
    const short=l.replace('Sangat ','S.').replace('Tinggi','T').replace('Rendah','R').replace('Sedang','Sed').replace('Cukup','C');
    ctx.fillText(short,bx+bW/2,H-pad+13);
  }});
}}

// ── PC scatter ──
function drawPcCanvas(){{
  const id=document.getElementById('page-clustering').classList.contains('active')?'bubble-canvas':'pc-canvas';
  const el=document.getElementById(id); if(!el) return;
  const {{ctx,W,H}}=initCanvas(id,380);
  const pad=44;
  const x1=Math.min(...STUDENTS.map(s=>s.pc1))-0.5, x2=Math.max(...STUDENTS.map(s=>s.pc1))+0.5;
  const y1=Math.min(...STUDENTS.map(s=>s.pc2))-0.5, y2=Math.max(...STUDENTS.map(s=>s.pc2))+0.5;
  const sX=v=>pad+(v-x1)/(x2-x1)*(W-2*pad);
  const sY=v=>H-pad-(v-y1)/(y2-y1)*(H-2*pad);
  ctx.fillStyle=MPL_BG; ctx.fillRect(0,0,W,H);
  // Axes
  ctx.strokeStyle='#2a3344'; ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.stroke();
  ctx.beginPath();ctx.moveTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#5a6a80';ctx.font='10px DM Sans';ctx.textAlign='center';
  ctx.fillText(`PC1 (${{EVR[0]?EVR[0].toFixed(1):'?'}}%)`,W/2,H-pad+16);
  ctx.save();ctx.translate(14,H/2);ctx.rotate(-Math.PI/2);
  ctx.fillText(`PC2 (${{EVR[1]?EVR[1].toFixed(1):'?'}}%)`,0,0);ctx.restore();
  STUDENTS.forEach(s=>{{
    const cx=sX(s.pc1), cy=sY(s.pc2), r=7;
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.fillStyle=COLORS[s.cluster]+'cc';
    ctx.strokeStyle=COLORS[s.cluster]; ctx.lineWidth=1.5;
    ctx.fill();ctx.stroke();
  }});
  if(id==='pc-canvas') buildBubbleLegend('pc-legend');
}}

// ── Bubble Chart ──
function drawBubble(filtered){{
  const {{ctx,W,H}}=initCanvas('bubble-canvas',400);
  const pad=44;
  const pts=filtered;
  if(!pts.length) return;
  const x1=Math.min(...pts.map(s=>s.pc1))-0.5, x2=Math.max(...pts.map(s=>s.pc1))+0.5;
  const y1=Math.min(...pts.map(s=>s.pc2))-0.5, y2=Math.max(...pts.map(s=>s.pc2))+0.5;
  const sX=v=>pad+(v-x1)/(x2-x1)*(W-2*pad);
  const sY=v=>H-pad-(v-y1)/(y2-y1)*(H-2*pad);
  ctx.fillStyle=MPL_BG; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#2a3344';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.stroke();
  ctx.beginPath();ctx.moveTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#5a6a80';ctx.font='10px DM Sans';ctx.textAlign='center';
  ctx.fillText('PC1',W/2,H-6);
  ctx.save();ctx.translate(14,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('PC2',0,0);ctx.restore();
  pts.forEach(s=>{{
    const cx=sX(s.pc1), cy=sY(s.pc2), r=s.bubble_size/2.2;
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.fillStyle=COLORS[s.cluster]+'99';
    ctx.strokeStyle=COLORS[s.cluster]; ctx.lineWidth=1.5;
    ctx.fill();ctx.stroke();
  }});
  const canvas=document.getElementById('bubble-canvas');
  canvas._pts=pts.map(s=>{{return{{...s,cx:sX(s.pc1),cy:sY(s.pc2),r:s.bubble_size/2.2}}}});
  buildBubbleLegend('bubble-legend');
}}

function buildBubbleLegend(id){{
  const el=document.getElementById(id); if(!el) return;
  el.innerHTML=LABELS.map(l=>`<div class="legend-item">
    <div class="legend-dot" style="background:${{COLORS[l]}}"></div>${{l}}</div>`).join('');
}}

// Bubble interactions
(function(){{
  const canvas=document.getElementById('bubble-canvas'), tip=document.getElementById('tooltip');
  function pos(e){{const r=canvas.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}}
  function hit(px,py){{return(canvas._pts||[]).find(p=>Math.hypot(p.cx-px,p.cy-py)<=p.r+4)}}
  canvas.addEventListener('mousemove',e=>{{
    const[px,py]=pos(e),h=hit(px,py);
    if(h){{canvas.style.cursor='pointer';tip.style.display='block';
      tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';
      tip.innerHTML=`<b>${{h.name}}</b><br>NRP: ${{h.id}}<br>IPS: ${{h.ips.toFixed(3)}}<br>Cluster: ${{h.cluster}}`;
    }}else{{canvas.style.cursor='crosshair';tip.style.display='none';}}
  }});
  canvas.addEventListener('mouseleave',()=>tip.style.display='none');
  canvas.addEventListener('click',e=>{{const[px,py]=pos(e),h=hit(px,py);if(h)openModal(h);}});
}})();

// ── Heatmap table ──
function buildHeatmap(){{
  const feats=Object.keys(Object.values(SUMMARY)[0]).filter(k=>k!=='count');
  let html=`<thead><tr><th>Cluster</th><th>Jumlah</th>${{feats.map(f=>`<th>${{f}}</th>`).join('')}}</tr></thead><tbody>`;
  LABELS.forEach(l=>{{
    const s=SUMMARY[l], c=COLORS[l];
    html+=`<tr><td style="font-weight:700;color:${{c}};border-left:3px solid ${{c}}">${{l}}</td><td>${{s.count}}</td>`;
    feats.forEach(f=>html+=`<td>${{s[f]!==null&&s[f]!==undefined?s[f]:'-'}}</td>`);
    html+='</tr>';
  }});
  document.getElementById('heatmap-table').innerHTML=html+'</tbody>';
}}

// ── Eval heatmap ──
const EVAL_DATA = {{
  methods:['KMeans','FCM','FPCM','MFPCM','KMedoids','PCM'],
  ks:[2,3,4,5,6,7,8,9,10],
  sil:{{
    'KMeans':[0.3227,0.3249,0.3345,0.3485,0.3297,0.3107,0.3216,0.3170,0.3183],
    'FCM':[0.3227,0.3043,0.3369,0.3485,0.3273,0.3243,0.2937,0.2610,0.2476],
    'FPCM':[0.3227,0.3043,0.3260,0.2673,0.3238,0.2574,0.2732,0.2592,0.1593],
    'MFPCM':[0.3227,0.3043,0.2801,0.2451,0.2839,0.2839,0.3227,0.3227,0.3227],
    'KMedoids':[0.3054,0.2513,0.3247,0.2598,0.1583,0.2069,0.2713,0.2660,0.2713],
    'PCM':[0.1309,0.1309,0.1093,-0.0304,0.0583,-0.0246,0.0843,0.0687,0.2244]
  }}
}};
function drawEvalHeatmap(){{
  const {{ctx,W,H}}=initCanvas('eval-heatmap-canvas',260);
  const methods=EVAL_DATA.methods, ks=EVAL_DATA.ks;
  const padL=72, padT=36, padR=20, padB=28;
  const cellW=(W-padL-padR)/ks.length, cellH=(H-padT-padB)/methods.length;
  ctx.fillStyle=MPL_BG;ctx.fillRect(0,0,W,H);
  methods.forEach((m,i)=>{{
    ks.forEach((k,j)=>{{
      const v=EVAL_DATA.sil[m]?.[j]||0;
      const t=Math.max(0,Math.min(1,(v+0.1)/0.6));
      const r=Math.round(13+t*(77-13)),g=Math.round(27+t*(166-27)),b=Math.round(42+t*(255-42));
      ctx.fillStyle=`rgb(${{r}},${{g}},${{b}})`;
      ctx.fillRect(padL+j*cellW,padT+i*cellH,cellW-1,cellH-1);
      ctx.fillStyle=v>0.3?'white':'#8892a4';
      ctx.font=`8.5px DM Sans`;ctx.textAlign='center';
      ctx.fillText(v>-0.1?v.toFixed(3):'',padL+j*cellW+cellW/2,padT+i*cellH+cellH/2+3);
    }});
    ctx.fillStyle='#8892a4';ctx.font='10px DM Sans';ctx.textAlign='right';
    ctx.fillText(m,padL-6,padT+i*cellH+cellH/2+3);
  }});
  ks.forEach((k,j)=>{{
    ctx.fillStyle='#5a6a80';ctx.font='9px DM Sans';ctx.textAlign='center';
    ctx.fillText('k='+k,padL+j*cellW+cellW/2,padT-8);
  }});
  // threshold line
  ctx.strokeStyle='#fc5c65';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
  // just label
  ctx.fillStyle='#fc5c65';ctx.font='8px DM Sans';ctx.textAlign='left';
  ctx.fillText('≥0.3',W-padR-28,padT+2);
  ctx.setLineDash([]);
}}

// ── Course bars ──
let activeCourseCluster = LABELS[0];
function buildCourseClusterTabs(){{
  const wrap=document.getElementById('course-cluster-tabs');
  if(!wrap||wrap.children.length) return;
  LABELS.forEach(l=>{{
    const btn=document.createElement('button');
    btn.className='tab-btn'+(l===activeCourseCluster?' active':'');
    btn.style.cssText='border:1px solid var(--border);border-radius:6px;padding:5px 14px';
    btn.textContent=l;
    btn.onclick=()=>{{
      activeCourseCluster=l;
      document.querySelectorAll('#course-cluster-tabs .tab-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      drawCourseBar(l);
    }};
    wrap.appendChild(btn);
  }});
}}
function drawCourseBar(lbl){{
  const {{ctx,W,H}}=initCanvas('course-bar-canvas',340);
  const sub=STUDENTS.filter(s=>s.cluster===lbl);
  if(!sub.length) return;
  // aggregate course means
  const allCourses={{}};
  sub.forEach(s=>Object.entries(s.courses).forEach(([k,v])=>{{
    if(!allCourses[k]) allCourses[k]=[]; allCourses[k].push(v);
  }}));
  let means=Object.entries(allCourses).map(([k,vs])=>([k,vs.reduce((a,v)=>a+v,0)/vs.length]));
  means.sort((a,b)=>b[1]-a[1]); means=means.slice(0,14);
  const pad=36, labelW=160, barMax=W-labelW-pad-50;
  const rowH=Math.max(18,(H-2*pad)/means.length);
  ctx.fillStyle=MPL_BG;ctx.fillRect(0,0,W,H);
  const color=COLORS[lbl];
  means.forEach(([name,val],i)=>{{
    const by=pad+i*rowH, bw=(val/4)*barMax;
    ctx.fillStyle=color+'33';
    ctx.fillRect(labelW,by,barMax,rowH-3);
    ctx.fillStyle=color+'cc';
    ctx.beginPath();ctx.roundRect(labelW,by,bw,rowH-3,[2]);ctx.fill();
    ctx.fillStyle='#8E9BAE';ctx.font=`${{Math.min(10,rowH-4)}}px DM Sans`;ctx.textAlign='right';
    const short=name.length>22?name.slice(0,22)+'…':name;
    ctx.fillText(short,labelW-6,by+rowH/2+3);
    ctx.fillStyle='#566A7F';ctx.font=`bold ${{Math.min(10,rowH-4)}}px DM Sans`;ctx.textAlign='left';
    ctx.fillText(val.toFixed(2),labelW+bw+6,by+rowH/2+3);
  }});
  ctx.fillStyle='var(--blue)';ctx.font='bold 11px DM Sans';ctx.textAlign='center';
  ctx.fillText(`Rata-Rata Nilai — Cluster ${{lbl}}`,W/2,pad-14);
}}

// ── Advice ──
(function buildAdvice(){{
  const TIPS={{
    'Sangat Tinggi':{{icon:'🏆',c:'#28C76F',tips:['Proyek riset & kompetisi nasional/internasional.','Fast-track S2 atau beasiswa.','Libatkan sebagai tutor sebaya.']}},
    'Tinggi':{{icon:'⭐',c:'#00CFE8',tips:['Materi pengayaan & sertifikasi profesional.','Dorong partisipasi seminar.','Rekomendasikan program magang industri.']}},
    'Sedang':{{icon:'📘',c:'#7367F0',tips:['Bimbingan rutin setiap 2 minggu.','Sumber belajar tambahan.','Fokus perbaikan mata kuliah lemah.']}},
    'Cukup':{{icon:'⚠️',c:'#FF9F43',tips:['Konseling akademik 1×/bulan.','Program remedial terstruktur.','Manfaatkan kehadiran baik untuk keaktifan.']}},
    'Rendah':{{icon:'🚨',c:'#EA5455',tips:['Intervensi segera & monitoring mingguan.','Koordinasi dengan orang tua/wali.','Evaluasi beban SKS sesuai kemampuan.']}},
  }};
  const wrap=document.getElementById('advice-grid');
  LABELS.forEach(l=>{{
    const a=TIPS[l]||{{icon:'📋',c:'#8E9BAE',tips:[]}}, n=STUDENTS.filter(s=>s.cluster===l).length;
    const sub=STUDENTS.filter(s=>s.cluster===l);
    const avgIPS=(sub.reduce((s,st)=>s+st.ips,0)/sub.length).toFixed(3);
    wrap.innerHTML+=`<div class="advice-card" style="background:${{a.c}}12;border-color:${{a.c}}44">
      <h4 style="color:${{a.c}}">${{a.icon}} ${{l}} <span style="font-weight:300;font-size:.78rem">(n=${{n}}, IPK=${{avgIPS}})</span></h4>
      <ul>${{a.tips.map(t=>`<li>${{t}}</li>`).join('')}}</ul>
    </div>`;
  }});
}})();

// ── Radar summary ──
function drawRadar(){{
  const {{ctx,W,H}}=initCanvas('summary-radar',350);
  const metrics=['IPS Kum','IPS S1','IPS S2','IPS S3','IPS S4','Absensi'];
  const n=metrics.length, angles=metrics.map((_,i)=>i*2*Math.PI/n-Math.PI/2);
  const cx=W/2, cy=H/2, maxR=Math.min(W,H)/2-50;
  ctx.fillStyle=MPL_BG;ctx.fillRect(0,0,W,H);
  // Grid
  [0.25,0.5,0.75,1].forEach(t=>{{
    ctx.strokeStyle='#F0F0F0';ctx.lineWidth=1;ctx.beginPath();
    angles.forEach((a,i)=>{{const r=maxR*t;const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
    ctx.closePath();ctx.stroke();
  }});
  angles.forEach(a=>{{ctx.strokeStyle='#F0F0F0';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+maxR*Math.cos(a),cy+maxR*Math.sin(a));ctx.stroke();}});
  metrics.forEach((m,i)=>{{
    const a=angles[i], x=cx+(maxR+20)*Math.cos(a), y=cy+(maxR+20)*Math.sin(a);
    ctx.fillStyle='#8E9BAE';ctx.font='9px DM Sans';ctx.textAlign='center';ctx.fillText(m,x,y+3);
  }});
  LABELS.forEach(l=>{{
    const sub=STUDENTS.filter(s=>s.cluster===l);
    if(!sub.length)return;
    const avg=(key)=>sub.reduce((a,s)=>a+(s.ips_sem?.[key]||s.ips),0)/sub.length;
    const vals=[
      (sub.reduce((a,s)=>a+s.ips,0)/sub.length-2.5)/1.5,
      (avg('Sem 1')-2.5)/1.5,(avg('Sem 2')-2.5)/1.5,
      (avg('Sem 3')-2.5)/1.5,(avg('Sem 4')-2.5)/1.5,
      (sub.reduce((a,s)=>a+(s.absensi||99),0)/sub.length-95)/5,
    ].map(v=>Math.max(0,Math.min(1,v)));
    ctx.strokeStyle=COLORS[l];ctx.lineWidth=2;ctx.fillStyle=COLORS[l]+'22';
    ctx.beginPath();
    angles.forEach((a,i)=>{{const r=maxR*vals[i];const x=cx+r*Math.cos(a),y=cy+r*Math.sin(a);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}});
    ctx.closePath();ctx.stroke();ctx.fill();
  }});
  // Legend
  let lx=cx-LABELS.length*55;
  LABELS.forEach(l=>{{
    ctx.fillStyle=COLORS[l];ctx.beginPath();ctx.arc(lx,H-16,5,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#8E9BAE';ctx.font='9px DM Sans';ctx.textAlign='left';ctx.fillText(l,lx+9,H-12);
    lx+=Math.max(60,l.length*8);
  }});
}}

// ── Student list ──
(function buildStudentFilter(){{
  const sel=document.getElementById('stu-filter');
  LABELS.forEach(l=>{{const o=document.createElement('option');o.value=l;o.textContent=l;sel.appendChild(o);}});
}})();
function renderStudents(){{
  const q=document.getElementById('stu-search').value.toLowerCase();
  const f=document.getElementById('stu-filter').value;
  let list=STUDENTS;
  if(f!=='Semua') list=list.filter(s=>s.cluster===f);
  if(q) list=list.filter(s=>s.name.toLowerCase().includes(q)||String(s.id).includes(q));
  document.getElementById('stu-count').textContent=`Menampilkan ${{list.length}} mahasiswa`;
  const wrap=document.getElementById('stu-grid'); wrap.innerHTML='';
  list.forEach(s=>{{
    const c=COLORS[s.cluster]||'#888';
    const card=document.createElement('div');
    card.className='stu-card';
    card.style.borderTopColor=c;
    card.innerHTML=`<div class="stu-name">${{s.name}}</div>
      <div class="stu-nrp">${{s.id}} · ${{s.angkatan}}</div>
      <div class="stu-meta">
        <span style="background:${{c}}22;color:${{c}};padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:700">${{s.cluster}}</span>
        <span style="font-size:.78rem;color:var(--muted)">IPK <b style="color:var(--text)">${{s.ips.toFixed(3)}}</b></span>
      </div>`;
    card.onclick=()=>openModal(s);
    wrap.appendChild(card);
  }});
  if(!list.length) wrap.innerHTML='<p style="color:var(--muted);padding:20px">Tidak ada data.</p>';
}}

// ── Modal ──
function openModal(s){{
  const c=COLORS[s.cluster]||'#888';
  const courseRows=Object.entries(s.courses||{{}})
    .sort((a,b)=>a[1]-b[1])
    .map(([n,v])=>{{
      const cls=v>=3.5?'grade-A':v>=3?'grade-B':v>=2.5?'grade-C':'grade-D';
      return `<tr><td>${{n}}</td><td class="course-grade ${{cls}}">${{v}}</td></tr>`;
    }}).join('');
  const semRows=Object.entries(s.ips_sem||{{}}).map(([k,v])=>`<tr><td>${{k}}</td><td class="grade-B">${{v}}</td></tr>`).join('');
  document.getElementById('modal-body').innerHTML=`
    <div class="modal-name">${{s.name}}</div>
    <span class="cl-badge" style="background:${{c}}">${{s.cluster}}</span>
    <div class="modal-grid">
      <div class="modal-item"><div class="mi-lbl">NRP</div><div class="mi-val" style="font-family:var(--mono)">${{s.id}}</div></div>
      <div class="modal-item"><div class="mi-lbl">Angkatan</div><div class="mi-val">${{s.angkatan}}</div></div>
      <div class="modal-item"><div class="mi-lbl">Jenis Kelamin</div><div class="mi-val">${{s.jk}}</div></div>
      <div class="modal-item"><div class="mi-lbl">Asal</div><div class="mi-val" style="font-size:.85rem">${{s.asal}}</div></div>
      <div class="modal-item" style="background:${{c}}18;border:1px solid ${{c}}44">
        <div class="mi-lbl">IPK Kumulatif</div>
        <div class="mi-val" style="color:${{c}};font-size:1.5rem">${{s.ips.toFixed(3)}}</div>
      </div>
      ${{s.absensi!==null?`<div class="modal-item"><div class="mi-lbl">Absensi Rata-Rata</div><div class="mi-val">${{s.absensi?.toFixed(1)}}%</div></div>`:''}}
    </div>
    ${{semRows?`<p style="font-size:.82rem;font-weight:600;color:var(--blue);margin:12px 0 8px">IPS per Semester</p>
      <table class="course-tbl"><thead><tr><th>Semester</th><th>IPS</th></tr></thead><tbody>${{semRows}}</tbody></table>`:''}}
    ${{courseRows?`<p style="font-size:.82rem;font-weight:600;color:var(--blue);margin:14px 0 8px">Nilai Mata Kuliah</p>
      <table class="course-tbl"><thead><tr><th>Mata Kuliah</th><th>Nilai</th></tr></thead><tbody>${{courseRows}}</tbody></table>`:''}}
  `;
  document.getElementById('modal-overlay').classList.add('active');
}}
function closeModal(){{document.getElementById('modal-overlay').classList.remove('active');}}
document.getElementById('modal-overlay').addEventListener('click',e=>{{if(e.target===e.currentTarget)closeModal();}});

// ── Init ──
window.addEventListener('DOMContentLoaded',()=>{{
  setTimeout(()=>{{drawBar();drawIpsBar();}},80);
  renderStudents();
}});
window.addEventListener('resize',()=>setTimeout(()=>{{drawBar();drawIpsBar();}},100));
</script>
</body>
</html>"""

    out_path = f'{OUT}/dashboard_enhanced.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[OK] Enhanced dashboard: {out_path}")
    return out_path

if __name__ == '__main__':
    build_dashboard_enhanced()
