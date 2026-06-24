"""
DASHBOARD STREAMLIT — Clustering Akademik Mahasiswa
Analisis berbasis PRINCALS + Fuzzy Clustering
By Naufal
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Clustering Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #566A7F;
}

/* Main bg */
.main { background: #FFFFFF; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* Global headings */
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: #7367F0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F5F3FF;
    border-right: 1px solid #F0F0F0;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] label span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #566A7F !important;
}
section[data-testid="stSidebar"] label:hover,
section[data-testid="stSidebar"] label:hover span,
section[data-testid="stSidebar"] label:hover p {
    color: #5E50EE !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.8rem;
}

/* Header hero */
.hero {
    background: linear-gradient(135deg, #F5F3FF 0%, #E8E4FF 100%);
    border: 1px solid #F0F0F0;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, #7367F022 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #7367F0;
    line-height: 1.1;
    margin: 0;
}
.hero-title span { color: #5E50EE; }
.hero-sub {
    color: #566A7F;
    font-size: 0.92rem;
    margin-top: 10px;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: #FFFFFF;
    border: 1px solid #7367F088;
    color: #7367F0;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 8px;
    margin-top: 12px;
}
.by-line {
    position: absolute;
    bottom: 18px; right: 24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #7367F088;
    letter-spacing: 0.08em;
}

/* Metric cards */
.metric-row { display: flex; gap: 14px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #FFFFFF;
    border: 1px solid #F0F0F0;
    box-shadow: 0 4px 12px #EAEAEA;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.metric-card:hover { border-color: #5E50EE; }
.metric-card .accent {
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 12px 0 0 12px;
}
.metric-card .mc-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #7367F0;
    line-height: 1;
}
.metric-card .mc-lbl {
    font-size: 0.75rem;
    color: #8E9BAE;
    margin-top: 6px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #566A7F;
    margin: 2rem 0 1rem;
    padding-bottom: 8px;
    border-bottom: 1px solid #F0F0F0;
}
.section-header span { color: #7367F0; }

/* Cluster badge */
.cl-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    color: white;
}

/* Student card */
.stu-card {
    background: #FFFFFF;
    border: 1px solid #F0F0F0;
    box-shadow: 0 2px 8px #EAEAEA;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: all .18s;
}
.stu-card:hover { border-color: #5E50EE; transform: translateX(3px); }

/* Tab styling */
div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #8E9BAE;
    font-size: 0.88rem;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7367F0;
    border-bottom-color: #7367F0;
}

/* Info box */
.info-box {
    background: #F5F3FF;
    border: 1px solid #E8E4FF;
    border-left: 3px solid #7367F0;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.84rem;
    color: #566A7F;
    line-height: 1.6;
}

/* Chart wrapper */
.chart-wrapper {
    background: #FFFFFF;
    border: 1px solid #F0F0F0;
    box-shadow: 0 4px 12px #EAEAEA;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: #8E9BAE;
    font-size: 0.78rem;
    border-top: 1px solid #F0F0F0;
    margin-top: 3rem;
    font-family: 'JetBrains Mono', monospace;
}
.footer span { color: #7367F0; }
</style>

""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df        = pd.read_pickle(os.path.join(BASE, 'output/df_labeled.pkl'))
    X_pc      = pd.read_pickle(os.path.join(BASE, 'output/X_princals.pkl'))
    X_scaled  = pd.read_pickle(os.path.join(BASE, 'output/X_scaled.pkl'))
    results   = pd.read_pickle(os.path.join(BASE, 'output/clustering_results.pkl'))
    feat_cols = pd.read_pickle(os.path.join(BASE, 'output/feature_cols.pkl')).tolist()
    var_info  = pd.read_pickle(os.path.join(BASE, 'output/princals_info.pkl'))
    best      = pd.read_pickle(os.path.join(BASE, 'output/best_model.pkl'))
    min_len   = min(len(df), len(X_pc))
    return (df.iloc[:min_len].reset_index(drop=True),
            X_pc.iloc[:min_len].reset_index(drop=True),
            X_scaled.iloc[:min_len].reset_index(drop=True),
            results, feat_cols, var_info, best)

df, X_pc, X_scaled, results, feat_cols, var_info, best = load_data()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PALETTE = {
    'Sangat Tinggi': '#28C76F',
    'Tinggi'       : '#00CFE8',
    'Sedang'       : '#7367F0',
    'Cukup'        : '#FF9F43',
    'Rendah'       : '#EA5455',
    'Sangat Rendah': '#EA5455',
    'Kritis'       : '#EA5455',
}
ORDER   = ['Sangat Tinggi', 'Tinggi', 'Sedang', 'Cukup', 'Rendah']
SEM_IPS = ['nilai ips', 'IPS', 'IPS.1', 'IPS.2']
SEM_LBL = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
ABSEN   = ['Rata-Rata Absen Mahasiswa','ABSENSI RATA RATA',
           'ABSENSI RATA_RATA','ABSENSI RATA RATA.1','ABSENSI RATA RATA.2']
ABSEN_L = ['Sem 1','Sem 2','Sem 3','Sem 4','Sem 5']
SKIP    = {'NRP','Nama Mahasiswa','Angkatan Tahun','Prodi','JK','Asal Kab/Kota',
           'cluster','cluster_label','Prodi_enc','Asal Kab/Kota_enc'}
SKIP_IA = {c for c in df.columns if 'IPS' in c.upper() or 'ABSEN' in c.upper()}
COURSE  = [c for c in df.columns if c not in SKIP and c not in SKIP_IA
           and not c.endswith('_enc') and not c.endswith('.1')]
 
MPL_STYLE = {
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#EAEAEA',
    'axes.labelcolor':  '#566A7F',
    'xtick.color':      '#566A7F',
    'ytick.color':      '#566A7F',
    'text.color':       '#566A7F',
    'grid.color':       '#F0F0F0',
    'grid.alpha':       0.8,
}
plt.rcParams.update(MPL_STYLE)

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.2rem;color:#7367F0'>
            🎓 Clustering<br><span style='color:#5E50EE'>Akademik</span>
        </div>
        <div style='font-size:0.72rem;color:#8E9BAE;margin-top:4px;
                    font-family:"JetBrains Mono",monospace'>by Naufal</div>
    </div>
    <hr style='border:none;border-top:1px solid #F0F0F0;margin:8px 0 16px'>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigasi", [
        "🏠  Overview",
        "🔬  EDA",
        "🌀  PRINCALS",
        "🫧  Clustering",
        "📊  Evaluasi",
        "👥  Mahasiswa",
        "💡  Rekomendasi",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""<div style='font-size:0.75rem;color:#566A7F;line-height:1.8'>
        <b style='color:#7367F0'>Model Terbaik</b><br>
        Method &nbsp;&nbsp;: KMeans<br>
        k &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 5 cluster<br>
        Silhouette: 0.3485<br>
        BSS/TSS &nbsp;: 74.46%
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style='font-size:0.7rem;color:#8E9BAE;text-align:center;
                font-family:"JetBrains Mono",monospace'>
        PRINCALS · FCM · PCM · FPCM<br>MFPCM · DBSCAN · K-Medoids
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO (shown on all pages)
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Academic<br><span>Clustering</span> Dashboard</p>
    <p class="hero-sub">Segmentasi mahasiswa berbasis PRINCALS + 7 metode clustering &nbsp;·&nbsp; Analisis multi-kriteria</p>
    <span class="hero-badge">PRINCALS</span>
    <span class="hero-badge">FCM / FPCM / MFPCM</span>
    <span class="hero-badge">K-Means / K-Medoids</span>
    <span class="hero-badge">DBSCAN</span>
    <div class="by-line">by Naufal</div>
</div>
""", unsafe_allow_html=True)

page = nav.split("  ")[1]

# ═══════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════
if page == "Overview":
    # Metric cards
    n_total   = len(df)
    avg_ips   = df['Rata-Rata IPS'].mean()
    n_cluster = df['cluster_label'].nunique()
    sil_val   = 0.3485
    bss_val   = 74.46

    m_data = [
        {'val': n_total,        'lbl': 'Total Mahasiswa', 'color': '#4da6ff'},
        {'val': n_cluster,      'lbl': 'Jumlah Cluster',  'color': '#43e97b'},
        {'val': f'{avg_ips:.3f}','lbl': 'Rata-Rata IPK', 'color': '#f9ca24'},
        {'val': f'{sil_val:.4f}','lbl': 'Silhouette',    'color': '#fd9644'},
        {'val': f'{bss_val:.1f}%','lbl': 'BSS/TSS',      'color': '#fc5c65'},
    ]
    cards_html = '<div class="metric-row">'
    for m in m_data:
        cards_html += f"""<div class="metric-card">
            <div class="accent" style="background:{m['color']}"></div>
            <div class="mc-val">{m['val']}</div>
            <div class="mc-lbl">{m['lbl']}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="section-header">Distribusi <span>Cluster</span></div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        labels_ord = [l for l in ORDER if l in df['cluster_label'].values]
        counts     = [df[df['cluster_label']==l].shape[0] for l in labels_ord]
        colors_bar = [PALETTE[l] for l in labels_ord]
        bars = ax.barh(labels_ord, counts, color=colors_bar, height=0.55, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{cnt} mhs', va='center', fontsize=10, color='#c8d8e8', fontweight='600')
        ax.set_xlabel('Jumlah Mahasiswa')
        ax.set_xlim(0, max(counts) * 1.25)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top','right','left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-header">Rata-Rata <span>IPS per Cluster</span></div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ips_means = [df[df['cluster_label']==l]['Rata-Rata IPS'].mean() for l in labels_ord]
        scatter_colors = [PALETTE[l] for l in labels_ord]
        bars2 = ax.bar(range(len(labels_ord)), ips_means, color=scatter_colors, alpha=0.85, width=0.6)
        ax.set_xticks(range(len(labels_ord)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels_ord], fontsize=8.5)
        ax.set_ylim(2.8, 4.0)
        ax.set_ylabel('Rata-Rata IPS Kumulatif')
        for bar, v in zip(bars2, ips_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=9, color='#c8d8e8', fontweight='600')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # IPS trend per semester
    st.markdown('<div class="section-header">Tren <span>IPS per Semester</span></div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for lbl in labels_ord:
        sub = df[df['cluster_label'] == lbl]
        vals = [sub[c].mean() for c in SEM_IPS if c in df.columns]
        ax.plot(SEM_LBL[:len(vals)], vals, 'o-', color=PALETTE[lbl],
                linewidth=2.5, markersize=7, label=lbl, alpha=0.9)
        ax.fill_between(SEM_LBL[:len(vals)], vals, alpha=0.05, color=PALETTE[lbl])
    ax.set_ylabel('Rata-Rata IPS')
    ax.set_ylim(3.0, 4.1)
    ax.legend(fontsize=9, framealpha=0.1, labelcolor='#c8d8e8')
    ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════
elif page == "EDA":
    st.markdown('<div class="section-header">Exploratory Data <span>Analysis</span></div>',
                unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
    EDA dilakukan untuk memahami karakteristik data sebelum clustering:
    distribusi variabel, deteksi outlier, korelasi antar fitur, dan pola awal.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊  Distribusi IPS", "📉  Distribusi Absensi",
        "🔴  Outlier", "🌡️  Korelasi", "📚  Nilai Mata Kuliah"
    ])

    # ── TAB 1: Distribusi IPS ──
    with tab1:
        st.markdown("#### Distribusi IPS per Semester & Rata-Rata IPS Kumulatif")
        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        ips_plot_cols = SEM_IPS + ['Rata-Rata IPS']
        ips_plot_lbls = SEM_LBL + ['Rata-Rata IPS']
        axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(5)]
        for ax, col, lbl in zip(axes, ips_plot_cols, ips_plot_lbls):
            if col not in df.columns: continue
            vals = df[col].dropna()
            ax.hist(vals, bins=10, color='#4da6ff', alpha=0.7, edgecolor='#0d1b2a', linewidth=0.8)
            ax.axvline(vals.mean(), color='#f9ca24', linewidth=2, linestyle='--', label=f'μ={vals.mean():.3f}')
            ax.axvline(vals.median(), color='#43e97b', linewidth=1.5, linestyle=':', label=f'med={vals.median():.3f}')
            ax.set_title(lbl, fontsize=10, fontweight='600', color='#e8f4fd')
            ax.set_xlabel('Nilai IPS', fontsize=8)
            ax.legend(fontsize=7.5, framealpha=0.1, labelcolor='#c8d8e8')
            ax.spines[['top','right']].set_visible(False)
            skew = stats.skew(vals)
            ax.text(0.97, 0.92, f'skew={skew:.2f}', transform=ax.transAxes,
                    ha='right', fontsize=7.5, color='#8892a4')
        # Hide last empty
        fig.add_subplot(gs[1, 2]).set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Summary stats table
        st.markdown("#### Statistik Deskriptif IPS")
        ips_stats = df[SEM_IPS + ['Rata-Rata IPS']].describe().T
        ips_stats.columns = ['n','mean','std','min','25%','50%','75%','max']
        ips_stats = ips_stats.round(4)
        ips_stats.index = SEM_LBL + ['Rata-Rata IPS']
        st.dataframe(ips_stats.style.background_gradient(subset=['mean'], cmap='Blues'),
                     use_container_width=True)

    # ── TAB 2: Distribusi Absensi ──
    with tab2:
        st.markdown("#### Distribusi Absensi per Semester")
        absen_exist = [c for c in ABSEN if c in df.columns]
        fig, axes = plt.subplots(1, len(absen_exist), figsize=(14, 4))
        if len(absen_exist) == 1: axes = [axes]
        for ax, col, lbl in zip(axes, absen_exist, ABSEN_L[:len(absen_exist)]):
            vals = df[col].dropna()
            ax.hist(vals, bins=12, color='#43e97b', alpha=0.7, edgecolor='#0d1b2a')
            ax.axvline(vals.mean(), color='#f9ca24', lw=2, ls='--', label=f'μ={vals.mean():.1f}%')
            ax.set_title(lbl, fontsize=10, fontweight='600', color='#e8f4fd')
            ax.set_xlabel('Absensi (%)', fontsize=8)
            ax.legend(fontsize=7.5, framealpha=0.1, labelcolor='#c8d8e8')
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Distribusi Absensi per Semester', fontsize=12, color='#e8f4fd', y=1.02)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Absensi by cluster
        st.markdown("#### Rata-Rata Absensi per Cluster")
        fig, ax = plt.subplots(figsize=(10, 4))
        absen_mean = [df[df['cluster_label']==l][absen_exist].mean().mean() for l in ORDER if l in df['cluster_label'].values]
        lbl_ord = [l for l in ORDER if l in df['cluster_label'].values]
        bars = ax.bar(lbl_ord, absen_mean, color=[PALETTE[l] for l in lbl_ord], alpha=0.85, width=0.55)
        for bar, v in zip(bars, absen_mean):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{v:.2f}%',
                    ha='center', fontsize=9, color='#c8d8e8', fontweight='600')
        ax.set_ylim(95, 101)
        ax.set_ylabel('Rata-Rata Absensi (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── TAB 3: Outlier ──
    with tab3:
        st.markdown("#### Deteksi Outlier — Boxplot per Fitur")
        fig, ax = plt.subplots(figsize=(13, 5))
        plot_data = [df[c].dropna().values for c in feat_cols if c in df.columns and c != 'Angkatan Tahun']
        plot_lbls = [c.replace('Rata-Rata ','') for c in feat_cols if c in df.columns and c != 'Angkatan Tahun']
        bp = ax.boxplot(plot_data, vert=True, patch_artist=True,
                        medianprops=dict(color='#f9ca24', linewidth=2.5),
                        whiskerprops=dict(color='#4da6ff', linewidth=1.5),
                        capprops=dict(color='#4da6ff', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='#fc5c65', markersize=5, alpha=0.7))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor('#1d6fa422')
            patch.set_edgecolor('#4da6ff88')
        ax.set_xticks(range(1, len(plot_lbls)+1))
        ax.set_xticklabels(plot_lbls, rotation=45, ha='right', fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # IQR outlier table
        st.markdown("#### Jumlah Outlier per Fitur (Metode IQR)")
        out_rows = []
        for col in feat_cols:
            if col not in df.columns or col == 'Angkatan Tahun': continue
            vals = df[col].dropna()
            Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
            IQR = Q3 - Q1
            n_out = int(((vals < Q1 - 1.5*IQR) | (vals > Q3 + 1.5*IQR)).sum())
            z_out = int((np.abs(stats.zscore(vals)) > 3).sum())
            out_rows.append({'Fitur': col, 'Outlier IQR': n_out, 'Outlier Z-score': z_out,
                             'Q1': round(Q1,3), 'Q3': round(Q3,3), 'IQR': round(IQR,3)})
        out_df = pd.DataFrame(out_rows)
        st.dataframe(out_df.style.highlight_max(subset=['Outlier IQR'], color='#fc5c6533'),
                     use_container_width=True)

    # ── TAB 4: Korelasi ──
    with tab4:
        st.markdown("#### Heatmap Korelasi Antar Fitur")
        fc_valid = [c for c in feat_cols if c in df.columns and c != 'Angkatan Tahun']
        corr = df[fc_valid].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = LinearSegmentedColormap.from_list('naufal', ['#fc5c65','#141824','#4da6ff'])
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8, colors='#8892a4')
        ax.set_xticks(range(len(fc_valid)))
        ax.set_yticks(range(len(fc_valid)))
        lbl_short = [c.replace('Rata-Rata ','').replace('ABSENSI ','Abs ').replace('Mahasiswa','Mhs') for c in fc_valid]
        ax.set_xticklabels(lbl_short, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(lbl_short, fontsize=8)
        for i in range(len(fc_valid)):
            for j in range(len(fc_valid)):
                v = corr.values[i,j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white' if abs(v) > 0.5 else '#8892a4')
        ax.set_title('Korelasi Pearson Antar Fitur', fontsize=12, color='#e8f4fd', pad=12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Scatter: IPS kumulatif vs IPS terkini
        st.markdown("#### Scatter: IPS Kumulatif vs IPS Terkini (Sem 1)")
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl in [l for l in ORDER if l in df['cluster_label'].values]:
            sub = df[df['cluster_label'] == lbl]
            ax.scatter(sub['Rata-Rata IPS'], sub['nilai ips'], c=PALETTE[lbl],
                       s=80, alpha=0.85, label=lbl, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Rata-Rata IPS (Kumulatif)')
        ax.set_ylabel('IPS Semester 1')
        ax.legend(fontsize=9, framealpha=0.1, labelcolor='#c8d8e8')
        ax.grid(alpha=0.25)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── TAB 5: Nilai Mata Kuliah ──
    with tab5:
        st.markdown("#### Rata-Rata Nilai Mata Kuliah per Cluster")
        selected_cluster = st.selectbox("Pilih Cluster:", [l for l in ORDER if l in df['cluster_label'].values])
        sub = df[df['cluster_label'] == selected_cluster]
        course_means = sub[COURSE].mean().sort_values(ascending=True)
        top_n = st.slider("Tampilkan N Mata Kuliah:", 5, len(course_means), 15)
        course_means = course_means.tail(top_n)

        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
        color_c = PALETTE.get(selected_cluster, '#4da6ff')
        bars = ax.barh(course_means.index, course_means.values,
                       color=color_c, alpha=0.8, height=0.6)
        for bar, v in zip(bars, course_means.values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.2f}', va='center', fontsize=8.5, color='#c8d8e8')
        ax.set_xlim(0, 4.2)
        ax.set_xlabel('Rata-Rata Nilai')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Radar chart: compare all clusters on avg course performance
        st.markdown("#### Perbandingan Rata-Rata Nilai — Top 8 Mata Kuliah")
        top8 = df[COURSE].mean().nlargest(8).index.tolist()
        angles = np.linspace(0, 2*np.pi, len(top8), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_facecolor('#1a2130')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for lbl in [l for l in ORDER if l in df['cluster_label'].values]:
            sub = df[df['cluster_label'] == lbl]
            vals = [sub[c].mean() for c in top8]
            vals += vals[:1]
            ax.plot(angles, vals, color=PALETTE[lbl], linewidth=2, label=lbl)
            ax.fill(angles, vals, color=PALETTE[lbl], alpha=0.07)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[:12] for c in top8], fontsize=8, color='#8892a4')
        ax.set_ylim(2.5, 4.1)
        ax.set_yticks([3.0, 3.5, 4.0])
        ax.set_yticklabels(['3.0','3.5','4.0'], fontsize=7, color='#5a6a80')
        ax.grid(color='#2a3344', alpha=0.6)
        ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1.15),
                  framealpha=0.1, labelcolor='#c8d8e8')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════
# PAGE: PRINCALS
# ═══════════════════════════════════════
elif page == "PRINCALS":
    st.markdown('<div class="section-header">Transformasi <span>PRINCALS</span></div>',
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    <b>PRINCALS (Principal Components Analysis by ALS)</b> melakukan reduksi dimensi dengan
    optimal scaling non-linear untuk data campuran. Implementasi menggunakan rank-based
    transformation + PCA (aproksimasi ALS). Jumlah komponen dipilih berdasarkan
    <b>cumulative variance ≥ 80%</b>.
    </div>""", unsafe_allow_html=True)

    n_comp = int(var_info['n_components'])
    cumvar = float(var_info['cumulative_variance'])
    evr    = var_info['explained_variance_ratio']

    c1, c2, c3 = st.columns(3)
    c1.metric("Dimensi Awal",  f"{len(feat_cols)}")
    c2.metric("Komponen Dipilih", f"{n_comp}")
    c3.metric("Cumulative Variance", f"{cumvar*100:.2f}%")

    # Scree plot
    st.markdown("#### Scree Plot & Cumulative Variance")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = range(1, len(evr)+1)
    axes[0].bar(x, [e*100 for e in evr], color='#4da6ff', alpha=0.8, edgecolor='#0d1b2a')
    axes[0].axvline(n_comp, color='#fc5c65', ls='--', lw=2, label=f'n={n_comp}')
    axes[0].set_xlabel('Komponen'); axes[0].set_ylabel('Variance (%)')
    axes[0].set_title('Variance per Komponen', color='#e8f4fd')
    axes[0].legend(labelcolor='#c8d8e8', framealpha=0.1)
    axes[0].spines[['top','right']].set_visible(False)

    cumvars = np.cumsum(evr) * 100
    axes[1].plot(x, cumvars, 'o-', color='#4da6ff', lw=2.5)
    axes[1].fill_between(x, cumvars, alpha=0.12, color='#4da6ff')
    axes[1].axhline(80, color='#fc5c65', ls='--', lw=1.5, label='80% threshold')
    axes[1].axvline(n_comp, color='#43e97b', ls='--', lw=1.5, label=f'n={n_comp}')
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel('Jumlah Komponen'); axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].set_title('Cumulative Variance', color='#e8f4fd')
    axes[1].legend(labelcolor='#c8d8e8', framealpha=0.1)
    axes[1].spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # PC scatter
    st.markdown("#### Scatter Plot PC1 vs PC2 (pewarnaan per cluster)")
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in [l for l in ORDER if l in df['cluster_label'].values]:
        idx = df[df['cluster_label'] == lbl].index
        ax.scatter(X_pc.loc[idx, 'PC1'], X_pc.loc[idx, 'PC2'],
                   c=PALETTE[lbl], s=90, alpha=0.85, label=lbl,
                   edgecolors='white', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
    ax.legend(fontsize=9, framealpha=0.1, labelcolor='#c8d8e8')
    ax.grid(alpha=0.2)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════
# PAGE: CLUSTERING
# ═══════════════════════════════════════
elif page == "Clustering":
    st.markdown('<div class="section-header">Hasil <span>Clustering</span></div>',
                unsafe_allow_html=True)

    # Bubble chart
    st.markdown("#### Bubble Chart — PC1 vs PC2 (ukuran = IPS kumulatif)")
    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl in [l for l in ORDER if l in df['cluster_label'].values]:
        idx  = df[df['cluster_label'] == lbl].index
        x    = X_pc.loc[idx, 'PC1'].values
        y    = X_pc.loc[idx, 'PC2'].values
        size = (df.loc[idx, 'Rata-Rata IPS'].values) ** 3 * 12
        ax.scatter(x, y, s=size, c=PALETTE[lbl], alpha=0.75, label=lbl,
                   edgecolors='white', linewidth=0.6)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9, framealpha=0.1, labelcolor='#c8d8e8')
    ax.grid(alpha=0.2)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Bubble Chart Cluster (PC1 vs PC2)', color='#e8f4fd')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Heatmap cluster profile
    st.markdown("#### Heatmap Profil Cluster")
    fc_heat = [c for c in feat_cols if c in df.columns and c != 'Angkatan Tahun']
    lbl_ord = [l for l in ORDER if l in df['cluster_label'].values]
    heat_data = np.array([[df[df['cluster_label']==l][c].mean() for c in fc_heat] for l in lbl_ord])
    # Normalize per column for color
    heat_norm = (heat_data - heat_data.min(axis=0)) / (heat_data.max(axis=0) - heat_data.min(axis=0) + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 4))
    cmap2 = LinearSegmentedColormap.from_list('naufal2', ['#0d1b2a','#1d6fa4','#4da6ff'])
    im = ax.imshow(heat_norm, cmap=cmap2, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Normalized')
    ax.set_xticks(range(len(fc_heat)))
    ax.set_xticklabels([c.replace('Rata-Rata ','').replace('ABSENSI ','Abs ') for c in fc_heat],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(lbl_ord)))
    ax.set_yticklabels(lbl_ord, fontsize=9)
    for i in range(len(lbl_ord)):
        for j in range(len(fc_heat)):
            ax.text(j, i, f'{heat_data[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if heat_norm[i,j] > 0.5 else '#8892a4')
    ax.set_title('Profil Rata-Rata per Cluster', color='#e8f4fd')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════
# PAGE: EVALUASI
# ═══════════════════════════════════════
elif page == "Evaluasi":
    st.markdown('<div class="section-header">Evaluasi <span>Cluster</span></div>',
                unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
    <b>Silhouette Coefficient</b> mengukur seberapa mirip objek dengan clusternya sendiri dibanding
    cluster lain. Nilai ≥ 0.3 diterima. &nbsp;·&nbsp;
    <b>BSS/TSS</b> mengukur proporsi variansi yang dijelaskan oleh cluster:
    ≥ 50% = cukup baik, ≥ 75% = sangat baik.
    </div>""", unsafe_allow_html=True)

    df_k = results[~results['method'].str.startswith('DBSCAN')].copy()
    methods = sorted(df_k['method'].unique())
    col_map = {'KMeans':'#4da6ff','FCM':'#43e97b','FPCM':'#f9ca24',
               'MFPCM':'#fd9644','KMedoids':'#fc5c65','PCM':'#a29bfe'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method in methods:
        sub = df_k[df_k['method'] == method].sort_values('k')
        c   = col_map.get(method, '#888')
        axes[0].plot(sub['k'], sub['silhouette'], 'o-', color=c, lw=2, label=method, alpha=0.9)
        axes[1].plot(sub['k'], sub['bss_tss_ratio'], 'o-', color=c, lw=2, label=method, alpha=0.9)

    axes[0].axhline(0.3, color='#fc5c65', ls='--', lw=1.5, label='min=0.3')
    axes[0].set_title('Silhouette per k', color='#e8f4fd')
    axes[0].set_xlabel('k'); axes[0].set_ylabel('Silhouette')
    axes[0].legend(fontsize=8, framealpha=0.1, labelcolor='#c8d8e8', ncol=2)
    axes[0].grid(alpha=0.25); axes[0].spines[['top','right']].set_visible(False)
    axes[0].set_xticks(range(2,11))

    axes[1].axhline(50,  color='#fd9644', ls='--', lw=1.5, label='50% (cukup)')
    axes[1].axhline(75,  color='#43e97b', ls='--', lw=1.5, label='75% (sangat baik)')
    axes[1].set_title('BSS/TSS per k', color='#e8f4fd')
    axes[1].set_xlabel('k'); axes[1].set_ylabel('BSS/TSS (%)')
    axes[1].legend(fontsize=8, framealpha=0.1, labelcolor='#c8d8e8', ncol=2)
    axes[1].grid(alpha=0.25); axes[1].spines[['top','right']].set_visible(False)
    axes[1].set_xticks(range(2,11))

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Heatmap silhouette
    st.markdown("#### Heatmap Silhouette Coefficient")
    pivot = df_k.pivot_table(index='method', columns='k', values='silhouette', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap3 = LinearSegmentedColormap.from_list('sil', ['#0d1b2a','#1d6fa4','#4da6ff','#43e97b'])
    im = ax.imshow(pivot.values, cmap=cmap3, aspect='auto', vmin=0, vmax=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'k={k}' for k in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i,j]
            if not np.isnan(v):
                col = 'white' if v > 0.25 else '#5a6a80'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.5, color=col)
    ax.set_title('Silhouette Coefficient Heatmap', color='#e8f4fd')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Best models table
    st.markdown("#### Top 10 Model (Silhouette ≥ 0.3 & BSS/TSS ≥ 50%)")
    top = results[(results['silhouette'] >= 0.3) & (results['bss_tss_ratio'] >= 50)]\
          .sort_values(['silhouette','bss_tss_ratio'], ascending=False).head(10)
    if top.empty:
        top = results.sort_values('silhouette', ascending=False).head(10)
    st.dataframe(top[['method','k','silhouette','bss_tss_ratio','BSS','TSS']].style
                 .highlight_max(subset=['silhouette','bss_tss_ratio'], color='#1d6fa444')
                 .format({'silhouette':'{:.4f}','bss_tss_ratio':'{:.2f}%','BSS':'{:.3f}','TSS':'{:.3f}'}),
                 use_container_width=True)

# ═══════════════════════════════════════
# PAGE: MAHASISWA
# ═══════════════════════════════════════
elif page == "Mahasiswa":
    st.markdown('<div class="section-header">Data <span>Mahasiswa</span></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("🔍 Cari nama atau NRP", placeholder="Ketik nama atau NRP...")
    with c2:
        filter_cl = st.selectbox("Filter Cluster", ['Semua'] + [l for l in ORDER if l in df['cluster_label'].values])

    filtered = df.copy()
    if filter_cl != 'Semua':
        filtered = filtered[filtered['cluster_label'] == filter_cl]
    if search:
        filtered = filtered[
            filtered['Nama Mahasiswa'].str.lower().str.contains(search.lower(), na=False) |
            filtered['NRP'].astype(str).str.contains(search)
        ]

    st.caption(f"Menampilkan {len(filtered)} mahasiswa")

    # Student detail expander
    for _, row in filtered.iterrows():
        lbl   = row['cluster_label']
        color = PALETTE.get(lbl, '#888')
        with st.expander(f"**{row['Nama Mahasiswa']}** &nbsp;·&nbsp; NRP: {row['NRP']} &nbsp;·&nbsp; 🏷️ {lbl}"):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("IPS Kumulatif", f"{row['Rata-Rata IPS']:.3f}")
            cc2.metric("IPS Sem 1", f"{row['nilai ips']:.3f}")
            cc3.metric("IPS Sem 2", f"{row['IPS']:.3f}" if 'IPS' in row.index else "-")
            cc4.metric("Absensi", f"{row.get('Rata-Rata Absen Mahasiswa', 0):.1f}%")

            # IPS trend
            ips_vals = [row.get(c, None) for c in SEM_IPS]
            ips_vals = [v for v in ips_vals if v is not None]
            if ips_vals:
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.plot(SEM_LBL[:len(ips_vals)], ips_vals, 'o-', color=color, lw=2.5, ms=8)
                ax.fill_between(SEM_LBL[:len(ips_vals)], ips_vals, alpha=0.15, color=color)
                ax.set_ylim(min(2.5, min(ips_vals)-0.1), 4.1)
                ax.set_ylabel('IPS', fontsize=9)
                ax.grid(alpha=0.25); ax.spines[['top','right']].set_visible(False)
                ax.set_title(f'Tren IPS — {row["Nama Mahasiswa"]}', fontsize=9, color='#e8f4fd')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Top & bottom courses
            course_vals = {c: row.get(c, None) for c in COURSE if c in row.index}
            course_vals = {k: v for k, v in course_vals.items() if v is not None and not np.isnan(v)}
            if course_vals:
                cv_series = pd.Series(course_vals).sort_values()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**🔴 5 Nilai Terendah**")
                    for name, val in cv_series.head(5).items():
                        st.markdown(f"<small>{name}: <b style='color:#fc5c65'>{val:.2f}</b></small>", unsafe_allow_html=True)
                with col_b:
                    st.markdown("**🟢 5 Nilai Tertinggi**")
                    for name, val in cv_series.tail(5).iloc[::-1].items():
                        st.markdown(f"<small>{name}: <b style='color:#43e97b'>{val:.2f}</b></small>", unsafe_allow_html=True)

# ═══════════════════════════════════════
# PAGE: REKOMENDASI
# ═══════════════════════════════════════
elif page == "Rekomendasi":
    st.markdown('<div class="section-header">Rekomendasi <span>untuk Dosen</span></div>',
                unsafe_allow_html=True)

    ADVICE = {
        'Sangat Tinggi': {
            'icon': '🏆', 'color': '#4da6ff',
            'desc': 'Mahasiswa berprestasi tinggi dengan konsistensi akademik sangat baik.',
            'tips': [
                'Berikan tantangan tambahan: proyek riset, kompetisi nasional/internasional.',
                'Pertimbangkan jalur fast-track atau beasiswa lanjutan studi S2.',
                'Libatkan sebagai asisten praktikum atau tutor sebaya.',
                'Dorong publikasi ilmiah atau partisipasi konferensi.',
            ]
        },
        'Tinggi': {
            'icon': '⭐', 'color': '#43e97b',
            'desc': 'Performa akademik di atas rata-rata dengan potensi berkembang lebih jauh.',
            'tips': [
                'Pertahankan momentum; beri akses materi pengayaan & sertifikasi profesional.',
                'Dorong partisipasi aktif dalam seminar dan proyek interdisiplin.',
                'Monitoring ringan tiap bulan untuk memastikan konsistensi.',
                'Rekomendasikan program magang di industri relevan.',
            ]
        },
        'Sedang': {
            'icon': '📘', 'color': '#f9ca24',
            'desc': 'Performa rata-rata dengan absensi baik — potensi peningkatan signifikan.',
            'tips': [
                'Lakukan bimbingan rutin setiap 2 minggu untuk identifikasi hambatan.',
                'Rekomendasikan sumber belajar tambahan (modul, video, kelompok belajar).',
                'Fokus pada mata kuliah dengan nilai terendah untuk perbaikan terarah.',
                'Dorong konsistensi IPS antar semester.',
            ]
        },
        'Cukup': {
            'icon': '⚠️', 'color': '#fd9644',
            'desc': 'IPK kumulatif rendah namun IPS terkini & absensi cukup baik — ada sinyal perbaikan.',
            'tips': [
                'Jadwalkan konseling akademik individual minimal 1×/bulan.',
                'Identifikasi mata kuliah bermasalah dan siapkan program remedial.',
                'Manfaatkan kehadiran yang baik — dorong keaktifan di kelas.',
                'Evaluasi apakah ada kendala non-akademik (ekonomi, kesehatan).',
            ]
        },
        'Rendah': {
            'icon': '🚨', 'color': '#fc5c65',
            'desc': 'IPK dan IPS terkini rendah, absensi juga perlu perhatian — butuh intervensi segera.',
            'tips': [
                'Intervensi segera: konseling akademik intensif & monitoring kehadiran mingguan.',
                'Koordinasi dengan orang tua/wali untuk dukungan di luar kampus.',
                'Program bimbingan khusus atau pendampingan intensif.',
                'Pertimbangkan evaluasi beban SKS apakah sesuai kemampuan.',
            ]
        },
    }

    for lbl in [l for l in ORDER if l in df['cluster_label'].values]:
        a     = ADVICE.get(lbl, {})
        color = a.get('color', '#888')
        n_mhs = df[df['cluster_label'] == lbl].shape[0]

        with st.expander(f"{a.get('icon','')} **{lbl}** — {n_mhs} mahasiswa", expanded=(lbl == 'Rendah')):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"<div class='info-box' style='border-left-color:{color}'>"
                            f"<b>{a.get('desc','')}</b></div>", unsafe_allow_html=True)
                for tip in a.get('tips', []):
                    st.markdown(f"<small>✦ {tip}</small>", unsafe_allow_html=True)
            with c2:
                sub = df[df['cluster_label'] == lbl]
                ips_avg  = sub['Rata-Rata IPS'].mean()
                ips_std  = sub['Rata-Rata IPS'].std()
                absen_avg = sub[[c for c in ABSEN if c in df.columns]].mean().mean()
                st.metric("Avg IPK", f"{ips_avg:.3f}")
                st.metric("Std IPK", f"{ips_std:.3f}")
                st.metric("Avg Absensi", f"{absen_avg:.2f}%")

                # Mini IPS trend
                fig, ax = plt.subplots(figsize=(4, 2))
                for _, row in sub.iterrows():
                    ips_v = [row.get(c) for c in SEM_IPS if c in row.index]
                    ips_v = [v for v in ips_v if v is not None]
                    ax.plot(range(len(ips_v)), ips_v, color=color, alpha=0.3, lw=1.2)
                # Mean line
                means = [sub[c].mean() for c in SEM_IPS if c in df.columns]
                ax.plot(range(len(means)), means, color=color, lw=2.5, label='Rata-Rata')
                ax.set_xticks(range(len(means)))
                ax.set_xticklabels(SEM_LBL[:len(means)], fontsize=7)
                ax.set_ylim(2.5, 4.1)
                ax.grid(alpha=0.2); ax.spines[['top','right']].set_visible(False)
                ax.set_title('Tren IPS', fontsize=8, color='#e8f4fd')
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Academic Clustering Dashboard &nbsp;·&nbsp;
    <span>PRINCALS + FCM + FPCM + MFPCM + K-Means + K-Medoids + DBSCAN</span>
    &nbsp;·&nbsp; by <span>Naufal</span>
</div>
""", unsafe_allow_html=True)
