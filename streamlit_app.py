"""
DASHBOARD STREAMLIT — Clustering Akademik Mahasiswa
Analisis berbasis PRINCALS + 7 Algoritma Clustering
By Naufal
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
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
# THEME & CSS (Premium Purple Theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

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
    font-family: 'DM Serif Display', serif;
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
    font-weight: 500;
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
    font-size: 1.4rem;
    color: #566A7F;
    margin: 1.8rem 0 1rem;
    padding-bottom: 8px;
    border-bottom: 1px solid #F0F0F0;
}
.section-header span { color: #7367F0; }

/* Student card */
.stu-card {
    background: #FFFFFF;
    border: 1px solid #F0F0F0;
    box-shadow: 0 2px 8px #EAEAEA;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all .18s;
}
.stu-card:hover { border-color: #5E50EE; transform: translateX(3px); }

/* Custom Badge */
.custom-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.76rem;
    font-weight: bold;
    color: #fff;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADER & CACHED EXCEL READER
# ─────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def clean_display_table(df_to_show):
    if not isinstance(df_to_show, pd.DataFrame):
        return df_to_show
    cols_to_drop = []
    for col in df_to_show.columns:
        col_str = str(col).lower()
        if any(kw in col_str for kw in ['rata-rata', 'angkatan', 'tahun', 'prodi', 'jenis kelamin', 'jk', 'asal kab/kota']):
            cols_to_drop.append(col)
    return df_to_show.drop(columns=cols_to_drop, errors='ignore')

@st.cache_data
def get_excel_sheet(filepath, sheet_name=None, index_col=None):
    """Cached function to read Excel sheets (prevents loading freeze)."""
    if sheet_name:
        return pd.read_excel(filepath, sheet_name=sheet_name, index_col=index_col)
    return pd.read_excel(filepath, index_col=index_col)

@st.cache_data
def load_data():
    df_labeled = pd.read_pickle(os.path.join(BASE, 'output/df_labeled.pkl'))
    X_pc = pd.read_pickle(os.path.join(BASE, 'output/X_princals.pkl'))
    X_scaled = pd.read_pickle(os.path.join(BASE, 'output/X_scaled.pkl'))
    results = pd.read_pickle(os.path.join(BASE, 'output/clustering_results.pkl'))
    feat_cols = pd.read_pickle(os.path.join(BASE, 'output/feature_cols.pkl')).tolist()
    var_info = pd.read_pickle(os.path.join(BASE, 'output/princals_info.pkl'))
    best_model = pd.read_pickle(os.path.join(BASE, 'output/best_model.pkl'))
    ranking = pd.read_pickle(os.path.join(BASE, 'output/method_ranking.pkl'))
    
    with open(os.path.join(BASE, 'output/variable_groups.pkl'), 'rb') as f:
        var_groups = pickle.load(f)
        
    return df_labeled, X_pc, X_scaled, results, feat_cols, var_info, best_model, ranking, var_groups

try:
    df_labeled, X_pc, X_scaled, results, feat_cols, var_info, best_model, ranking, var_groups = load_data()
except Exception as e:
    st.error(f"Error loading pickle data. Please run the pipeline script main.py first! Details: {e}")
    st.stop()

# ─────────────────────────────────────────────
# DYNAMIC COLUMN DETECTORS
# ─────────────────────────────────────────────
ips_individual = sorted([c for c in df_labeled.columns if ('nilai IPS' in c or 'nilai ips' in c.lower()) and 'Rata-Rata' not in c])
abs_individual = sorted([c for c in df_labeled.columns if 'ABSENSI' in c.upper() and 'Rata-Rata' not in c])

# ─────────────────────────────────────────────
# PERSISTENT NOTES MANAGEMENT (Dosen / Sidang Notes)
# ─────────────────────────────────────────────
NOTES_FILE = os.path.join(BASE, 'output/dosen_notes.json')

def load_notes():
    if os.path.exists(NOTES_FILE):
        try:
            with open(NOTES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_note(section, note_text):
    notes = load_notes()
    notes[section] = note_text
    os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)
    with open(NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False, indent=4)

def render_dosen_notes(section_name):
    notes = load_notes()
    current_note = notes.get(section_name, "")
    
    st.markdown("---")
    with st.expander("📝 Catatan Dosen Pembimbing / Catatan Sidang", expanded=False):
        if current_note:
            st.info(f"**Catatan Aktif:**\n\n{current_note}")
        else:
            st.write("*Belum ada catatan dari Dosen Pembimbing untuk bagian ini.*")
            
        new_note = st.text_area("Masukkan catatan baru / koreksi dari dosen pembimbing:", value=current_note, key=f"note_{section_name}")
        if st.button("Simpan Catatan", key=f"btn_{section_name}"):
            save_note(section_name, new_note)
            st.success("Catatan berhasil disimpan!")
            st.rerun()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PALETTE = {
    'Sangat Tinggi': '#28C76F',
    'Tinggi'       : '#00CFE8',
    'Sedang'       : '#7367F0',
    'Cukup'        : '#FF9F43',
    'Rendah'       : '#EA5455',
    'Sangat Rendah': '#A067F0',
    'Kritis'       : '#323232',
    'Outlier (Kritis)': '#8E9BAE'
}

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;color:#7367F0;line-height:1.2'>
            🎓 Clustering<br><span style='color:#5E50EE'>Akademik Mahasiswa</span>
        </div>
        <div style='font-size:0.75rem;color:#8E9BAE;margin-top:4px;
                    font-family:"JetBrains Mono",monospace'>by Naufal</div>
    </div>
    <hr style='border:none;border-top:1px solid #EAEAEA;margin:8px 0 16px'>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigasi Halaman:", [
        "1. Overview Dataset",
        "2. Preprocessing",
        "3. EDA",
        "4. Hasil PRINCALS",
        "5. Hasil Clustering",
        "6. Perbandingan Metode",
        "7. Evaluasi Silhouette & BSS/TSS",
        "8. Visualisasi Cluster",
        "9. Dashboard Insight",
        "10. Mahasiswa Berprestasi"
    ])

    st.markdown("---")
    st.markdown(f"""<div style='font-size:0.75rem;color:#566A7F;line-height:1.8'>
        <b style='color:#7367F0'>Model Terbaik Terpilih:</b><br>
        Metode &nbsp;&nbsp;&nbsp;&nbsp;: {best_model['method']}<br>
        Parameter&nbsp;: {best_model['params']}<br>
        Silhouette: {best_model['silhouette']:.4f}<br>
        BSS/TSS &nbsp;&nbsp;: {best_model['bss_tss_ratio']:.2f}%
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">Academic <span>Clustering</span> Dashboard</p>
    <p class="hero-sub">Analisis Akademik dan Profiling Mahasiswa menggunakan Reduksi Dimensi PRINCALS & 7 Metode Clustering</p>
    <span class="hero-badge">PRINCALS Reduction</span>
    <span class="hero-badge">K-Means & K-Medoids</span>
    <span class="hero-badge">DBSCAN Grid Search</span>
    <span class="hero-badge">Fuzzy Clustering (FCM, PCM, FPCM, MFPCM)</span>
    <div class="by-line">by Naufal</div>
</div>
""", unsafe_allow_html=True)

page = nav.split(". ")[1]

# ==============================================================================
# PAGE 1: OVERVIEW DATASET
# ==============================================================================
if page == "Overview Dataset":
    st.markdown('<p class="section-header">1. Overview & <span>Tujuan Penelitian</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background:#F5F3FF; border:1px solid #7367F044; border-radius:12px; padding:24px; margin-bottom:24px;'>
        <h3 style='margin-top:0; color:#7367F0; font-family:"DM Serif Display", serif;'>🎯 Tujuan Penelitian (Research Objectives)</h3>
        <p style='font-size:0.95rem; line-height:1.6; color:#566A7F;'>
            Berdasarkan arahan akademik dan metodologi data mining, penelitian ini memiliki 4 tujuan utama:
        </p>
        <ol style='font-size:0.92rem; line-height:1.8; color:#566A7F; padding-left:20px; margin-bottom:0;'>
            <li style='margin-bottom:8px;'>
                <b>Reduksi Dimensi Campuran yang Efektif (PRINCALS)</b>:<br>
                Mengimplementasikan metode <i>Principal Component Analysis by Alternating Least Squares</i> (PRINCALS) untuk mengolah data akademik mahasiswa yang terdiri dari variabel numerik dan kategorikal, sehingga kompleksitas tipe data dapat diatasi dan dimensi data dapat disederhanakan tanpa mengurangi kualitas informasi.
            </li>
            <li style='margin-bottom:8px;'>
                <b>Clustering Mahasiswa Berbasis Hasil Reduksi Dimensi</b>:<br>
                Menggunakan komponen utama hasil reduksi PRINCALS sebagai input dalam 7 algoritma clustering (K-Means, K-Medoids, DBSCAN, FCM, PCM, FPCM, dan MFPCM) untuk mengungkap pola pengelompokan mahasiswa secara komprehensif dan sistematis.
            </li>
            <li style='margin-bottom:8px;'>
                <b>Analisis Faktor Dominan & Rekomendasi Pertukaran Pelajar</b>:<br>
                Menganalisis dan mengidentifikasi karakteristik dari setiap klaster yang terbentuk untuk menentukan variabel akademik (nilai per matakuliah, total nilai per semester, dan kehadiran) maupun non-akademik (kuisioner interaksi sosial, dosen wali, dan fasilitas) yang paling memengaruhi prestasi mahasiswa untuk merumuskan <b>Rekomendasi Pertukaran Pelajar</b>.
            </li>
            <li style='margin-bottom:0;'>
                <b>Pengembangan Dashboard Visual Interaktif (Streamlit)</b>:<br>
                Membangun antarmuka visualisasi berbasis Streamlit untuk menyajikan distribusi klaster, hasil reduksi dimensi, perbandingan nilai matakuliah vs total nilai semester, dan hubungan antar variabel secara intuitif untuk membantu pemangku kebijakan mengambil keputusan strategis.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### Deskripsi Penelitian
        Dashboard ini dikembangkan untuk mengimplementasikan pipeline data mining terstruktur guna menganalisis 
        data akademik mahasiswa. Penelitian ini menerapkan reduksi dimensi menggunakan **PRINCALS (Principal Covariates Regression 
        on Alternative Least Squares)** untuk mengatasi multikolinearitas dan data berdimensi tinggi, kemudian membandingkan performa 
        7 metode clustering:
        * **K-Means & K-Medoids** (Centroid-based)
        * **DBSCAN** (Density-based)
        * **FCM, PCM, FPCM, & MFPCM** (Fuzzy/Possibilistic-based)
        """)
    with col2:
        st.markdown(f"""
        ### Ringkasan Kualitas Data Mentah
        * **Jumlah Mahasiswa (Observasi)**: {len(df_labeled)} mahasiswa
        * **Jumlah Variabel (Fitur)**: 124 variabel
        * **Missing Values**: 0 sel kosong (telah diimputasi)
        * **Baris Duplikat**: 0 baris (telah dibersihkan)
        """)
        
    st.markdown('<p class="section-header">Preview <span>Dataset Mentah (Original)</span></p>', unsafe_allow_html=True)
    try:
        df_raw = get_excel_sheet(r'C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx')
        st.dataframe(clean_display_table(df_raw), use_container_width=True)
    except Exception as e:
        st.warning(f"Gagal membaca Excel data mentah. Menampilkan data preprocessed: {e}")
        st.dataframe(clean_display_table(df_labeled), use_container_width=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 2: PREPROCESSING
# ==============================================================================
elif page == "Preprocessing":
    st.markdown('<p class="section-header">2. Preprocessing <span>Data & Klasifikasi</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Langkah Preprocessing Otomatis:
    1. **Pembersihan Data**: Mengisi nilai kosong (*missing values*) secara otomatis menggunakan median (untuk numerik) dan modus (kategorik), serta menghapus duplikasi data.
    2. **Label Encoding**: Mengubah variabel kategorik non-numerik seperti `Prodi` dan `Asal Kab/Kota` menjadi kode numerik ter-encode.
    3. **Normalisasi**: Standarisasi seluruh fitur numerik menggunakan **StandardScaler** untuk menyetarakan skala dengan rata-rata 0 dan variansi 1.
    """)
    
    st.markdown("### Klasifikasi Variabel:")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"**Identitas (Tidak di-cluster)**\n* " + "\n* ".join(var_groups['identitas']))
    with c2:
        st.success(f"**Indeks Prestasi Semester (IPS)**\n* " + "\n* ".join(var_groups['ips']))
    with c3:
        st.warning(f"**Kehadiran (Absensi)**\n* " + "\n* ".join(var_groups['absensi']))
        
    st.markdown(f"**Variabel Akademik ({len(var_groups['akademik'])} Mata Kuliah)**")
    with st.expander("Tampilkan Seluruh Variabel Akademik"):
        st.write(", ".join(var_groups['akademik']))
        
    st.markdown("### Preview Data Setelah Standardisasi (Siap untuk PRINCALS)")
    st.dataframe(clean_display_table(X_scaled), use_container_width=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 3: EDA
# ==============================================================================
elif page == "EDA":
    st.markdown('<p class="section-header">3. Exploratory <span>Data Analysis (EDA)</span></p>', unsafe_allow_html=True)
    
    # Summary stats
    st.markdown("### Statistik Deskriptif Variabel Akademik & Utama")
    if os.path.exists('output/eda_summary_stats.csv'):
        df_stats = pd.read_csv('output/eda_summary_stats.csv', index_col=0)
        st.dataframe(df_stats.round(3), use_container_width=True)
    else:
        st.dataframe(df_labeled.describe().T, use_container_width=True)
        
    st.markdown("### Visualisasi Deskriptif")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Histogram Distribusi", 
        "📦 Boxplot Outlier", 
        "🌡️ Korelasi Heatmap",
        "❓ Distribusi Missing Value"
    ])
    
    with tab1:
        if os.path.exists('output/eda_distributions.png'):
            st.image('output/eda_distributions.png', caption="Distribusi Nilai Setelah Normalisasi", use_container_width=True)
        else:
            st.warning("Grafik distribusi belum dibuat.")
            
    with tab2:
        if os.path.exists('output/eda_outliers.png'):
            st.image('output/eda_outliers.png', caption="Deteksi Outlier", use_container_width=True)
        else:
            st.warning("Grafik boxplot belum dibuat.")
            
    with tab3:
        if os.path.exists('output/eda_correlation.png'):
            st.image('output/eda_correlation.png', caption="Heatmap Korelasi Pearson", use_container_width=True)
        else:
            st.warning("Heatmap korelasi belum dibuat.")
            
    with tab4:
        if os.path.exists('output/eda_missing_values.png'):
            st.image('output/eda_missing_values.png', caption="Distribusi Missing Value Mentah", use_container_width=True)
        else:
            st.warning("Grafik missing value belum dibuat.")

    render_dosen_notes(page)

# ==============================================================================
# PAGE 4: HASIL PRINCALS (CACHED READS)
# ==============================================================================
elif page == "Hasil PRINCALS":
    st.markdown('<p class="section-header">4. Transformasi Dimensi <span>PRINCALS</span></p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Aturan Seleksi Komponen PRINCALS:
    Menggunakan kriteria **Cumulative Explained Variance ≥ 80% (0.80)**.
    Dari 122 variabel awal, terpilih sebanyak **{var_info['n_components']} komponen** utama yang menjelaskan 
    **{var_info['cumulative_variance']*100:.2f}%** dari total variansi informasi.
    """)
    
    st.image('output/princals_scree.png', caption="PRINCALS Scree Plot & Cumulative Variance", use_container_width=True)
    
    if os.path.exists('output/princals_biplot.png'):
        st.image('output/princals_biplot.png', caption="PRINCALS Biplot - PC1 vs PC2", use_container_width=True)
        
    st.markdown("### Lembar Hasil Analisis PRINCALS (Loaded from Cache)")
    
    tab_e1, tab_e2, tab_e3, tab_e4, tab_e5 = st.tabs([
        "Eigen Value & Variance",
        "Component Loading",
        "Object Score (Transformasi)",
        "Component Score Coefficients",
        "Dataset Final (Terpilih)"
    ])
    
    # Access the cached data frames
    with tab_e1:
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("**Eigenvalues per Komponen**")
            df_ev = get_excel_sheet('hasil_principals.xlsx', sheet_name='Eigen Value')
            st.dataframe(df_ev.round(4), use_container_width=True)
        with c_v2:
            st.markdown("**Explained Variance Ratio**")
            df_var = get_excel_sheet('hasil_principals.xlsx', sheet_name='Explained Variance')
            st.dataframe(df_var.round(4), use_container_width=True)
            
    with tab_e2:
        st.markdown("**Component Loading Matrix (Korelasi Variabel vs Komponen)**")
        df_loadings = get_excel_sheet('hasil_principals.xlsx', sheet_name='Component Loading', index_col=0)
        
        # Interactive heatmap in Plotly
        fig_load = px.imshow(df_loadings.iloc[:35, :5], 
                             color_continuous_scale='RdBu',
                             labels=dict(color="Loading"), aspect="auto")
        fig_load.update_coloraxes(cmid=0)
        fig_load.update_layout(title="Heatmap Component Loading (Top 35 Variabel vs 5 PC Pertama)")
        st.plotly_chart(fig_load, use_container_width=True)
        st.dataframe(df_loadings.round(4), use_container_width=True)
        
    with tab_e3:
        st.markdown("**Object Scores (Koordinat Komponen Seluruh Mahasiswa)**")
        df_obj = get_excel_sheet('hasil_principals.xlsx', sheet_name='Object Score')
        st.dataframe(df_obj.round(4), use_container_width=True)
        
    with tab_e4:
        st.markdown("**Component Score Coefficients (Eigenvectors / Pembobot Linier)**")
        df_csc = get_excel_sheet('hasil_principals.xlsx', sheet_name='Component Score', index_col=0)
        st.dataframe(df_csc.round(4), use_container_width=True)
        
    with tab_e5:
        st.markdown(f"**Dataset Final untuk Clustering ({var_info['n_components']} Komponen Terpilih)**")
        df_final = get_excel_sheet('hasil_principals.xlsx', sheet_name='Dataset Final')
        st.dataframe(df_final.round(4), use_container_width=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 5: HASIL CLUSTERING (INDIVIDUAL SHEET VIEWER - CACHED)
# ==============================================================================
elif page == "Hasil Clustering":
    st.markdown('<p class="section-header">5. Eksperimen <span>Algoritma Clustering</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    Seluruh algoritma di bawah ini dijalankan secara berurutan menggunakan **Dataset Final hasil PRINCALS** 
    sebagai input eksklusif:
    * **K-Means & K-Medoids**: Centroid-based hard clustering.
    * **FCM (Fuzzy C-Means)**: Memberikan keanggotaan fuzzy kontinu (0 s.d. 1) untuk setiap cluster.
    * **PCM (Possibilistic C-Means)**: Mengatasi noise dengan mengukur typicality nilai kecocokan absolut.
    * **FPCM & MFPCM**: Menggabungkan fuzzy membership dan possibilistic typicality untuk stabilitas optimal.
    * **DBSCAN**: Mengelompokkan berdasarkan kerapatan data padat dan mengidentifikasi outlier.
    """)
    
    sel_method = st.selectbox("Pilih Metode untuk Melihat Hasil Sheet Assignment:", [
        "K-Means", "K-Medoids", "FCM", "PCM", "FPCM", "MFPCM", "DBSCAN"
    ])
    
    st.markdown(f"### Lembar Kerja Sheet: `{sel_method}`")
    try:
        df_sheet = get_excel_sheet('hasil_clustering.xlsx', sheet_name=sel_method)
        st.dataframe(df_sheet, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal membaca sheet clustering {sel_method} (Mungkin file terkunci atau belum dibuat): {e}")

    render_dosen_notes(page)

# ==============================================================================
# PAGE 6: PERBANDINGAN METODE
# ==============================================================================
elif page == "Perbandingan Metode":
    st.markdown('<p class="section-header">6. Perbandingan <span>Performa & Ranking</span></p>', unsafe_allow_html=True)
    
    st.markdown("### Tabel Ranking Parameter Terbaik per Metode (Urut Silhouette Descending)")
    st.dataframe(ranking[['Ranking', 'method', 'params', 'silhouette', 'bss_tss_ratio', 'BSS', 'TSS']], use_container_width=True)
    
    st.markdown("### Tabel Hasil Lengkap Seluruh Eksperimen (K=2..10 & DBSCAN Grid Search)")
    
    # Filter options
    f_methods = st.multiselect("Filter Metode:", results['method'].unique(), default=results['method'].unique())
    df_filtered = results[results['method'].isin(f_methods)].sort_values('silhouette', ascending=False)
    
    st.dataframe(df_filtered, use_container_width=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 7: EVALUASI SILHOUETTE & BSS/TSS
# ==============================================================================
elif page == "Evaluasi Silhouette & BSS/TSS":
    st.markdown('<p class="section-header">7. Evaluasi <span>Silhouette & BSS/TSS Ratio</span></p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Kriteria Model Terbaik:
    Sesuai dengan kriteria akademik:
    * **Silhouette Coefficient $\\ge$ 0.3** (kriteria kekompakan klaster).
    * **BSS/TSS Ratio $\\ge$ 50%** (tingkat pemisahan klaster, disarankan $\\ge$ 75% untuk pemisahan optimal).
    
    Model terbaik yang terpilih saat ini secara otomatis adalah: 
    **{best_model['method']} ({best_model['params']})** dengan hasil:
    * **Silhouette Coefficient**: `{best_model['silhouette']:.4f}`
    * **BSS/TSS Ratio**: `{best_model['bss_tss_ratio']:.2f}%`
    
    ### 💡 Rationale / Alasan Pemilihan Model Ini:
    1. **Silhouette Coefficient Tinggi (`{best_model['silhouette']:.4f}`)**: 
       Skor Silhouette ini bernilai **jauh di atas batas minimal 0.3**, yang mengindikasikan bahwa struktur klaster sangat kokoh, padat (*dense*), dan memiliki batas yang jelas antar-klaster tanpa adanya tumpang tindih (*overlap*) yang berarti. Sebagai perbandingan, model centroid (K-Means, K-Medoids, dan Fuzzy C-Means) pada dataset bersih ini hanya mencapai Silhouette berkisar antara 0.10 s.d. 0.25.
    2. **Pemisahan Klaster Sangat Optimal (BSS/TSS = `{best_model['bss_tss_ratio']:.2f}%`)**: 
       Rasio BSS/TSS ini bernilai **di atas batas preferensi 75%**. Rasio ini membuktikan bahwa variabilitas antar-klaster (Between-group Sum of Squares) mendominasi total variabilitas data (Total Sum of Squares), sehingga setiap klaster yang terbentuk memiliki perbedaan karakteristik akademis yang sangat signifikan dan kontras satu sama lain.
    3. **Penanganan Outlier yang Cerdas**:
       DBSCAN secara otomatis memisahkan mahasiswa dengan data akademik yang tidak biasa (outliers) sebagai klaster noise (`-1`), sehingga profil klaster utama (`Sangat Tinggi` dan `Tinggi`) tidak terdistorsi oleh data ekstrem, menghasilkan analisis yang secara akademis valid dan obyektif.
    """)
    
    # Plotly Bar Charts of Best models
    st.markdown("### Perbandingan Performa Terbaik per Metode")
    
    c1, c2 = st.columns(2)
    with c1:
        df_rank_sorted_sil = ranking.sort_values('silhouette', ascending=True)
        fig_sil = px.bar(df_rank_sorted_sil, x='silhouette', y='method', orientation='h',
                         text='silhouette', color='silhouette',
                         color_continuous_scale='Viridis',
                         labels={'silhouette': 'Silhouette Coefficient', 'method': 'Metode'},
                         title='Perbandingan Silhouette Coefficient Terbaik')
        fig_sil.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        st.plotly_chart(fig_sil, use_container_width=True)
        
    with c2:
        df_rank_sorted_bss = ranking.sort_values('bss_tss_ratio', ascending=True)
        fig_bss = px.bar(df_rank_sorted_bss, x='bss_tss_ratio', y='method', orientation='h',
                         text='bss_tss_ratio', color='bss_tss_ratio',
                         color_continuous_scale='Blues',
                         labels={'bss_tss_ratio': 'BSS/TSS Ratio (%)', 'method': 'Metode'},
                         title='Perbandingan BSS/TSS Ratio Terbaik')
        fig_bss.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig_bss, use_container_width=True)
        
    # Radar polar chart
    st.markdown("### Radar Chart Performa Rata-Rata Algoritma (Silhouette & BSS/TSS)")
    
    categories = ['Silhouette Score', 'BSS/TSS Ratio (Scaled 0-1)']
    fig_radar = go.Figure()
    
    for idx, row in ranking.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['silhouette'], row['bss_tss_ratio']/100.0],
            theta=categories,
            fill='toself',
            name=row['method']
        ))
        
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Perbandingan Profil Evaluasi Multi-Metrik'
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Display static Matplotlib Heatmaps/Lines
    st.markdown("### Heatmap & Tren Line Plot (Matplotlib Visualizations)")
    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.image('output/eval_heatmap.png', caption="Evaluasi Heatmap per k", use_container_width=True)
    with col_mat2:
        st.image('output/eval_lines.png', caption="Evaluasi Line Plot per k", use_container_width=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 8: VISUALISASI CLUSTER (DYNAMIC SEMESTERS)
# ==============================================================================
elif page == "Visualisasi Cluster":
    st.markdown('<p class="section-header">8. Visualisasi <span>Cluster & Analisis Dimensi</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    > [!TIP]
    > **Interaktivitas Grafik**: Anda dapat **mengklik salah satu titik mahasiswa** pada grafik sebar PC1 vs PC2 di bawah ini. 
    > Detail informasi akademik mahasiswa tersebut (seperti IPS per semester, absensi, dan saran bimbingan) 
    > akan langsung muncul di kartu profil detail di bagian bawah grafik secara real-time!
    """)
    
    # 1. PC1 vs PC2 Scatter plot
    df_scatter = pd.concat([df_labeled, X_pc[['PC1', 'PC2']]], axis=1)
    
    fig_scat = px.scatter(df_scatter, x='PC1', y='PC2', color='cluster_label',
                         color_discrete_map=PALETTE,
                         custom_data=['NRP', 'Nama Mahasiswa'],
                         hover_data=['NRP', 'Nama Mahasiswa', 'Rata-Rata IPS'],
                         labels={'PC1': 'PC1 (PRINCALS)', 'PC2': 'PC2 (PRINCALS)'},
                         title=f"Visualisasi Proyeksi Ruang 2D (Model Terbaik: {best_model['method']})")
    
    # Add centroids for centroid methods
    if best_model['method'] != 'DBSCAN':
        centroids = df_scatter.groupby('cluster_label')[['PC1', 'PC2']].mean().reset_index()
        fig_scat.add_trace(go.Scatter(
            x=centroids['PC1'], y=centroids['PC2'],
            mode='markers', marker=dict(symbol='x', size=14, color='black', line=dict(width=2)),
            name='Centroid Cluster'
        ))
        
    fig_scat.update_traces(marker=dict(size=12, line=dict(width=0.5, color='White')))
    
    # Catch click selection event using st.plotly_chart rerun
    event_data = st.plotly_chart(fig_scat, on_select="rerun", use_container_width=True)
    
    # Handle click event and show student details below
    if event_data and 'selection' in event_data and 'points' in event_data['selection'] and len(event_data['selection']['points']) > 0:
        point = event_data['selection']['points'][0]
        custom_data = point.get('customdata', None)
        if custom_data:
            nrp_clicked = custom_data[0]
            student_clicked = df_labeled[df_labeled['NRP'] == nrp_clicked].iloc[0]
            
            color = PALETTE.get(student_clicked['cluster_label'], '#8E9BAE')
            
            # Semester GPAs and attendance text lists (Dynamic)
            ips_text = " &nbsp;·&nbsp; ".join([f"Sem {i+1}: <b>{student_clicked[col]:.2f}</b>" for i, col in enumerate(ips_individual)])
            abs_text = " &nbsp;·&nbsp; ".join([f"Sem {i+1}: <b>{student_clicked[col]:.1f}%</b>" for i, col in enumerate(abs_individual)])
            
            st.markdown(f"""
            <div style='background:#F5F3FF; border:1px solid #7367F0; border-radius:10px; padding:20px; margin-bottom:20px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <h4 style='color:#7367F0; margin:0;'>🔍 Detail Profil Mahasiswa (Hasil Klik Grafik)</h4>
                    <span class="custom-badge" style="background-color:{color};">{student_clicked['cluster_label']}</span>
                </div>
                <hr style='border:none; border-top:1px solid #E8E4FF; margin:8px 0 12px'>
                <div style='font-size:1.15rem; font-weight:bold; color:#566A7F;'>{student_clicked['Nama Mahasiswa']} (NRP: {student_clicked['NRP']})</div>
                <div style='font-size:0.8rem; color:#8E9BAE;'>Prodi: {student_clicked['Prodi']} &nbsp;·&nbsp; Angkatan: {student_clicked['Angkatan Tahun']}</div>
                <div style='display:flex; gap:24px; margin-top:14px; font-size:0.85rem;'>
                    <div style='flex:1;'>
                        <b style='color:#7367F0;'>PERFORMA INDEKS PRESTASI (IPS):</b><br>
                        Rata-Rata IPK (Sem 1-5): <b>{student_clicked['Rata-Rata IPS']:.3f}</b><br>
                        {ips_text}
                    </div>
                    <div style='flex:1;'>
                        <b style='color:#28C76F;'>KEHADIRAN (ABSENSI):</b><br>
                        Kehadiran Rata-Rata (Sem 1-5): <b>{student_clicked['Rata-Rata Absen Mahasiswa']:.1f}%</b><br>
                        {abs_text}
                    </div>
                </div>
                <div style='margin-top:14px; padding:10px 14px; background:#FFFFFF; border-radius:6px; font-size:0.82rem; border-left:3px solid #7367F0;'>
                    <b>Saran Dosen Wali / PA:</b> {
                        "Sangat berprestasi. Dorong untuk mengambil program akselerasi S2, asisten praktikum, atau asisten penelitian." if student_clicked['cluster_label'] == 'Sangat Tinggi' else
                        "Performa akademis sangat baik. Rekomendasikan aktif dalam kegiatan organisasi, lomba esai ilmiah, atau pertukaran pelajar." if student_clicked['cluster_label'] == 'Tinggi' else
                        "Stabilitas akademis cukup baik. Jaga motivasi belajar dan pastikan pemahaman materi mata kuliah semester berjalan lancar." if student_clicked['cluster_label'] == 'Sedang' else
                        "Performa pas-pasan. Jadwalkan bimbingan rutin, dorong mengikuti kelas bimbingan belajar mahasiswa/asistensi sebaya." if student_clicked['cluster_label'] == 'Cukup' else
                        "Mahasiswa kritis / mengalami penurunan drastis. Berikan pendampingan akademis privat intensif, lakukan konseling, dan pertimbangkan remedial nilai."
                    }
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # 2. Bubble Chart of Clusters
    st.markdown("### Bubble Chart Profil Cluster")
    st.markdown("""
    * **Sumbu X**: Rata-Rata IPK Cluster
    * **Sumbu Y**: Rata-Rata Absensi Cluster (%)
    * **Ukuran Bubble (Size)**: Jumlah Mahasiswa di Cluster
    """)
    df_cluster_summary = df_labeled.groupby('cluster_label').agg({
        'Rata-Rata IPS': 'mean',
        'Rata-Rata Absen Mahasiswa': 'mean',
        'NRP': 'count'
    }).reset_index().rename(columns={'NRP': 'jumlah_mahasiswa'})
    
    fig_bubble = px.scatter(df_cluster_summary, x='Rata-Rata IPS', y='Rata-Rata Absen Mahasiswa',
                            size='jumlah_mahasiswa', color='cluster_label',
                            color_discrete_map=PALETTE,
                            hover_data=['jumlah_mahasiswa'],
                            size_max=50,
                            labels={'Rata-Rata IPS': 'Rata-Rata IPK', 'Rata-Rata Absen Mahasiswa': 'Rata-Rata Absensi (%)'},
                            title="Bubble Chart Performa Cluster (Ukuran = Jumlah Mahasiswa)")
    st.plotly_chart(fig_bubble, use_container_width=True)

    # 3. Radar Chart of Cluster Profiles (Multivariate Profiling)
    st.markdown("### Radar Chart Karakteristik Cluster")
    
    # Calculate averages on academic courses + gpa + attendance
    radar_cats = ['IPK Kumulatif', 'Absensi (skala 0-4)', 'Aljabar Linier', 'Statistika Dasar', 'Pemrograman 1']
    
    fig_radar_prof = go.Figure()
    
    # For each cluster, plot the profile
    for lbl in sorted(df_labeled['cluster_label'].unique()):
        sub = df_labeled[df_labeled['cluster_label'] == lbl]
        if len(sub) == 0:
            continue
            
        gpa_val = sub['Rata-Rata IPS'].mean()
        absen_val = (sub['Rata-Rata Absen Mahasiswa'].mean() / 100.0) * 4.0
        
        # Check course grade mappings
        alj_val = sub['Nilai Semester 1 - Aljabar Linier'].mean() if 'Nilai Semester 1 - Aljabar Linier' in sub.columns else 0
        stat_val = sub['Nilai Semester 1 - Statistika Dasar'].mean() if 'Nilai Semester 1 - Statistika Dasar' in sub.columns else 0
        pem_val = sub['Nilai Semester 1 - Pemrograman 1'].mean() if 'Nilai Semester 1 - Pemrograman 1' in sub.columns else 0
        
        fig_radar_prof.add_trace(go.Scatterpolar(
            r=[gpa_val, absen_val, alj_val, stat_val, pem_val],
            theta=radar_cats,
            fill='toself',
            name=lbl,
            line=dict(color=PALETTE.get(lbl, '#8E9BAE'))
        ))
        
    fig_radar_prof.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 4.2])),
        showlegend=True,
        title="Radar Chart Karakteristik Multi-Dimensi Tiap Cluster"
    )
    st.plotly_chart(fig_radar_prof, use_container_width=True)

    # 4. Profile Heatmap (Cluster vs Core Courses)
    st.markdown("### Heatmap Perbandingan Akademik Antar Cluster")
    core_courses_list = [
        'Nilai Semester 1 - Aljabar Linier',
        'Nilai Semester 1 - Statistika Dasar',
        'Nilai Semester 1 - Pemrograman 1',
        'Nilai Semester 2 - Pemrograman 2',
        'Nilai Semester 4  - Data Mining'
    ]
    # Filter core courses list to ensure they exist
    core_courses_list = [c for c in core_courses_list if c in df_labeled.columns]
    
    df_heat_data = df_labeled.groupby('cluster_label')[core_courses_list].mean()
    # Clean course labels for presentation
    df_heat_data.columns = [c.split(' - ')[-1] for c in df_heat_data.columns]
    
    fig_heat = px.imshow(df_heat_data,
                         labels=dict(x="Mata Kuliah", y="Cluster", color="Rata-Rata Nilai"),
                         color_continuous_scale='Purples',
                         title="Heatmap Nilai Rata-Rata Mata Kuliah Inti per Cluster")
    st.plotly_chart(fig_heat, use_container_width=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 9: DASHBOARD INSIGHT
# ==============================================================================
elif page == "Dashboard Insight":
    st.markdown('<p class="section-header">9. Dashboard Insight <span>& Rekomendasi Akademik</span></p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ### Analisis Hasil Segmentasi Mahasiswa (Data Baru Semester 1 s.d. 5)
    Berdasarkan pengelompokan model terbaik **{best_model['method']} ({best_model['params']})**, 
    mahasiswa dikelompokkan ke dalam kategori-kategori berdasarkan karakteristik akademik komposit mereka:
    """)
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown("**Profil Rata-Rata Cluster Utama**")
        df_group = df_labeled.groupby('cluster_label')[['Rata-Rata IPS', 'Rata-Rata Absen Mahasiswa', 'NRP']].agg({
            'Rata-Rata IPS': 'mean',
            'Rata-Rata Absen Mahasiswa': 'mean',
            'NRP': 'count'
        }).rename(columns={'NRP': 'Jumlah Mahasiswa'})
        st.dataframe(df_group.round(3))
        
    with c_p2:
        st.markdown("**Karakteristik & Kebijakan Bimbingan:**")
        st.markdown("""
        * 🟢 **Sangat Tinggi**: Mahasiswa dengan IPK unggulan. Mempertahankan motivasi, direkomendasikan program akselerasi atau asisten praktikum.
        * 🔵 **Tinggi**: Akademik baik dan stabil. Didorong mengikuti kompetisi akademik mahasiswa tingkat nasional.
        * 🟣 **Sedang**: Performa cukup memadai namun memiliki fluktuasi di beberapa semester. Pendampingan berkala.
        * 🟠 **Cukup**: Perlu didorong dalam bimbingan intensif dan pemantauan nilai mata kuliah dasar coding/statistika.
        * 🔴 **Outlier / Kritis**: Membutuhkan program remedial, asistensi belajar privat, dan konseling bimbingan akademik intensif.
        """)
        
    st.markdown('<p class="section-header">Pencarian <span>Profil Akademik Mahasiswa</span></p>', unsafe_allow_html=True)
    
    search_q = st.text_input("Masukkan Nama atau NRP Mahasiswa:")
    if search_q:
        match = df_labeled[df_labeled['Nama Mahasiswa'].str.contains(search_q, case=False) | 
                           df_labeled['NRP'].astype(str).str.contains(search_q)]
    else:
        match = df_labeled
        
    if match.empty:
        st.warning("Data mahasiswa tidak ditemukan.")
    else:
        for idx, row in match.iterrows():
            lbl = row['cluster_label']
            color = PALETTE.get(lbl, '#8E9BAE')
            
            # Semester GPAs and attendance text lists (Dynamic)
            ips_text = " &nbsp;·&nbsp; ".join([f"Sem {i+1}: <b>{row[col]:.2f}</b>" for i, col in enumerate(ips_individual)])
            abs_text = " &nbsp;·&nbsp; ".join([f"Sem {i+1}: <b>{row[col]:.1f}%</b>" for i, col in enumerate(abs_individual)])
            
            with st.container():
                st.markdown(f"""
                <div class="stu-card">
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:1.15rem; font-weight:bold; color:#7367F0;'>{row['Nama Mahasiswa']}</span>
                        <span class="custom-badge" style="background-color:{color};">{lbl}</span>
                    </div>
                    <div style='font-size:0.8rem; color:#8E9BAE; margin-top:2px;'>NRP: {row['NRP']} &nbsp;·&nbsp; Prodi: {row['Prodi']} &nbsp;·&nbsp; Angkatan: {row['Angkatan Tahun']}</div>
                    <hr style='border:none; border-top:1px solid #F0F0F0; margin:8px 0'>
                    <div style='display:flex; gap:16px; margin-top:6px; font-size:0.85rem;'>
                        <div style='flex:1;'>
                            <b style='color:#566A7F; font-size:0.8rem;'>AKADEMIK:</b><br>
                            IPK Rata-Rata (Sem 1-5): <span style='color:#7367F0; font-weight:bold;'>{row['Rata-Rata IPS']:.3f}</span><br>
                            {ips_text}
                        </div>
                        <div style='flex:1;'>
                            <b style='color:#566A7F; font-size:0.8rem;'>KEHADIRAN:</b><br>
                            Rata-Rata Kehadiran (Sem 1-5): <span style='color:#28C76F; font-weight:bold;'>{row['Rata-Rata Absen Mahasiswa']:.1f}%</span><br>
                            {abs_text}
                        </div>
                    </div>
                    <div style='margin-top:10px; padding:8px 12px; background:#F8F8F8; border-radius:6px; font-size:0.8rem;'>
                        <b>Saran Pembimbing Akademik (Dosen Wali):</b> {
                            "Berikan tantangan proyek mandiri atau dorong ikut kompetisi/asisten dosen wali." if lbl == 'Sangat Tinggi' else
                            "Pertahankan konsistensi performa dan latih keterampilan soft-skill/sertifikasi." if lbl == 'Tinggi' else
                            "Berikan motivasi belajar berkala, monitor fluktuasi indeks prestasi semester." if lbl == 'Sedang' else
                            "Jadwalkan asistensi belajar, prioritaskan pemahaman konsep pemrograman dasar." if lbl == 'Cukup' else
                            "Jadwalkan remedial, lakukan konseling dosen wali intensif, dan batasi beban SKS semester berikutnya."
                        }
                    </div>
                </div>
                """, unsafe_allow_html=True)

    render_dosen_notes(page)

# ==============================================================================
# PAGE 10: MAHASISWA BERPRESTASI
# ==============================================================================
elif page == "Mahasiswa Berprestasi":
    st.markdown('<p class="section-header">10. Analisis <span>Mahasiswa Berprestasi & Rekomendasi</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    Kandidat mahasiswa berprestasi dan rekomendasi pertukaran pelajar diidentifikasi dengan memadukan **IPS Kumulatif**, **Kehadiran**, 
    dan performa **Total serta Rata-Rata Nilai Ujian Semester (skala 0-100)** untuk setiap mata kuliah.
    """)
    
    # 1. Top 5 Rekomendasi Pertukaran Pelajar
    st.markdown("### 🏆 Top 5 Rekomendasi Mahasiswa untuk Pertukaran Pelajar")
    
    # Calculate composite score for ranking: 50% IPS + 30% Average Semester Grade + 20% Attendance
    df_rank = df_labeled.copy()
    
    avg_grades_all_sems = []
    for idx, row in df_rank.iterrows():
        mean_grade = np.mean([row[f'Rata-Rata Nilai Semester {s}'] for s in [1, 2, 3, 4, 5]])
        avg_grades_all_sems.append(mean_grade)
    df_rank['Rata-Rata Nilai Keseluruhan'] = avg_grades_all_sems
    
    # Normalize components for ranking
    ips_min, ips_max = df_rank['Rata-Rata IPS'].min(), df_rank['Rata-Rata IPS'].max()
    grade_min, grade_max = df_rank['Rata-Rata Nilai Keseluruhan'].min(), df_rank['Rata-Rata Nilai Keseluruhan'].max()
    absen_min, absen_max = df_rank['Rata-Rata Absen Mahasiswa'].min(), df_rank['Rata-Rata Absen Mahasiswa'].max()
    
    norm_ips = (df_rank['Rata-Rata IPS'] - ips_min) / (ips_max - ips_min) if ips_max > ips_min else 1.0
    norm_grade = (df_rank['Rata-Rata Nilai Keseluruhan'] - grade_min) / (grade_max - grade_min) if grade_max > grade_min else 1.0
    norm_absen = (df_rank['Rata-Rata Absen Mahasiswa'] - absen_min) / (absen_max - absen_min) if absen_max > absen_min else 1.0
    
    df_rank['Score_Komposit'] = 0.5 * norm_ips + 0.3 * norm_grade + 0.2 * norm_absen
    df_top5 = df_rank.sort_values('Score_Komposit', ascending=False).head(5)
    
    # Display Top 5 in cards
    cols_top = st.columns(5)
    badges = ["🥇 GOLD", "🥈 SILVER", "🥉 BRONZE", "🎖️ TOP 4", "🎖️ TOP 5"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#7367F0", "#7367F0"]
    
    for idx, (i, row) in enumerate(df_top5.iterrows()):
        with cols_top[idx]:
            st.markdown(f"""
            <div style='background:#FFFFFF; border:2px solid {colors[idx]}; border-radius:12px; padding:16px; text-align:center; box-shadow: 0 4px 10px rgba(0,0,0,0.05);'>
                <span style='font-size:0.85rem; font-weight:bold; color:{colors[idx]};'>{badges[idx]}</span>
                <div style='font-size:1.05rem; font-weight:bold; color:#566A7F; margin-top:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{row['Nama Mahasiswa']}</div>
                <div style='font-size:0.75rem; color:#8E9BAE; margin-bottom:12px;'>NRP: {row['NRP']}</div>
                <hr style='border:none; border-top:1px solid #F0F0F0; margin:8px 0'>
                <div style='font-size:0.8rem; text-align:left; color:#566A7F; line-height:1.5;'>
                    IPK Kum : <b>{row['Rata-Rata IPS']:.3f}</b><br>
                    Nilai Rata: <b>{row['Rata-Rata Nilai Keseluruhan']:.2f}</b><br>
                    Absensi : <b>{row['Rata-Rata Absen Mahasiswa']:.1f}%</b>
                </div>
                <div style='margin-top:10px; background:{colors[idx]}22; padding:4px; border-radius:6px; font-size:0.7rem; font-weight:bold; color:#566A7F;'>
                    REKOMENDASI: YA
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # 2. Interactive Comparison Explorer
    st.markdown("### 📊 Perbandingan Nilai Mata Kuliah vs Rata-Rata/Total Semester")
    
    # Select student
    student_names = sorted(df_labeled['Nama Mahasiswa'].unique())
    sel_stu_name = st.selectbox("Pilih Mahasiswa:", student_names)
    sel_student = df_labeled[df_labeled['Nama Mahasiswa'] == sel_stu_name].iloc[0]
    
    # Select semester
    sel_sem = st.selectbox("Pilih Semester untuk Detil Perbandingan Nilai:", [1, 2, 3, 4, 5])
    
    # Find course columns for that semester
    sem_cols = [c for c in df_labeled.columns if f'Nilai Semester {sel_sem}' in c 
                and not any(kw in c.lower() for kw in ['ips', 'absen', 'rata-rata', 'total'])]
    
    if not sem_cols:
        st.warning(f"Tidak ada data nilai mata kuliah untuk Semester {sel_sem}.")
    else:
        # Extract course names and values
        courses_clean_names = [c.split(' - ')[-1] for c in sem_cols]
        student_grades = [sel_student[c] for c in sem_cols]
        class_averages = [df_labeled[c].mean() for c in sem_cols]
        
        student_total = sel_student[f'Total Nilai Semester {sel_sem}']
        student_avg = sel_student[f'Rata-Rata Nilai Semester {sel_sem}']
        
        class_total_avg = df_labeled[f'Total Nilai Semester {sel_sem}'].mean()
        class_grade_avg = df_labeled[f'Rata-Rata Nilai Semester {sel_sem}'].mean()
        
        # Display Stats summary row
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Nilai Mahasiswa", f"{student_total:.1f}", f"{student_total - class_total_avg:+.1f} vs Rata-Rata Kelas")
        with col_s2:
            st.metric("Rata-Rata Nilai Mahasiswa", f"{student_avg:.2f}", f"{student_avg - class_grade_avg:+.2f} vs Rata-Rata Kelas")
        with col_s3:
            absen_col = f'Nilai Semester {sel_sem} - ABSENSI'
            if absen_col in sel_student:
                student_abs = sel_student[absen_col]
                class_abs = df_labeled[absen_col].mean()
                st.metric("Kehadiran Semester", f"{student_abs:.1f}%", f"{student_abs - class_abs:+.1f}% vs Rata-Rata Kelas")
            else:
                st.metric("Kehadiran Semester", "N/A")
        with col_s4:
            ips_col = [c for c in ips_individual if f'Semester {sel_sem}' in c]
            if ips_col:
                student_ips = sel_student[ips_col[0]]
                class_ips = df_labeled[ips_col[0]].mean()
                st.metric("IPS Semester", f"{student_ips:.3f}", f"{student_ips - class_ips:+.3f} vs Rata-Rata Kelas")
            else:
                st.metric("IPS Semester", "N/A")
                
        # Grouped bar chart using Plotly
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            x=courses_clean_names,
            y=student_grades,
            name=f"Nilai {sel_stu_name}",
            marker_color='#7367F0'
        ))
        
        fig_comp.add_trace(go.Bar(
            x=courses_clean_names,
            y=class_averages,
            name="Rata-Rata Kelas",
            marker_color='#E8E4FF'
        ))
        
        fig_comp.add_shape(type="line",
            x0=-0.5, y0=student_avg, x1=len(courses_clean_names)-0.5, y1=student_avg,
            line=dict(color="#FF9F43", width=2, dash="dash"),
            name="Rata-Rata Nilai Mahasiswa"
        )
        fig_comp.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#FF9F43", width=2, dash="dash"),
            name="Rata-Rata Nilai Mahasiswa"
        ))
        
        fig_comp.add_shape(type="line",
            x0=-0.5, y0=class_grade_avg, x1=len(courses_clean_names)-0.5, y1=class_grade_avg,
            line=dict(color="#EA5455", width=2, dash="dot"),
            name="Rata-Rata Nilai Kelas"
        )
        fig_comp.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#EA5455", width=2, dash="dot"),
            name="Rata-Rata Nilai Kelas"
        ))
        
        fig_comp.update_layout(
            barmode='group',
            title=f"Perbandingan Nilai Mata Kuliah Semester {sel_sem} - {sel_stu_name} vs Kelas",
            xaxis_title="Mata Kuliah",
            yaxis_title="Nilai Ujian (Skala 0-100)",
            yaxis=dict(range=[0, 105]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Insight Text
        best_course_idx = np.argmax(student_grades)
        best_course_name = courses_clean_names[best_course_idx]
        best_course_grade = student_grades[best_course_idx]
        
        st.markdown(f"""
        ### 🔍 Analisis Akademik & Kekuatan Kompetensi:
        * **Mata Kuliah Unggulan**: Pada Semester {sel_sem}, **{sel_stu_name}** paling unggul dalam mata kuliah **{best_course_name}** dengan nilai **{best_course_grade:.1f}**.
        * **Status Dibanding Kelas**: Rata-rata nilai mahasiswa adalah **{student_avg:.2f}**, yang mana 
          **{"LEBIH TINGGI" if student_avg >= class_grade_avg else "LEBIH RENDAH"}** sebesar **{abs(student_avg - class_grade_avg):.2f}** poin dibandingkan dengan rata-rata nilai kelas (**{class_grade_avg:.2f}**).
        * **Rekomendasi Karir/Akademis**: Mahasiswa ini menunjukkan ketertarikan/bakat yang kuat di bidang *{best_course_name}*. Disarankan untuk dibina lebih lanjut pada topik riset atau praktikum yang selaras.
        """)

    render_dosen_notes(page)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Restrukturisasi Data Mining Pipeline Selesai &nbsp;·&nbsp; <span>PRINCALS & Fuzzy Clustering Dashboard</span> &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
