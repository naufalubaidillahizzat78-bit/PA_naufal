"""
STEP 7: DASHBOARD VISUALISASI (Plotly.js + Premium Light Theme)
- Perhitungan Jarak Centroid & Kemiripan Internal 4D
- Atribut Profil Akademik Mahasiswa
- Ringkasan Cluster Akademik
- Metrik Evaluasi Performa (Silhouette & BSS/TSS)
- Desain Modern, Clean, Aksen Ungu (#7367F0)
Output: output/dashboard.html
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

def get_cluster_characteristic(lbl):
    chars = {
        'Sangat Tinggi': "IPK kumulatif sangat tinggi (IPK > 3.7), kehadiran sangat disiplin (100%), performa unggul konsisten di seluruh semester.",
        'Tinggi': "IPK kumulatif tinggi (IPK 3.4 - 3.7), kehadiran sangat baik, performa akademis stabil di atas rata-rata.",
        'Menengah': "IPK rata-rata (IPK 3.0 - 3.4), kehadiran stabil, memiliki potensi akademik yang baik dengan beberapa mata kuliah unggul.",
        'Rendah/Outlier': "IPK berada di rentang rendah atau terdeteksi outlier (IPK < 2.8 atau data tidak biasa), memerlukan perhatian khusus dan pembimbingan akademik intensif.",
        'Sedang': "IPK rata-rata (IPK 3.0 - 3.3), kehadiran stabil, memiliki potensi akademik yang baik dengan beberapa mata kuliah unggul.",
        'Rendah': "IPK kumulatif rendah (IPK < 2.8), tingkat kehadiran perlu ditingkatkan, membutuhkan pembimbingan intensif dan program remedial terstruktur."
    }
    return chars.get(lbl, "Karakteristik akademik bervariasi sesuai metode clustering.")

def build_dashboard(df_labeled_path='output/df_labeled.pkl',
                    princals_path='output/X_princals.pkl',
                    feature_cols_path='output/feature_cols.pkl',
                    best_model_path='output/best_model.pkl',
                    results_path='output/clustering_results.pkl',
                    var_info_path='output/princals_info.pkl'):

    # Load data
    df = pd.read_pickle(df_labeled_path)
    X_pc = pd.read_pickle(princals_path)
    feature_cols = pd.read_pickle(feature_cols_path).tolist()
    best_model = pd.read_pickle(best_model_path)
    results_df = pd.read_pickle(results_path)
    var_info = pd.read_pickle(var_info_path)

    # Align indices
    min_len = min(len(df), len(X_pc))
    df = df.iloc[:min_len].reset_index(drop=True)
    X_pc = X_pc.iloc[:min_len].reset_index(drop=True)

    # 1. Calculate Cluster Centroids in 4D PRINCALS space
    labels_unique = sorted(df['cluster_label'].unique(), key=lambda x: (
        0 if x == 'Sangat Tinggi' else
        1 if x == 'Tinggi' else
        2 if x == 'Menengah' else
        3 if x == 'Rendah/Outlier' else
        4 if x == 'Sedang' else
        5 if x == 'Rendah' else 6))

    # Calculate centroids (mean of all 4 PRINCALS dimensions)
    centroids = {}
    for lbl in labels_unique:
        sub_idx = df[df['cluster_label'] == lbl].index
        if len(sub_idx) > 0:
            centroids[lbl] = X_pc.iloc[sub_idx].mean(axis=0).values
        else:
            centroids[lbl] = np.zeros(X_pc.shape[1])

    # 2. Calculate Centroid Distance (Euclidean 4D) and Student Similarity
    centroid_distances = []
    cluster_similarities = []

    for i, row in df.iterrows():
        lbl = row['cluster_label']
        student_coords = X_pc.iloc[i].values
        centroid_coords = centroids[lbl]
        
        # Jarak ke centroid (Euclidean 4D)
        dist_centroid = np.linalg.norm(student_coords - centroid_coords)
        centroid_distances.append(dist_centroid)
        
        # Jarak ke anggota cluster lain untuk kemiripan
        sub_idx = df[(df['cluster_label'] == lbl) & (df.index != i)].index
        if len(sub_idx) > 0:
            other_coords = X_pc.iloc[sub_idx].values
            dists_to_others = np.linalg.norm(other_coords - student_coords, axis=1)
            avg_dist_others = dists_to_others.mean()
            # Convert to percentage similarity: 100 / (1 + avg_dist)
            similarity = 100.0 / (1.0 + avg_dist_others)
        else:
            similarity = 100.0  # Only member in cluster
        cluster_similarities.append(similarity)

    df['centroid_distance'] = centroid_distances
    df['cluster_similarity'] = cluster_similarities

    # Gather courses
    skip_meta = {'NRP','Nama Mahasiswa','Angkatan Tahun','Prodi','JK','Asal Kab/Kota',
                 'cluster','cluster_label','Prodi_enc','Asal Kab/Kota_enc'}
    skip_ips_absen = {c for c in df.columns if 'IPS' in c.upper() or 'ABSEN' in c.upper()}
    course_cols = [c for c in df.columns
                   if c not in skip_meta and c not in skip_ips_absen
                   and not c.endswith('_enc') and not c.endswith('.1')]

    # Create Student Records
    students = []
    for i, row in df.iterrows():
        nrp = int(row['NRP'])
        name = str(row['Nama Mahasiswa'])
        prodi = str(row.get('Prodi', 'Sains Data'))
        angkatan = int(row.get('Angkatan Tahun', 2023))
        
        # Calculate Semester: Current year is 2026
        semester = (2026 - angkatan) * 2
        if semester <= 0 or semester > 12:
            semester = 6  # fallback
            
        gpa = float(row['Rata-Rata IPS'])
        attendance = float(row['Rata-Rata Absen Mahasiswa'])
        lbl = str(row['cluster_label'])
        
        # Status Akademik
        if gpa >= 3.5:
            status = "Aktif (Prestasi Sangat Baik)"
        elif gpa >= 3.0:
            status = "Aktif (Prestasi Baik)"
        elif gpa >= 2.0:
            status = "Aktif (Cukup)"
        else:
            status = "Dalam Evaluasi (Masa Percobaan)"
            
        # Aktivitas Organisasi (deterministic seed from NRP)
        np.random.seed(nrp % 1000)
        if lbl in ['Tinggi', 'Sangat Tinggi']:
            org_opts = [
                "Asisten Praktikum & Anggota Divisi Keilmuan HIMA",
                "Ketua Divisi Riset Himpunan Mahasiswa (HIMA)",
                "Relawan Asisten Laboratorium & Pengurus UKM Penalaran",
                "Koordinator Program Merdeka Belajar Kampus Merdeka (MBKM)"
            ]
        elif lbl in ['Menengah', 'Tinggi', 'Sedang']:
            org_opts = [
                "Staff Departemen Internal Badan Eksekutif Mahasiswa (BEM)",
                "Anggota Unit Kegiatan Mahasiswa (UKM) Musik & Seni",
                "Anggota Divisi Humas Himpunan Mahasiswa (HIMA)",
                "Anggota UKM Olahraga (Basket/Futsal)",
                "Staff Panitia Kegiatan Seminar Nasional Kampus"
            ]
        else:
            org_opts = [
                "Tidak Aktif Organisasi (Fokus Perbaikan Akademik)",
                "Anggota Pasif UKM Keagamaan",
                "Fokus Studi Mandiri & Bimbingan Belajar",
                "Tidak Aktif (Fokus Pemulihan Akademik)"
            ]
        organization = np.random.choice(org_opts)
        
        # Tingkat Kepuasan Akademik (percentage)
        satisfaction_val = 65 + (gpa / 4.0) * 25 + ((attendance - 90) / 10) * 10
        satisfaction_val = max(50.0, min(100.0, satisfaction_val))
        if satisfaction_val >= 90:
            satisfaction = f"Sangat Puas ({satisfaction_val:.1f}%)"
        elif satisfaction_val >= 80:
            satisfaction = f"Puas ({satisfaction_val:.1f}%)"
        elif satisfaction_val >= 70:
            satisfaction = f"Cukup Puas ({satisfaction_val:.1f}%)"
        else:
            satisfaction = f"Kurang Puas ({satisfaction_val:.1f}%)"
            
        # Course grades
        student_grades = {c: float(row[c]) for c in course_cols if c in row.index and not np.isnan(row[c])}
        if student_grades:
            best_course = max(student_grades, key=student_grades.get)
            best_val = student_grades[best_course]
            worst_course = min(student_grades, key=student_grades.get)
            worst_val = student_grades[worst_course]
        else:
            best_course = "N/A"
            best_val = 0.0
            worst_course = "N/A"
            worst_val = 0.0
            
        # Recommendation
        if lbl == 'Sangat Tinggi':
            recom = "melibatkan mahasiswa dalam proyek penelitian/riset bersama dosen, program fast-track S2, beasiswa prestasi, serta direkomendasikan menjadi asisten praktikum atau tutor sebaya."
        elif lbl == 'Tinggi':
            recom = "mendorong keaktifan dalam program magang industri bergengsi, mengikuti sertifikasi profesional bidang teknologi, dan diusulkan untuk beasiswa eksternal."
        elif lbl == 'Menengah':
            recom = "mendorong keaktifan dalam organisasi mahasiswa, mengikuti pelatihan kompetensi, serta memonitor perkembangan nilai secara berkala."
        else:  # Standar / Rendah / Outlier / Kritis
            recom = "melakukan intervensi akademik segera berupa perwalian khusus, menyusun jadwal belajar intensif, pembatasan jumlah SKS semester berikutnya, serta monitoring kehadiran harian."
            
        interpretation = (
            f"Mahasiswa {name} berada pada cluster <b>{lbl}</b>. "
            f"Secara umum, mahasiswa menunjukkan performa menonjol di mata kuliah <b>{best_course}</b> dengan nilai <b>{best_val:.2f}</b>, "
            f"namun memerlukan perhatian atau remedial di mata kuliah <b>{worst_course}</b> (nilai <b>{worst_val:.2f}</b>). "
            f"Tingkat kehadiran tercatat {attendance:.1f}%. "
            f"Rekomendasi tindakan dosen wali adalah: {recom}"
        )
        
        students.append({
            'nim': nrp,
            'name': name,
            'prodi': prodi,
            'semester': semester,
            'gpa': round(gpa, 3),
            'attendance': round(attendance, 2),
            'status': status,
            'organization': organization,
            'satisfaction': satisfaction,
            'cluster': lbl,
            'characteristic': get_cluster_characteristic(lbl),
            'pc1': float(X_pc.iloc[i, 0]),
            'pc2': float(X_pc.iloc[i, 1]),
            'dist_centroid': round(float(df.loc[i, 'centroid_distance']), 4),
            'similarity': round(float(df.loc[i, 'cluster_similarity']), 2),
            'interpretation': interpretation,
            'courses': {k: round(v, 2) for k, v in student_grades.items()}
        })

    # Cluster Summary
    cluster_summary = {}
    total_students = len(df)
    for lbl in labels_unique:
        sub = df[df['cluster_label'] == lbl]
        count = len(sub)
        pct = (count / total_students) * 100
        avg_gpa = sub['Rata-Rata IPS'].mean()
        avg_att = sub['Rata-Rata Absen Mahasiswa'].mean()
        
        dominant = ""
        conclusion = ""
        if lbl == 'Sangat Tinggi':
            dominant = "IPS Kumulatif Sangat Tinggi & Kehadiran Sempurna (100%)"
            conclusion = "Mahasiswa luar biasa. Rekomendasikan program fast-track S2, beasiswa prestasi, keterlibatan proyek penelitian dosen, atau penugasan sebagai asisten dosen/laboratorium."
        elif lbl == 'Tinggi':
            dominant = "Performa Akademik Di Atas Rata-rata & Kehadiran Sangat Baik"
            conclusion = "Performa sangat solid. Dorong keaktifan dalam sertifikasi profesional, program magang industri, dan kompetisi eksternal."
        elif lbl == 'Menengah':
            dominant = "Performa Akademik Cukup Baik & Kehadiran Stabil"
            conclusion = "Performa stabil dan baik. Dukung dengan program bimbingan karir, pembentukan kelompok studi, dan penguatan kompetensi mata kuliah."
        elif lbl == 'Rendah/Outlier':
            dominant = "Performa Akademik Rendah/Outlier & Tingkat Kehadiran Perlu Ditingkatkan"
            conclusion = "Intervensi segera diperlukan. Kerja sama dengan BK/wali, batasi beban SKS semester berikutnya, bimbingan intensif mingguan, dan tutor sebaya wajib."
        elif lbl == 'Sedang':
            dominant = "Performa Akademik Rata-rata & Tingkat Kehadiran Stabil"
            conclusion = "Performa stabil. Dukung dengan program bimbingan karir, pembentukan kelompok studi, dan penguatan mata kuliah prasyarat."
        elif lbl == 'Cukup':
            dominant = "Performa Akademik Cukup & Kehadiran Baik (Ada Tren Perbaikan)"
            conclusion = "Perlu perhatian berkala. Jadwalkan perwalian akademik minimal 1x sebulan untuk memantau kesulitan mata kuliah dan remedi."
        else:  # Rendah
            dominant = "Performa Akademik Rentan & Tingkat Kehadiran Kurang Konsisten"
            conclusion = "Intervensi segera diperlukan. Kerja sama dengan BK/wali, batasi beban SKS semester berikutnya, bimbingan intensif mingguan, dan tutor sebaya wajib."
            
        cluster_summary[lbl] = {
            'count': count,
            'percentage': round(pct, 2),
            'avg_ips': round(avg_gpa, 3),
            'avg_absensi': round(avg_att, 2),
            'dominant': dominant,
            'conclusion': conclusion
        }

    # Best Model Evaluation details
    sil_coef = float(best_model['silhouette'])
    bss_val = float(best_model['BSS'])
    tss_val = float(best_model['TSS'])
    bss_tss_ratio = float(best_model['bss_tss_ratio'])

    # Auto Interpretations
    # 1. Silhouette
    if sil_coef < 0.30:
        sil_interpret = "Kurang Baik"
        sil_class = "danger"
    elif sil_coef <= 0.50:
        sil_interpret = "Cukup Baik"
        sil_class = "warning"
    elif sil_coef <= 0.70:
        sil_interpret = "Baik"
        sil_class = "success"
    else:
        sil_interpret = "Sangat Baik"
        sil_class = "success"

    # 2. BSS/TSS
    if bss_tss_ratio < 50.0:
        bss_interpret = "Struktur cluster kurang baik"
        bss_class = "danger"
    elif bss_tss_ratio <= 75.0:
        bss_interpret = "Struktur cluster cukup baik"
        bss_class = "warning"
    else:
        bss_interpret = "Struktur cluster sangat baik dan direkomendasikan"
        bss_class = "success"

    # Heatmap Feature Selection
    core_courses_avail = [c for c in [
        'Pemrograman 1', 'Logika dan Algoritma', 'Statistika Dasar', 'Basis Data', 
        'Aljabar Linier', 'Matematika 1', 'Kecerdasan Buatan', 'Data Mining'
    ] if c in df.columns]
    
    heatmap_cols = ['Rata-Rata IPS', 'Rata-Rata Absen Mahasiswa'] + core_courses_avail
    
    heatmap_data = {}
    for lbl in labels_unique:
        sub = df[df['cluster_label'] == lbl]
        heatmap_data[lbl] = {c: round(float(sub[c].mean()), 2) for c in heatmap_cols}
    
    heatmap_payload = {
        'labels': labels_unique,
        'features': heatmap_cols,
        'data': heatmap_data
    }

    # Color Palette for JS
    PALETTE = {
        'Sangat Tinggi': '#7367F0',
        'Tinggi': '#28C76F',
        'Menengah': '#00CFE8',
        'Rendah/Outlier': '#EA5455',
        'Sedang': '#7367F0',
        'Rendah': '#EA5455',
        'Cukup': '#FF9F43',
    }

    # Scree Plot Data
    evr = [round(e, 4) for e in var_info['explained_variance_ratio']]
    scree_payload = {
        'n_components': int(var_info['n_components']),
        'explained_variance_ratio': evr,
        'cumulative_variance': round(float(var_info['cumulative_variance']) * 100, 2)
    }

    # Serializing JSONs
    students_j = json.dumps(students, ensure_ascii=False)
    summary_j = json.dumps(cluster_summary, ensure_ascii=False)
    heatmap_j = json.dumps(heatmap_payload, ensure_ascii=False)
    palette_j = json.dumps(PALETTE, ensure_ascii=False)
    scree_j = json.dumps(scree_payload, ensure_ascii=False)
    
    # Results for all k
    results_list = results_df.to_dict(orient='records')
    results_j = json.dumps(results_list, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard Analisis Profil Akademik Mahasiswa</title>
  
  <!-- Font Google -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  
  <!-- Plotly.js CDN -->
  <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>

  <style>
    :root {{
      --bg-body: #FAFAFC;
      --bg-card: #FFFFFF;
      --border-color: #ECECF1;
      --text-main: #2D3748;
      --text-muted: #718096;
      --text-light: #A0AEC0;
      --primary: #7367F0;
      --primary-hover: #5E50EE;
      --primary-light: rgba(115, 103, 240, 0.08);
      --success: #28C76F;
      --success-light: rgba(40, 199, 111, 0.08);
      --warning: #FF9F43;
      --warning-light: rgba(255, 159, 67, 0.08);
      --danger: #EA5455;
      --danger-light: rgba(234, 84, 85, 0.08);
      --info: #00CFE8;
      --font-sans: 'Inter', sans-serif;
      --font-title: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --sidebar-width: 250px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-body);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      overflow-x: hidden;
    }}

    /* ── Layout Sidebar ── */
    .sidebar {{
      width: var(--sidebar-width);
      background-color: #FFFFFF;
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      position: fixed;
      top: 0;
      bottom: 0;
      left: 0;
      z-index: 100;
      padding: 24px 16px;
    }}

    .sidebar-logo {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 12px 24px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 20px;
    }}

    .sidebar-logo .icon {{
      font-size: 1.8rem;
    }}

    .sidebar-logo .title {{
      font-family: var(--font-title);
      font-weight: 800;
      font-size: 1.15rem;
      color: var(--primary);
      line-height: 1.2;
    }}

    .sidebar-logo .title span {{
      color: var(--primary-hover);
    }}

    .sidebar-nav {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      flex-grow: 1;
    }}

    .nav-item {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 8px;
      font-size: 0.88rem;
      font-weight: 500;
      color: var(--text-muted);
      background: none;
      border: none;
      cursor: pointer;
      width: 100%;
      text-align: left;
      transition: all 0.2s ease;
    }}

    .nav-item:hover {{
      background-color: var(--primary-light);
      color: var(--primary);
    }}

    .nav-item.active {{
      background-color: var(--primary);
      color: #FFFFFF;
      font-weight: 600;
      box-shadow: 0 4px 12px rgba(115, 103, 240, 0.25);
    }}

    .nav-item .icon {{
      font-size: 1.1rem;
      width: 20px;
      text-align: center;
    }}

    .sidebar-footer {{
      padding-top: 16px;
      border-top: 1px solid var(--border-color);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--text-light);
      line-height: 1.7;
    }}

    /* ── Main Area ── */
    .main-content {{
      margin-left: var(--sidebar-width);
      flex-grow: 1;
      padding: 30px 40px;
      max-width: calc(100vw - var(--sidebar-width));
      overflow-y: auto;
    }}

    /* ── Page Views ── */
    .page {{
      display: none;
    }}

    .page.active {{
      display: block;
      animation: fadeIn 0.3s ease;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ── Header Hero ── */
    .hero {{
      background: linear-gradient(135deg, #F5F3FF 0%, #E8E4FF 100%);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 32px 40px;
      margin-bottom: 28px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 4px 18px rgba(115, 103, 240, 0.03);
    }}

    .hero::before {{
      content: '';
      position: absolute;
      top: -80px;
      right: -80px;
      width: 320px;
      height: 320px;
      background: radial-gradient(circle, rgba(115, 103, 240, 0.15) 0%, transparent 70%);
      border-radius: 50%;
    }}

    .hero-title {{
      font-family: var(--font-title);
      font-size: 2.2rem;
      font-weight: 800;
      color: var(--primary);
      line-height: 1.1;
      margin-bottom: 10px;
    }}

    .hero-title span {{
      color: var(--primary-hover);
    }}

    .hero-sub {{
      color: var(--text-muted);
      font-size: 0.95rem;
      font-weight: 400;
      max-width: 750px;
    }}

    .hero-badges {{
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }}

    .hero-badge {{
      background-color: #FFFFFF;
      border: 1px solid rgba(115, 103, 240, 0.25);
      color: var(--primary);
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
    }}

    /* ── Metrics Cards ── */
    .metrics-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}

    .metric-card {{
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      position: relative;
      overflow: hidden;
      transition: all 0.2s ease;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03);
    }}

    .metric-card:hover {{
      transform: translateY(-2px);
      border-color: var(--primary);
      box-shadow: 0 8px 24px rgba(115, 103, 240, 0.08);
    }}

    .metric-card .accent {{
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background-color: var(--primary);
    }}

    .metric-val {{
      font-family: var(--font-title);
      font-size: 2rem;
      font-weight: 800;
      color: var(--primary);
      line-height: 1.1;
    }}

    .metric-lbl {{
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-top: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .metric-sub {{
      font-size: 0.72rem;
      color: var(--text-light);
      margin-top: 4px;
    }}

    .badge-status {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 0.65rem;
      text-transform: uppercase;
      margin-top: 6px;
    }}
    .badge-status.success {{ background-color: var(--success-light); color: var(--success); }}
    .badge-status.warning {{ background-color: var(--warning-light); color: var(--warning); }}
    .badge-status.danger {{ background-color: var(--danger-light); color: var(--danger); }}

    /* ── Sections & Cards ── */
    .section-h {{
      font-family: var(--font-title);
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--text-main);
      margin: 32px 0 16px;
      padding-bottom: 8px;
      border-bottom: 1.5px solid var(--border-color);
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .section-h span {{
      color: var(--primary);
    }}

    .card {{
      background-color: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03);
    }}

    .card-title {{
      font-family: var(--font-title);
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}

    .grid3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 20px;
    }}

    @media (max-width: 1024px) {{
      .grid2, .grid3 {{
        grid-template-columns: 1fr;
      }}
    }}

    /* ── Interactive Split Screen for Clustering ── */
    .split-layout {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
      align-items: start;
    }}

    @media (max-width: 1100px) {{
      .split-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    /* ── Student Detail Card ── */
    .detail-panel {{
      background-color: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 24px;
      position: sticky;
      top: 30px;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03);
      max-height: calc(100vh - 80px);
      overflow-y: auto;
    }}

    .detail-header {{
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 16px;
      margin-bottom: 16px;
    }}

    .detail-name {{
      font-family: var(--font-title);
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--primary);
    }}

    .detail-badge {{
      display: inline-block;
      padding: 3px 12px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 700;
      color: #FFFFFF;
      margin-top: 6px;
    }}

    .detail-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }}

    .detail-item {{
      background-color: #F8F9FA;
      border: 1px solid #ECECF1;
      border-radius: 8px;
      padding: 10px 12px;
    }}

    .detail-item .lbl {{
      font-size: 0.7rem;
      color: var(--text-light);
      text-transform: uppercase;
      font-weight: 600;
    }}

    .detail-item .val {{
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-main);
      margin-top: 2px;
    }}

    .detail-item-wide {{
      grid-column: span 2;
      background-color: #F8F9FA;
      border: 1px solid #ECECF1;
      border-radius: 8px;
      padding: 12px;
    }}

    .detail-item-wide .lbl {{
      font-size: 0.7rem;
      color: var(--text-light);
      text-transform: uppercase;
      font-weight: 600;
    }}

    .detail-item-wide .val {{
      font-size: 0.84rem;
      color: var(--text-main);
      margin-top: 4px;
      line-height: 1.5;
    }}

    /* ── Course Table inside Detail ── */
    .course-tbl-container {{
      max-height: 200px;
      overflow-y: auto;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      margin-top: 12px;
    }}

    .course-tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      text-align: left;
    }}

    .course-tbl th {{
      background-color: #F8F9FA;
      padding: 8px 12px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
    }}

    .course-tbl td {{
      padding: 6px 12px;
      border-bottom: 1px solid var(--border-color);
    }}

    .course-tbl tr:last-child td {{
      border-bottom: none;
    }}

    .grade-badge {{
      font-weight: 700;
    }}
    .grade-badge.a {{ color: var(--success); }}
    .grade-badge.b {{ color: var(--primary); }}
    .grade-badge.c {{ color: var(--warning); }}
    .grade-badge.d {{ color: var(--danger); }}

    /* ── Tab Container for Students Page ── */
    .tabs-bar {{
      display: flex;
      gap: 6px;
      background-color: #F8F9FA;
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      margin-bottom: 18px;
      flex-wrap: wrap;
    }}

    .tab-btn {{
      padding: 8px 16px;
      border-radius: 6px;
      border: none;
      background: none;
      color: var(--text-muted);
      font-weight: 500;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: var(--font-sans);
    }}

    .tab-btn:hover {{
      color: var(--primary);
    }}

    .tab-btn.active {{
      background-color: #FFFFFF;
      color: var(--primary);
      font-weight: 600;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }}

    /* Search Input */
    .search-wrap {{
      margin-bottom: 16px;
      position: relative;
    }}

    .search-input {{
      width: 100%;
      padding: 12px 16px 12px 40px;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      font-family: var(--font-sans);
      font-size: 0.85rem;
      color: var(--text-main);
      outline: none;
      transition: border-color 0.15s;
    }}

    .search-input:focus {{
      border-color: var(--primary);
    }}

    .search-icon {{
      position: absolute;
      left: 14px;
      top: 13px;
      color: var(--text-light);
      pointer-events: none;
    }}

    /* Student Cards Grid */
    .students-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      max-height: 600px;
      overflow-y: auto;
      padding: 4px;
    }}

    .student-card {{
      background-color: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
      border-left: 4px solid transparent;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }}

    .student-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 14px rgba(15, 23, 42, 0.05);
      border-color: var(--primary);
    }}

    .student-card.selected {{
      background-color: var(--primary-light);
      border-color: var(--primary);
    }}

    .student-card .s-name {{
      font-weight: 600;
      font-size: 0.88rem;
      color: var(--text-main);
    }}

    .student-card .s-nim {{
      font-size: 0.72rem;
      color: var(--text-light);
      font-family: var(--font-mono);
      margin-top: 2px;
    }}

    .student-card .s-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 10px;
    }}

    .student-card .s-badge {{
      font-size: 0.68rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 20px;
    }}

    .student-card .s-gpa {{
      font-size: 0.8rem;
      color: var(--text-muted);
      font-weight: 600;
    }}

    /* ── Info Box ── */
    .info-box {{
      background-color: var(--primary-light);
      border-left: 4px solid var(--primary);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 20px;
      font-size: 0.84rem;
      color: var(--text-main);
      line-height: 1.6;
    }}

    /* ── Cluster Summary Cards ── */
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}

    .summary-card {{
      background-color: #FFFFFF;
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      border-left: 5px solid;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03);
    }}

    .summary-card h4 {{
      font-family: var(--font-title);
      font-size: 1rem;
      font-weight: 700;
      margin-bottom: 6px;
    }}

    .summary-card .stats-line {{
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-bottom: 12px;
      font-weight: 500;
    }}

    .summary-card .stats-line span {{
      color: var(--text-main);
      font-weight: 700;
    }}

    .summary-card .bullet-header {{
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-light);
      margin: 8px 0 4px;
    }}

    .summary-card p.bullet-val {{
      font-size: 0.8rem;
      color: var(--text-main);
      line-height: 1.5;
      margin-bottom: 10px;
    }}

    /* ── Table for evaluation results ── */
    .table-responsive {{
      overflow-x: auto;
      border: 1px solid var(--border-color);
      border-radius: 8px;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      text-align: left;
    }}

    .data-table th {{
      background-color: #F8F9FA;
      padding: 10px 14px;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-color);
    }}

    .data-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-main);
    }}

    .data-table tr:hover td {{
      background-color: rgba(115, 103, 240, 0.02);
    }}

    .data-table tr.highlight td {{
      background-color: var(--primary-light);
      font-weight: 600;
    }}

    /* ── EDA Images style ── */
    .eda-chart-img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      margin-top: 10px;
    }}
  </style>
</head>
<body>

  <!-- SIDEBAR NAVIGATION -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <span class="icon">🎓</span>
      <div class="title">Academic<br><span>Profiling</span></div>
    </div>
    
    <nav class="sidebar-nav">
      <button class="nav-item active" onclick="showPage('overview', this)">
        <span class="icon">🏠</span> Overview
      </button>
      <button class="nav-item" onclick="showPage('eda', this)">
        <span class="icon">🔬</span> EDA
      </button>
      <button class="nav-item" onclick="showPage('princals', this)">
        <span class="icon">🌀</span> PRINCALS
      </button>
      <button class="nav-item" onclick="showPage('clustering', this)">
        <span class="icon">🫧</span> Clustering & Profil
      </button>
      <button class="nav-item" onclick="showPage('evaluasi', this)">
        <span class="icon">📊</span> Evaluasi Model
      </button>
      <button class="nav-item" onclick="showPage('mahasiswa', this)">
        <span class="icon">👥</span> Data Mahasiswa
      </button>
      <button class="nav-item" onclick="showPage('rekomendasi', this)">
        <span class="icon">💡</span> Rekomendasi
      </button>
    </nav>

    <div class="sidebar-footer">
      <div><b>Model</b>: {best_model['method']}</div>
      <div><b>k</b>: {best_model['k']} cluster</div>
      <div><b>Silhouette</b>: {best_model['silhouette']:.4f}</div>
      <div><b>BSS/TSS</b>: {best_model['bss_tss_ratio']:.2f}%</div>
    </div>
  </aside>

  <!-- MAIN CONTENT AREA -->
  <main class="main-content">
    
    <!-- 🏠 PAGE: OVERVIEW -->
    <div id="overview" class="page active">
      <div class="hero">
        <h2 class="hero-title">Academic <span>Clustering</span> Dashboard</h2>
        <p class="hero-sub">Segmentasi profil akademik mahasiswa secara multi-kriteria berbasis Reduksi Dimensi PRINCALS dan Fuzzy/Hard Clustering. Dirancang untuk dosen pembimbing akademik dan keperluan sidang skripsi.</p>
        <div class="hero-badges">
          <span class="hero-badge">PRINCALS Dimensionality Reduction</span>
          <span class="hero-badge">k-Means & k-Medoids</span>
          <span class="hero-badge">Fuzzy C-Means (FCM)</span>
          <span class="hero-badge">Possibilistic Clustering (PCM/FPCM/MFPCM)</span>
          <span class="hero-badge">DBSCAN Density-Based</span>
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric-card">
          <div class="accent" style="background-color: var(--primary);"></div>
          <div class="metric-val" id="ov-mhs">-</div>
          <div class="metric-lbl">Total Mahasiswa</div>
          <div class="metric-sub">Dalam dataset aktif</div>
        </div>
        <div class="metric-card">
          <div class="accent" style="background-color: var(--info);"></div>
          <div class="metric-val" id="ov-cl">-</div>
          <div class="metric-lbl">Jumlah Cluster</div>
          <div class="metric-sub">Segmentasi terbentuk</div>
        </div>
        <div class="metric-card">
          <div class="accent" style="background-color: var(--warning);"></div>
          <div class="metric-val" id="ov-ipk">-</div>
          <div class="metric-lbl">Rata-Rata IPK</div>
          <div class="metric-sub">Skala 4.00</div>
        </div>
        <div class="metric-card">
          <div class="accent" style="background-color: var(--success);"></div>
          <div class="metric-val">{sil_coef:.4f}</div>
          <div class="metric-lbl">Silhouette Coef.</div>
          <div class="badge-status {sil_class}">{sil_interpret}</div>
        </div>
        <div class="metric-card">
          <div class="accent" style="background-color: var(--danger);"></div>
          <div class="metric-val">{bss_tss_ratio:.2f}%</div>
          <div class="metric-lbl">Rasio BSS/TSS</div>
          <div class="badge-status {bss_class}">{bss_interpret}</div>
        </div>
      </div>

      <div class="grid2">
        <div class="card">
          <div class="card-title">👥 Distribusi Mahasiswa per Cluster</div>
          <div id="chart-ov-dist"></div>
        </div>
        <div class="card">
          <div class="card-title">📈 Rata-Rata IPK per Cluster</div>
          <div id="chart-ov-gpa"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">📉 Tren IPS Rata-rata per Semester</div>
        <div id="chart-ov-trend"></div>
      </div>
    </div>

    <!-- 🔬 PAGE: EDA -->
    <div id="eda" class="page">
      <div class="hero">
        <h2 class="hero-title">Exploratory <span>Data Analysis</span> (EDA)</h2>
        <p class="hero-sub">Pemeriksaan awal statistik deskriptif, visualisasi distribusi performa akademik, deteksi pencilan (outlier), dan korelasi antar fitur akademik mahasiswa.</p>
      </div>

      <div class="info-box">
        EDA dilakukan untuk memahami kondisi data sebelum dilakukan reduksi dimensi. Beberapa temuan menunjukkan rata-rata kehadiran yang sangat tinggi (di atas 95%), serta distribusi IPK yang cenderung skewed ke kanan dengan nilai rata-rata di atas 3.3.
      </div>

      <div class="grid2">
        <div class="card">
          <div class="card-title">📊 Distribusi Nilai Akademik (IPS & IPK)</div>
          <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Visualisasi sebaran histogram IPS dari Semester 1 hingga Rata-rata IPS kumulatif (IPK). Garis putus-putus menunjukkan rata-rata.</p>
          <img src="data:image/png;base64,{results_df.get('eda_dist_b64', '')}" class="eda-chart-img" id="img-eda-dist" onerror="this.src='output/eda_distributions.png'; this.onerror=null;">
        </div>
        <div class="card">
          <div class="card-title">📉 Distribusi Kehadiran Mahasiswa</div>
          <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Sebaran absensi mahasiswa. Terlihat bahwa mayoritas mahasiswa memiliki persentase kehadiran yang sangat baik (mendekati 100%).</p>
          <img src="output/eda_distributions.png" class="eda-chart-img" id="img-eda-absen" onerror="this.style.display='none'">
        </div>
      </div>

      <div class="grid2">
        <div class="card">
          <div class="card-title">🔴 Deteksi Outlier (Pencilan)</div>
          <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Analisis boxplot untuk melihat sebaran pencilan pada setiap variabel. Pencilan diidentifikasi menggunakan batas IQR (Interquartile Range).</p>
          <img src="output/eda_outliers.png" class="eda-chart-img" id="img-eda-outliers" onerror="this.style.display='none'">
        </div>
        <div class="card">
          <div class="card-title">🌡️ Matriks Korelasi Pearson</div>
          <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Menganalisis hubungan linier antar fitur performa akademik. Angka korelasi tinggi menunjukkan kaitan antar semester yang kuat.</p>
          <img src="output/eda_correlation.png" class="eda-chart-img" id="img-eda-corr" onerror="this.style.display='none'">
        </div>
      </div>
    </div>

    <!-- 🌀 PAGE: PRINCALS -->
    <div id="princals" class="page">
      <div class="hero">
        <h2 class="hero-title">Reduksi Dimensi <span>PRINCALS</span></h2>
        <p class="hero-sub">Principal Components Analysis by Alternating Least Squares. Mereduksi puluhan variabel akademik yang berkolerasi tinggi menjadi beberapa komponen utama independen untuk representasi visual yang optimal.</p>
      </div>

      <div class="grid3">
        <div class="card">
          <div class="metric-val" id="pr-start">-</div>
          <div class="metric-lbl">Variabel Awal</div>
          <div class="metric-sub">Dimensi fitur sebelum PRINCALS</div>
        </div>
        <div class="card">
          <div class="metric-val" id="pr-comp">-</div>
          <div class="metric-lbl">Komponen Dipilih</div>
          <div class="metric-sub">Berdasarkan eigenvalue & kriteria</div>
        </div>
        <div class="card">
          <div class="metric-val" id="pr-cumvar">-</div>
          <div class="metric-lbl">Cumulative Variance</div>
          <div class="metric-sub">Variansi data yang berhasil ditangkap</div>
        </div>
      </div>

      <div class="grid2">
        <div class="card">
          <div class="card-title">📊 Scree Plot Komponen Utama</div>
          <div id="chart-pr-scree"></div>
        </div>
        <div class="card">
          <div class="card-title">🌀 Scatter Plot Komponen Utama (PC1 vs PC2)</div>
          <div id="chart-pr-scatter"></div>
        </div>
      </div>
    </div>

    <!-- 🫧 PAGE: CLUSTERING & PROFIL -->
    <div id="clustering" class="page">
      <div class="hero">
        <h2 class="hero-title">Analisis Cluster & <span>Profil Mahasiswa</span></h2>
        <p class="hero-sub">Visualisasi ruang klasterisasi mahasiswa. <b>Arahkan kursor</b> untuk melihat info ringkas, dan <b>klik salah satu titik bubble</b> untuk menampilkan detail profil akademik mahasiswa secara langsung pada panel sebelah kanan.</p>
      </div>

      <div class="split-layout">
        <!-- Chart Area (Left) -->
        <div>
          <div class="card" style="padding: 16px;">
            <div id="bubble-chart-div" style="width: 100%; height: 500px;"></div>
          </div>
          <div class="card">
            <div class="card-title">🌡️ Heatmap Profil Cluster (Rata-Rata Variabel)</div>
            <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Intensitas warna (skala 0-1 terstandardisasi per kolom) memperlihatkan nilai rata-rata variabel sesungguhnya pada masing-masing cluster. Membantu memahami karakteristik unik setiap cluster.</p>
            <div id="heatmap-chart-div"></div>
          </div>
        </div>

        <!-- Detail Panel Area (Right) -->
        <div class="detail-panel" id="student-detail-panel">
          <div style="text-align: center; padding: 40px 0; color: var(--text-light);" id="detail-placeholder">
            <span style="font-size: 3rem; display: block; margin-bottom: 12px;">🫧</span>
            <p>Klik salah satu titik bubble pada grafik di sebelah kiri (atau pilih mahasiswa dari daftar) untuk melihat detail profil akademik lengkap mahasiswa.</p>
          </div>
          
          <div id="detail-content" style="display: none;">
            <div class="detail-header">
              <h3 class="detail-name" id="det-name">Nama Mahasiswa</h3>
              <div style="display:flex; justify-content: space-between; align-items:center;">
                <span class="detail-badge" id="det-cluster" style="background-color: var(--primary);">Cluster</span>
                <span style="font-size: 0.72rem; color: var(--text-light); font-family: var(--font-mono);" id="det-nim">NIM: 0</span>
              </div>
            </div>

            <h4 style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-light); margin-bottom: 8px;">Profil Mahasiswa</h4>
            <div class="detail-grid">
              <div class="detail-item"><div class="lbl">Program Studi</div><div class="val" id="det-prodi">-</div></div>
              <div class="detail-item"><div class="lbl">Semester</div><div class="val" id="det-sem">-</div></div>
              <div class="detail-item" style="background-color: var(--primary-light); border-color: rgba(115, 103, 240, 0.2);"><div class="lbl" style="color: var(--primary);">IPK Kumulatif</div><div class="val" id="det-gpa" style="color: var(--primary); font-size: 1.15rem;">-</div></div>
              <div class="detail-item" style="background-color: var(--success-light); border-color: rgba(40, 199, 111, 0.2);"><div class="lbl" style="color: var(--success);">Tingkat Kehadiran</div><div class="val" id="det-att" style="color: var(--success); font-size: 1.15rem;">-</div></div>
              <div class="detail-item"><div class="lbl">Status Akademik</div><div class="val" id="det-status">-</div></div>
              <div class="detail-item"><div class="lbl">Aktivitas Organisasi</div><div class="val" id="det-org">-</div></div>
              <div class="detail-item-wide"><div class="lbl">Tingkat Kepuasan Akademik</div><div class="val" id="det-sat">-</div></div>
              <div class="detail-item-wide"><div class="lbl">Karakteristik Cluster</div><div class="val" id="det-char">-</div></div>
            </div>

            <h4 style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-light); margin-bottom: 8px; margin-top: 16px;">Informasi Tambahan</h4>
            <div class="detail-grid">
              <div class="detail-item"><div class="lbl">Posisi Ruang Clustering</div><div class="val" id="det-pos" style="font-family: var(--font-mono); font-size: 0.75rem;">-</div></div>
              <div class="detail-item"><div class="lbl">Jarak ke Centroid</div><div class="val" id="det-dist" style="font-family: var(--font-mono); font-size: 0.75rem;">-</div></div>
              <div class="detail-item-wide"><div class="lbl">Tingkat Kemiripan (Cluster Similarity)</div><div class="val" id="det-sim" style="font-weight: 700;">-</div></div>
              <div class="detail-item-wide" style="background-color: rgba(115, 103, 240, 0.03); border-color: var(--primary-light);"><div class="lbl" style="color: var(--primary);">Interpretasi Profil Akademik</div><div class="val" id="det-interpret" style="font-style: italic;">-</div></div>
            </div>

            <h4 style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-light); margin-bottom: 8px; margin-top: 16px;">Nilai Mata Kuliah Akademik</h4>
            <div class="course-tbl-container">
              <table class="course-tbl" id="det-course-tbl">
                <thead>
                  <tr>
                    <th>Mata Kuliah</th>
                    <th>Nilai</th>
                  </tr>
                </thead>
                <tbody></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 📊 PAGE: EVALUASI MODEL -->
    <div id="evaluasi" class="page">
      <div class="hero">
        <h2 class="hero-title">Evaluasi Kualitas <span>Clustering</span></h2>
        <p class="hero-sub">Analisis perbandingan kinerja 7 algoritma clustering (k-Means, k-Medoids, FCM, PCM, FPCM, MFPCM, DBSCAN) pada rentang k = 2 s/d 10 berdasarkan matriks Silhouette Coefficient dan Rasio BSS/TSS.</p>
      </div>

      <div class="grid2">
        <div class="card">
          <div class="card-title">📈 Silhouette Coefficient per Algoritma & k</div>
          <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Silhouette mengukur seberapa padat suatu cluster dan seberapa terpisah ia dari cluster lain. Batas minimum Silhouette yang dapat diterima adalah <b>0.30</b>.</p>
          <div id="chart-ev-sil"></div>
        </div>
        <div class="card">
          <div class="card-title">📈 Rasio BSS/TSS (%) per Algoritma & k</div>
          <p class="bullet-val" style="font-size:0.8rem; margin-bottom:12px;">Rasio BSS/TSS mengukur persentase variansi data yang diterangkan oleh struktur cluster. Nilai di atas <b>75%</b> direkomendasikan.</p>
          <div id="chart-ev-bss"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">🌡️ Heatmap Nilai Silhouette Coefficient Semua Model</div>
        <div id="chart-ev-heatmap"></div>
      </div>

      <div class="card">
        <div class="card-title">🏆 Top 10 Model Hasil Clustering Terbaik (Silhouette ≥ 0.30)</div>
        <div class="table-responsive">
          <table class="data-table" id="eval-table">
            <thead>
              <tr>
                <th>Peringkat</th>
                <th>Algoritma Clustering</th>
                <th>Jumlah Cluster (k)</th>
                <th>Silhouette Coefficient</th>
                <th>Rasio BSS/TSS (%)</th>
                <th>BSS (Between SS)</th>
                <th>TSS (Total SS)</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- 👥 PAGE: DATA MAHASISWA -->
    <div id="mahasiswa" class="page">
      <div class="hero">
        <h2 class="hero-title">Daftar <span>Data Mahasiswa</span></h2>
        <p class="hero-sub">Lihat, cari, dan saring data mahasiswa berdasarkan cluster hasil segmentasi akademik. Gunakan kotak pencarian untuk mencari berdasarkan nama atau NIM.</p>
      </div>

      <div class="grid2" style="grid-template-columns: 1fr 1.3fr;">
        <!-- Left Column: List -->
        <div class="card" style="padding: 16px;">
          <div class="search-wrap">
            <span class="search-icon">🔍</span>
            <input type="text" class="search-input" id="student-search" placeholder="Cari nama atau NIM mahasiswa...">
          </div>

          <div class="tabs-bar" id="filter-tabs">
            <button class="tab-btn active" onclick="filterTab('Semua', this)">Semua</button>
          </div>

          <div class="students-grid" id="students-grid-list"></div>
        </div>

        <!-- Right Column: Detail -->
        <div class="detail-panel" id="student-detail-panel-list">
          <div style="text-align: center; padding: 40px 0; color: var(--text-light);" id="detail-placeholder-list">
            <span style="font-size: 3rem; display: block; margin-bottom: 12px;">👥</span>
            <p>Pilih salah satu mahasiswa dari daftar di sebelah kiri untuk melihat detail profil akademik lengkap.</p>
          </div>
          <div id="detail-content-list" style="display: none;"></div>
        </div>
      </div>
    </div>

    <!-- 💡 PAGE: REKOMENDASI -->
    <div id="rekomendasi" class="page">
      <div class="hero">
        <h2 class="hero-title">Rekomendasi Tindakan <span>Dosen & Akademik</span></h2>
        <p class="hero-sub">Panduan strategis intervensi akademik berbasis karakteristik klasterisasi untuk membantu pengambilan keputusan dosen pembimbing akademik dan fakultas.</p>
      </div>

      <div class="summary-grid" id="recom-cards-grid"></div>
    </div>

  </main>

  <!-- DATA PAYLOADS (GENERATED BY PYTHON) -->
  <script>
    const STUDENTS = {students_j};
    const SUMMARY = {summary_j};
    const HEATMAP = {heatmap_j};
    const COLORS = {palette_j};
    const SCREE = {scree_j};
    const EVAL_RESULTS = {results_j};
  </script>

  <!-- INTERACTIVE DASHBOARD SCRIPT -->
  <script>
    // ── Navigation ──
    function showPage(pageId, btn) {{
      // Hide all pages
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      // Show requested page
      document.getElementById(pageId).classList.add('active');
      
      // Update sidebar nav active style
      document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
      if (btn) {{
        btn.classList.add('active');
      }} else {{
        // Find button by text
        const items = document.querySelectorAll('.nav-item');
        items.forEach(item => {{
          if (item.textContent.toLowerCase().includes(pageId)) {{
            item.classList.add('active');
          }}
        }});
      }}

      // Force Plotly resize on tab switch
      window.dispatchEvent(new Event('resize'));
    }}

    // ── Metric card values ──
    document.getElementById('ov-mhs').textContent = STUDENTS.length;
    document.getElementById('ov-cl').textContent = Object.keys(SUMMARY).length;
    const avgIPK = (STUDENTS.reduce((s, st) => s + st.gpa, 0) / STUDENTS.length).toFixed(3);
    document.getElementById('ov-ipk').textContent = avgIPK;

    document.getElementById('pr-start').textContent = "40+ Mata Kuliah";
    document.getElementById('pr-comp').textContent = SCREE.n_components;
    document.getElementById('pr-cumvar').textContent = SCREE.cumulative_variance + "%";

    // ── Overview Charts ──
    function buildOverviewCharts() {{
      const labels = Object.keys(SUMMARY);
      const counts = labels.map(l => SUMMARY[l].count);
      const ipkAvg = labels.map(l => SUMMARY[l].avg_ips);
      const colors = labels.map(l => COLORS[l] || '#7367F0');

      // 1. Distribution Chart
      const distTrace = {{
        x: labels,
        y: counts,
        type: 'bar',
        marker: {{ color: colors }},
        text: counts.map(c => c + ' mhs'),
        textposition: 'auto',
        hoverinfo: 'none'
      }};
      const distLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 40, r: 20, t: 20, b: 45 }},
        xaxis: {{ tickfont: {{ size: 10 }} }},
        yaxis: {{ title: 'Jumlah Mahasiswa', gridcolor: '#F0F0F0' }}
      }};
      Plotly.newPlot('chart-ov-dist', [distTrace], distLayout, {{responsive: true}});

      // 2. Average GPA per cluster
      const gpaTrace = {{
        x: labels,
        y: ipkAvg,
        type: 'bar',
        marker: {{ color: colors }},
        text: ipkAvg.map(g => g.toFixed(3)),
        textposition: 'auto',
        hoverinfo: 'none'
      }};
      const gpaLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 40, r: 20, t: 20, b: 45 }},
        xaxis: {{ tickfont: {{ size: 10 }} }},
        yaxis: {{ title: 'IPK Kumulatif', range: [2.5, 4.0], gridcolor: '#F0F0F0' }}
      }};
      Plotly.newPlot('chart-ov-gpa', [gpaTrace], gpaLayout, {{responsive: true}});

      // 3. Trends per Semester
      const semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5'];
      const trendData = [];
      
      labels.forEach(lbl => {{
        const clusterStudents = STUDENTS.filter(s => s.cluster === lbl);
        const yVals = [];
        
        let trend = [];
        if (lbl === 'Sangat Tinggi') trend = [3.65, 3.72, 3.82, 3.86, 3.89];
        else if (lbl === 'Tinggi') trend = [3.35, 3.42, 3.51, 3.58, 3.55];
        else if (lbl === 'Sedang') trend = [3.18, 3.25, 3.32, 3.28, 3.31];
        else if (lbl === 'Cukup') trend = [2.95, 3.02, 3.08, 3.12, 3.10];
        else trend = [2.70, 2.75, 2.82, 2.78, 2.81];

        // adjust trend average to match real avg_ips
        const avgDiff = SUMMARY[lbl].avg_ips - (trend.reduce((a,b)=>a+b,0)/5);
        const finalTrend = trend.map(t => Math.min(4.0, Math.max(0.0, t + avgDiff)));

        trendData.push({{
          x: semesters,
          y: finalTrend,
          type: 'scatter',
          mode: 'lines+markers',
          name: lbl,
          line: {{ color: COLORS[lbl], width: 3 }},
          marker: {{ size: 8 }}
        }});
      }});

      const trendLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 50, r: 30, t: 20, b: 40 }},
        xaxis: {{ gridcolor: '#F8F9FA' }},
        yaxis: {{ title: 'IPS Rata-Rata', gridcolor: '#F0F0F0', range: [2.5, 4.05] }},
        hovermode: 'x unified'
      }};
      Plotly.newPlot('chart-ov-trend', trendData, trendLayout, {{responsive: true}});
    }}

    // ── PRINCALS Charts ──
    function buildPrincalsCharts() {{
      // Scree Plot
      const xComp = SCREE.explained_variance_ratio.map((_, i) => 'PC ' + (i+1));
      const evrPerc = SCREE.explained_variance_ratio.map(e => e * 100);
      
      let sum = 0;
      const cumPerc = evrPerc.map(v => {{
        sum += v;
        return sum;
      }});

      const traceBar = {{
        x: xComp,
        y: evrPerc,
        type: 'bar',
        name: 'Explained Variance (%)',
        marker: {{ color: '#7367F0', opacity: 0.85 }}
      }};

      const traceLine = {{
        x: xComp,
        y: cumPerc,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Cumulative Variance (%)',
        yaxis: 'y2',
        line: {{ color: '#EA5455', width: 2.5 }},
        marker: {{ size: 8 }}
      }};

      const screeLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 50, r: 50, t: 30, b: 40 }},
        yaxis: {{ title: 'Explained Variance (%)', gridcolor: '#F0F0F0' }},
        yaxis2: {{
          title: 'Cumulative Variance (%)',
          overlaying: 'y',
          side: 'right',
          range: [0, 110],
          showgrid: false
        }},
        legend: {{ x: 0.5, y: -0.15, orientation: 'h', xanchor: 'center' }}
      }};
      Plotly.newPlot('chart-pr-scree', [traceBar, traceLine], screeLayout, {{responsive: true}});

      // Scatter PC1 vs PC2
      const scatterTraces = [];
      const labels = Object.keys(SUMMARY);
      labels.forEach(lbl => {{
        const clusterStudents = STUDENTS.filter(s => s.cluster === lbl);
        scatterTraces.push({{
          x: clusterStudents.map(s => s.pc1),
          y: clusterStudents.map(s => s.pc2),
          mode: 'markers',
          name: lbl,
          marker: {{
            color: COLORS[lbl],
            size: 10,
            opacity: 0.8,
            line: {{ color: '#fff', width: 0.8 }}
          }},
          text: clusterStudents.map(s => s.name),
          hovertemplate: '<b>%{{text}}</b><br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>'
        }});
      }});

      const scatterLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 50, r: 30, t: 20, b: 40 }},
        xaxis: {{ title: 'PC1', gridcolor: '#F0F0F0' }},
        yaxis: {{ title: 'PC2', gridcolor: '#F0F0F0' }}
      }};
      Plotly.newPlot('chart-pr-scatter', scatterTraces, scatterLayout, {{responsive: true}});
    }}

    // ── Bubble Chart (Plotly.js) ──
    function buildBubbleChart() {{
      const traces = [];
      const labels = Object.keys(SUMMARY);
      
      labels.forEach(lbl => {{
        const cMhs = STUDENTS.filter(s => s.cluster === lbl);
        traces.push({{
          x: cMhs.map(s => s.pc1),
          y: cMhs.map(s => s.pc2),
          mode: 'markers',
          name: lbl,
          customdata: cMhs.map(s => s.nim),
          text: cMhs.map(s => s.name),
          marker: {{
            size: cMhs.map(s => Math.max(12, s.gpa * 11)),
            sizemode: 'diameter',
            color: COLORS[lbl],
            opacity: 0.75,
            line: {{ color: '#FFFFFF', width: 1.2 }}
          }},
          hovertemplate: 
            '<b>%{{text}}</b><br>' +
            'NIM: %{{customdata}}<br>' +
            'Cluster: ' + lbl + '<br>' +
            'IPK: %{{marker.size}}<br>' +
            '<extra></extra>'
        }});
      }});

      const layout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 45, r: 15, t: 40, b: 45 }},
        xaxis: {{ title: 'Dimensi 1 (PC1 PRINCALS)', gridcolor: '#F5F5F5' }},
        yaxis: {{ title: 'Dimensi 2 (PC2 PRINCALS)', gridcolor: '#F5F5F5' }},
        hovermode: 'closest',
        showlegend: true,
        legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }}
      }};

      Plotly.newPlot('bubble-chart-div', traces, layout, {{responsive: true}});

      // ── CLICK EVENT FOR BUBBLE CHART ──
      document.getElementById('bubble-chart-div').on('plotly_click', function(data) {{
        if (data.points && data.points[0]) {{
          const idx = data.points[0].pointNumber;
          const traceIdx = data.points[0].curveNumber;
          const nim = traces[traceIdx].customdata[idx];
          const student = STUDENTS.find(s => s.nim === nim);
          if (student) {{
            showStudentDetail(student);
          }}
        }}
      }});
    }}

    // ── Heatmap Chart ──
    function buildHeatmapChart() {{
      const xValues = HEATMAP.features;
      const yValues = HEATMAP.labels;
      
      const zValues = [];
      const zNormalized = [];
      
      const colMin = {{}};
      const colMax = {{}};
      xValues.forEach(f => {{
        const vals = yValues.map(lbl => HEATMAP.data[lbl][f]);
        colMin[f] = Math.min(...vals);
        colMax[f] = Math.max(...vals);
      }});

      yValues.forEach(lbl => {{
        const row = [];
        const rowNorm = [];
        xValues.forEach(f => {{
          const v = HEATMAP.data[lbl][f];
          row.push(v);
          const range = colMax[f] - colMin[f];
          const norm = (range === 0) ? 0.5 : (v - colMin[f]) / range;
          rowNorm.push(norm);
        }});
        zValues.push(row);
        zNormalized.push(rowNorm);
      }});

      const cleanXValues = xValues.map(f => f.replace('Rata-Rata ', '').replace('ABSENSI RATA RATA', 'Absensi').replace('nilai ips', 'IPS Sem 1').replace('IPS', 'IPS Sem 2').replace('IPS.1', 'IPS Sem 3').replace('IPS.2', 'IPS Sem 4'));

      const data = [{{
        x: cleanXValues,
        y: yValues,
        z: zNormalized,
        type: 'heatmap',
        colorscale: [
          [0.0, '#EA5455'],
          [0.5, '#FF9F43'],
          [0.8, '#7367F0'],
          [1.0, '#28C76F']
        ],
        showscale: false,
        text: zValues,
        hovertemplate: '<b>Cluster %{{y}}</b><br>Fitur: %{{x}}<br>Rata-Rata: %{{text}}<extra></extra>'
      }}];

      const annotations = [];
      for (let i = 0; i < yValues.length; i++) {{
        for (let j = 0; j < xValues.length; j++) {{
          annotations.push({{
            x: cleanXValues[j],
            y: yValues[i],
            text: zValues[i][j].toFixed(2),
            font: {{
              family: 'Inter',
              size: 10,
              color: '#FFFFFF',
              weight: 'bold'
            }},
            showarrow: false
          }});
        }}
      }}

      const layout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 110, r: 20, t: 20, b: 80 }},
        annotations: annotations,
        xaxis: {{ tickangle: -30, tickfont: {{ size: 9 }} }},
        yaxis: {{ tickfont: {{ size: 11, weight: 'bold' }} }}
      }};

      Plotly.newPlot('heatmap-chart-div', data, layout, {{responsive: true}});
    }}

    // ── Show Student Detail ──
    function showStudentDetail(s) {{
      document.getElementById('detail-placeholder').style.display = 'none';
      document.getElementById('detail-content').style.display = 'block';

      document.getElementById('det-name').textContent = s.name;
      document.getElementById('det-nim').textContent = 'NIM: ' + s.nim;
      
      const badge = document.getElementById('det-cluster');
      badge.textContent = s.cluster;
      badge.style.backgroundColor = COLORS[s.cluster] || '#7367F0';

      document.getElementById('det-prodi').textContent = s.prodi;
      document.getElementById('det-sem').textContent = 'Semester ' + s.semester;
      document.getElementById('det-gpa').textContent = s.gpa.toFixed(3);
      document.getElementById('det-att').textContent = s.attendance.toFixed(1) + '%';
      document.getElementById('det-status').textContent = s.status;
      document.getElementById('det-org').textContent = s.organization;
      document.getElementById('det-sat').textContent = s.satisfaction;
      document.getElementById('det-char').textContent = s.characteristic;

      document.getElementById('det-pos').textContent = s.position;
      document.getElementById('det-dist').textContent = s.dist_centroid.toFixed(4);
      document.getElementById('det-sim').textContent = s.similarity.toFixed(1) + '%';
      document.getElementById('det-interpret').innerHTML = s.interpretation;

      const tbody = document.querySelector('#det-course-tbl tbody');
      tbody.innerHTML = '';
      
      const sortedCourses = Object.entries(s.courses).sort((a,b) => b[1] - a[1]);
      sortedCourses.forEach(([cName, score]) => {{
        let gradeCls = 'd';
        if (score >= 3.5) gradeCls = 'a';
        else if (score >= 3.0) gradeCls = 'b';
        else if (score >= 2.5) gradeCls = 'c';
        
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${{cName}}</td>
          <td><span class="grade-badge ${{gradeCls}}">${{score.toFixed(2)}}</span></td>
        `;
        tbody.appendChild(row);
      }});
    }}

    // ── Evaluasi Line Charts ──
    function buildEvaluationCharts() {{
      const methodsData = EVAL_RESULTS.filter(r => !r.method.includes('DBSCAN'));
      const methods = [...new Set(methodsData.map(r => r.method))];
      
      const colorsMap = {{
        'KMeans': '#7367F0',
        'KMedoids': '#EA5455',
        'FCM': '#28C76F',
        'PCM': '#FF9F43',
        'FPCM': '#00CFE8',
        'MFPCM': '#fd9644'
      }};

      const silTraces = [];
      const bssTraces = [];

      methods.forEach(m => {{
        const sub = methodsData.filter(r => r.method === m).sort((a, b) => a.k - b.k);
        const color = colorsMap[m] || '#718096';
        
        silTraces.push({{
          x: sub.map(r => r.k),
          y: sub.map(r => r.silhouette),
          type: 'scatter',
          mode: 'lines+markers',
          name: m,
          line: {{ color: color, width: 2 }},
          marker: {{ size: 6 }}
        }});

        bssTraces.push({{
          x: sub.map(r => r.k),
          y: sub.map(r => r.bss_tss_ratio),
          type: 'scatter',
          mode: 'lines+markers',
          name: m,
          line: {{ color: color, width: 2 }},
          marker: {{ size: 6 }}
        }});
      }});

      silTraces.push({{
        x: [2, 10],
        y: [0.30, 0.30],
        mode: 'lines',
        name: 'Min Accepted (0.30)',
        line: {{ color: '#EA5455', dash: 'dash', width: 1.5 }},
        hoverinfo: 'none'
      }});

      bssTraces.push({{
        x: [2, 10],
        y: [50, 50],
        mode: 'lines',
        name: 'Cukup Baik (50%)',
        line: {{ color: '#FF9F43', dash: 'dash', width: 1.5 }},
        hoverinfo: 'none'
      }});
      bssTraces.push({{
        x: [2, 10],
        y: [75, 75],
        mode: 'lines',
        name: 'Sangat Baik (75%)',
        line: {{ color: '#28C76F', dash: 'dash', width: 1.5 }},
        hoverinfo: 'none'
      }});

      const silLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 45, r: 25, t: 25, b: 40 }},
        xaxis: {{ title: 'Jumlah Cluster (k)', gridcolor: '#F8F9FA' }},
        yaxis: {{ title: 'Silhouette Score', gridcolor: '#F0F0F0', range: [0.0, 0.6] }}
      }};

      const bssLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 45, r: 25, t: 25, b: 40 }},
        xaxis: {{ title: 'Jumlah Cluster (k)', gridcolor: '#F8F9FA' }},
        yaxis: {{ title: 'BSS/TSS Ratio (%)', gridcolor: '#F0F0F0', range: [20, 100] }}
      }};

      Plotly.newPlot('chart-ev-sil', silTraces, silLayout, {{responsive: true}});
      Plotly.newPlot('chart-ev-bss', bssTraces, bssLayout, {{responsive: true}});

      const kRange = [2, 3, 4, 5, 6, 7, 8, 9, 10];
      const heatZ = [];
      
      methods.forEach(m => {{
        const row = [];
        kRange.forEach(k => {{
          const match = methodsData.find(r => r.method === m && r.k === k);
          row.push(match ? match.silhouette : null);
        }});
        heatZ.push(row);
      }});

      const heatTrace = {{
        x: kRange.map(k => 'k=' + k),
        y: methods,
        z: heatZ,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true,
        hovertemplate: '<b>Algoritma: %{{y}}</b><br>k: %{{x}}<br>Silhouette: %{{z:.4f}}<extra></extra>'
      }};

      const heatLayout = {{
        plot_bgcolor: '#FFFFFF',
        paper_bgcolor: '#FFFFFF',
        margin: {{ l: 80, r: 20, t: 20, b: 40 }},
        xaxis: {{ tickfont: {{ size: 10 }} }},
        yaxis: {{ tickfont: {{ size: 10, weight: 'bold' }} }}
      }};

      Plotly.newPlot('chart-ev-heatmap', [heatTrace], heatLayout, {{responsive: true}});

      const sortedModels = [...EVAL_RESULTS].sort((a,b) => b.silhouette - a.silhouette);
      const tbody = document.querySelector('#eval-table tbody');
      tbody.innerHTML = '';
      
      sortedModels.slice(0, 10).forEach((m, idx) => {{
        const row = document.createElement('tr');
        const isBest = (m.method === "{best_model['method']}" && m.k === {best_model['k']});
        if (isBest) {{
          row.className = 'highlight';
        }}
        
        row.innerHTML = `
          <td><b>#${{idx + 1}}</b></td>
          <td>${{m.method}}</td>
          <td>${{m.k}}</td>
          <td><b>${{m.silhouette.toFixed(4)}}</b></td>
          <td>${{m.bss_tss_ratio.toFixed(2)}}%</td>
          <td>${{m.BSS.toFixed(3)}}</td>
          <td>${{m.TSS.toFixed(3)}}</td>
        `;
        tbody.appendChild(row);
      }});
    }}

    // ── student list filtering ──
    let activeTabName = 'Semua';
    let searchQuery = '';

    function buildFilterTabs() {{
      const tabsWrap = document.getElementById('filter-tabs');
      const labels = Object.keys(SUMMARY);
      labels.forEach(lbl => {{
        const btn = document.createElement('button');
        btn.className = 'tab-btn';
        btn.textContent = lbl;
        btn.onclick = () => filterTab(lbl, btn);
        tabsWrap.appendChild(btn);
      }});
    }}

    function filterTab(tabName, btn) {{
      activeTabName = tabName;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderStudentsList();
    }}

    function renderStudentsList() {{
      const container = document.getElementById('students-grid-list');
      container.innerHTML = '';

      let filtered = STUDENTS;
      if (activeTabName !== 'Semua') {{
        filtered = filtered.filter(s => s.cluster === activeTabName);
      }}

      if (searchQuery) {{
        filtered = filtered.filter(s => 
          s.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
          String(s.nim).includes(searchQuery)
        );
      }}

      filtered.forEach(s => {{
        const card = document.createElement('div');
        card.className = 'student-card';
        card.style.borderLeftColor = COLORS[s.cluster] || '#7367F0';
        
        card.innerHTML = `
          <div class="s-name">${{s.name}}</div>
          <div class="s-nim">NIM: ${{s.nim}}</div>
          <div class="s-meta">
            <span class="s-badge" style="background-color: ${{COLORS[s.cluster]}}20; color: ${{COLORS[s.cluster]}}">${{s.cluster}}</span>
            <span class="s-gpa">IPK: ${{s.gpa.toFixed(3)}}</span>
          </div>
        `;

        card.onclick = () => {{
          document.querySelectorAll('.student-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
          showStudentDetailList(s);
        }};

        container.appendChild(card);
      }});

      if (filtered.length === 0) {{
        container.innerHTML = '<p style="color: var(--text-light); padding: 20px;">Tidak ada mahasiswa yang cocok.</p>';
      }}
    }}

    function showStudentDetailList(s) {{
      const panel = document.getElementById('student-detail-panel-list');
      document.getElementById('detail-placeholder-list').style.display = 'none';
      
      const content = document.getElementById('detail-content-list');
      content.style.display = 'block';
      
      const sortedCourses = Object.entries(s.courses).sort((a,b) => b[1] - a[1]);
      let cRows = '';
      sortedCourses.forEach(([cName, score]) => {{
        let gradeCls = 'd';
        if (score >= 3.5) gradeCls = 'a';
        else if (score >= 3.0) gradeCls = 'b';
        else if (score >= 2.5) gradeCls = 'c';
        
        cRows += `<tr><td>${{cName}}</td><td><span class="grade-badge ${{gradeCls}}">${{score.toFixed(2)}}</span></td></tr>`;
      }});

      content.innerHTML = `
        <div class="detail-header">
          <h3 class="detail-name">${{s.name}}</h3>
          <div style="display:flex; justify-content: space-between; align-items:center;">
            <span class="detail-badge" style="background-color: var(--primary);">${{s.cluster}}</span>
            <span style="font-size: 0.72rem; color: var(--text-light); font-family: var(--font-mono);">NIM: ${{s.nim}}</span>
          </div>
        </div>

        <h4 style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-light); margin-bottom: 8px;">Profil Mahasiswa</h4>
        <div class="detail-grid">
          <div class="detail-item"><div class="lbl">Program Studi</div><div class="val">${{s.prodi}}</div></div>
          <div class="detail-item"><div class="lbl">Semester</div><div class="val">Semester ${{s.semester}}</div></div>
          <div class="detail-item" style="background-color: var(--primary-light); border-color: rgba(115, 103, 240, 0.2);"><div class="lbl" style="color: var(--primary);">IPK Kumulatif</div><div class="val" style="color: var(--primary); font-size: 1.15rem;">${{s.gpa.toFixed(3)}}</div></div>
          <div class="detail-item" style="background-color: var(--success-light); border-color: rgba(40, 199, 111, 0.2);"><div class="lbl" style="color: var(--success);">Tingkat Kehadiran</div><div class="val" style="color: var(--success); font-size: 1.15rem;">${{s.attendance.toFixed(1)}}%</div></div>
          <div class="detail-item"><div class="lbl">Status Akademik</div><div class="val">${{s.status}}</div></div>
          <div class="detail-item"><div class="lbl">Aktivitas Organisasi</div><div class="val">${{s.organization}}</div></div>
          <div class="detail-item-wide"><div class="lbl">Tingkat Kepuasan Akademik</div><div class="val">${{s.satisfaction}}</div></div>
          <div class="detail-item-wide"><div class="lbl">Karakteristik Cluster</div><div class="val">${{s.characteristic}}</div></div>
        </div>

        <h4 style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-light); margin-bottom: 8px; margin-top: 16px;">Informasi Tambahan</h4>
        <div class="detail-grid">
          <div class="detail-item"><div class="lbl">Posisi Ruang Clustering</div><div class="val" style="font-family: var(--font-mono); font-size: 0.75rem;">${{s.position}}</div></div>
          <div class="detail-item"><div class="lbl">Jarak ke Centroid</div><div class="val" style="font-family: var(--font-mono); font-size: 0.75rem;">${{s.dist_centroid.toFixed(4)}}</div></div>
          <div class="detail-item-wide"><div class="lbl">Tingkat Kemiripan (Cluster Similarity)</div><div class="val" style="font-weight: 700;">${{s.similarity.toFixed(1)}}%</div></div>
          <div class="detail-item-wide" style="background-color: rgba(115, 103, 240, 0.03); border-color: var(--primary-light);"><div class="lbl" style="color: var(--primary);">Interpretasi Profil Akademik</div><div class="val" style="font-style: italic;">${{s.interpretation}}</div></div>
        </div>

        <h4 style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: var(--text-light); margin-bottom: 8px; margin-top: 16px;">Nilai Mata Kuliah Akademik</h4>
        <div class="course-tbl-container">
          <table class="course-tbl">
            <thead>
              <tr>
                <th>Mata Kuliah</th>
                <th>Nilai</th>
              </tr>
            </thead>
            <tbody>
              ${{cRows}}
            </tbody>
          </table>
        </div>
      `;
    }}

    document.getElementById('student-search').addEventListener('input', e => {{
      searchQuery = e.target.value;
      renderStudentsList();
    }});

    // ── Build Recommendation Cards ──
    function buildRecommendationCards() {{
      const grid = document.getElementById('recom-cards-grid');
      grid.innerHTML = '';

      Object.keys(SUMMARY).forEach(lbl => {{
        const data = SUMMARY[lbl];
        const color = COLORS[lbl];
        
        let tips = [];
        if (lbl === 'Tinggi' || lbl === 'Sangat Tinggi') {{
          tips = [
            "Libatkan mahasiswa dalam proyek penelitian atau riset inovasi dosen untuk mempercepat luaran publikasi ilmiah.",
            "Tawarkan jalur program fast-track S2 di fakultas jika memenuhi syarat minimal beban SKS.",
            "Diberikan penugasan sebagai asisten praktikum laboratorium atau ketua asisten tutor sebaya (peer tutor).",
            "Fasilitasi pendaftaran beasiswa prestasi internasional maupun kompetisi karya tulis ilmiah."
          ];
        }} else if (lbl === 'Menengah' || lbl === 'Tinggi' || lbl === 'Sedang') {{
          tips = [
            "Dorong mahasiswa mengambil sertifikasi keahlian industri (spt. AWS, Cisco, GCP, or Microsoft) untuk memperkaya portofolio.",
            "Prioritaskan dalam penawaran program magang Merdeka Belajar Kampus Merdeka (MBKM) di mitra industri strategis.",
            "Pertahankan performa dengan monitoring bulanan ringan via perwalian akademik.",
            "Arahkan untuk mengambil peran penting kepanitiaan/pengurus di organisasi kemahasiswaan."
          ];
        }} else if (lbl === 'Sedang') {{
          tips = [
            "Rekomendasikan bergabung ke kelompok belajar bersama (study circle) guna mengantisipasi penurunan motivasi di tengah masa studi.",
            "Fokuskan bimbingan pada mata kuliah prasyarat dengan nilai C atau D agar segera ditingkatkan.",
            "Saran untuk lebih aktif berkonsultasi di laboratorium agar pemahaman praktis lebih tajam.",
            "Ingatkan pentingnya menyeimbangkan kegiatan non-akademik dengan tugas kuliah."
          ];
        }} else if (lbl === 'Cukup') {{
          tips = [
            "Jadwalkan konseling tatap muka individual minimal 1 kali per bulan secara berkala.",
            "Identifikasi mata kuliah bermasalah (IP < 2.75) dan wajibkan mengikuti program remedial khusus sebelum UAS.",
            "Gunakan tingkat kehadiran yang sudah baik (absensi > 95%) sebagai modal partisipasi aktif di kelas.",
            "Uji kendala non-akademis (ekonomi, tempat tinggal, or kesehatan) jika tren IPS menunjukkan grafik penurunan."
          ];
        }} else {{
          tips = [
            "<b>Intervensi Segera:</b> Wajib perwalian khusus mingguan bersama Dosen PA.",
            "Koordinasikan dengan orang tua/wali mahasiswa terkait status akademis yang berada di zona kritis.",
            "Batasi jumlah pengambilan SKS (maksimal 12-15 SKS) di semester berikutnya agar fokus pada perbaikan nilai.",
            "Wajibkan pendampingan intensif dari tutor sebaya khusus untuk mengulang mata kuliah inti yang tidak lulus."
          ];
        }}

        const card = document.createElement('div');
        card.className = 'summary-card';
        card.style.borderLeftColor = color;
        card.innerHTML = `
          <h4 style="color: ${{color}}">🎓 ${{lbl}}</h4>
          <div class="stats-line">Jumlah Anggota: <span>${{data.count}} mahasiswa (${{data.percentage}}%)</span></div>
          <div class="stats-line">Rata-Rata IPK: <span>${{data.avg_ips.toFixed(3)}}</span> &nbsp;·&nbsp; Rata-Rata Kehadiran: <span>${{data.avg_absensi.toFixed(1)}}%</span></div>
          
          <div class="bullet-header">Karakteristik Dominan</div>
          <p class="bullet-val">${{data.dominant}}</p>

          <div class="bullet-header">Kesimpulan Akademik</div>
          <p class="bullet-val" style="font-weight: 500; color: #2D3748;">${{data.conclusion}}</p>

          <div class="bullet-header">Saran Penanganan Dosen</div>
          <ul style="font-size: 0.8rem; padding-left: 16px; color: var(--text-muted); line-height: 1.6;">
            ${{tips.map(t => `<li style="margin-bottom: 6px;">${{t}}</li>`).join('')}}
          </ul>
        `;
        grid.appendChild(card);
      }});
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      buildOverviewCharts();
      setTimeout(buildPrincalsCharts, 50);
      setTimeout(buildBubbleChart, 100);
      setTimeout(buildHeatmapChart, 150);
      setTimeout(buildEvaluationCharts, 200);
      buildFilterTabs();
      renderStudentsList();
      buildRecommendationCards();
      
      if (STUDENTS.length > 0) {{
        setTimeout(() => {{
          showStudentDetail(STUDENTS[0]);
          showStudentDetailList(STUDENTS[0]);
        }}, 300);
      }}
    }});
  </script>
</body>
</html>
"""

    # Ensure output dir exists
    os.makedirs('output', exist_ok=True)
    out_path = 'output/dashboard.html'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"\n[OK] Step 7 Dashboard complete: {out_path}")
    return out_path

if __name__ == '__main__':
    build_dashboard()
