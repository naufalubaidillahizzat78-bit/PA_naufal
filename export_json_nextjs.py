import os
import json
import pandas as pd
import numpy as np

# Konfigurasi Path
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'output')
NEXT_PUBLIC_DATA_DIR = os.path.join(BASE, 'nextjs_data')

def export_to_nextjs():
    print("Memulai export data ke JSON untuk Next.js...")
    os.makedirs(NEXT_PUBLIC_DATA_DIR, exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_pickle(f'{OUT}/df_labeled.pkl')
        X_pc = pd.read_pickle(f'{OUT}/X_princals.pkl')
        feat_cols = pd.read_pickle(f'{OUT}/feature_cols.pkl').tolist()
        var_info = pd.read_pickle(f'{OUT}/princals_info.pkl')
    except FileNotFoundError as e:
        print(f"Error: {e}. Pastikan file pickle sudah di-generate dari step sebelumnya.")
        return

    # Menyesuaikan panjang data jika ada perbedaan
    min_len = min(len(df), len(X_pc))
    df = df.iloc[:min_len].reset_index(drop=True)
    X_pc = X_pc.iloc[:min_len].reset_index(drop=True)

    # Konstanta dan Variabel Bantu
    PALETTE = {
        'Sangat Tinggi': '#28C76F',
        'Tinggi': '#00CFE8',
        'Sedang': '#7367F0',
        'Cukup': '#FF9F43',
        'Rendah': '#EA5455',
        'Sangat Rendah': '#EA5455',
        'Kritis': '#EA5455',
    }
    ORDER   = ['Sangat Tinggi','Tinggi','Sedang','Cukup','Rendah']
    lbl_ord = [l for l in ORDER if l in df['cluster_label'].values]
    
    SEM_IPS = ['nilai ips','IPS','IPS.1','IPS.2']
    SKIP = {'NRP','Nama Mahasiswa','Angkatan Tahun','Prodi','JK','Asal Kab/Kota',
            'cluster','cluster_label','Prodi_enc','Asal Kab/Kota_enc'}
    SKIP_IA = {c for c in df.columns if 'IPS' in c.upper() or 'ABSEN' in c.upper()}
    COURSE = [c for c in df.columns if c not in SKIP and c not in SKIP_IA
              and not c.endswith('_enc') and not c.endswith('.1')]
    
    # 2. Proses data Students
    students = []
    for i, row in df.iterrows():
        pc1 = float(X_pc.iloc[i,0]) if X_pc.shape[1]>0 else 0
        pc2 = float(X_pc.iloc[i,1]) if X_pc.shape[1]>1 else 0
        ips_val = float(row['Rata-Rata IPS']) if 'Rata-Rata IPS' in row.index else 0
        absen_v = float(row['Rata-Rata Absen Mahasiswa']) if 'Rata-Rata Absen Mahasiswa' in row.index else None
        
        courses = {c: round(float(row[c]),2) for c in COURSE if c in row.index and not np.isnan(row[c])}
        ips_sem = {}
        for j, col in enumerate(SEM_IPS):
            if col in row.index: ips_sem[f'Sem {j+1}'] = round(float(row[col]),3)
            
        students.append({
            'id': int(row['NRP']),
            'name': str(row['Nama Mahasiswa']),
            'angkatan': int(row.get('Angkatan Tahun',0)),
            'jk': 'Laki-laki' if row.get('JK',1)==1 else 'Perempuan',
            'asal': str(row.get('Asal Kab/Kota','-')),
            'cluster': str(row['cluster_label']),
            'ips': ips_val,
            'absensi': absen_v,
            'pc1': pc1,
            'pc2': pc2,
            'bubble_size': max(8, ips_val*12),
            'courses': courses,
            'ips_sem': ips_sem
        })

    # 3. Proses Overview dan Cluster Summary
    cluster_summary = {}
    ABSEN = ['Rata-Rata Absen Mahasiswa','ABSENSI RATA RATA','ABSENSI RATA_RATA',
             'ABSENSI RATA RATA.1','ABSENSI RATA RATA.2']
             
    for lbl in lbl_ord:
        sub = df[df['cluster_label']==lbl]
        absen_exist = [c for c in ABSEN if c in df.columns]
        cluster_summary[lbl] = {
            'count': int(len(sub)),
            'avg_ips': round(float(sub['Rata-Rata IPS'].mean()),3) if 'Rata-Rata IPS' in sub.columns else 0,
            'avg_absensi': round(float(sub[absen_exist].mean().mean()),2) if absen_exist else None,
        }
        for j, col in enumerate(SEM_IPS):
            if col in sub.columns:
                cluster_summary[lbl][f'IPS Sem {j+1}'] = round(float(sub[col].mean()),3)

    overview = {
        "metrics": {
            "total_mahasiswa": len(df),
            "jumlah_cluster": len(lbl_ord),
            "rata_rata_ipk": round(float(df['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else 0), 3),
            "n_comp_princals": int(var_info['n_components']),
            "cumulative_variance_princals": round(float(var_info['cumulative_variance'])*100, 2)
        },
        "clusters": lbl_ord,
        "palette": PALETTE
    }

    ADVICE = {
        'Sangat Tinggi':{'icon':'🏆','tips':['Proyek riset & kompetisi nasional/internasional.']},
        'Tinggi':{'icon':'⭐','tips':['Akses materi pengayaan & sertifikasi profesional.']},
        'Sedang':{'icon':'📘','tips':['Bimbingan rutin setiap 2 minggu.']},
        'Cukup':{'icon':'⚠️','tips':['Konseling akademik individual 1×/bulan.']},
        'Rendah':{'icon':'🚨','tips':['Intervensi segera & monitoring kehadiran mingguan.']},
    }

    # 4. Save to JSON files
    with open(f'{NEXT_PUBLIC_DATA_DIR}/students.json', 'w', encoding='utf-8') as f:
        json.dump(students, f, ensure_ascii=False, indent=2)

    with open(f'{NEXT_PUBLIC_DATA_DIR}/cluster_summary.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

    with open(f'{NEXT_PUBLIC_DATA_DIR}/overview.json', 'w', encoding='utf-8') as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    with open(f'{NEXT_PUBLIC_DATA_DIR}/advice.json', 'w', encoding='utf-8') as f:
        json.dump(ADVICE, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Ekspor selesai! Semua file Next.js tersimpan di folder: {NEXT_PUBLIC_DATA_DIR}")
    print("  -> Salin folder 'nextjs_data' ini ke folder 'public/data/' di project Next.js Anda.")

if __name__ == '__main__':
    export_to_nextjs()
