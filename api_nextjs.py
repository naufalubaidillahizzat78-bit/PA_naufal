import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI(title="Academic Clustering API for Next.js",
              description="API Backend untuk menyediakan data hasil clustering ke frontend Next.js")

# Izinkan CORS agar Next.js dapat fetch data tanpa error cors policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti dengan domain Next.js kamu, contoh: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'output')

def get_data():
    df = pd.read_pickle(f'{OUT}/df_labeled.pkl')
    X_pc = pd.read_pickle(f'{OUT}/X_princals.pkl')
    feat_cols = pd.read_pickle(f'{OUT}/feature_cols.pkl').tolist()
    var_info = pd.read_pickle(f'{OUT}/princals_info.pkl')
    
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
    
    min_len = min(len(df), len(X_pc))
    df = df.iloc[:min_len].reset_index(drop=True)
    X_pc = X_pc.iloc[:min_len].reset_index(drop=True)
    return df, X_pc, feat_cols, var_info, PALETTE, lbl_ord

@app.get("/")
def read_root():
    return {"message": "Welcome to Academic Clustering API. Akses /docs untuk melihat dokumentasi API."}

@app.get("/api/overview")
def get_overview():
    try:
        df, X_pc, feat_cols, var_info, PALETTE, lbl_ord = get_data()
        
        n_comp = int(var_info['n_components'])
        cumvar = round(float(var_info['cumulative_variance'])*100, 2)
        avgIPS = round(float(df['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else 0), 3)

        return {
            "total_mahasiswa": len(df),
            "jumlah_cluster": len(lbl_ord),
            "rata_rata_ipk": avgIPS,
            "silhouette": 0.3485,
            "bss_tss_ratio": "74.46%",
            "method": "KMeans k=5",
            "princals": {
                "n_components": n_comp,
                "cumulative_variance": cumvar
            },
            "clusters": lbl_ord,
            "palette": PALETTE
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/students")
def get_students():
    try:
        df, X_pc, feat_cols, var_info, PALETTE, lbl_ord = get_data()
        
        SEM_IPS = ['nilai ips','IPS','IPS.1','IPS.2']
        SKIP = {'NRP','Nama Mahasiswa','Angkatan Tahun','Prodi','JK','Asal Kab/Kota',
                'cluster','cluster_label','Prodi_enc','Asal Kab/Kota_enc'}
        SKIP_IA = {c for c in df.columns if 'IPS' in c.upper() or 'ABSEN' in c.upper()}
        COURSE = [c for c in df.columns if c not in SKIP and c not in SKIP_IA
                  and not c.endswith('_enc') and not c.endswith('.1')]
        
        students = []
        for i, row in df.iterrows():
            pc1 = float(X_pc.iloc[i,0]) if X_pc.shape[1]>0 else 0
            pc2 = float(X_pc.iloc[i,1]) if X_pc.shape[1]>1 else 0
            ips_val = float(row['Rata-Rata IPS']) if 'Rata-Rata IPS' in row.index else 0
            absen_v = float(row['Rata-Rata Absen Mahasiswa']) if 'Rata-Rata Absen Mahasiswa' in row.index else None
            
            courses = {c: round(float(row[c]),2) for c in COURSE if c in row.index and not np.isnan(row[c])}
            
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
                'courses': courses
            })
            
        return {"data": students}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/clusters")
def get_cluster_stats():
    try:
        df, X_pc, feat_cols, var_info, PALETTE, lbl_ord = get_data()
        
        SEM_IPS = ['nilai ips','IPS','IPS.1','IPS.2']
        ABSEN = ['Rata-Rata Absen Mahasiswa','ABSENSI RATA RATA','ABSENSI RATA_RATA',
                 'ABSENSI RATA RATA.1','ABSENSI RATA RATA.2']
        
        cluster_summary = {}
        for lbl in lbl_ord:
            sub = df[df['cluster_label']==lbl]
            absen_exist = [c for c in ABSEN if c in df.columns]
            
            stat = {
                'count': int(len(sub)),
                'avg_ips': round(float(sub['Rata-Rata IPS'].mean()),3) if 'Rata-Rata IPS' in sub.columns else 0,
                'avg_absensi': round(float(sub[absen_exist].mean().mean()),2) if absen_exist else None,
            }
            
            for j, col in enumerate(SEM_IPS):
                if col in sub.columns:
                    stat[f'IPS Sem {j+1}'] = round(float(sub[col].mean()),3)
            cluster_summary[lbl] = stat
            
        return {"data": cluster_summary}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/advice")
def get_advice():
    # Mengembalikan rekomendasi untuk tiap cluster
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
    return {"data": ADVICE}

if __name__ == "__main__":
    print("Membuka API server pada http://localhost:8000")
    print("Pastikan untuk install: pip install fastapi uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=8000)
