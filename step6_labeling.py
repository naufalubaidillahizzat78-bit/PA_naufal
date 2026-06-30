"""
STEP 6: LABELING CLUSTER
- Beri label human-readable pada cluster berdasarkan composite score akademik:
  50% IPS kumulatif + 30% IPS terkini + 20% Absensi
- Analisis profil tiap cluster
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

def compute_composite_score(df, cluster_id):
    """
    Skor akademik multi-kriteria per cluster:
      50% → Rata-Rata IPS (kumulatif)
      30% → IPS Semester 5 (tren terkini)
      20% → Absensi rata-rata (dinormalisasi ke skala 0-4)
    """
    sub = df[df['cluster'] == cluster_id]
    if len(sub) == 0:
        return 0.0
        
    # 1. Rata-Rata IPS Kumulatif
    ips_cum = sub['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else 0.0
    
    # 2. IPS Semester 5
    sem5_cols = [c for c in df.columns if 'Semester 5' in c and ('nilai IPS' in c or 'nilai ips' in c.lower())]
    if sem5_cols:
        ips_sem5 = sub[sem5_cols[0]].mean()
    else:
        ips_individual = sorted([c for c in df.columns if ('nilai IPS' in c or 'nilai ips' in c.lower()) and 'Rata-Rata' not in c])
        ips_sem5 = sub[ips_individual[-1]].mean() if ips_individual else ips_cum
        
    # 3. Persentase Kehadiran
    absen_val = sub['Rata-Rata Absen Mahasiswa'].mean() if 'Rata-Rata Absen Mahasiswa' in df.columns else 100.0
    absen_norm = (absen_val / 100.0) * 4.0   # normalisasi ke skala IPS (0-4)

    return 0.50 * ips_cum + 0.30 * ips_sem5 + 0.20 * absen_norm

def assign_labels(df, labels, feature_cols):
    df = df.copy()
    df['cluster'] = labels

    # Identify all unique clusters including noise/outliers (-1)
    cluster_ids = sorted(list(df['cluster'].unique()))
    
    # Compute composite scores for all clusters
    scores = {c: compute_composite_score(df, c) for c in cluster_ids}
    sorted_clusters = sorted(cluster_ids, key=lambda c: scores[c], reverse=True)

    rank_labels = {}
    n_clusters = len(sorted_clusters)
    
    if n_clusters == 1:
        rank_labels[sorted_clusters[0]] = 'Tinggi'
    elif n_clusters == 2:
        rank_labels[sorted_clusters[0]] = 'Tinggi'
        rank_labels[sorted_clusters[1]] = 'Rendah'
    else:
        # Highest score -> Tinggi
        rank_labels[sorted_clusters[0]] = 'Tinggi'
        # Lowest score -> Rendah
        rank_labels[sorted_clusters[-1]] = 'Rendah'
        # Middle score(s) -> Sedang
        for c in sorted_clusters[1:-1]:
            rank_labels[c] = 'Sedang'

    df['cluster_label'] = df['cluster'].map(rank_labels)

    # Validation check: Tinggi > Sedang > Rendah
    label_scores = {}
    for lbl in ['Tinggi', 'Sedang', 'Rendah']:
        sub = df[df['cluster_label'] == lbl]
        if len(sub) > 0:
            ips_cum = sub['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else 0.0
            
            sem5_cols = [c for c in df.columns if 'Semester 5' in c and ('nilai IPS' in c or 'nilai ips' in c.lower())]
            if sem5_cols:
                ips_sem5 = sub[sem5_cols[0]].mean()
            else:
                ips_individual = sorted([c for c in df.columns if ('nilai IPS' in c or 'nilai ips' in c.lower()) and 'Rata-Rata' not in c])
                ips_sem5 = sub[ips_individual[-1]].mean() if ips_individual else ips_cum
                
            absen_val = sub['Rata-Rata Absen Mahasiswa'].mean() if 'Rata-Rata Absen Mahasiswa' in df.columns else 100.0
            absen_norm = (absen_val / 100.0) * 4.0
            
            label_scores[lbl] = 0.50 * ips_cum + 0.30 * ips_sem5 + 0.20 * absen_norm
        else:
            label_scores[lbl] = None

    print("\n[Validation] Average Composite Scores by Label:")
    for lbl, score in label_scores.items():
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {lbl}: {score_str}")

    # Check monotonicity
    valid = True
    if label_scores['Tinggi'] is not None and label_scores['Sedang'] is not None and label_scores['Tinggi'] <= label_scores['Sedang']:
        valid = False
    if label_scores['Sedang'] is not None and label_scores['Rendah'] is not None and label_scores['Sedang'] <= label_scores['Rendah']:
        valid = False
    if label_scores['Tinggi'] is not None and label_scores['Rendah'] is not None and label_scores['Tinggi'] <= label_scores['Rendah']:
        valid = False

    if not valid:
        print("\n[WARNING] INKONSISTENSI DETEKSI: Urutan skor komposit tidak konsisten (Tinggi <= Sedang atau Sedang <= Rendah)!")
    else:
        print("\n[OK] VALIDASI BERHASIL: Skor komposit konsisten (Tinggi > Sedang > Rendah).")
        # Assert to guarantee correctness
        assert (label_scores['Sedang'] is None or label_scores['Tinggi'] > label_scores['Sedang']), "Tinggi must be greater than Sedang"
        assert (label_scores['Sedang'] is None or label_scores['Rendah'] is None or label_scores['Sedang'] > label_scores['Rendah']), "Sedang must be greater than Rendah"
        assert (label_scores['Rendah'] is None or label_scores['Tinggi'] > label_scores['Rendah']), "Tinggi must be greater than Rendah"

    # Print scoring details
    print("\n[Label] Composite Score (50% IPS kum + 30% IPS Sem 5 + 20% Absensi):")
    print(f"  {'Cluster':>8}  {'Composite':>10}  {'Label'}")
    print("  " + "-"*40)
    for c in sorted_clusters:
        print(f"  {c:>8}  {scores[c]:>10.4f}  -> {rank_labels[c]}")

    return df, rank_labels

def profile_clusters(df, feature_cols):
    """Buat profil statistik tiap cluster."""
    # Group by label
    grouped = df.groupby('cluster_label')
    
    # We will pick a few representative features to write in the text report
    absen_cols = [c for c in df.columns if 'ABSEN' in c.upper()]
    ips_cols = [c for c in df.columns if 'IPS' in c.upper() or 'nilai ips' in c.lower()]
    
    rep_cols = []
    if 'Rata-Rata IPS' in df.columns: rep_cols.append('Rata-Rata IPS')
    if absen_cols: rep_cols.append(absen_cols[0])
    
    profile = grouped[feature_cols].mean()
    profile['jumlah_mahasiswa'] = grouped['NRP'].count()
    
    print("\n[Label] Distribusi Cluster:")
    print(df['cluster_label'].value_counts().to_string())
    
    print("\n[Label] Profil Cluster (Rata-Rata):")
    cols_to_show = [c for c in rep_cols if c in profile.columns] + ['jumlah_mahasiswa']
    print(profile[cols_to_show].round(3).to_string())
    
    return profile

def run_labeling(df_path='output/df_cleaned.pkl',
                 best_model_path='output/best_model.pkl',
                 label_store_path='output/label_store.pkl',
                 feature_cols_path='output/feature_cols.pkl'):

    df = pd.read_pickle(df_path)
    best = pd.read_pickle(best_model_path)
    with open(label_store_path, 'rb') as f:
        label_store = pickle.load(f)
    feature_cols = pd.read_pickle(feature_cols_path).tolist()

    method = best['method']
    params = best['params']

    # Map params back to label_store key
    if method == 'DBSCAN':
        parts = params.split(',')
        eps_val = parts[0].split('=')[1]
        min_s_val = parts[1].split('=')[1]
        key = f'DBSCAN_eps{eps_val}_min{min_s_val}'
    else:
        k_val = params.split('=')[1]
        key = f'{method}_k{k_val}'

    labels = label_store[key]
    print(f"\n[Label] Using best model: {method} ({params}), key='{key}'")

    df_labeled, rank_labels = assign_labels(df, labels, feature_cols)
    profile = profile_clusters(df_labeled, feature_cols)

    # Save outputs
    df_labeled.to_pickle('output/df_labeled.pkl')
    df_labeled.to_csv('output/df_labeled.csv', index=False)
    profile.to_csv('output/cluster_profiles.csv')
    with open('output/rank_labels.pkl', 'wb') as f:
        pickle.dump(rank_labels, f)

    print(f"\n[OK] Step 6 Labeling complete -> output/df_labeled.pkl & output/df_labeled.csv")
    return df_labeled, rank_labels

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/best_model.pkl'):
        print("Jalankan step 1-5 terlebih dahulu.")
    else:
        run_labeling()
