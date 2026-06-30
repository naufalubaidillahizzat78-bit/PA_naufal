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
      30% → Rata-rata IPS semester terkini (nilai ips, IPS, IPS.1, IPS.2)
      20% → Absensi rata-rata (dinormalisasi ke skala 0-4)
    """
    sub = df[df['cluster'] == cluster_id]
    if len(sub) == 0:
        return 0.0
        
    ips_recent_cols = [c for c in df.columns if ('nilai IPS' in c or 'nilai ips' in c.lower()) and 'Rata-Rata' not in c]
    absen_cols = [c for c in df.columns if 'ABSEN' in c.upper() and 'Rata-Rata' not in c]

    ips_cum   = sub['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else sub[ips_recent_cols].mean().mean()
    ips_rec   = sub[ips_recent_cols].mean().mean() if ips_recent_cols else ips_cum
    absen_raw = sub[absen_cols].mean().mean() if absen_cols else 100.0
    absen_norm = (absen_raw / 100.0) * 4.0   # normalisasi ke skala IPS (0-4)

    return 0.50 * ips_cum + 0.30 * ips_rec + 0.20 * absen_norm

def assign_labels(df, labels, feature_cols):
    df = df.copy()
    df['cluster'] = labels

    # Identify non-noise clusters
    cluster_ids = sorted([c for c in df['cluster'].unique() if c != -1])
    
    # Compute composite scores for non-noise clusters
    scores = {c: compute_composite_score(df, c) for c in cluster_ids}
    sorted_clusters = sorted(cluster_ids, key=lambda c: scores[c], reverse=True)

    label_map_academic = {
        1: 'Sangat Tinggi',
        2: 'Tinggi',
        3: 'Sedang',
        4: 'Cukup',
        5: 'Rendah',
        6: 'Sangat Rendah',
        7: 'Kritis',
    }

    rank_labels = {}
    for rank, cluster_id in enumerate(sorted_clusters, start=1):
        rank_labels[cluster_id] = label_map_academic.get(rank, f'Cluster-{rank}')
        
    # Map noise points in DBSCAN
    if -1 in df['cluster'].unique():
        rank_labels[-1] = 'Outlier (Kritis)'

    df['cluster_label'] = df['cluster'].map(rank_labels)

    # Print scoring details
    ips_recent_cols = [c for c in df.columns if ('nilai IPS' in c or 'nilai ips' in c.lower()) and 'Rata-Rata' not in c]
    print("\n[Label] Composite Score (50% IPS kum + 30% IPS terkini + 20% Absensi):")
    print(f"  {'Cluster':>8}  {'Composite':>10}  {'IPS_kum':>8}  {'IPS_rec':>8}  {'Label'}")
    print("  " + "-"*60)
    for c in sorted_clusters:
        sub = df[df['cluster'] == c]
        ips_cum = sub['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else 0
        ips_rec = sub[ips_recent_cols].mean().mean() if ips_recent_cols else 0
        print(f"  {c:>8}  {scores[c]:>10.4f}  {ips_cum:>8.3f}  {ips_rec:>8.3f}  -> {rank_labels[c]}")
        
    if -1 in rank_labels:
        print(f"  {-1:>8}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  -> {rank_labels[-1]} (DBSCAN Noise)")

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
