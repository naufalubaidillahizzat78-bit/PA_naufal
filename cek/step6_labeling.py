"""
STEP 6: LABELING CLUSTER
- Beri label pada cluster berdasarkan karakteristik akademik
  (Tinggi / Sedang / Rendah / dst.)
- Analisis profil tiap cluster
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_composite_score(df, cluster_id):
    """
    Skor akademik multi-kriteria per cluster:
      50% → Rata-Rata IPS (kumulatif — representasi performa keseluruhan)
      30% → Rata-rata IPS semester terkini (nilai ips, IPS, IPS.1, IPS.2)
      20% → Absensi rata-rata (dinormalisasi ke skala 0-4)
    Bobot ini mencegah mahasiswa dengan IPS semester awal buruk
    mendapat label tinggi hanya karena IPS semester akhir membaik.
    """
    sub = df[df['cluster'] == cluster_id]
    ips_recent_cols = [c for c in ['nilai ips', 'IPS', 'IPS.1', 'IPS.2'] if c in df.columns]
    absen_cols = [c for c in df.columns if 'ABSEN' in c.upper()]

    ips_cum   = sub['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else sub[ips_recent_cols].mean().mean()
    ips_rec   = sub[ips_recent_cols].mean().mean() if ips_recent_cols else ips_cum
    absen_raw = sub[absen_cols].mean().mean() if absen_cols else 100.0
    absen_norm = (absen_raw / 100.0) * 4.0   # normalisasi ke skala IPS (0-4)

    return 0.50 * ips_cum + 0.30 * ips_rec + 0.20 * absen_norm


def assign_labels(df, labels, feature_cols, n_clusters):
    """
    Label cluster menggunakan composite score multi-kriteria:
      50% IPS kumulatif + 30% IPS terkini + 20% Absensi
    Ini mencegah cluster dengan IPS terkini tinggi namun
    IPK kumulatif rendah mendapat label yang salah.
    """
    df = df.copy()
    df['cluster'] = labels

    cluster_ids = sorted(df['cluster'].unique())
    scores = {c: compute_composite_score(df, c) for c in cluster_ids}
    sorted_clusters = sorted(scores, key=lambda c: scores[c], reverse=True)

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

    df['cluster_label'] = df['cluster'].map(rank_labels)

    # Print scoring detail
    ips_recent_cols = [c for c in ['nilai ips', 'IPS', 'IPS.1', 'IPS.2'] if c in df.columns]
    print("\n[Label] Composite Score (50% IPS kum + 30% IPS terkini + 20% Absensi):")
    print(f"  {'Cluster':>8}  {'Composite':>10}  {'IPS_kum':>8}  {'IPS_rec':>8}  {'Label'}")
    print("  " + "-"*60)
    for c in sorted_clusters:
        sub = df[df['cluster'] == c]
        ips_cum = sub['Rata-Rata IPS'].mean() if 'Rata-Rata IPS' in df.columns else 0
        ips_rec = sub[ips_recent_cols].mean().mean() if ips_recent_cols else 0
        print(f"  {c:>8}  {scores[c]:>10.4f}  {ips_cum:>8.3f}  {ips_rec:>8.3f}  → {rank_labels[c]}")

    return df, rank_labels


def profile_clusters(df, feature_cols):
    """Buat profil statistik tiap cluster."""
    profile = df.groupby('cluster_label')[feature_cols].agg(['mean', 'std', 'count'])
    profile.columns = ['_'.join(col) for col in profile.columns]
    # Also count per label
    count = df['cluster_label'].value_counts().rename('jumlah_mahasiswa')
    print("\n[Label] Distribusi Cluster:")
    print(count.to_string())
    print("\n[Label] Profil Cluster (mean):")
    mean_cols = [c for c in profile.columns if c.endswith('_mean')]
    print(profile[mean_cols].round(3).to_string())
    return profile


def run_labeling(df_path='output/df_cleaned.pkl',
                 best_model_path='output/best_model.pkl',
                 label_store_path='output/label_store.pkl',
                 feature_cols_path='output/feature_cols.pkl'):

    df = pd.read_pickle(df_path)
    best = pd.read_pickle(best_model_path)
    label_store = pd.read_pickle(label_store_path)
    feature_cols = pd.read_pickle(feature_cols_path).tolist()

    method = best['method']
    k = int(best['k'])
    key = f'{method}_k{k}'

    # Handle DBSCAN key format
    if key not in label_store:
        key = [k for k in label_store.keys() if method.replace('(', '').replace(')', '').replace('=', '').replace('.', '') in k.replace('(', '').replace(')', '').replace('=', '').replace('.', '')]
        key = key[0] if key else list(label_store.keys())[0]

    labels = label_store[key]
    print(f"\n[Label] Using model: {method}, k={k}, key='{key}'")

    # Remove DBSCAN noise (-1 label)
    if -1 in labels:
        mask = labels != -1
        df_valid = df[mask].copy()
        labels_valid = labels[mask]
        print(f"[Label] DBSCAN noise removed: {(~mask).sum()} points")
    else:
        df_valid = df.copy()
        labels_valid = labels

    df_labeled, rank_labels = assign_labels(df_valid, labels_valid, feature_cols, k)
    profile = profile_clusters(df_labeled, feature_cols)

    # Save
    df_labeled.to_pickle('output/df_labeled.pkl')
    df_labeled.to_csv('output/df_labeled.csv', index=False)
    profile.to_csv('output/cluster_profiles.csv')
    pd.Series(rank_labels).to_pickle('output/rank_labels.pkl')

    print(f"\n✅ Step 6 Labeling complete → output/df_labeled.csv")
    return df_labeled, rank_labels

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/best_model.pkl'):
        print("Jalankan step 1-5 terlebih dahulu.")
    else:
        run_labeling()
