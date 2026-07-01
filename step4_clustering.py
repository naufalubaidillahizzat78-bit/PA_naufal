"""
STEP 4: CLUSTERING EXPERIMENTS
- Metode: KMeans, KMedoids, FCM, PCM, FPCM, MFPCM (K = 2..10)
- DBSCAN Grid Search (eps: 4.0..10.0, min_samples: 2..3)
- Hitung Silhouette Coefficient & BSS/TSS Ratio untuk seluruh eksperimen
- Input: Dataset dari PRINCALS (hanya PC1, PC2, ... tanpa nama asli)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
import warnings
import os
import pickle

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# Fuzzy C-Means (FCM)
# ══════════════════════════════════════════════════════════════════
def fcm(X, c, m=1.2, max_iter=200, eps=1e-6, seed=42):
    """
    FCM: Fuzzy C-Means.
    m=1.2 dan inisialisasi centroid dari data untuk mencegah collapse.
    """
    np.random.seed(seed)
    n, d = X.shape
    idx = np.random.choice(n, c, replace=False)
    V = X[idx].copy()

    for _ in range(max_iter):
        dist = np.zeros((c, n))
        for i in range(c):
            dist[i] = np.linalg.norm(X - V[i], axis=1)
        dist = np.maximum(dist, 1e-10)

        U = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U[i] += (dist[i] / dist[k]) ** (2 / (m - 1))
        U = 1.0 / U

        Um = U ** m
        V_new = (Um @ X) / Um.sum(axis=1, keepdims=True)

        if np.linalg.norm(V_new - V) < eps:
            break
        V = V_new

    labels = np.argmax(U, axis=0)
    return labels, U, V


# ══════════════════════════════════════════════════════════════════
# Possibilistic C-Means (PCM)
# ══════════════════════════════════════════════════════════════════
def pcm(X, c, m=1.1, max_iter=200, eps=1e-6, seed=42):
    """
    PCM: Possibilistic C-Means.
    m=1.1 untuk mencegah coincident clusters collapse.
    """
    np.random.seed(seed)
    labels_fcm, U_fcm, V = fcm(X, c, m=1.2, seed=seed)
    V = V.copy()
    n, d = X.shape

    dist2 = np.zeros((c, n))
    for i in range(c):
        dist2[i] = np.linalg.norm(X - V[i], axis=1) ** 2
    Um = U_fcm ** 1.2
    eta = (Um * dist2).sum(axis=1) / Um.sum(axis=1)
    eta = np.maximum(eta, 1e-10)
    T = np.zeros((c, n))

    for _ in range(max_iter):
        for i in range(c):
            d2 = np.linalg.norm(X - V[i], axis=1) ** 2
            T[i] = 1.0 / (1.0 + (d2 / eta[i]) ** (1 / (m - 1)))

        Tm = T ** m
        V_new = (Tm @ X) / Tm.sum(axis=1, keepdims=True)
        if np.linalg.norm(V_new - V) < eps:
            break
        V = V_new

    labels = np.argmax(T, axis=0)
    return labels, T, V


# ══════════════════════════════════════════════════════════════════
# Fuzzy Possibilistic C-Means (FPCM)
# ══════════════════════════════════════════════════════════════════
def fpcm(X, c, m=1.2, eta_param=1.2, max_iter=200, eps=1e-6, seed=42):
    """
    FPCM: Fuzzy Possibilistic C-Means.
    m=1.2 dan eta_param=1.2 untuk mencegah collapse.
    """
    np.random.seed(seed)
    n, d = X.shape
    idx = np.random.choice(n, c, replace=False)
    V = X[idx].copy()

    U = np.random.dirichlet(np.ones(c), size=n).T
    U = U / U.sum(axis=0)
    T = np.random.rand(c, n)
    T = T / T.sum(axis=0)

    for _ in range(max_iter):
        Tm = T ** eta_param
        Um = U ** m
        V_new = ((Um + Tm) @ X) / ((Um + Tm).sum(axis=1, keepdims=True))

        dist = np.zeros((c, n))
        for i in range(c):
            dist[i] = np.linalg.norm(X - V_new[i], axis=1)
        dist = np.maximum(dist, 1e-10)

        U_new = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U_new[i] += (dist[i] / dist[k]) ** (2 / (m - 1))
        U_new = 1.0 / U_new

        T_new = np.zeros((c, n))
        for i in range(c):
            T_new[i] = 1.0 / (1.0 + dist[i] ** (2 / (eta_param - 1)))

        if np.linalg.norm(U_new - U) + np.linalg.norm(T_new - T) < eps:
            break
        U, T, V = U_new, T_new, V_new

    labels = np.argmax(U, axis=0)
    return labels, U, T, V


# ══════════════════════════════════════════════════════════════════
# Modified FPCM (MFPCM)
# ══════════════════════════════════════════════════════════════════
def mfpcm(X, c, m=1.2, alpha=0.9, max_iter=200, eps=1e-6, seed=42):
    """
    MFPCM: Modified Fuzzy Possibilistic C-Means.
    alpha=0.9 dan m=1.2 dengan inisialisasi centroid.
    """
    np.random.seed(seed)
    n, d = X.shape
    idx = np.random.choice(n, c, replace=False)
    V = X[idx].copy()
    U = np.zeros((c, n))
    T = np.zeros((c, n))

    for _ in range(max_iter):
        dist = np.zeros((c, n))
        for i in range(c):
            dist[i] = np.linalg.norm(X - V[i], axis=1) ** 2
        dist = np.maximum(dist, 1e-10)

        U_new = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U_new[i] += (dist[i] / dist[k]) ** (1 / (m - 1))
        U_new = 1.0 / U_new

        eta = ((U_new ** m) * dist).sum(axis=1) / (U_new ** m).sum(axis=1)
        eta = np.maximum(eta, 1e-10)

        T_new = np.zeros((c, n))
        for i in range(c):
            T_new[i] = 1.0 / (1.0 + dist[i] / eta[i])

        W = alpha * U_new + (1 - alpha) * T_new
        V_new = (W ** m @ X) / (W ** m).sum(axis=1, keepdims=True)

        if np.linalg.norm(V_new - V) < eps:
            break
        V, U, T = V_new, U_new, T_new

    labels = np.argmax(U + T, axis=0)
    return labels, U, T, V


# ══════════════════════════════════════════════════════════════════
# K-Medoids (Partitioning Around Medoids - PAM)
# ══════════════════════════════════════════════════════════════════
def kmedoids(X, k, max_iter=100, seed=42):
    np.random.seed(seed)
    n = len(X)
    medoid_idx = np.random.choice(n, k, replace=False)
    D = pairwise_distances(X)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoid_idx], axis=1)
        new_medoids = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[c])
                continue
            sub_D = D[np.ix_(members, members)]
            best = members[np.argmin(sub_D.sum(axis=1))]
            new_medoids.append(best)
        new_medoids = np.array(new_medoids)
        if np.all(new_medoids == medoid_idx):
            break
        medoid_idx = new_medoids

    labels = np.argmin(D[:, medoid_idx], axis=1)
    return labels, medoid_idx


# ══════════════════════════════════════════════════════════════════
# BSS / TSS Calculation
# ══════════════════════════════════════════════════════════════════
def compute_bss_tss(X, labels):
    X = np.array(X)
    grand_mean = X.mean(axis=0)
    TSS = np.sum((X - grand_mean) ** 2)
    BSS = 0.0
    for c in np.unique(labels):
        if c == -1:  # Skip DBSCAN noise
            continue
        members = X[labels == c]
        if len(members) == 0:
            continue
        cluster_mean = members.mean(axis=0)
        BSS += len(members) * np.sum((cluster_mean - grand_mean) ** 2)
    ratio = BSS / TSS if TSS > 0 else 0
    return BSS, TSS, ratio


def run_clustering(X_princals_path='output/X_princals.pkl'):
    print("\n" + "=" * 100)
    print("                      STEP 4: CLUSTERING EXPERIMENTS")
    print("=" * 100)

    X = pd.read_pickle(X_princals_path).values
    n_obs, n_features = X.shape

    print(f"\n[INFO] Input shape       : {n_obs} observasi × {n_features} fitur (PC)")
    print(f"[INFO] Kolom input       : PC1, PC2, ..., PC{n_features} (TANPA nama variabel asli)")
    print(f"[INFO] K range           : 2 - 10")
    print(f"[INFO] Metode centroid   : K-Means, K-Medoids, FCM, PCM, FPCM, MFPCM")
    print(f"[INFO] Metode density    : DBSCAN (Grid Search)")

    k_range = range(2, 11)
    results = []
    label_store = {}
    no = 0
    centroid_methods = ['K-Means', 'K-Medoids', 'FCM', 'PCM', 'FPCM', 'MFPCM']

    # ═══════════════════════════════════════════════════════════════
    # 1. CENTROID-BASED METHODS — tampilkan per K
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 100)
    print("  CENTROID-BASED METHODS  (K-Means, K-Medoids, FCM, PCM, FPCM, MFPCM)")
    print("═" * 100)

    for k in k_range:
        methods = {
            'K-Means':   lambda k=k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X),
            'K-Medoids': lambda k=k: kmedoids(X, k)[0],
            'FCM':       lambda k=k: fcm(X, k)[0],
            'PCM':       lambda k=k: pcm(X, k)[0],
            'FPCM':      lambda k=k: fpcm(X, k)[0],
            'MFPCM':     lambda k=k: mfpcm(X, k)[0],
        }

        print(f"\n  ┌── K = {k} {'─' * 88}")
        print(f"  │  {'No':<4} {'Metode':<12} {'Silhouette':>12} {'BSS/TSS':>10} {'BSS':>14} {'TSS':>14}")
        print(f"  │  {'─' * 57}")

        for name, func in methods.items():
            try:
                labels = func()
                n_unique = len(np.unique(labels))
                if n_unique < 2:
                    continue
                sil = silhouette_score(X, labels)
                BSS, TSS, ratio = compute_bss_tss(X, labels)
                no += 1
                results.append({
                    'no': no,
                    'method': name,
                    'k': k,
                    'params': f'k={k}',
                    'silhouette': round(sil, 4),
                    'bss_tss_ratio': round(ratio * 100, 2),
                    'BSS': round(BSS, 4),
                    'TSS': round(TSS, 4),
                    'n_clusters_actual': n_unique
                })
                label_store[f'{name}_k{k}'] = labels
                print(f"  │  {no:<4} {name:<12} {sil:>12.4f} {ratio*100:>9.2f}% {BSS:>14.4f} {TSS:>14.4f}")
            except Exception as e:
                print(f"  │  {'?':<4} {name:<12}  ERROR: {str(e)[:50]}")
        print(f"  └──")

    # ═══════════════════════════════════════════════════════════════
    # 2. DBSCAN GRID SEARCH
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 100)
    print("  DBSCAN  —  Grid Search (eps × min_samples)")
    print("═" * 100)
    print(f"  {'No':<4} {'Parameter':<22} {'K':>4} {'Silhouette':>12} {'BSS/TSS':>10} {'BSS':>14} {'TSS':>14} {'Noise':>7}")
    print("  " + "─" * 80)

    eps_grid = np.round(np.arange(4.0, 10.1, 0.2), 1)
    min_samples_grid = [2, 3]

    for eps_val in eps_grid:
        for min_samples_val in min_samples_grid:
            try:
                db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                labels_db = db.fit_predict(X)

                unique_labels = np.unique(labels_db)
                valid_labels = unique_labels[unique_labels != -1]
                n_clusters_actual = len(valid_labels)

                if n_clusters_actual < 2:
                    continue

                mask = labels_db != -1
                if mask.sum() < 2:
                    continue

                sil = silhouette_score(X[mask], labels_db[mask])
                BSS, TSS, ratio = compute_bss_tss(X[mask], labels_db[mask])
                no += 1

                param_str = f'eps={eps_val},min_s={min_samples_val}'
                results.append({
                    'no': no,
                    'method': 'DBSCAN',
                    'k': n_clusters_actual,
                    'params': param_str,
                    'silhouette': round(sil, 4),
                    'bss_tss_ratio': round(ratio * 100, 2),
                    'BSS': round(BSS, 4),
                    'TSS': round(TSS, 4),
                    'n_clusters_actual': n_clusters_actual
                })
                label_store[f'DBSCAN_eps{eps_val}_min{min_samples_val}'] = labels_db
                n_noise = (labels_db == -1).sum()
                print(f"  {no:<4} {param_str:<22} {n_clusters_actual:>4} {sil:>12.4f} {ratio*100:>9.2f}% {BSS:>14.4f} {TSS:>14.4f} {n_noise:>7}")
            except Exception as e:
                pass

    # ═══════════════════════════════════════════════════════════════
    # CREATE RESULTS DATAFRAME
    # ═══════════════════════════════════════════════════════════════
    df_results = pd.DataFrame(results)

    if df_results.empty:
        print("\n[WARNING] Tidak ada model valid yang terbentuk!")
        return df_results, label_store

    # ═══════════════════════════════════════════════════════════════════════════
    # TABEL PERBANDINGAN SILHOUETTE COEFFICIENT  (Metode × K)
    # ═══════════════════════════════════════════════════════════════════════════
    centroid_df = df_results[df_results['method'] != 'DBSCAN']
    dbscan_df   = df_results[df_results['method'] == 'DBSCAN']

    print("\n" + "═" * 100)
    print("  SILHOUETTE COEFFICIENT — Perbandingan Semua Metode")
    print("═" * 100)

    # Pivot tabel Silhouette centroid (metode × k)
    sil_pivot = centroid_df.pivot_table(
        index='method', columns='k', values='silhouette', aggfunc='first'
    )
    sil_pivot.columns = [f'k={c}' for c in sil_pivot.columns]
    sil_pivot.index.name = 'Metode'
    print("\n  [ Centroid Methods ]")
    print(sil_pivot.round(4).to_string())

    # DBSCAN silhouette — best per k
    if not dbscan_df.empty:
        sil_db = dbscan_df.groupby('k')['silhouette'].max().reset_index()
        sil_db.columns = ['Jumlah Cluster', 'Silhouette Terbaik']
        print("\n  [ DBSCAN — Silhouette Terbaik per Jumlah Cluster ]")
        print(sil_db.to_string(index=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # TABEL PERBANDINGAN BSS/TSS  (Metode × K)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 100)
    print("  BSS/TSS RATIO (%)  —  Perbandingan Semua Metode")
    print("═" * 100)

    # Pivot tabel BSS/TSS centroid
    bss_pivot = centroid_df.pivot_table(
        index='method', columns='k', values='bss_tss_ratio', aggfunc='first'
    )
    bss_pivot.columns = [f'k={c}' for c in bss_pivot.columns]
    bss_pivot.index.name = 'Metode'
    print("\n  [ Centroid Methods ] (%)")
    print(bss_pivot.round(2).to_string())

    # DBSCAN BSS/TSS — best per k
    if not dbscan_df.empty:
        bss_db = dbscan_df.groupby('k')['bss_tss_ratio'].max().reset_index()
        bss_db.columns = ['Jumlah Cluster', 'BSS/TSS Terbaik (%)']
        print("\n  [ DBSCAN — BSS/TSS Terbaik per Jumlah Cluster ]")
        print(bss_db.to_string(index=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL TERBAIK OVERALL
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 100)
    print("  MODEL TERBAIK  (Silhouette Coefficient Tertinggi)")
    print("═" * 100)

    best_sil_idx = df_results['silhouette'].idxmax()
    best_sil = df_results.loc[best_sil_idx]

    print("\n┌─────────────────────────────────────────────────────────────────────────────────────┐")
    print("│  MODEL TERBAIK (Berdasarkan Silhouette Coefficient Tertinggi)                       │")
    print("├─────────────────────────────────────────────────────────────────────────────────────┤")
    print(f"│  Metode            : {best_sil['method']:<60}│")
    print(f"│  Parameter         : {best_sil['params']:<60}│")
    print(f"│  Jumlah Cluster    : {best_sil['n_clusters_actual']:<60}│")
    print(f"│  Silhouette Score  : {best_sil['silhouette']:<60}│")
    bss_tss_str = f"{best_sil['bss_tss_ratio']:.2f}%"
    print(f"│  BSS/TSS Ratio     : {bss_tss_str:<60}│")
    print(f"│  BSS               : {best_sil['BSS']:<60}│")
    print(f"│  TSS               : {best_sil['TSS']:<60}│")
    print("└─────────────────────────────────────────────────────────────────────────────────────┘")

    # ═══════════════════════════════════════════════════════════════
    # SAVE OUTPUTS
    # ═══════════════════════════════════════════════════════════════
    df_results.to_pickle('output/clustering_results.pkl')
    df_results.to_csv('output/clustering_results.csv', index=False)
    with open('output/label_store.pkl', 'wb') as f:
        pickle.dump(label_store, f)

    # Save best model info
    best_info = {
        'method': best_sil['method'],
        'params': best_sil['params'],
        'k': int(best_sil['n_clusters_actual']),
        'silhouette': float(best_sil['silhouette']),
        'bss_tss_ratio': float(best_sil['bss_tss_ratio']),
        'label_key': f"{best_sil['method']}_k{int(best_sil['n_clusters_actual'])}"
    }
    # Find the correct label key
    for key in label_store.keys():
        if best_sil['method'] in key:
            test_labels = label_store[key]
            if len(np.unique(test_labels)) == int(best_sil['n_clusters_actual']):
                best_info['label_key'] = key
                break

    pd.Series(best_info).to_pickle('output/best_clustering.pkl')

    print("\n" + "=" * 100)
    print("                         FILE OUTPUT CLUSTERING")
    print("=" * 100)
    print("  ✓ output/clustering_results.pkl  - Semua hasil eksperimen")
    print("  ✓ output/clustering_results.csv  - Semua hasil eksperimen (CSV)")
    print("  ✓ output/label_store.pkl         - Semua label cluster per model")
    print("  ✓ output/best_clustering.pkl     - Info model terbaik")
    print("=" * 100)

    return df_results, label_store


if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)

    if not os.path.exists('output/X_princals.pkl'):
        print("[INFO] Menjalankan Step 1 & 3 terlebih dahulu...")
        from step1_preprocessing import run_preprocessing
        from step3_princals import run_princals
        run_preprocessing(r'C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx')
        X_scaled = pd.read_pickle('output/X_scaled.pkl')
        feature_cols = pd.read_pickle('output/feature_cols.pkl').tolist()
        run_princals(X_scaled, feature_cols)

    run_clustering()
