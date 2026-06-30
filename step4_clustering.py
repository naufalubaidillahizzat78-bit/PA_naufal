"""
STEP 4: CLUSTERING EXPERIMENTS
- Metode: KMeans, KMedoids, FCM, PCM, FPCM, MFPCM (K = 2..10)
- DBSCAN Grid Search (eps: 8.0..12.0, min_samples: 2..3)
- Hitung Silhouette Coefficient & BSS/TSS Ratio untuk seluruh eksperimen
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# Fuzzy C-Means (FCM)
# ══════════════════════════════════════════════════════
def fcm(X, c, m=1.2, max_iter=200, eps=1e-6, seed=42):
    """
    FCM: Fuzzy C-Means.
    Default m=1.2 and centroid initialization to prevent collapse.
    """
    np.random.seed(seed)
    n, d = X.shape
    # Init centroids by picking c random data points (prevents collapse to center)
    idx = np.random.choice(n, c, replace=False)
    V = X[idx].copy()

    for _ in range(max_iter):
        # Update membership
        dist = np.array([[np.linalg.norm(X[j] - V[i]) for j in range(n)] for i in range(c)])
        dist = np.maximum(dist, 1e-10)
        U = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U[i] += (dist[i] / dist[k]) ** (2 / (m - 1))
        U = 1.0 / U

        # Update centroids
        Um = U ** m
        V_new = (Um @ X) / Um.sum(axis=1, keepdims=True)

        if np.linalg.norm(V_new - V) < eps:
            break
        V = V_new

    labels = np.argmax(U, axis=0)
    return labels, U, V


# ══════════════════════════════════════════════════════
# Possibilistic C-Means (PCM)
# ══════════════════════════════════════════════════════
def pcm(X, c, m=1.1, max_iter=200, eps=1e-6, seed=42):
    """
    PCM: Possibilistic C-Means.
    Default m=1.1 to prevent coincident clusters collapse.
    """
    np.random.seed(seed)
    # Init with FCM to get initial centroids and U
    labels_fcm, U_fcm, V = fcm(X, c, m=1.2, seed=seed)
    V = V.copy()
    n, d = X.shape
    
    # Compute eta (typicality scale)
    dist2 = np.array([[np.linalg.norm(X[j] - V[i])**2 for j in range(n)] for i in range(c)])
    Um = U_fcm ** 1.2
    eta = (Um * dist2).sum(axis=1) / Um.sum(axis=1)
    eta = np.maximum(eta, 1e-10)
    T = np.zeros((c, n))

    for _ in range(max_iter):
        # Update typicality
        for i in range(c):
            d2 = np.array([np.linalg.norm(X[j] - V[i])**2 for j in range(n)])
            T[i] = 1.0 / (1.0 + (d2 / eta[i]) ** (1 / (m - 1)))

        # Update centroids
        Tm = T ** m
        V_new = (Tm @ X) / Tm.sum(axis=1, keepdims=True)
        if np.linalg.norm(V_new - V) < eps:
            break
        V = V_new

    labels = np.argmax(T, axis=0)
    return labels, T, V


# ══════════════════════════════════════════════════════
# Fuzzy Possibilistic C-Means (FPCM)
# ══════════════════════════════════════════════════════
def fpcm(X, c, m=1.2, eta_param=1.2, max_iter=200, eps=1e-6, seed=42):
    """
    FPCM: Fuzzy Possibilistic C-Means.
    Default m=1.2 and eta_param=1.2 to prevent collapse.
    """
    np.random.seed(seed)
    n, d = X.shape
    # Init centroids by picking c random data points
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
        dist = np.array([[np.linalg.norm(X[j] - V_new[i]) for j in range(n)] for i in range(c)])
        dist = np.maximum(dist, 1e-10)

        # Update U
        U_new = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U_new[i] += (dist[i] / dist[k]) ** (2 / (m - 1))
        U_new = 1.0 / U_new

        # Update T
        T_new = np.zeros((c, n))
        for i in range(c):
            T_new[i] = 1.0 / (1.0 + dist[i] ** (2 / (eta_param - 1)))

        if np.linalg.norm(U_new - U) + np.linalg.norm(T_new - T) < eps:
            break
        U, T, V = U_new, T_new, V_new

    labels = np.argmax(U, axis=0)
    return labels, U, T, V


# ══════════════════════════════════════════════════════
# Modified FPCM (MFPCM)
# ══════════════════════════════════════════════════════
def mfpcm(X, c, m=1.2, alpha=0.9, max_iter=200, eps=1e-6, seed=42):
    """
    MFPCM: Modified Fuzzy Possibilistic C-Means.
    Default alpha=0.9 and m=1.2 with centroid initialization.
    """
    np.random.seed(seed)
    n, d = X.shape
    # Init centroids by picking c random data points
    idx = np.random.choice(n, c, replace=False)
    V = X[idx].copy()

    for _ in range(max_iter):
        dist = np.array([[np.linalg.norm(X[j] - V[i])**2 for j in range(n)] for i in range(c)])
        dist = np.maximum(dist, 1e-10)

        # Update U
        U_new = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U_new[i] += (dist[i] / dist[k]) ** (1 / (m - 1))
        U_new = 1.0 / U_new

        # Update T
        eta = ((U_new ** m) * dist).sum(axis=1) / (U_new ** m).sum(axis=1)
        eta = np.maximum(eta, 1e-10)
        T_new = np.zeros((c, n))
        for i in range(c):
            T_new[i] = 1.0 / (1.0 + dist[i] / eta[i])

        # Combined membership weights
        W = alpha * U_new + (1 - alpha) * T_new
        V_new = (W ** m @ X) / (W ** m).sum(axis=1, keepdims=True)

        if np.linalg.norm(V_new - V) < eps:
            break
        V, U, T = V_new, U_new, T_new

    labels = np.argmax(U + T, axis=0)
    return labels, U, T, V


# ══════════════════════════════════════════════════════
# K-Medoids (Partitioning Around Medoids - PAM)
# ══════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════
# BSS / TSS Calculation
# ══════════════════════════════════════════════════════
def compute_bss_tss(X, labels):
    X = np.array(X)
    grand_mean = X.mean(axis=0)
    TSS = np.sum((X - grand_mean) ** 2)
    BSS = 0.0
    for c in np.unique(labels):
        if c == -1: # Skip DBSCAN noise
            continue
        members = X[labels == c]
        if len(members) == 0:
            continue
        cluster_mean = members.mean(axis=0)
        BSS += len(members) * np.sum((cluster_mean - grand_mean) ** 2)
    ratio = BSS / TSS if TSS > 0 else 0
    return BSS, TSS, ratio


def run_clustering(X_princals_path='output/X_princals.pkl'):
    X = pd.read_pickle(X_princals_path).values
    k_range = range(2, 11)

    results = []
    label_store = {}

    print(f"\n[Clustering] Input features shape: {X.shape}")
    print(f"[Clustering] Centroid methods k range: {list(k_range)}")
    print("-" * 80)
    print(f"{'Method':<12} {'k/Params':<15} {'Silhouette':>12} {'BSS/TSS':>12} {'BSS':>14} {'TSS':>14}")
    print("-" * 80)

    # 1. Run Centroid Methods
    for k in k_range:
        methods = {
            'K-Means': lambda k=k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X),
            'K-Medoids': lambda k=k: kmedoids(X, k)[0],
            'FCM': lambda k=k: fcm(X, k)[0],
            'PCM': lambda k=k: pcm(X, k)[0],
            'FPCM': lambda k=k: fpcm(X, k)[0],
            'MFPCM': lambda k=k: mfpcm(X, k)[0],
        }

        for name, func in methods.items():
            try:
                labels = func()
                n_unique = len(np.unique(labels))
                if n_unique < 2:
                    continue
                sil = silhouette_score(X, labels)
                BSS, TSS, ratio = compute_bss_tss(X, labels)
                results.append({
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
                print(f"{name:<12} {f'k={k}':<15} {sil:>12.4f} {ratio*100:>11.2f}% {BSS:>14.4f} {TSS:>14.4f}")
            except Exception as e:
                print(f"{name:<12} {f'k={k}':<15} ERROR: {e}")

    # 2. Run DBSCAN Grid Search (eps scaled to fit the 14D standard space pairwise distance distribution)
    eps_grid = np.round(np.arange(4.0, 10.1, 0.1), 1)
    min_samples_grid = [2, 3]
    
    for eps_val in eps_grid:
        for min_samples_val in min_samples_grid:
            try:
                db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                labels_db = db.fit_predict(X)
                
                # Check valid clusters (excluding noise label -1)
                unique_labels = np.unique(labels_db)
                valid_labels = unique_labels[unique_labels != -1]
                n_clusters_actual = len(valid_labels)
                
                if n_clusters_actual < 2:
                    continue
                
                # Metric calculation (on non-noise elements)
                mask = labels_db != -1
                if mask.sum() < 2:
                    continue
                    
                sil = silhouette_score(X[mask], labels_db[mask])
                BSS, TSS, ratio = compute_bss_tss(X[mask], labels_db[mask])
                
                results.append({
                    'method': 'DBSCAN',
                    'k': n_clusters_actual,
                    'params': f'eps={eps_val},min_s={min_samples_val}',
                    'silhouette': round(sil, 4),
                    'bss_tss_ratio': round(ratio * 100, 2),
                    'BSS': round(BSS, 4),
                    'TSS': round(TSS, 4),
                    'n_clusters_actual': n_clusters_actual
                })
                label_store[f'DBSCAN_eps{eps_val}_min{min_samples_val}'] = labels_db
                print(f"{'DBSCAN':<12} {f'eps={eps_val},min_s={min_samples_val}':<15} {sil:>12.4f} {ratio*100:>11.2f}% {BSS:>14.4f} {TSS:>14.4f}")
            except Exception as e:
                pass

    df_results = pd.DataFrame(results)
    
    # Save outputs
    df_results.to_pickle('output/clustering_results.pkl')
    df_results.to_csv('output/clustering_results.csv', index=False)
    with open('output/label_store.pkl', 'wb') as f:
        pickle.dump(label_store, f)

    print("\n" + "=" * 80)
    print("HASIL TERBAIK (Silhouette tertinggi):")
    if not df_results.empty:
        best = df_results.loc[df_results['silhouette'].idxmax()]
        print(best.to_string())
    else:
        print("Tidak ada model valid yang terbentuk.")

    print(f"\n[OK] Step 4 Clustering complete -> output/clustering_results.pkl & output/label_store.pkl")
    return df_results, label_store

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/X_princals.pkl'):
        from step1_preprocessing import run_preprocessing
        from step3_princals import run_princals
        _, X_s, fc, _ = run_preprocessing(r'C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx')
        run_princals(X_s, fc)
    run_clustering()
