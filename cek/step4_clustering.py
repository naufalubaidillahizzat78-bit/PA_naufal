"""
STEP 4: CLUSTERING
Metode: FCM, PCM, FPCM, MFPCM, DBSCAN, K-Medoids, K-Means
k = 2 sampai k = 10
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# Fuzzy C-Means (FCM)
# ══════════════════════════════════════════════════════
def fcm(X, c, m=2, max_iter=200, eps=1e-6, seed=42):
    np.random.seed(seed)
    n, d = X.shape
    # Init membership matrix
    U = np.random.dirichlet(np.ones(c), size=n).T  # shape (c, n)
    U = U / U.sum(axis=0)

    for _ in range(max_iter):
        # Compute centroids
        Um = U ** m
        V = (Um @ X) / Um.sum(axis=1, keepdims=True)  # (c, d)

        # Update membership
        dist = np.array([[np.linalg.norm(X[j] - V[i]) for j in range(n)] for i in range(c)])
        dist = np.maximum(dist, 1e-10)
        U_new = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U_new[i] += (dist[i] / dist[k]) ** (2 / (m - 1))
        U_new = 1.0 / U_new

        if np.linalg.norm(U_new - U) < eps:
            break
        U = U_new

    labels = np.argmax(U, axis=0)
    return labels, U, V


# ══════════════════════════════════════════════════════
# Possibilistic C-Means (PCM)
# ══════════════════════════════════════════════════════
def pcm(X, c, m=2, max_iter=200, eps=1e-6, seed=42):
    np.random.seed(seed)
    # Init with FCM
    labels_fcm, U_fcm, V = fcm(X, c, m=m, seed=seed)
    V = V.copy()
    n, d = X.shape
    # Compute eta (typicality scale)
    dist2 = np.array([[np.linalg.norm(X[j] - V[i])**2 for j in range(n)] for i in range(c)])
    Um = U_fcm ** m
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
def fpcm(X, c, m=2, eta_param=2, max_iter=200, eps=1e-6, seed=42):
    np.random.seed(seed)
    n, d = X.shape
    U = np.random.dirichlet(np.ones(c), size=n).T
    U = U / U.sum(axis=0)
    T = np.random.rand(c, n)
    T = T / T.sum(axis=0)
    V = (U ** m @ X) / (U ** m).sum(axis=1, keepdims=True)

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
        U, T = U_new, T_new

    labels = np.argmax(U, axis=0)
    return labels, U, T, V


# ══════════════════════════════════════════════════════
# Modified FPCM (MFPCM) — weighted combination
# ══════════════════════════════════════════════════════
def mfpcm(X, c, m=2, alpha=0.5, max_iter=200, eps=1e-6, seed=42):
    np.random.seed(seed)
    n, d = X.shape
    U = np.random.dirichlet(np.ones(c), size=n).T
    U = U / U.sum(axis=0)
    T = np.random.rand(c, n)

    V = (U ** m @ X) / (U ** m).sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        dist = np.array([[np.linalg.norm(X[j] - V[i])**2 for j in range(n)] for i in range(c)])
        dist = np.maximum(dist, 1e-10)

        # Update U (fuzzy part)
        U_new = np.zeros((c, n))
        for i in range(c):
            for k in range(c):
                U_new[i] += (dist[i] / dist[k]) ** (1 / (m - 1))
        U_new = 1.0 / U_new

        # Update T (possibilistic part)
        eta = ((U_new ** m) * dist).sum(axis=1) / (U_new ** m).sum(axis=1)
        eta = np.maximum(eta, 1e-10)
        T_new = np.zeros((c, n))
        for i in range(c):
            T_new[i] = 1.0 / (1.0 + dist[i] / eta[i])

        # Combined membership
        W = alpha * U_new + (1 - alpha) * T_new
        V_new = (W ** m @ X) / (W ** m).sum(axis=1, keepdims=True)

        if np.linalg.norm(V_new - V) < eps:
            break
        V, U, T = V_new, U_new, T_new

    labels = np.argmax(U + T, axis=0)
    return labels, U, T, V


# ══════════════════════════════════════════════════════
# K-Medoids (PAM)
# ══════════════════════════════════════════════════════
def kmedoids(X, k, max_iter=100, seed=42):
    np.random.seed(seed)
    n = len(X)
    medoid_idx = np.random.choice(n, k, replace=False)
    D = pairwise_distances(X)

    for _ in range(max_iter):
        # Assign
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
# BSS / TSS evaluation
# ══════════════════════════════════════════════════════
def compute_bss_tss(X, labels):
    X = np.array(X)
    grand_mean = X.mean(axis=0)
    TSS = np.sum((X - grand_mean) ** 2)
    BSS = 0.0
    for c in np.unique(labels):
        members = X[labels == c]
        if len(members) == 0:
            continue
        cluster_mean = members.mean(axis=0)
        BSS += len(members) * np.sum((cluster_mean - grand_mean) ** 2)
    ratio = BSS / TSS if TSS > 0 else 0
    return BSS, TSS, ratio


# ══════════════════════════════════════════════════════
# Main: run all methods k=2..10
# ══════════════════════════════════════════════════════
def run_clustering(X_princals_path='output/X_princals.pkl'):
    X = pd.read_pickle(X_princals_path).values
    k_range = range(2, 11)

    results = []
    label_store = {}

    print(f"\n[Clustering] Data shape: {X.shape}")
    print(f"[Clustering] k range: {list(k_range)}")
    print("-" * 75)
    print(f"{'Method':<12} {'k':<4} {'Silhouette':>12} {'BSS/TSS':>12} {'BSS':>14} {'TSS':>14}")
    print("-" * 75)

    for k in k_range:
        methods = {
            'KMeans': lambda k=k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X),
            'KMedoids': lambda k=k: kmedoids(X, k)[0],
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
                    'method': name, 'k': k,
                    'silhouette': round(sil, 4),
                    'bss_tss_ratio': round(ratio * 100, 2),
                    'BSS': round(BSS, 4),
                    'TSS': round(TSS, 4),
                    'n_clusters_actual': n_unique
                })
                label_store[f'{name}_k{k}'] = labels
                print(f"{name:<12} {k:<4} {sil:>12.4f} {ratio*100:>11.2f}% {BSS:>14.4f} {TSS:>14.4f}")
            except Exception as e:
                print(f"{name:<12} {k:<4} ERROR: {e}")

    # DBSCAN (no k loop — auto)
    for eps_val in [0.3, 0.5, 0.7]:
        try:
            db = DBSCAN(eps=eps_val, min_samples=2)
            labels_db = db.fit_predict(X)
            n_unique = len(np.unique(labels_db[labels_db != -1]))
            if n_unique < 2:
                continue
            mask = labels_db != -1
            if mask.sum() < 2:
                continue
            sil = silhouette_score(X[mask], labels_db[mask])
            BSS, TSS, ratio = compute_bss_tss(X[mask], labels_db[mask])
            k_eff = n_unique
            results.append({
                'method': f'DBSCAN(eps={eps_val})', 'k': k_eff,
                'silhouette': round(sil, 4),
                'bss_tss_ratio': round(ratio * 100, 2),
                'BSS': round(BSS, 4), 'TSS': round(TSS, 4),
                'n_clusters_actual': k_eff
            })
            label_store[f'DBSCAN_eps{eps_val}'] = labels_db
            print(f"{'DBSCAN':<12} {k_eff:<4} {sil:>12.4f} {ratio*100:>11.2f}% (eps={eps_val})")
        except Exception as e:
            print(f"DBSCAN eps={eps_val} ERROR: {e}")

    df_results = pd.DataFrame(results)
    df_results.to_pickle('output/clustering_results.pkl')
    df_results.to_csv('output/clustering_results.csv', index=False)
    pd.to_pickle(label_store, 'output/label_store.pkl')

    print("\n" + "=" * 75)
    print("HASIL TERBAIK (Silhouette tertinggi):")
    best = df_results.loc[df_results['silhouette'].idxmax()]
    print(best.to_string())

    print("\nHASIL TERBAIK (BSS/TSS ≥ 75%):")
    good = df_results[df_results['bss_tss_ratio'] >= 75].sort_values('silhouette', ascending=False)
    print(good.head(5).to_string() if not good.empty else "  Tidak ada yang ≥75%")

    print(f"\n✅ Step 4 Clustering complete → output/clustering_results.csv")
    return df_results, label_store

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/X_princals.pkl'):
        from step1_preprocessing import run_preprocessing
        from step3_princals import run_princals
        _, X_s, fc, _ = run_preprocessing('/mnt/user-data/uploads/data_base_cleaned.xlsx')
        run_princals(X_s, fc)
    run_clustering()
