"""
STEP 3: TRANSFORMASI DATA - PRINCALS (via PCA non-linear approximation)
- Reduksi dimensi data campuran (numerik + kategorik)
- Cumulative variance >= 80%
- Implementasi menggunakan sklearn PCA + preprocessing optimal
  (PRINCALS sesungguhnya menggunakan Alternating Least Squares (ALS)
   yang diapproksimasi di sini menggunakan PCA setelah optimal scaling)
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def optimal_scaling(X):
    """
    Optimal scaling: transform setiap variabel ke skala yang memaksimalkan
    korelasi linear → aproksimasi transformasi PRINCALS non-linear.
    Untuk variabel dengan skala terbatas (0-1 setelah normalisasi), 
    gunakan rank-based normalization sebagai transformasi non-linear.
    """
    X_opt = X.copy()
    for col in X.columns:
        vals = X[col].values
        # Rank-based non-linear transform (aproksimasi PRINCALS optimal scaling)
        ranks = pd.Series(vals).rank(method='average')
        # Normal score transformation
        n = len(vals)
        X_opt[col] = (ranks - 0.5) / n
    return X_opt


def run_princals(X_scaled, feature_cols, target_variance=0.80):
    print(f"\n[PRINCALS] Input shape: {X_scaled.shape}")
    print(f"[PRINCALS] Target cumulative variance: {target_variance*100:.0f}%")

    # Step 1: Optimal scaling (non-linear transform ALS approximation)
    X_opt = optimal_scaling(X_scaled[feature_cols])
    print("[PRINCALS] Optimal scaling (rank-based) applied")

    # Step 2: Standardize sebelum PCA
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_opt)

    # Step 3: PCA (ALS optimization approximated by eigen decomposition)
    pca = PCA()
    pca.fit(X_std)

    # Step 4: Pilih jumlah komponen berdasarkan cumulative variance >= 80%
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= target_variance) + 1)
    print(f"[PRINCALS] Komponen dipilih: {n_components} (cumvar: {cumvar[n_components-1]*100:.2f}%)")

    # Step 5: Transform data
    pca_final = PCA(n_components=n_components)
    X_princals = pca_final.fit_transform(X_std)
    X_princals_df = pd.DataFrame(
        X_princals,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    # ── Scree plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Variance per component
    axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1),
                pca.explained_variance_ratio_ * 100,
                color='steelblue', alpha=0.8, edgecolor='white')
    axes[0].axvline(n_components, color='red', linestyle='--', label=f'n={n_components}')
    axes[0].set_xlabel('Komponen')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot - Variance per Komponen')
    axes[0].legend()

    # Cumulative variance
    axes[1].plot(range(1, len(cumvar)+1), cumvar * 100, 'o-', color='steelblue', linewidth=2)
    axes[1].axhline(target_variance*100, color='red', linestyle='--', alpha=0.7, label=f'{target_variance*100:.0f}% threshold')
    axes[1].axvline(n_components, color='green', linestyle='--', label=f'n={n_components}')
    axes[1].fill_between(range(1, len(cumvar)+1), cumvar*100, alpha=0.15, color='steelblue')
    axes[1].set_xlabel('Jumlah Komponen')
    axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].set_ylim(0, 105)

    plt.suptitle('PRINCALS - Reduksi Dimensi', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/princals_scree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[PRINCALS] Scree plot tersimpan → output/princals_scree.png")

    # ── Loading matrix ──
    loadings = pd.DataFrame(
        pca_final.components_.T,
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    loadings.to_csv('output/princals_loadings.csv')
    print("[PRINCALS] Loadings:")
    print(loadings.round(3).to_string())

    # ── Biplot PC1 vs PC2 ──
    if n_components >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(X_princals_df['PC1'], X_princals_df['PC2'],
                   alpha=0.7, s=80, color='steelblue', edgecolors='white', linewidth=0.5)
        # Loading arrows
        scale = 3
        for i, feat in enumerate(feature_cols):
            ax.annotate('', xy=(loadings.iloc[i, 0]*scale, loadings.iloc[i, 1]*scale),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            ax.text(loadings.iloc[i, 0]*scale*1.1, loadings.iloc[i, 1]*scale*1.1,
                    feat, fontsize=7, color='darkred')
        ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Biplot PRINCALS - PC1 vs PC2', fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/princals_biplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[PRINCALS] Biplot tersimpan → output/princals_biplot.png")

    # Save
    X_princals_df.to_pickle('output/X_princals.pkl')
    variance_info = {
        'n_components': n_components,
        'cumulative_variance': float(cumvar[n_components-1]),
        'explained_variance_ratio': pca_final.explained_variance_ratio_.tolist()
    }
    pd.Series(variance_info).to_pickle('output/princals_info.pkl')

    print(f"\n✅ Step 3 PRINCALS complete → {n_components} komponen, "
          f"cumvar={cumvar[n_components-1]*100:.2f}%")
    print(f"   Dimensi: {X_scaled.shape[1]} → {n_components}")
    return X_princals_df, pca_final, variance_info

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/X_scaled.pkl'):
        from step1_preprocessing import run_preprocessing
        run_preprocessing('/mnt/user-data/uploads/data_base_cleaned.xlsx')
    X_scaled = pd.read_pickle('output/X_scaled.pkl')
    feature_cols = pd.read_pickle('output/feature_cols.pkl').tolist()
    run_princals(X_scaled, feature_cols)
