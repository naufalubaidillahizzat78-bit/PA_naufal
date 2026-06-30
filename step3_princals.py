"""
STEP 3: TRANSFORMASI DATA - PRINCALS (via PCA non-linear approximation)
- Reduksi dimensi menggunakan PCA setelah Optimal Scaling (ALS Approximation)
- Cumulative variance >= 80%
- Output ke Excel workbook hasil_principals.xlsx
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# Set plotting style
MPL_STYLE = {
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#EAEAEA',
    'axes.labelcolor':  '#566A7F',
    'xtick.color':      '#566A7F',
    'ytick.color':      '#566A7F',
    'text.color':       '#566A7F',
    'grid.color':       '#F0F0F0',
    'grid.alpha':       0.8,
}
plt.rcParams.update(MPL_STYLE)

def optimal_scaling(X):
    """
    Optimal scaling: transform setiap variabel ke skala yang memaksimalkan
    korelasi linear → aproksimasi transformasi PRINCALS non-linear.
    Menggunakan normalisasi berbasis ranking (Rank-based optimal scaling).
    """
    X_opt = X.copy()
    for col in X.columns:
        vals = X[col].values
        ranks = pd.Series(vals).rank(method='average')
        n = len(vals)
        X_opt[col] = (ranks - 0.5) / n
    return X_opt

def run_princals(X_scaled, feature_cols, target_variance=0.80):
    print(f"\n[PRINCALS] Input shape: {X_scaled.shape}")
    print(f"[PRINCALS] Target cumulative variance: {target_variance*100:.0f}%")

    # Step 1: Optimal scaling (non-linear transform ALS approximation)
    X_opt = optimal_scaling(X_scaled[feature_cols])
    print("[PRINCALS] Optimal scaling (rank-based) applied")

    # Step 2: Standardize setelah optimal scaling
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_opt)

    # Step 3: PCA (ALS optimization approximated by eigen decomposition)
    pca = PCA()
    pca.fit(X_std)

    # Hitung Eigen Value (Variansi tiap komponen)
    eigenvalues = pca.explained_variance_
    
    # Hitung Explained Variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumvar = np.cumsum(explained_variance_ratio)

    # Pilih jumlah komponen berdasarkan cumulative variance >= 80%
    n_components = int(np.argmax(cumvar >= target_variance) + 1)
    print(f"[PRINCALS] Komponen dipilih: {n_components} (cumvar: {cumvar[n_components-1]*100:.2f}%)")

    # Step 4: Transform data ke ruang komponen baru (Object Scores)
    pca_final = PCA(n_components=n_components)
    X_princals = pca_final.fit_transform(X_std)
    
    # Menghitung Object Scores & Component Scores untuk seluruh komponen
    X_princals_all = pca.transform(X_std)
    
    # DataFrame Object Scores (Semua komponen)
    columns_all = [f'PC{i+1}' for i in range(len(eigenvalues))]
    df_object_scores_all = pd.DataFrame(X_princals_all, columns=columns_all)
    
    # DataFrame Object Scores (Komponen terpilih saja)
    columns_selected = [f'PC{i+1}' for i in range(n_components)]
    df_object_scores_selected = pd.DataFrame(X_princals, columns=columns_selected)

    # Hitung Component Loadings (Loadings = Eigenvectors * sqrt(Eigenvalues))
    # Loadings untuk seluruh komponen
    loadings_all = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=feature_cols,
        columns=columns_all
    )
    # Loadings untuk komponen terpilih saja
    loadings_selected = loadings_all[columns_selected].copy()

    # Hitung Component Score (Koefisien bobot / Eigenvectors)
    component_scores_all = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=columns_all
    )

    # ── Scree plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Variance per component
    axes[0].bar(range(1, len(explained_variance_ratio)+1),
                explained_variance_ratio * 100,
                color='#7367F0', alpha=0.8, edgecolor='white')
    axes[0].axvline(n_components, color='#EA5455', linestyle='--', label=f'Selected (n={n_components})')
    axes[0].set_xlabel('Komponen Utama')
    axes[0].set_ylabel('Explained Variance (%)')
    axes[0].set_title('Scree Plot - Variance per Komponen', fontweight='bold', color='#7367F0')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Cumulative variance
    axes[1].plot(range(1, len(cumvar)+1), cumvar * 100, 'o-', color='#7367F0', linewidth=2)
    axes[1].axhline(target_variance*100, color='#EA5455', linestyle='--', alpha=0.7, label=f'Threshold ({target_variance*100:.0f}%)')
    axes[1].axvline(n_components, color='#28C76F', linestyle='--', label=f'Selected (n={n_components})')
    axes[1].fill_between(range(1, len(cumvar)+1), cumvar*100, alpha=0.15, color='#7367F0')
    axes[1].set_xlabel('Jumlah Komponen')
    axes[1].set_ylabel('Cumulative Variance (%)')
    axes[1].set_title('Cumulative Variance Explained', fontweight='bold', color='#7367F0')
    axes[1].legend()
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('PRINCALS - Reduksi Dimensi & Analisis Variansi', fontsize=14, fontweight='bold', color='#566A7F', y=0.98)
    plt.tight_layout()
    plt.savefig('output/princals_scree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[PRINCALS] Scree plot saved -> output/princals_scree.png")

    # ── Biplot PC1 vs PC2 ──
    if len(eigenvalues) >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(df_object_scores_selected['PC1'], df_object_scores_selected['PC2'],
                   alpha=0.7, s=80, color='#7367F0', edgecolors='white', linewidth=0.5)
        # Loading arrows
        scale = 3.5
        # Plot only top 15 features with highest loading magnitude in PC1 or PC2 to avoid clutter
        loading_mags = np.sqrt(loadings_selected['PC1']**2 + loadings_selected['PC2']**2)
        top_loadings_idx = loading_mags.nlargest(15).index
        
        for feat in feature_cols:
            x_loading = loadings_selected.loc[feat, 'PC1'] * scale
            y_loading = loadings_selected.loc[feat, 'PC2'] * scale
            if feat in top_loadings_idx:
                ax.annotate('', xy=(x_loading, y_loading), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='#EA5455', lw=1.5))
                ax.text(x_loading * 1.1, y_loading * 1.1, feat, fontsize=7, color='#EA5455', fontweight='bold')
            
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)')
        ax.set_title('Biplot PRINCALS - PC1 vs PC2', fontweight='bold', color='#7367F0')
        plt.tight_layout()
        plt.savefig('output/princals_biplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[PRINCALS] Biplot saved -> output/princals_biplot.png")

    # Save to Excel: hasil_principals.xlsx (exactly 8 sheets)
    # Prep dataframes
    df_eigenvalues = pd.DataFrame({
        'Komponen': [f'PC{i+1}' for i in range(len(eigenvalues))],
        'Eigen Value': eigenvalues
    })
    df_explained_variance = pd.DataFrame({
        'Komponen': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
        'Individual Explained Variance (%)': explained_variance_ratio * 100,
        'Cumulative Explained Variance (%)': cumvar * 100
    })

    excel_path = 'hasil_principals.xlsx'
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            X_scaled.to_excel(writer, sheet_name='Data Preprocessing', index=False)
            df_eigenvalues.to_excel(writer, sheet_name='Eigen Value', index=False)
            df_explained_variance.to_excel(writer, sheet_name='Explained Variance', index=False)
            loadings_all.to_excel(writer, sheet_name='Component Loading')
            df_object_scores_all.to_excel(writer, sheet_name='Object Score', index=False)
            component_scores_all.to_excel(writer, sheet_name='Component Score')
            df_object_scores_all.to_excel(writer, sheet_name='Transformasi PRINCALS', index=False)
            df_object_scores_selected.to_excel(writer, sheet_name='Dataset Final', index=False)
        print(f"[PRINCALS] Excel saved -> {excel_path} (8 sheets)")
    except PermissionError:
        print(f"[PRINCALS] [WARN] Gagal menulis ke '{excel_path}' karena file sedang dibuka oleh program lain (seperti Excel). Silakan tutup file tersebut agar data baru dapat diperbarui.")

    # Save pickles for downstream steps
    df_object_scores_selected.to_pickle('output/X_princals.pkl')
    # Save all components in case needed
    df_object_scores_all.to_pickle('output/X_princals_all.pkl')
    # Save loadings
    loadings_selected.to_csv('output/princals_loadings.csv')
    loadings_all.to_pickle('output/princals_loadings_all.pkl')

    variance_info = {
        'n_components': n_components,
        'cumulative_variance': float(cumvar[n_components-1]),
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'eigenvalues': eigenvalues.tolist()
    }
    pd.Series(variance_info).to_pickle('output/princals_info.pkl')

    print(f"\n[OK] Step 3 PRINCALS complete -> {n_components} components selected, "
          f"cumvar={cumvar[n_components-1]*100:.2f}%")
    print(f"   Reduced dimensions: {X_scaled.shape[1]} -> {n_components}")
    return df_object_scores_selected, pca_final, variance_info

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/X_scaled.pkl'):
        from step1_preprocessing import run_preprocessing
        run_preprocessing(r'C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx')
    X_scaled = pd.read_pickle('output/X_scaled.pkl')
    feature_cols = pd.read_pickle('output/feature_cols.pkl').tolist()
    run_princals(X_scaled, feature_cols)

