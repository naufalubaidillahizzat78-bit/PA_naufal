"""
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
- Distribusi data
- Deteksi outlier (IQR & Z-score)
- Pola awal sebelum clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def plot_distributions(X_scaled, feature_cols):
    n = len(feature_cols)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        axes[i].hist(X_scaled[col], bins=10, color='steelblue', edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=9, fontweight='bold')
        axes[i].set_xlabel('Normalized Value')
        axes[i].set_ylabel('Frequency')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Distribusi Fitur (Setelah Normalisasi)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('output/eda_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Distribusi tersimpan → output/eda_distributions.png")

def detect_outliers(X_scaled, feature_cols):
    results = {}
    for col in feature_cols:
        # IQR method
        Q1, Q3 = X_scaled[col].quantile(0.25), X_scaled[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = ((X_scaled[col] < Q1 - 1.5*IQR) | (X_scaled[col] > Q3 + 1.5*IQR)).sum()
        # Z-score method
        z = np.abs(stats.zscore(X_scaled[col]))
        outliers_z = (z > 3).sum()
        results[col] = {'IQR_outliers': int(outliers_iqr), 'Zscore_outliers': int(outliers_z)}

    df_out = pd.DataFrame(results).T
    print("\n[EDA] Deteksi Outlier:")
    print(df_out.to_string())

    # Boxplot
    fig, ax = plt.subplots(figsize=(14, 5))
    X_scaled[feature_cols].boxplot(ax=ax, vert=True, patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6))
    ax.set_title('Boxplot Deteksi Outlier (Fitur Ternormalisasi)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    plt.savefig('output/eda_outliers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Boxplot outlier tersimpan → output/eda_outliers.png")
    return df_out

def correlation_heatmap(X_scaled, feature_cols):
    corr = X_scaled[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_cols, fontsize=8)
    ax.set_title('Korelasi Antar Fitur', fontweight='bold')
    # Annotate
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center', fontsize=6,
                    color='white' if abs(corr.values[i, j]) > 0.6 else 'black')
    plt.tight_layout()
    plt.savefig('output/eda_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Korelasi tersimpan → output/eda_correlation.png")

def summary_stats(df, feature_cols):
    stats_df = df[feature_cols].describe().T
    stats_df['cv'] = stats_df['std'] / stats_df['mean']
    stats_df.to_csv('output/eda_summary_stats.csv')
    print("\n[EDA] Summary Statistics:")
    print(stats_df[['mean', 'std', 'min', 'max', 'cv']].round(3).to_string())

def run_eda(X_scaled_path='output/X_scaled.pkl', df_path='output/df_cleaned.pkl',
            feature_cols_path='output/feature_cols.pkl'):
    X_scaled = pd.read_pickle(X_scaled_path)
    df = pd.read_pickle(df_path)
    feature_cols = pd.read_pickle(feature_cols_path).tolist()

    summary_stats(df, feature_cols)
    plot_distributions(X_scaled, feature_cols)
    outlier_report = detect_outliers(X_scaled, feature_cols)
    correlation_heatmap(X_scaled, feature_cols)

    print("\n✅ Step 2 EDA complete → output/eda_*.png")
    return outlier_report

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    # Run preprocessing first if pickles don't exist
    import os
    if not os.path.exists('output/X_scaled.pkl'):
        from step1_preprocessing import run_preprocessing
        run_preprocessing('/mnt/user-data/uploads/data_base_cleaned.xlsx')
    run_eda()
