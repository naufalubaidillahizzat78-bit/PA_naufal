"""
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
- Hitung statistik deskriptif dan simpan ke CSV
- Visualisasi kualitas data (Missing Values)
- Visualisasi distribusi fitur utama
- Deteksi outlier menggunakan Boxplot
- Heatmap matriks korelasi fitur utama
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import warnings

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

def plot_missing_values():
    """Plot missing values distribution before cleaning."""
    if not os.path.exists('output/raw_missing_info.pkl'):
        print("[EDA] raw_missing_info.pkl not found. Skipping missing value plot.")
        return
        
    missing_info = pd.read_pickle('output/raw_missing_info.pkl')
    # Filter only columns with missing values to make plot readable
    missing_only = missing_info[missing_info > 0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if not missing_only.empty:
        missing_only.sort_values(ascending=False).plot(kind='bar', color='#EA5455', edgecolor='white', ax=ax)
        ax.set_ylabel('Number of Missing Cells')
        ax.set_title('Distribusi Nilai Hilang per Variabel (Sebelum Preprocessing)', fontweight='bold', fontsize=12, color='#7367F0')
    else:
        # If no missing values
        ax.text(0.5, 0.5, 'Tidak ditemukan missing values dalam dataset!', 
                ha='center', va='center', fontsize=12, color='#28C76F', fontweight='bold')
        ax.set_axis_off()
        
    plt.tight_layout()
    plt.savefig('output/eda_missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Missing value plot saved -> output/eda_missing_values.png")

def summary_statistics(df, feature_cols):
    """Compute descriptive statistics and save to CSV."""
    stats_df = df[feature_cols].describe().T
    stats_df['cv'] = stats_df['std'] / stats_df['mean']
    stats_df.to_csv('output/eda_summary_stats.csv')
    print("\n[EDA] Summary Statistics saved -> output/eda_summary_stats.csv")
    print(stats_df[['mean', 'std', 'min', 'max', 'cv']].head(10).round(3).to_string())
    return stats_df

def plot_distributions(X_scaled, key_features):
    """Plot histograms for key features."""
    n = len(key_features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.2))
    axes = axes.flatten()
    
    for i, col in enumerate(key_features):
        axes[i].hist(X_scaled[col], bins=10, color='#7367F0', edgecolor='white', alpha=0.8)
        axes[i].set_title(col, fontsize=10, fontweight='bold', color='#566A7F')
        axes[i].set_xlabel('Standardized Value', fontsize=8)
        axes[i].set_ylabel('Frequency', fontsize=8)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    fig.suptitle('Distribusi Fitur Utama (Setelah Preprocessing & Standardisasi)', 
                 fontsize=14, fontweight='bold', color='#7367F0', y=1.01)
    plt.tight_layout()
    plt.savefig('output/eda_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Distributions plot saved -> output/eda_distributions.png")

def plot_outliers(df, key_features):
    """Plot boxplot for outlier detection of key features."""
    fig, ax = plt.subplots(figsize=(14, 6))
    df[key_features].boxplot(ax=ax, vert=True, patch_artist=True,
                             boxprops=dict(facecolor='#7367F0', alpha=0.6, color='#7367F0'),
                             medianprops=dict(color='#EA5455', linewidth=1.5),
                             whiskerprops=dict(color='#8E9BAE'),
                             capprops=dict(color='#8E9BAE'))
    
    ax.set_title('Boxplot Deteksi Outlier (Fitur Utama Sebelum Normalisasi)', 
                 fontweight='bold', fontsize=13, color='#7367F0')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('output/eda_outliers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Outliers boxplot saved -> output/eda_outliers.png")

def plot_correlation(df, key_features):
    """Plot correlation heatmap of key features."""
    corr = df[key_features].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(key_features)))
    ax.set_yticks(range(len(key_features)))
    ax.set_xticklabels(key_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(key_features, fontsize=8)
    ax.set_title('Matriks Korelasi (Fitur Utama)', fontweight='bold', fontsize=13, color='#7367F0')
    
    # Annotate values
    for i in range(len(key_features)):
        for j in range(len(key_features)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > 0.6 else 'black')
                    
    plt.tight_layout()
    plt.savefig('output/eda_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] Correlation matrix heatmap saved -> output/eda_correlation.png")

def run_eda(X_scaled_path='output/X_scaled.pkl', df_path='output/df_cleaned.pkl',
            feature_cols_path='output/feature_cols.pkl'):
    X_scaled = pd.read_pickle(X_scaled_path)
    df = pd.read_pickle(df_path)
    feature_cols = pd.read_pickle(feature_cols_path).tolist()

    # Load variable groups to find key representative features
    with open('output/variable_groups.pkl', 'rb') as f:
        groups = pickle.load(f)

    # Choose a representative set of key features (10-12 variables)
    ips_keys = [c for c in groups['ips'] if 'Rata-Rata IPS' in c or 'nilai IPS' in c][:2]
    absen_keys = [c for c in groups['absensi'] if 'Absen' in c or 'ABSENSI' in c][:2]
    course_keys = [c for c in groups['akademik']][:6]
    cat_keys = [c for c in X_scaled.columns if c.endswith('_enc')][:2]
    
    key_features = ips_keys + absen_keys + course_keys + cat_keys
    # Deduplicate and ensure they exist
    key_features = list(dict.fromkeys([c for c in key_features if c in X_scaled.columns]))

    print(f"[EDA] Selected {len(key_features)} key features for static plotting: {key_features}")

    # Generate plots and statistics
    summary_statistics(df, feature_cols)
    plot_missing_values()
    plot_distributions(X_scaled, key_features)
    plot_outliers(df, key_features)
    plot_correlation(df, key_features)

    print("\n[OK] Step 2 EDA complete -> output/eda_*.png, output/eda_summary_stats.csv")

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/X_scaled.pkl'):
        from step1_preprocessing import run_preprocessing
        run_preprocessing()
    run_eda()
