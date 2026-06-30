"""
STEP 5: CLUSTERING EVALUATION & RANKING
- Menentukan model terbaik berdasarkan kriteria user (Silhouette >= 0.3 & BSS/TSS >= 50%)
- Visualisasi performa (Heatmaps & Line plots)
- Menyimpan hasil ke Excel workbook hasil_clustering.xlsx (10 sheets)
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

def plot_evaluation_heatmaps(df_results):
    """Plot heatmaps of Silhouette and BSS/TSS per K for centroid methods."""
    df_k = df_results[df_results['method'] != 'DBSCAN'].copy()
    if df_k.empty:
        return
    k_range = sorted(df_k['k'].unique())
    methods = df_k['method'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, label, cmap, val_format in zip(
        axes,
        ['silhouette', 'bss_tss_ratio'],
        ['Silhouette Coefficient', 'BSS/TSS Ratio (%)'],
        ['RdYlGn', 'Blues'],
        ['{:.3f}', '{:.1f}%']
    ):
        pivot = df_k.pivot_table(index='method', columns='k', values=metric, aggfunc='mean')
        
        # Plot using imshow
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                       vmin=pivot.values.min(), vmax=pivot.values.max())
        plt.colorbar(im, ax=ax)
        
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f'k={k}' for k in pivot.columns], fontsize=9)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(f'{label} per Metode dan k', fontweight='bold', color='#7367F0', fontsize=12)
        
        # Annotate cell values
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isnan(val):
                    continue
                txt = val_format.format(val)
                ax.text(j, i, txt, ha='center', va='center', fontsize=8, 
                        color='black', fontweight='bold')

    plt.suptitle('Evaluasi Performa Clustering (Metode Centroid)', fontsize=14, fontweight='bold', color='#566A7F', y=0.98)
    plt.tight_layout()
    plt.savefig('output/eval_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Eval] Heatmap saved → output/eval_heatmap.png")

def plot_evaluation_lines(df_results):
    """Plot line charts of metrics per K for centroid methods."""
    df_k = df_results[df_results['method'] != 'DBSCAN'].copy()
    if df_k.empty:
        return
    methods = df_k['method'].unique()
    colors = ['#7367F0', '#00CFE8', '#28C76F', '#FF9F43', '#EA5455', '#A067F0']

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, metric, title, threshold, thresh_lbl in zip(
        axes,
        ['silhouette', 'bss_tss_ratio'],
        ['Silhouette Coefficient per k', 'BSS/TSS Ratio per k'],
        [0.3, 50.0],
        ['Min Acceptable (0.3)', 'Cukup Baik (50%)']
    ):
        for idx, method in enumerate(methods):
            sub = df_k[df_k['method'] == method].sort_values('k')
            ax.plot(sub['k'], sub[metric], 'o-', label=method, color=colors[idx % len(colors)], linewidth=2)
            
        ax.axhline(threshold, color='#EA5455', linestyle='--', linewidth=1.2, label=thresh_lbl)
        if metric == 'bss_tss_ratio':
            ax.axhline(75.0, color='#28C76F', linestyle='--', linewidth=1.2, label='Sangat Baik (75%)')
            
        ax.set_xlabel('Jumlah Cluster (k)', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(title, fontweight='bold', color='#7367F0', fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.set_xticks(sorted(df_k['k'].unique()))
        ax.grid(alpha=0.3, linestyle=':')

    plt.suptitle('Evaluasi Tren Performa Clustering', fontsize=14, fontweight='bold', color='#566A7F', y=0.98)
    plt.tight_layout()
    plt.savefig('output/eval_lines.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Eval] Line plots saved → output/eval_lines.png")

def select_best_model(df_results):
    """
    Pilih model terbaik berdasarkan kriteria user:
    - Silhouette Coefficient >= 0.3 (minimum)
    - BSS/TSS ratio >= 50% (sebaiknya >= 75%)
    - Jika tidak ada model dengan silhouette >= 0.3, pilih yang memiliki BSS/TSS >= 50% 
      dengan Silhouette tertinggi.
    """
    # 1. Filter candidates with silhouette >= 0.3
    candidates = df_results[df_results['silhouette'] >= 0.3].copy()
    
    if not candidates.empty:
        # Prioritize BSS/TSS >= 75%
        good_bss_75 = candidates[candidates['bss_tss_ratio'] >= 75]
        if not good_bss_75.empty:
            best = good_bss_75.sort_values('silhouette', ascending=False).iloc[0]
        else:
            # Fallback to BSS/TSS >= 50%
            good_bss_50 = candidates[candidates['bss_tss_ratio'] >= 50]
            if not good_bss_50.empty:
                best = good_bss_50.sort_values('silhouette', ascending=False).iloc[0]
            else:
                best = candidates.sort_values('silhouette', ascending=False).iloc[0]
    else:
        print("[Eval] [WARN] Tidak ada model dengan Silhouette >= 0.3. Menerapkan relaksasi pencarian...")
        # 2. Relieve silhouette constraint, look for BSS/TSS >= 50% first
        # Try BSS/TSS >= 75%
        good_bss_75 = df_results[df_results['bss_tss_ratio'] >= 75].copy()
        if not good_bss_75.empty:
            best = good_bss_75.sort_values('silhouette', ascending=False).iloc[0]
        else:
            # Try BSS/TSS >= 50%
            good_bss_50 = df_results[df_results['bss_tss_ratio'] >= 50].copy()
            if not good_bss_50.empty:
                best = good_bss_50.sort_values('silhouette', ascending=False).iloc[0]
            else:
                # Absolute fallback
                best = df_results.sort_values('silhouette', ascending=False).iloc[0]
                
    print("\n" + "="*60)
    print("MODEL TERBAIK TERPILIH (OPTIMAL SELECTION):")
    print(f"  Method    : {best['method']}")
    print(f"  k         : {best['k']}")
    print(f"  Params    : {best['params']}")
    print(f"  Silhouette: {best['silhouette']:.4f}")
    print(f"  BSS/TSS   : {best['bss_tss_ratio']:.2f}%")
    print("="*60)
    
    pd.Series(best).to_pickle('output/best_model.pkl')
    return best

def run_evaluation(results_path='output/clustering_results.pkl', 
                   label_store_path='output/label_store.pkl',
                   df_path='output/df_cleaned.pkl'):
    
    df_results = pd.read_pickle(results_path)
    with open(label_store_path, 'rb') as f:
        label_store = pickle.load(f)
    df = pd.read_pickle(df_path)

    # 1. Compile Heatmaps and Line Plots
    plot_evaluation_heatmaps(df_results)
    plot_evaluation_lines(df_results)

    # 2. Find best configuration per method & construct rankings
    best_per_method = []
    methods_list = ['K-Means', 'K-Medoids', 'DBSCAN', 'FCM', 'PCM', 'FPCM', 'MFPCM']
    
    for m in methods_list:
        sub = df_results[df_results['method'] == m]
        if not sub.empty:
            # Sort by silhouette desc, then bss_tss desc
            best_cfg = sub.sort_values(['silhouette', 'bss_tss_ratio'], ascending=[False, False]).iloc[0]
            best_per_method.append(best_cfg)
            
    df_ranking = pd.DataFrame(best_per_method)
    # Sort ranking by silhouette score descending
    df_ranking = df_ranking.sort_values(['silhouette', 'bss_tss_ratio'], ascending=[False, False]).reset_index(drop=True)
    df_ranking['Ranking'] = df_ranking.index + 1
    
    # Save Ranking dataframe
    df_ranking.to_pickle('output/method_ranking.pkl')
    
    # Best overall model based on specific user criteria
    best_model = select_best_model(df_results)

    # 3. Create Excel sheets for hasil_clustering.xlsx
    excel_path = 'hasil_clustering.xlsx'
    
    # Base columns for student assignment sheets
    student_base = df[['NRP', 'Nama Mahasiswa']].copy()
    
    # Dict to hold sheets data
    sheets = {}
    
    # ── Sheets 1 to 7: Centroid-based and DBSCAN assignments ──
    for m in ['K-Means', 'K-Medoids', 'FCM', 'PCM', 'FPCM', 'MFPCM']:
        sheet_df = student_base.copy()
        for k in range(2, 11):
            key = f'{m}_k{k}'
            if key in label_store:
                sheet_df[f'Cluster_k{k}'] = label_store[key]
        sheets[m] = sheet_df
        
    # DBSCAN assignment sheet
    dbscan_df = student_base.copy()
    dbscan_keys = [k for k in label_store.keys() if k.startswith('DBSCAN')]
    for key in dbscan_keys:
        col_name = key.replace('DBSCAN_', '')
        dbscan_df[col_name] = label_store[key]
    sheets['DBSCAN'] = dbscan_df

    # ── Sheet 8: Perbandingan Seluruh Metode ──
    sheets['Perbandingan Seluruh Metode'] = df_results.copy()

    # ── Sheet 9: Ranking Metode ──
    sheets['Ranking Metode'] = df_ranking.copy()

    # ── Sheet 10: Dataset dengan Label Cluster Terbaik ──
    # Assign labels of the best overall model
    best_method = best_model['method']
    best_params = best_model['params']
    
    # Map params back to label_store key
    if best_method == 'DBSCAN':
        parts = best_params.split(',')
        eps_val = parts[0].split('=')[1]
        min_s_val = parts[1].split('=')[1]
        best_key = f'DBSCAN_eps{eps_val}_min{min_s_val}'
    else:
        k_val = best_params.split('=')[1]
        best_key = f'{best_method}_k{k_val}'
        
    best_labels = label_store[best_key]
    
    # Combine with original dataset
    df_best_labeled = df.copy()
    df_best_labeled['cluster'] = best_labels
    
    # Drop aggregate/demographic/encoded columns from Excel export as requested
    drop_cols = [c for c in df_best_labeled.columns if any(kw in c.lower() for kw in ['rata-rata', 'angkatan', 'tahun', 'prodi', 'jenis kelamin', 'jk', 'asal kab/kota'])]
    df_best_labeled = df_best_labeled.drop(columns=drop_cols, errors='ignore')
    
    # Add to sheets
    sheets['Dataset dengan Label Cluster Terbaik'] = df_best_labeled

    # Write to Excel
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for name in ['K-Means', 'K-Medoids', 'DBSCAN', 'FCM', 'PCM', 'FPCM', 'MFPCM', 
                         'Perbandingan Seluruh Metode', 'Ranking Metode', 'Dataset dengan Label Cluster Terbaik']:
                sheets[name].to_excel(writer, sheet_name=name, index=False)
        print(f"[Eval] Excel saved -> {excel_path} (10 sheets)")
    except PermissionError:
        print(f"[Eval] [WARN] Gagal menulis ke '{excel_path}' karena file sedang dibuka oleh program lain (seperti Excel). Silakan tutup file tersebut agar data baru dapat diperbarui.")
    return best_model

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/clustering_results.pkl'):
        from step1_preprocessing import run_preprocessing
        from step3_princals import run_princals
        from step4_clustering import run_clustering
        _, X_s, fc, _ = run_preprocessing(r'C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx')
        run_princals(X_s, fc)
        run_clustering()
    run_evaluation()
