"""
STEP 5: EVALUASI CLUSTER
- BSS/TSS ratio (persentase, threshold: 50% cukup baik, 75% sangat baik)
- Silhouette Coefficient (minimum 0.3)
- Penentuan model terbaik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')


def plot_evaluation_heatmaps(df_results):
    methods = df_results['method'].unique().tolist()
    # Filter non-DBSCAN for k-based plots
    df_k = df_results[~df_results['method'].str.startswith('DBSCAN')].copy()
    k_range = sorted(df_k['k'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, label, threshold in zip(
        axes,
        ['silhouette', 'bss_tss_ratio'],
        ['Silhouette Coefficient', 'BSS/TSS Ratio (%)'],
        [0.3, 50]
    ):
        pivot = df_k.pivot_table(index='method', columns='k', values=metric, aggfunc='mean')
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                       vmin=pivot.values.min(), vmax=pivot.values.max())
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(k_range)))
        ax.set_xticklabels([f'k={k}' for k in k_range])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'{label}\n(threshold={threshold})', fontweight='bold')
        # Annotate values
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isnan(val):
                    continue
                txt = f'{val:.3f}' if metric == 'silhouette' else f'{val:.1f}%'
                color = 'white' if (val < pivot.values.min() + (pivot.values.max()-pivot.values.min())*0.3) else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='black', fontweight='bold')

    plt.suptitle('Evaluasi Clustering: Silhouette & BSS/TSS', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/eval_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Eval] Heatmap evaluasi tersimpan → output/eval_heatmap.png")


def plot_evaluation_lines(df_results):
    df_k = df_results[~df_results['method'].str.startswith('DBSCAN')].copy()
    methods = df_k['method'].unique()
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, metric, title, threshold, thresh_label in zip(
        axes,
        ['silhouette', 'bss_tss_ratio'],
        ['Silhouette Coefficient per k', 'BSS/TSS Ratio per k'],
        [0.3, 50],
        ['Min acceptable (0.3)', '50% threshold']
    ):
        for i, method in enumerate(methods):
            sub = df_k[df_k['method'] == method].sort_values('k')
            ax.plot(sub['k'], sub[metric], 'o-', label=method, color=colors[i % len(colors)], linewidth=1.8)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=1.2, label=thresh_label)
        if metric == 'bss_tss_ratio':
            ax.axhline(75, color='green', linestyle='--', linewidth=1.2, label='75% (sangat baik)')
        ax.set_xlabel('Jumlah Cluster (k)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        ax.set_xticks(sorted(df_k['k'].unique()))
        ax.grid(alpha=0.3)

    plt.suptitle('Evaluasi Performa Clustering', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/eval_lines.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Eval] Line plots tersimpan → output/eval_lines.png")


def select_best_model(df_results):
    """Pilih model terbaik: silhouette tertinggi & BSS/TSS memenuhi threshold."""
    # Filter silhouette >= 0.3
    candidates = df_results[df_results['silhouette'] >= 0.3].copy()
    if candidates.empty:
        print("[Eval] ⚠️  Tidak ada model dengan silhouette ≥ 0.3. Menggunakan nilai terbaik yang ada.")
        candidates = df_results.copy()

    # Prefer BSS/TSS >= 50%
    good_bss = candidates[candidates['bss_tss_ratio'] >= 50]
    if not good_bss.empty:
        candidates = good_bss

    # Best = highest silhouette; tiebreak = highest bss_tss
    best = candidates.sort_values(['silhouette', 'bss_tss_ratio'], ascending=[False, False]).iloc[0]

    print("\n" + "="*60)
    print("MODEL TERBAIK TERPILIH:")
    print(f"  Method    : {best['method']}")
    print(f"  k         : {best['k']}")
    print(f"  Silhouette: {best['silhouette']:.4f}  (min=0.3)")
    print(f"  BSS/TSS   : {best['bss_tss_ratio']:.2f}%  (50%=cukup, 75%=sangat baik)")
    verdict = "✅ Sangat Baik" if best['bss_tss_ratio'] >= 75 else \
              "✅ Cukup Baik" if best['bss_tss_ratio'] >= 50 else "⚠️  Kurang"
    print(f"  Kualitas  : {verdict}")
    print("="*60)

    pd.Series(best).to_pickle('output/best_model.pkl')
    return best


def run_evaluation(results_path='output/clustering_results.pkl'):
    df_results = pd.read_pickle(results_path)

    print("\n[Eval] Full Results:")
    print(df_results.sort_values('silhouette', ascending=False).to_string(index=False))

    plot_evaluation_heatmaps(df_results)
    plot_evaluation_lines(df_results)
    best = select_best_model(df_results)

    print(f"\n✅ Step 5 Evaluation complete")
    return best

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/clustering_results.pkl'):
        from step1_preprocessing import run_preprocessing
        from step3_princals import run_princals
        from step4_clustering import run_clustering
        _, X_s, fc, _ = run_preprocessing('/mnt/user-data/uploads/data_base_cleaned.xlsx')
        run_princals(X_s, fc)
        run_clustering()
    run_evaluation()
