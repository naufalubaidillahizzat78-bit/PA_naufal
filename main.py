"""
MAIN RUNNER: Jalankan seluruh pipeline clustering secara berurutan.
Penggunaan: python main.py
"""

import os
import sys
import time

# Reconfigure stdout/stderr to UTF-8 to prevent encoding errors on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

DATA_PATH = r"C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx"

def banner(step, title):
    print("\n" + "="*65)
    print(f"  STEP {step}: {title}")
    print("="*65)

def main():
    os.makedirs('output', exist_ok=True)
    t0 = time.time()

    # ── Step 1: Preprocessing ──
    banner(1, "PREPROCESSING DATA")
    from step1_preprocessing import run_preprocessing
    df, X_scaled, feature_cols, scaler = run_preprocessing(DATA_PATH)

    # ── Step 2: EDA ──
    banner(2, "EXPLORATORY DATA ANALYSIS")
    from step2_eda import run_eda
    run_eda()

    # ── Step 3: PRINCALS ──
    banner(3, "TRANSFORMASI DATA - PRINCALS")
    from step3_princals import run_princals
    X_princals, pca_model, var_info = run_princals(X_scaled, feature_cols)

    # ── Step 4: Clustering ──
    banner(4, "CLUSTERING (FCM, PCM, FPCM, MFPCM, DBSCAN, K-Medoids, K-Means)")
    from step4_clustering import run_clustering
    df_results, label_store = run_clustering()

    # ── Step 5: Evaluation ──
    banner(5, "EVALUASI CLUSTER (BSS/TSS & Silhouette)")
    from step5_evaluation import run_evaluation
    best = run_evaluation()

    # ── Step 6: Labeling ──
    banner(6, "LABELING CLUSTER")
    from step6_labeling import run_labeling
    df_labeled, rank_labels = run_labeling()

    # ── Step 7: Dashboard ──
    banner(7, "DASHBOARD VISUALISASI")
    from step7_dashboard import build_dashboard
    build_dashboard()

    # ── Summary ──
    elapsed = time.time() - t0
    print("\n" + "="*65)
    print("  [OK]  PIPELINE SELESAI")
    print("="*65)
    print(f"  Waktu total  : {elapsed:.1f} detik")
    print(f"  Model terbaik: {best['method']}, k={int(best['k'])}")
    print(f"  Silhouette   : {best['silhouette']:.4f}")
    print(f"  BSS/TSS      : {best['bss_tss_ratio']:.2f}%")
    print(f"  Output       : output/")
    print()
    print("  File penting:")
    for f in ['output/dashboard.html',
              'output/df_labeled.csv',
              'output/clustering_results.csv',
              'output/princals_scree.png',
              'output/eval_heatmap.png']:
        exists = '[OK]' if os.path.exists(f) else '[X]'
        print(f"    {exists}  {f}")
    print()

if __name__ == '__main__':
    main()
