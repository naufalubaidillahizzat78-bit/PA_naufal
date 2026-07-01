"""
STEP 3: TRANSFORMASI DATA - PRINCALS (via PCA + Optimal Scaling)
=======================================================================
Mengikuti panduan prompt:
  TAHAP 2: PRINCALS
  - Input  : data ordinal (hasil binning 1-4 dari step1)
  - Output : Object Scores 2D (PC1, PC2) sebagai input clustering
  - Target : Eigenvalue Dimensi 1 & 2 MASING-MASING > 0.8
  - Jika eigenvalue < 0.8: saran tuning binning

Mode:
  n_components_fixed=2  -> untuk clustering (DEFAULT)
  n_components_fixed=N  -> untuk analisis lanjut (misal 45 dimensi)
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


def run_princals(X_scaled, feature_cols, target_variance=0.80, n_components_fixed=2):
    """
    Parameter
    ---------
    n_components_fixed : int (default=2)
        2   -> Object Scores 2D untuk clustering (REKOMENDASI prompt)
        None-> auto berdasarkan target_variance
        N   -> paksa ke N komponen
    """

    W = 100  # print width

    print("\n" + "=" * W)
    print("  STEP 3: PRINCALS  -  Principal Component Analysis with Optimal Scaling")
    print("=" * W)
    print(f"  Jumlah observasi  : {X_scaled.shape[0]}")
    print(f"  Jumlah variabel   : {len(feature_cols)}")
    if n_components_fixed:
        print(f"  Mode              : FIXED  ->  {len(feature_cols)} variabel -> {n_components_fixed} komponen")
    else:
        print(f"  Mode              : AUTO   ->  target cumvar >= {target_variance * 100:.0f}%")

    # ══════════════════════════════════════════════════════════════════════════════
    # TAHAP 1: DAFTAR VARIABEL INPUT (nama asli, dikelompokkan)
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * W)
    print(f"  TAHAP 1 : VARIABEL INPUT ({len(feature_cols)} variabel)")
    print("─" * W)

    # Kategorisasi variabel
    ips_cols   = [c for c in feature_cols if 'nilai IPS' in c or 'nilai ips' in c.lower()]
    absen_cols = [c for c in feature_cols if 'ABSEN' in c.upper() and 'Rata-Rata' not in c]
    matkul_cols = [c for c in feature_cols if c not in ips_cols and c not in absen_cols]

    groups = [
        ("📚 Nilai Mata Kuliah", matkul_cols),
        ("📈 IPS per Semester",  ips_cols),
        ("📋 Presensi/Absensi", absen_cols),
    ]

    no = 1
    for group_name, cols in groups:
        if not cols:
            continue
        print(f"\n  {group_name} ({len(cols)} variabel):")
        print("  " + "-" * 80)
        for c in cols:
            print(f"    [{no:>3}] {c}")
            no += 1
    print("─" * W)
    print(f"  Total variabel input ke PRINCALS : {len(feature_cols)}")

    # ══════════════════════════════════════════════════════════════════════════════
    # KOMPUTASI INTERNAL
    # ══════════════════════════════════════════════════════════════════════════════

    # Optimal Scaling (rank-based)
    X_opt = optimal_scaling(X_scaled[feature_cols])

    # Standardisasi
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_opt)

    # PCA full
    pca = PCA()
    pca.fit(X_std)
    eigenvalues           = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumvar                = np.cumsum(explained_variance_ratio)

    # ── Seleksi jumlah komponen ──────────────────────────────────────────────
    if n_components_fixed is not None:
        # Mode FIXED: paksa ke n_components_fixed
        n_components = min(int(n_components_fixed), len(eigenvalues))
        print(f"\n  [FIXED] Menggunakan tepat {n_components} komponen (sesuai permintaan).")
        print(f"  [INFO ] Dengan {n_components} komponen, variance yang dijelaskan: "
              f"{cumvar[n_components-1]*100:.2f}%")
    else:
        # Mode AUTO: pilih berdasarkan cumvar threshold
        n_components = int(np.argmax(cumvar >= target_variance) + 1)

    # PCA terpilih
    pca_final   = PCA(n_components=n_components)
    X_princals  = pca_final.fit_transform(X_std)
    X_princals_all = pca.transform(X_std)

    columns_all      = [f'PC{i+1}' for i in range(len(eigenvalues))]
    columns_selected = [f'PC{i+1}' for i in range(n_components)]

    df_object_scores_all      = pd.DataFrame(X_princals_all, columns=columns_all)
    df_object_scores_selected = pd.DataFrame(X_princals,    columns=columns_selected)

    # Loading & Score matrices (INDEX = nama variabel asli, KOLOM = PC1, PC2, ...)
    loadings_all = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=feature_cols,
        columns=columns_all
    )
    loadings_selected    = loadings_all[columns_selected].copy()
    component_scores_all = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=columns_all
    )

    # ============================================================
    # VALIDASI EIGENVALUE > 0.8 (Target Metrik Prompt)
    # ============================================================
    print("\n" + "-" * W)
    print("  VALIDASI TARGET METRIK: Eigenvalue > 0.8")
    print("-" * W)
    ev1 = eigenvalues[0]
    ev2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
    status1 = "[OK]  TERPENUHI" if ev1 > 0.8 else "[!!] BELUM TERPENUHI"
    status2 = "[OK]  TERPENUHI" if ev2 > 0.8 else "[!!] BELUM TERPENUHI"
    print(f"  Eigenvalue PC1 : {ev1:.6f}  {status1}")
    print(f"  Eigenvalue PC2 : {ev2:.6f}  {status2}")
    if ev1 <= 0.8 or ev2 <= 0.8:
        print("")
        print("  [SARAN TUNING] Eigenvalue belum mencapai > 0.8. Strategi:")
        print("    1. Perketat binning: gunakan 2 kategori (Tinggi vs Rendah)")
        print("    2. Hapus variabel dengan variance rendah sebelum PRINCALS")
        print("    3. Pastikan ada struktur kelompok nyata pada data")
    print("-" * W)
    mode_label = f"FIXED ({n_components} komponen)" if n_components_fixed else f"AUTO (threshold ≥ {target_variance*100:.0f}%)"
    print("\n" + "─" * W)
    print(f"  TAHAP 2 : EIGEN VALUE  &  SELEKSI KOMPONEN  [{mode_label}]")
    print("─" * W)
    print(f"  {'Komponen':<12} {'Eigen Value':>13} {'Variansi (%)':<14} {'Kumulatif (%)':<15} {'Status':>12}")
    print("  " + "-" * 68)
    for i, (ev, evr, cv) in enumerate(zip(eigenvalues, explained_variance_ratio, cumvar)):
        status = "★ DIPILIH" if i < n_components else ""
        print(f"  {columns_all[i]:<12} {ev:>13.6f} {evr*100:>13.4f}% {cv*100:>14.4f}% {status:>12}")
    print("─" * W)
    print(f"  ✓ {n_components} komponen terpilih  →  menjelaskan {cumvar[n_components-1]*100:.2f}% variasi total")
    print(f"  ✗ {len(eigenvalues) - n_components} komponen TIDAK dipilih  ({100 - cumvar[n_components-1]*100:.2f}% variasi diabaikan)")

    # ══════════════════════════════════════════════════════════════════════════════
    # TAHAP 3: COMPONENT LOADING
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * W)
    print(f"  TAHAP 3 : COMPONENT LOADING  ({len(feature_cols)} variabel × {n_components} komponen terpilih)")
    print("─" * W)
    print("  Keterangan:")
    print("   • BARIS  = Variabel asli (Mata Kuliah / IPS / Presensi)")
    print("   • KOLOM  = Komponen Utama (PC1, PC2, ...) yang terpilih")
    print("   • NILAI  = Korelasi variabel asli dengan komponen utama")
    print("   • Nilai ≥ |0.50| dianggap loading KUAT (kontribusi signifikan)")
    print("─" * W)

    pd.set_option('display.max_rows', 300)
    pd.set_option('display.width',    250)
    pd.set_option('display.float_format', lambda x: f'{x:+.4f}')
    print(loadings_selected.to_string())
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')

    # Variabel terpenting per PC (|loading| tertinggi)
    print("\n" + "─" * W)
    print("  Variabel dengan Kontribusi Terbesar per Komponen Utama (Top 5):")
    print("─" * W)
    for pc in columns_selected:
        top5 = loadings_selected[pc].abs().nlargest(5)
        print(f"\n  {pc}  (eigenvalue: {eigenvalues[int(pc[2:])-1]:.4f}  |  var: {explained_variance_ratio[int(pc[2:])-1]*100:.2f}%)")
        print(f"  {'No':<4} {'Variabel':<60} {'Loading':>10}")
        print("  " + "-" * 76)
        for rank, (var, _) in enumerate(top5.items(), 1):
            val = loadings_selected.loc[var, pc]
            category = "IPS" if var in ips_cols else ("Absensi" if var in absen_cols else "Matkul")
            print(f"  {rank:<4} {var:<60} {val:>+10.4f}  [{category}]")

    # ══════════════════════════════════════════════════════════════════════════════
    # TAHAP 4: TRANSFORMASI PRINCALS
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * W)
    print(f"  TAHAP 4 : TRANSFORMASI PRINCALS  →  Dataset Baru")
    print("─" * W)
    print("  Keterangan:")
    print("   • Seluruh variabel asli DIGANTIKAN oleh komponen utama")
    print("   • Output hanya berisi kolom PC (tanpa nama variabel asli)")
    print(f"   • Dimensi baru: {X_scaled.shape[0]} observasi × {n_components} komponen")
    print(f"   • Kolom: {', '.join(columns_selected)}")
    print("─" * W)

    pd.set_option('display.max_rows', 130)
    pd.set_option('display.width',    250)
    pd.set_option('display.float_format', lambda x: f'{x:+.4f}')
    print(df_object_scores_selected.to_string())
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    print("─" * W)

    # ══════════════════════════════════════════════════════════════════════════════
    # VISUALISASI
    # ══════════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors_bar = ['#7367F0' if i < n_components else '#D0D0D0' for i in range(len(explained_variance_ratio))]
    axes[0].bar(range(1, len(explained_variance_ratio) + 1),
                explained_variance_ratio * 100,
                color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
    axes[0].axvline(n_components + 0.5, color='#EA5455', linestyle='--', linewidth=1.5,
                    label=f'Batas pemilihan (n={n_components})')
    axes[0].set_xlabel('Komponen Utama', fontsize=11)
    axes[0].set_ylabel('Explained Variance (%)', fontsize=11)
    axes[0].set_title('Scree Plot — Variance per Komponen', fontweight='bold', color='#7367F0', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, 'o-', color='#7367F0', linewidth=2, markersize=5)
    axes[1].axhline(target_variance * 100, color='#EA5455', linestyle='--', alpha=0.8,
                    label=f'Threshold ({target_variance*100:.0f}%)')
    axes[1].axvline(n_components, color='#28C76F', linestyle='--', linewidth=1.5,
                    label=f'Selected (n={n_components})')
    axes[1].fill_between(range(1, len(cumvar) + 1), cumvar * 100, alpha=0.12, color='#7367F0')
    axes[1].set_xlabel('Jumlah Komponen', fontsize=11)
    axes[1].set_ylabel('Cumulative Variance (%)', fontsize=11)
    axes[1].set_title('Cumulative Variance Explained', fontweight='bold', color='#7367F0', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('PRINCALS — Reduksi Dimensi & Analisis Variansi', fontsize=14,
                 fontweight='bold', color='#566A7F', y=1.02)
    plt.tight_layout()
    plt.savefig('output/princals_scree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[INFO] Scree plot saved → output/princals_scree.png")

    # Biplot PC1 vs PC2 (top-15 loading arrows)
    if len(eigenvalues) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df_object_scores_selected['PC1'], df_object_scores_selected['PC2'],
                   alpha=0.7, s=80, color='#7367F0', edgecolors='white', linewidth=0.5)
        loading_mags   = np.sqrt(loadings_selected['PC1']**2 + loadings_selected['PC2']**2)
        top_loadings_idx = loading_mags.nlargest(min(15, len(feature_cols))).index
        scale = 3.5
        for feat in top_loadings_idx:
            xl = loadings_selected.loc[feat, 'PC1'] * scale
            yl = loadings_selected.loc[feat, 'PC2'] * scale
            ax.annotate('', xy=(xl, yl), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#EA5455', lw=1.5))
            # Short label: last segment after " - "
            label = feat.split(' - ')[-1] if ' - ' in feat else feat
            ax.text(xl * 1.1, yl * 1.1, label, fontsize=7, color='#EA5455', fontweight='bold')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=11)
        ax.set_title('Biplot PRINCALS — PC1 vs PC2', fontweight='bold', color='#7367F0', fontsize=12)
        plt.tight_layout()
        plt.savefig('output/princals_biplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[INFO] Biplot saved → output/princals_biplot.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # SAVE TO EXCEL (8 sheets)
    # ══════════════════════════════════════════════════════════════════════════════
    df_eigenvalues = pd.DataFrame({
        'Komponen':                  columns_all,
        'Eigen Value':               eigenvalues,
        'Variansi Individual (%)':   explained_variance_ratio * 100,
        'Variansi Kumulatif (%)':    cumvar * 100,
        'Dipilih (≥80%)':            ['Ya' if i < n_components else 'Tidak' for i in range(len(eigenvalues))]
    })

    # Daftar variabel dengan kategori (untuk referensi di Excel)
    df_var_list = pd.DataFrame({
        'No':       range(1, len(feature_cols)+1),
        'Variabel': feature_cols,
        'Kategori': ['IPS' if c in ips_cols else ('Presensi' if c in absen_cols else 'Mata Kuliah')
                     for c in feature_cols]
    })

    excel_path = 'hasil_principals.xlsx'
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Daftar Variabel Input (nama asli + kategori)
            df_var_list.to_excel(writer, sheet_name='Daftar Variabel', index=False)
            # Sheet 2: Data setelah preprocessing (nama asli)
            X_scaled[feature_cols].to_excel(writer, sheet_name='Data Preprocessing', index=False)
            # Sheet 3: Eigen Value (+ status dipilih)
            df_eigenvalues.to_excel(writer, sheet_name='Eigen Value', index=False)
            # Sheet 4: Explained Variance (sama dengan Eigen Value sheet)
            df_eigenvalues.to_excel(writer, sheet_name='Explained Variance', index=False)
            # Sheet 5: Component Loading (variabel asli → PC)
            loadings_all.to_excel(writer, sheet_name='Component Loading')
            # Sheet 6: Component Score Coefficients
            component_scores_all.to_excel(writer, sheet_name='Component Score')
            # Sheet 7: Object Scores semua PC
            df_object_scores_all.to_excel(writer, sheet_name='Object Score', index=False)
            # Sheet 8: Dataset Final (hanya PC terpilih)
            df_object_scores_selected.to_excel(writer, sheet_name='Dataset Final', index=False)
        print(f"[INFO] Excel saved → {excel_path} (8 sheets)")
    except PermissionError:
        print(f"[WARNING] Gagal menulis '{excel_path}' — tutup file Excel terlebih dahulu.")

    # ══════════════════════════════════════════════════════════════════════════════
    # SAVE PICKLES
    # ══════════════════════════════════════════════════════════════════════════════
    df_object_scores_selected.to_pickle('output/X_princals.pkl')
    df_object_scores_all.to_pickle('output/X_princals_all.pkl')
    loadings_selected.to_csv('output/princals_loadings.csv')
    loadings_all.to_pickle('output/princals_loadings_all.pkl')

    variance_info = {
        'n_components':            n_components,
        'cumulative_variance':     float(cumvar[n_components-1]),
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'eigenvalues':             eigenvalues.tolist()
    }
    pd.Series(variance_info).to_pickle('output/princals_info.pkl')

    # ══════════════════════════════════════════════════════════════════════════════
    # RINGKASAN AKHIR
    # ══════════════════════════════════════════════════════════════════════════════
    mode_str = f"FIXED ({n_components} komponen)" if n_components_fixed else f"AUTO (cumvar ≥ {target_variance*100:.0f}%)"
    print("\n" + "=" * W)
    print("  RINGKASAN STEP 3 PRINCALS")
    print("=" * W)
    print(f"  Variabel input (asli) : {len(feature_cols)}")
    print(f"    - Mata Kuliah       : {len(matkul_cols)}")
    print(f"    - IPS per Semester  : {len(ips_cols)}")
    print(f"    - Presensi/Absensi  : {len(absen_cols)}")
    print(f"  Mode seleksi          : {mode_str}")
    print(f"  Komponen terpilih     : {n_components}  (PC1 s.d. PC{n_components})")
    print(f"  Cumulative variance   : {cumvar[n_components-1]*100:.2f}%")
    print(f"  Reduksi dimensi       : {len(feature_cols)} variabel → {n_components} komponen (dimensi)")
    print(f"  Ukuran dataset baru   : {X_scaled.shape[0]} observasi × {n_components} PC")
    print("=" * W)

    return df_object_scores_selected, pca_final, variance_info

if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)

    if not os.path.exists('output/X_binned.pkl'):
        from step1_preprocessing import run_preprocessing
        run_preprocessing()

    # Gunakan X_binned (ordinal 1-4) untuk PRINCALS yang optimal
    if os.path.exists('output/X_binned.pkl'):
        X_input = pd.read_pickle('output/X_binned.pkl')
        print("[INFO] Menggunakan X_binned (ordinal 1-4) untuk PRINCALS")
    else:
        X_input = pd.read_pickle('output/X_scaled.pkl')
        print("[INFO] Menggunakan X_scaled (continuous)")

    feature_cols = pd.read_pickle('output/feature_cols.pkl').tolist()

    # Reduksi ke 2 Object Scores untuk clustering
    run_princals(X_input, feature_cols, n_components_fixed=2)
