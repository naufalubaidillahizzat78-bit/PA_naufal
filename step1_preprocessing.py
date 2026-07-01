"""
STEP 1: PREPROCESSING DATA
- Pembersihan data (missing value & duplikasi)
- Klasifikasi variabel (Identitas, Akademik, Kuesioner, IPS, Kehadiran)
- Encoding data kategorik (LabelEncoder)
- Normalisasi data numerik (StandardScaler)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

def load_and_clean(filepath):
    """
    Baca Excel — deteksi otomatis format header:
      * Multi-index (2 baris header) : data_base2.xlsx
      * Flat (1 baris header)        : data_sintetis_*.xlsx
    Strategi: baca baris ke-1 (index 1) dengan openpyxl.
      Jika semua nilai numerik/None → flat.
      Jika ada string (nama matkul/kategori) → multi-index.
    """
    import openpyxl
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    row0_vals, row1_vals = [], []
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=2, values_only=True)):
        if i == 0: row0_vals = list(row)
        if i == 1: row1_vals = list(row)
    wb.close()

    # Hitung sel string di baris ke-1 yang BUKAN milik kolom identitas biasa
    str_in_row1 = sum(
        1 for v in row1_vals
        if isinstance(v, str) and str(v).strip() not in ('', 'nan', 'None')
           and not str(v).strip().startswith('24')   # bukan NRP
    )
    is_multiindex = str_in_row1 > 3

    if is_multiindex:
        df = pd.read_excel(filepath, header=[0, 1])
        df.columns = [
            f'{c0} - {c1}' if not (pd.isna(c1) or 'Unnamed' in str(c1)) else c0
            for c0, c1 in df.columns
        ]
    else:
        df = pd.read_excel(filepath, header=0)

    df.columns = [str(c).strip() for c in df.columns]
    fmt = 'Multi-index' if is_multiindex else 'Flat'
    print(f"[1] Data loaded: {df.shape[0]} rows, {df.shape[1]} cols  [{fmt}]")

    # Clean text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan})

    # Save raw missing value stats before cleaning
    missing_info = df.isnull().sum()
    os.makedirs('output', exist_ok=True)
    missing_info.to_pickle('output/raw_missing_info.pkl')

    # Impute missing values
    missing_before = df.isnull().sum().sum()
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col].fillna(fill_val, inplace=True)
    print(f"[2] Missing values fixed: {missing_before} -> {df.isnull().sum().sum()}")

    # Duplicates
    dup = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[3] Duplicates removed: {dup}")

    # Calculate average IPS and average Absensi dynamically up to semester 5
    ips_individual = [c for c in df.columns if 'nilai IPS' in c or 'nilai ips' in c.lower()]
    abs_individual = [c for c in df.columns if 'ABSENSI' in c.upper()]
    
    print(f"[Prep] Found IPS columns: {ips_individual}")
    print(f"[Prep] Found ABSENSI columns: {abs_individual}")
    
    df['Rata-Rata IPS'] = df[ips_individual].mean(axis=1)
    df['Rata-Rata Absen Mahasiswa'] = df[abs_individual].mean(axis=1)
    df = df.copy() # De-fragment DataFrame
    
    print(f"[Prep] Calculated average IPS & Attendance (Semester 1 - 5).")
    return df

def categorize_and_encode(df):
    # Identitas (not used for clustering)
    identitas_cols = ['NRP', 'Nama Mahasiswa']
    identitas_cols = [c for c in identitas_cols if c in df.columns]

    # Categorical variables to encode
    cat_cols_to_encode = ['Prodi', 'Asal Kab/Kota']
    encoding_map = {}
    le = LabelEncoder()
    for col in cat_cols_to_encode:
        if col in df.columns:
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Gender (JK)
    if 'JK' in df.columns:
        df['JK_enc'] = df['JK']

    # Groupings
    # 1. IPS (individual and average)
    ips_cols = [c for c in df.columns if 'IPS' in c.upper() or 'nilai ips' in c.lower()]
    # 2. Absensi
    absensi_cols = [c for c in df.columns if 'ABSEN' in c.upper()]
    # 3. Kuesioner (Likert scores)
    kuesioner_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['kuesioner', 'kuisoner', 'kuisioner', 'likert', 'pertanyaan'])]
    # 4. Identitas & Meta
    identitas_all = identitas_cols + ['Prodi', 'Asal Kab/Kota', 'JK']
    # 5. Akademik: all numeric columns except identifiers, IPS, Absensi, Kuesioner, and the encoded features
    exclude_academic = set(identitas_all + ips_cols + absensi_cols + kuesioner_cols + ['Angkatan Tahun'])
    akademik_cols = [c for c in df.select_dtypes(include='number').columns
                     if c not in exclude_academic and not c.endswith('_enc')]

    groups = {
        'identitas': identitas_cols,
        'akademik': akademik_cols,
        'kuesioner': kuesioner_cols,
        'ips': ips_cols,
        'absensi': absensi_cols,
        'categorical': ['Prodi', 'Asal Kab/Kota', 'JK', 'Angkatan Tahun']
    }

    # Save groupings
    with open('output/variable_groups.pkl', 'wb') as f:
        pickle.dump(groups, f)

    print(f"[4] Variable Groupings:")
    print(f"    - Identitas: {len(identitas_cols)} columns ({identitas_cols})")
    print(f"    - Akademik : {len(akademik_cols)} columns")
    print(f"    - Kuesioner: {len(kuesioner_cols)} columns")
    print(f"    - IPS      : {len(ips_cols)} columns")
    print(f"    - Absensi  : {len(absensi_cols)} columns")

    # All features for PRINCALS (exclude aggregate averages like Rata-Rata and demographic data like Angkatan Tahun, Prodi, JK, Asal Kab/Kota)
    ips_individual = [c for c in df.columns if ('nilai IPS' in c or 'nilai ips' in c.lower()) and 'Rata-Rata' not in c]
    abs_individual = [c for c in df.columns if 'ABSENSI' in c.upper() and 'Rata-Rata' not in c]
    
    # We only cluster on academic performance (individual semester GPAs, absences, and course grades)
    feature_cols = akademik_cols + ips_individual + abs_individual

    # Calculate Total and Average Grade for each semester
    for sem in [1, 2, 3, 4, 5]:
        sem_cols = [c for c in df.columns if f'Nilai Semester {sem}' in c 
                    and not any(kw in c.lower() for kw in ['ips', 'absen', 'rata-rata'])]
        if sem_cols:
            df[f'Total Nilai Semester {sem}'] = df[sem_cols].sum(axis=1)
            df[f'Rata-Rata Nilai Semester {sem}'] = df[sem_cols].mean(axis=1)

    return df, feature_cols, groups, encoding_map

def bin_for_princals(df, feature_cols):
    """
    TAHAP 1 — Binning sesuai panduan prompt:
    Konversi setiap variabel kontinu ke skala ORDINAL 1-4 sebelum PRINCALS.
    PRINCALS bekerja optimal pada data ordinal karena menerapkan optimal scaling.

    Skema binning:
      - Kuisioner (1.00-4.00) : [<2.75=1, 2.75-3.25=2, 3.25-3.65=3, >=3.65=4]
      - Nilai matkul (56-99)  : [<65=1,   65-74=2,      75-84=3,     >=85=4  ]
      - IPS (2.00-4.00)       : [<2.75=1, 2.75-3.25=2, 3.25-3.75=3, >=3.75=4]
      - Absensi (85-100%)     : [<90=1,   90-95=2,      95-99=3,     >=99=4  ]
    """
    X_bin = df[feature_cols].copy().astype(float)

    for col in feature_cols:
        col_lo = col.lower()

        # --- Kuisioner ---
        if any(k in col_lo for k in ['kuisioner', 'kuisoner', 'kinerja']):
            X_bin[col] = pd.cut(
                X_bin[col],
                bins=[-np.inf, 2.75, 3.25, 3.65, np.inf],
                labels=[1, 2, 3, 4]
            ).astype(float)

        # --- IPS ---
        elif 'ips' in col_lo:
            X_bin[col] = pd.cut(
                X_bin[col],
                bins=[-np.inf, 2.75, 3.25, 3.75, np.inf],
                labels=[1, 2, 3, 4]
            ).astype(float)

        # --- Absensi ---
        elif 'absen' in col_lo:
            X_bin[col] = pd.cut(
                X_bin[col],
                bins=[-np.inf, 90.0, 95.0, 99.0, np.inf],
                labels=[1, 2, 3, 4]
            ).astype(float)

        # --- Nilai Matkul (integer 56-99) ---
        else:
            X_bin[col] = pd.cut(
                X_bin[col],
                bins=[-np.inf, 64.9, 74.9, 84.9, np.inf],
                labels=[1, 2, 3, 4]
            ).astype(float)

    # Isi NaN sisa binning dengan median
    X_bin = X_bin.fillna(X_bin.median())

    # Statistik binning
    kuis_c  = [c for c in feature_cols if any(k in c.lower() for k in ['kuisioner','kuisoner','kinerja'])]
    ips_c   = [c for c in feature_cols if 'ips' in c.lower()]
    absen_c = [c for c in feature_cols if 'absen' in c.lower()]
    nilai_c = [c for c in feature_cols if c not in kuis_c + ips_c + absen_c]
    print(f"[Bin] Binned {len(feature_cols)} kolom -> ordinal 1-4:")
    print(f"      Kuisioner : {len(kuis_c):>3} kolom  | bins: <2.75 / 2.75-3.25 / 3.25-3.65 / >=3.65")
    print(f"      Nilai     : {len(nilai_c):>3} kolom  | bins: <65 / 65-74 / 75-84 / >=85")
    print(f"      IPS       : {len(ips_c):>3} kolom  | bins: <2.75 / 2.75-3.25 / 3.25-3.75 / >=3.75")
    print(f"      Absensi   : {len(absen_c):>3} kolom  | bins: <90 / 90-95 / 95-99 / >=99")
    return X_bin


def normalize_standard(df, feature_cols):
    scaler = StandardScaler()
    X = df[feature_cols].copy()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    print(f"[5] Normalization (StandardScaler) done. Shape: {X_scaled.shape}")
    return X_scaled, scaler

def run_preprocessing(filepath=r'C:\Users\NITRO\Downloads\data_paa\test_akhir\APLIKASI_DASHBOARD_TA_FIX\data_base2.xlsx'):
    df = load_and_clean(filepath)
    df, feature_cols, groups, enc_map = categorize_and_encode(df)
    X_scaled, scaler = normalize_standard(df, feature_cols)

    # BINNING untuk PRINCALS (ordinal 1-4)
    X_binned = bin_for_princals(df, feature_cols)

    # Save outputs
    df.to_pickle('output/df_cleaned.pkl')
    X_scaled.to_pickle('output/X_scaled.pkl')
    X_binned.to_pickle('output/X_binned.pkl')   # <-- untuk PRINCALS
    pd.Series(feature_cols).to_pickle('output/feature_cols.pkl')

    with open('output/encoding_map.pkl', 'wb') as f:
        pickle.dump(enc_map, f)

    print("\n[OK] Step 1 Preprocessing complete")
    print("     -> output/df_cleaned.pkl")
    print("     -> output/X_scaled.pkl  (continuous, untuk EDA)")
    print("     -> output/X_binned.pkl  (ordinal 1-4, untuk PRINCALS)")
    return df, X_scaled, feature_cols, scaler

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    run_preprocessing()
