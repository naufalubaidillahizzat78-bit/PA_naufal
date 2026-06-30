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
    # Read Excel with two header rows (merged cells handle)
    df = pd.read_excel(filepath, header=[0, 1])
    
    # Flatten MultiIndex columns
    df.columns = [f'{c0} - {c1}' if not (pd.isna(c1) or 'Unnamed' in str(c1)) else c0 for c0, c1 in df.columns]
    df.columns = [c.strip() for c in df.columns]
    
    print(f"[1] Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

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

def normalize_standard(df, feature_cols):
    scaler = StandardScaler()
    X = df[feature_cols].copy()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    print(f"[5] Normalization (StandardScaler) done. Shape: {X_scaled.shape}")
    return X_scaled, scaler

def run_preprocessing(filepath=r'C:\Users\NITRO\Downloads\data_paa\test_akhir\cek2\data_base2.xlsx'):
    df = load_and_clean(filepath)
    df, feature_cols, groups, enc_map = categorize_and_encode(df)
    X_scaled, scaler = normalize_standard(df, feature_cols)

    # Save outputs
    df.to_pickle('output/df_cleaned.pkl')
    X_scaled.to_pickle('output/X_scaled.pkl')
    pd.Series(feature_cols).to_pickle('output/feature_cols.pkl')
    
    # Save encoder map
    with open('output/encoding_map.pkl', 'wb') as f:
        pickle.dump(enc_map, f)

    print("\n[OK] Step 1 Preprocessing complete -> output/df_cleaned.pkl, output/X_scaled.pkl")
    return df, X_scaled, feature_cols, scaler

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    run_preprocessing()
