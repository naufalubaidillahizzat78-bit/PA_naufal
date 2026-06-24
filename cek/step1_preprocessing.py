"""
STEP 1: PREPROCESSING DATA
- Pembersihan data (missing value & duplikasi)
- Normalisasi data numerik
- Encoding data kategorik
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_clean(filepath):
    df = pd.read_excel(filepath)
    print(f"[1] Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    # ── Missing values ──
    missing_before = df.isnull().sum().sum()
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print(f"[2] Missing values fixed: {missing_before} → {df.isnull().sum().sum()}")

    # ── Duplicates ──
    dup = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[3] Duplicates removed: {dup}")

    return df

def encode_categoricals(df):
    # JK already numeric (1/2), encode Prodi & Asal
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cat_cols = ['Prodi', 'Asal Kab/Kota']
    encoding_map = {}
    for col in cat_cols:
        if col in df.columns:
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"[4] Encoded: {cat_cols}")
    return df, encoding_map

def select_features(df):
    """Select core academic features for clustering."""
    # IPS per semester, rata-rata IPS, rata-rata absensi, JK
    feature_cols = [
        'nilai ips', 'IPS', 'IPS.1', 'IPS.2',          # IPS sem 1-4
        'Rata-Rata IPS',                                  # overall IPS
        'Rata-Rata Absen Mahasiswa', 'ABSENSI RATA RATA',
        'ABSENSI RATA_RATA', 'ABSENSI RATA RATA.1', 'ABSENSI RATA RATA.2',
        'JK', 'Angkatan Tahun'
    ]
    # Keep only existing
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"[5] Features selected ({len(feature_cols)}): {feature_cols}")
    return feature_cols

def normalize(df, feature_cols):
    scaler = MinMaxScaler()
    X = df[feature_cols].copy()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    print(f"[6] Normalization done. Shape: {X_scaled.shape}")
    return X_scaled, scaler

def run_preprocessing(filepath='../data_base_cleaned.xlsx'):
    df = load_and_clean(filepath)
    df, enc_map = encode_categoricals(df)
    feature_cols = select_features(df)
    X_scaled, scaler = normalize(df, feature_cols)

    # Save outputs
    df.to_pickle('output/df_cleaned.pkl')
    X_scaled.to_pickle('output/X_scaled.pkl')
    pd.Series(feature_cols).to_pickle('output/feature_cols.pkl')

    print("\n✅ Step 1 complete → output/df_cleaned.pkl, output/X_scaled.pkl")
    return df, X_scaled, feature_cols, scaler

if __name__ == '__main__':
    import os; os.makedirs('output', exist_ok=True)
    run_preprocessing('/mnt/user-data/uploads/data_base_cleaned.xlsx')
