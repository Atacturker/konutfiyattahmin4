import pandas as pd
import numpy as np

def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None

def preprocess_data(df):
    # Aykırı değerlerin filtrelenmesi
    lower_bound = df['Fiyat'].quantile(0.05)
    upper_bound = df['Fiyat'].quantile(0.95)
    df = df[(df['Fiyat'] >= lower_bound) & (df['Fiyat'] <= upper_bound)]

    # Balkon bilgisi varsa, eksik değerleri kaldır
    if 'Balkon' in df.columns:
        df = df.dropna(subset=['Balkon'])

    # Kategorik sütunları one-hot encoding ile dönüştür
    categorical_cols = ['İlçe', 'Mahalle', 'OdaSayısı']
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

    return df
