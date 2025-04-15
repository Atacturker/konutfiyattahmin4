# utils.py
import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Excel dosyasını yükler.
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None

def preprocess_data(df):
    """
    Veri temizleme ve ön işleme:
    - Kolon adlarını küçük harfe çekme,
    - 'fiyat' sütunundan "TL" ve virgülleri kaldırıp numeric hale getirme,
    - Uç değer filtrelemesi,
    - Sayısal sütunlardaki (metrekare, binayas, binakat, banyosayi, dairekat, balkonsayi) eksik değerler medyanla doldurulması,
    - Kategorik sütunlardaki eksik verilerin "Bilinmiyor" ile doldurulması,
    - One-hot encoding uygulanması.
    """
    # Kolon adlarını küçük harfe çekelim
    df.columns = [col.strip().lower() for col in df.columns]

    # 'fiyat' sütununu temizle (örneğin: "300,000TL" -> 300000)
    df['fiyat'] = (df['fiyat']
                   .astype(str)
                   .str.replace('TL', '', regex=False)
                   .str.replace(',', '', regex=False)
                   .str.strip()
                   .astype(float))

    # Uç değerleri filtrele (fiyat sütununa göre)
    lower_bound = df['fiyat'].quantile(0.05)
    upper_bound = df['fiyat'].quantile(0.95)
    df = df[(df['fiyat'] >= lower_bound) & (df['fiyat'] <= upper_bound)]

    # Sayısal sütunlar: medyan ile dolduralım
    numeric_columns = ['metrekare', 'binayas', 'binakat', 'banyosayi', 'dairekat', 'balkonsayi']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Kategorik sütunlar: eksik değerleri "Bilinmiyor" ile dolduralım
    categorical_cols = ['ilce', 'mahalle', 'tip', 'esya', 'odasayi', 'isitma', 'site', 'balkon']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Bilinmiyor')

    # Kategorik sütunları one-hot encoding ile dönüştürelim
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

    # Eğitimde kullanılacak özellik sütunları
    model_columns = df.drop('fiyat', axis=1).columns

    return df, model_columns

def transform_new_data(input_df, model_columns):
    """
    Kullanıcıdan gelen ham veri için aynı one-hot encoding dönüşümünü uygular.
    Eğitimde kullanılan sütunlara göre yeniden indeksleyip eksik değerleri 0 ile doldurur.
    """
    # Kategorik sütunlar, eğitimde kullanılanları burada da aynı şekilde one-hot işlemi yapılmalıdır.
    categorical_cols = ['ilce', 'mahalle', 'tip', 'esya', 'odasayi', 'isitma', 'site', 'balkon']
    df_processed = input_df.copy()
    
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Bilinmiyor')
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
    
    # Eğitimde kullanılan sütunlarla aynı düzeni yakalayalım
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)
    
    return df_processed
