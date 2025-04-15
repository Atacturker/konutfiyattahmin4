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
    - Balkon sütunu varsa eksik değerleri kaldırma,
    - Belirlenen kategorik sütunları one-hot encoding’e çevirme.
    """
    # Kolon adlarını küçük harfe çekelim
    df.columns = [col.strip().lower() for col in df.columns]

    # 'fiyat' sütununu temizle (örneğin: "300,000TL" -> 300000)
    df['fiyat'] = df['fiyat'].astype(str)\
                        .str.replace('TL', '', regex=False)\
                        .str.replace(',', '', regex=False)\
                        .str.strip()\
                        .astype(float)

    # Uç değerleri filtrele (fiyat sütununa göre)
    lower_bound = df['fiyat'].quantile(0.05)
    upper_bound = df['fiyat'].quantile(0.95)
    df = df[(df['fiyat'] >= lower_bound) & (df['fiyat'] <= upper_bound)]

    # Balkon bilgisi varsa; boş olanları kaldırıyoruz
    if 'balkon' in df.columns:
        df = df.dropna(subset=['balkon'])

    # Kategorik sütunlar (gereksinime göre güncelleyebilirsiniz)
    categorical_cols = ['ilce', 'mahalle', 'tip', 'esya', 'odasayi', 'isitma', 'site', 'balkon']
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
    Kullanıcıdan gelen ham veri (pandas DataFrame formatında) için aynı one-hot encoding dönüşümünü uygular.
    Mevcut sütunlara göre yeniden indeksleyerek eksik kolonları 0 olarak doldurur.
    Not: Bu fonksiyon, eğitimde hangi kolonların kullanıldığını bilen model_columns parametresi ile çalışır.
    """
    # Ham veride, one-hot dönüşümü uygulanması gereken kategorik sütunlar
    # (Eğitim öncesinde kullanılan kategorik kolonların adlarını burada belirtmeliyiz)
    categorical_cols = ['ilce', 'mahalle', 'tip', 'esya', 'odasayi', 'isitma', 'site', 'balkon']
    
    # Sayısal kolonlar varsayım olarak doğrudan alınıyor: 
    # metrekare, binayas, binakat, banyosayi, dairekat, balkonsayi
    df_processed = input_df.copy()
    
    # Uygun sütunlarda one-hot encoding yap
    for col in categorical_cols:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
    
    # Eğitimde kullanılan sütunlarla aynı düzeni yakalamak için reindex
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)
    
    return df_processed
