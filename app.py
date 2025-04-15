# app.py
import streamlit as st
import pandas as pd
from utils import load_data, preprocess_data, transform_new_data
from models import train_models

def main():
    st.title("Konut Fiyat Tahmin Uygulaması")

    # Veri yükleme
    raw_df = load_data('HouseData2.xlsx')
    if raw_df is None:
        st.error("Veri yüklenirken bir hata oluştu. Lütfen dosya yolunu ve dosya içeriğini kontrol edin.")
        st.stop()

    # Kategorik sütunlar ve seçeneklerin alınması: tip gibi sütunların tümünü string'e çevirebiliriz
    categorical_fields = ['ilce', 'mahalle', 'tip', 'esya', 'odasayi', 'isitma', 'site', 'balkon']
    options = {}
    for col in categorical_fields:
        if col in raw_df.columns:
            # Tüm değerler string'e çevriliyor ve sıralı hale getiriliyor
            options[col] = sorted(raw_df[col].dropna().unique(), key=lambda x: str(x))

    # Sayısal alanlar için min, max değerleri hesaplayın (Varsayım: metrekare, binayas, binakat, banyosayi, dairekat, balkonsayi)
    numeric_fields = ['metrekare', 'binayas', 'binakat', 'banyosayi', 'dairekat', 'balkonsayi']
    num_min = {}
    num_max = {}
    num_mean = {}
    for col in numeric_fields:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            num_min[col] = float(raw_df[col].min())
            num_max[col] = float(raw_df[col].max())
            num_mean[col] = float(raw_df[col].mean())

    st.sidebar.header("Konut Özellikleri Seçimi")

    # Kategorik özellikler için seçim kutuları
    selected_inputs = {}
    for col, opts in options.items():
        selected_inputs[col] = st.sidebar.selectbox(f"{col.capitalize()} Seçiniz", opts)

    # Sayısal özellikler için sayı girişi
    for col in numeric_fields:
        if col in raw_df.columns:
            selected_inputs[col] = st.sidebar.number_input(
                f"{col.capitalize()}",
                min_value=num_min[col],
                max_value=num_max[col],
                value=num_mean[col]
            )

    # Model seçimi
    model_option = st.sidebar.selectbox("Model Seçiniz", ["Karar Ağacı", "SVR", "Yapay Sinir Ağı"])

    st.write("### Model Eğitimi ve Değerlendirme")
    st.info("Model eğitim süreci arka planda çalışıyor, lütfen bekleyiniz...")

    # Ön işleme uygulanmış veriyi ve model sütunlarını alıyoruz
    df_processed, model_columns = preprocess_data(raw_df.copy())
    models, scores = train_models(df_processed)

    # Kullanıcı girişi: ham veri DataFrame oluşturma
    # Bu veri setinde 'fiyat' yer almayacak, çünkü tahmin edilecek.
    input_data = {col: [val] for col, val in selected_inputs.items()}
    input_df = pd.DataFrame(input_data)

    # Yeni kullanıcının girdiği veriyi, eğitimde kullanılan sütunlara uyarlıyoruz
    input_transformed = transform_new_data(input_df, model_columns)

    if st.button("Tahmin Et"):
        model = models[model_option]
        prediction = model.predict(input_transformed)[0]
        st.write(f"Seçilen ({model_option}) modele göre tahmini konut fiyatı: {prediction:,.2f} TL")
        st.write("Model başarı oranı (R² skoru):", round(scores[model_option], 2))

if __name__ == '__main__':
    main()
