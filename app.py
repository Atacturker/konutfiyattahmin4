import streamlit as st
from utils import load_data, preprocess_data
from models import train_models

def main():
    st.title("Konut Fiyat Tahmin Uygulaması")

    # Veri yükleme
    df = load_data('HouseData2.xlsx')
if df is None:
    st.error("Veri yüklenirken bir hata oluştu. Lütfen dosya yolunu ve dosya içeriğini kontrol edin.")
    st.stop()
df = preprocess_data(df)


    # Model eğitimi
    models, scores = train_models(df)

    # Kullanıcı arayüzü
    st.sidebar.header("Konut Özellikleri Seçimi")
    ilce_options = ['Kadıköy', 'Beşiktaş', 'Üsküdar']  # Örnek ilçe isimleri
    mahalle_options = ['Moda', 'Levent', 'Maslak']       # Örnek mahalle isimleri
    oda_options = ['2+1', '3+1', '4+1']                   # Örnek oda sayıları

    selected_ilce = st.sidebar.selectbox("İlçe Seçiniz", ilce_options)
    selected_mahalle = st.sidebar.selectbox("Mahalle Seçiniz", mahalle_options)
    selected_oda = st.sidebar.selectbox("Oda Sayısı Seçiniz", oda_options)

    # Model seçimi
    model_option = st.sidebar.selectbox("Model Seçiniz", list(models.keys()))

    if st.button("Tahmin Et"):
        model = models[model_option]
        # Özellikleri one-hot encoded forma dönüştür
        input_data = {}
        for col in df.columns:
            if col.startswith('ilce'):
                if col == f"ilce_{selected_ilce}":
                    input_data[col] = 1
                else:
                    input_data[col] = 0
            elif col.startswith('mahalle'):
                if col == f"mahalle_{selected_mahalle}":
                    input_data[col] = 1
                else:
                    input_data[col] = 0
            elif col.startswith('odasayi'):
                if col == f"odasayi_{selected_oda}":
                    input_data[col] = 1
                else:
                    input_data[col] = 0
            else:
                input_data[col] = 0  # Diğer sütunlar için varsayılan değer 0

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f"Seçilen ({model_option}) modele göre tahmini konut fiyatı: {prediction:.2f} TL")
        st.write("Model başarı oranı (R² skoru):", round(scores[model_option], 2))

if __name__ == '__main__':
    main()
