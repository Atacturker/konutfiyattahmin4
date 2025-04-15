import joblib
import pandas as pd
from utils import load_data, preprocess_data
from models import train_models

def main():
    # Excel dosyasından veriyi yükleyin
    raw_df = load_data('HouseData2.xlsx')
    if raw_df is None:
        print("Veri yüklenirken bir hata oluştu. Lütfen dosya yolunu ve dosya içeriğini kontrol edin.")
        return

    # Veriyi ön işleme tabi tutun
    df_processed, model_columns = preprocess_data(raw_df.copy())
    
    # Modelleri eğitin
    models, scores = train_models(df_processed)
    
    # Eğitilmiş modeli, skorları ve eğitimde kullanılan sütunları kaydet
    saved_model = {
        "models": models,
        "scores": scores,
        "model_columns": model_columns
    }
    
    # Joblib kullanarak modeli dosyaya kaydedin
    joblib.dump(saved_model, 'trained_models.pkl')
    print("Model başarıyla kaydedildi. 'trained_models.pkl' dosyası oluşturuldu.")

if __name__ == "__main__":
    main()
