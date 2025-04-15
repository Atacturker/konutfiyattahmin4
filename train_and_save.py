# train_and_save.py
import joblib
import pandas as pd
from utils import load_data, preprocess_data
from models import train_models

# Veriyi yükle
raw_df = load_data('HouseData2.xlsx')
df_processed, model_columns = preprocess_data(raw_df.copy())
models, scores = train_models(df_processed)

# Kaydetmek için bir sözlük oluşturuyoruz
saved_model = {
    "models": models,
    "scores": scores,
    "model_columns": model_columns
}

# Modelle kaydetme işlemi (joblib kullanıyoruz)
joblib.dump(saved_model, 'trained_models.pkl')
print("Model başarıyla kaydedildi.")
