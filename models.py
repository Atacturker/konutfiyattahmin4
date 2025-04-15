# models.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np

def train_models(df):
    """
    df: ön işlemden geçmiş DataFrame (one-hot encoding uygulanmış) ve hedef sütun 'fiyat' içermeli.
    Eğitim seti oluşturulup üç model eğitilmekte, her modelin skorları hesaplanmakta.
    """
    X = df.drop('fiyat', axis=1)
    y = df['fiyat']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}
    scores = {}

    # Karar Ağacı
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    score_dt = r2_score(y_test, y_pred_dt)
    models['Karar Ağacı'] = dt
    scores['Karar Ağacı'] = score_dt

    # Destek Vektör Regresyonu (SVR)
    svr = SVR()
    param_grid_svr = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
    }
    grid_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='r2')
    grid_svr.fit(X_train, y_train)
    best_svr = grid_svr.best_estimator_
    y_pred_svr = best_svr.predict(X_test)
    score_svr = r2_score(y_test, y_pred_svr)
    models['SVR'] = best_svr
    scores['SVR'] = score_svr

    # Yapay Sinir Ağı (MLPRegressor) - 40, 70 ve 100 nöronlu deneyler
    best_ann_score = -np.inf
    best_ann_model = None
    for neurons in [40, 70, 100]:
        ann = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=1000, random_state=42)
        ann.fit(X_train, y_train)
        y_pred_ann = ann.predict(X_test)
        score_ann = r2_score(y_test, y_pred_ann)
        if score_ann > best_ann_score:
            best_ann_score = score_ann
            best_ann_model = ann

    models['Yapay Sinir Ağı'] = best_ann_model
    scores['Yapay Sinir Ağı'] = best_ann_score

    return models, scores
