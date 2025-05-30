from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Глобальные переменные
ordinal_encoder = None
categorical_cols = []

def fit_encode_categorical_features(X):
    global ordinal_encoder, categorical_cols
    X = X.copy()
    
    # Определяем категориальные колонки
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype).startswith('category')]
    
    if not categorical_cols:
        ordinal_encoder = None
        return X
    
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat = X[categorical_cols].astype(str)
    X_encoded = ordinal_encoder.fit_transform(X_cat)
    
    X.loc[:, categorical_cols] = X_encoded
    return X

def transform_categorical_features(X):
    global ordinal_encoder, categorical_cols
    X = X.copy()
    if ordinal_encoder is None or not categorical_cols:
        return X
    X_cat = X[categorical_cols].astype(str)
    X_encoded = ordinal_encoder.transform(X_cat)
    X.loc[:, categorical_cols] = X_encoded
    return X

def train_models(X_train, y_train, selected_models):
    # Кодируем X_train
    X_train_encoded = fit_encode_categorical_features(X_train)

    models = {}
    for model_code in selected_models:
        if model_code == 'rf':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_encoded, y_train)
            models['Random Forest'] = rf
        elif model_code == 'gb':
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X_train_encoded, y_train)
            models['Gradient Boosting'] = gb
        elif model_code == 'nn':
            nn = MLPClassifier(max_iter=300, random_state=42)
            nn.fit(X_train_encoded, y_train)
            models['Neural Network'] = nn
        elif model_code == 'svm':
            svm = SVC(probability=True, random_state=42)
            svm.fit(X_train_encoded, y_train)
            models['Support Vector Machine'] = svm

    return models

def transform_test_data(X_test):
    # Применяем ту же трансформацию к тестовым данным
    return transform_categorical_features(X_test)
