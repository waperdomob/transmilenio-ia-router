"""
Módulo para entrenar los modelos de aprendizaje automático.
"""
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from config import MODELS_DIR, EPOCHS, BATCH_SIZE
from src.utils import ensure_dir_exists

def train_tiempo_viaje_model(data):
    """
    Entrena un modelo para predecir tiempos de viaje.
    
    Args:
        data: Diccionario con datos preparados
        
    Returns:
        Modelo entrenado (pipeline)
    """
    print("Entrenando modelo de predicción de tiempos de viaje...")
    
    # Pipeline con preprocesamiento y modelo
    pipeline = Pipeline([
        ('preprocessor', data['preprocessor']),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            random_state=42
        ))
    ])
    
    # Entrenar modelo
    pipeline.fit(data['X_train'], data['y_train'])
    
    # Guardar modelo
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'modelo_tiempos.joblib')
    joblib.dump(pipeline, model_path)
    
    print(f"Modelo de tiempos guardado en {model_path}")
    
    return pipeline

def train_congestion_model(data):
    """
    Entrena un modelo para predecir niveles de congestión.
    
    Args:
        data: Diccionario con datos preparados
        
    Returns:
        Tupla con (modelo, historial de entrenamiento)
    """
    print("Entrenando modelo de predicción de congestión...")
    
    # Preprocesar datos
    X_train_prep = data['preprocessor'].fit_transform(data['X_train'])
    
    # Crear modelo de red neuronal
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_prep.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train_prep, data['y_train'],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    # Guardar modelo y preprocesador
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'modelo_congestion.keras')
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor_congestion.joblib')
    
    model.save(model_path)
    joblib.dump(data['preprocessor'], preprocessor_path)
    
    print(f"Modelo de congestión guardado en {model_path}")
    print(f"Preprocesador guardado en {preprocessor_path}")
    
    return model, history

def train_rutas_model(data):
    """
    Entrena un modelo para predecir calidad de rutas.
    
    Args:
        data: Diccionario con datos preparados
        
    Returns:
        Modelo entrenado
    """
    print("Entrenando modelo de predicción de calidad de rutas...")
    
    # Escalar datos
    X_train_scaled = data['scaler'].fit_transform(data['X_train'])
    
    # Crear y entrenar modelo
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, data['y_train'])
    
    # Guardar modelo y scaler
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'modelo_rutas.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler_rutas.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(data['scaler'], scaler_path)
    
    print(f"Modelo de rutas guardado en {model_path}")
    print(f"Scaler guardado en {scaler_path}")
    
    return model