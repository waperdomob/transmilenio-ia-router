"""
Módulo para evaluar y visualizar los resultados de los modelos.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import VISUALIZATIONS_DIR
from src.utils import save_figure

def evaluate_tiempo_viaje_model(model, data):
    """
    Evalúa el modelo de tiempos de viaje.
    
    Args:
        model: Modelo entrenado (pipeline)
        data: Diccionario con datos preparados
        
    Returns:
        Diccionario con métricas y ruta a visualización
    """
    print("Evaluando modelo de tiempos de viaje...")
    
    # Predecir
    y_pred = model.predict(data['X_test'])
    
    # Calcular métricas
    mse = mean_squared_error(data['y_test'], y_pred)
    mae = mean_absolute_error(data['y_test'], y_pred)
    r2 = r2_score(data['y_test'], y_pred)
    
    print(f"Resultados - Modelo de Tiempos de Viaje:")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")
    print(f"Error absoluto medio (MAE): {mae:.4f}")
    print(f"Coeficiente de determinación (R²): {r2:.4f}")
    
    # Visualizar resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(data['y_test'], y_pred, alpha=0.5)
    plt.plot([data['y_test'].min(), data['y_test'].max()], 
             [data['y_test'].min(), data['y_test'].max()], 
             'r--', lw=2)
    plt.title('Predicción vs Real - Tiempos de Viaje')
    plt.xlabel('Tiempo real (minutos)')
    plt.ylabel('Tiempo predicho (minutos)')
    plt.grid(True)
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'prediccion_tiempos.png')
    
    return {
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2
        },
        'visualization_path': vis_path
    }

def evaluate_congestion_model(model, preprocessor, data):
    """
    Evalúa el modelo de congestión.
    
    Args:
        model: Modelo entrenado (red neuronal)
        preprocessor: Preprocesador ajustado
        data: Diccionario con datos preparados
        
    Returns:
        Diccionario con métricas y ruta a visualización
    """
    print("Evaluando modelo de congestión...")
    
    # Preprocesar datos de prueba
    X_test_prep = preprocessor.transform(data['X_test'])
    
    # Predecir
    y_pred = model.predict(X_test_prep).flatten()
    
    # Calcular métricas
    mse = mean_squared_error(data['y_test'], y_pred)
    mae = mean_absolute_error(data['y_test'], y_pred)
    r2 = r2_score(data['y_test'], y_pred)
    
    print(f"Resultados - Modelo de Congestión:")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")
    print(f"Error absoluto medio (MAE): {mae:.4f}")
    print(f"Coeficiente de determinación (R²): {r2:.4f}")
    
    # Visualizar resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(data['y_test'], y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.title('Predicción vs Real - Congestión')
    plt.xlabel('Congestión real')
    plt.ylabel('Congestión predicha')
    plt.grid(True)
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'prediccion_congestion.png')
    
    return {
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2
        },
        'visualization_path': vis_path
    }

def evaluate_rutas_model(model, scaler, data):
    """
    Evalúa el modelo de calidad de rutas.
    
    Args:
        model: Modelo entrenado (Random Forest)
        scaler: Transformador para escalar datos
        data: Diccionario con datos preparados
        
    Returns:
        Diccionario con métricas y rutas a visualizaciones
    """
    print("Evaluando modelo de calidad de rutas...")
    
    # Escalar datos de prueba
    X_test_scaled = scaler.transform(data['X_test'])
    
    # Predecir
    y_pred = model.predict(X_test_scaled)
    
    # Calcular métricas
    mse = mean_squared_error(data['y_test'], y_pred)
    mae = mean_absolute_error(data['y_test'], y_pred)
    r2 = r2_score(data['y_test'], y_pred)
    
    print(f"Resultados - Modelo de Calidad de Rutas:")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")
    print(f"Error absoluto medio (MAE): {mae:.4f}")
    print(f"Coeficiente de determinación (R²): {r2:.4f}")
    
    # Visualizar importancia de características
    plt.figure(figsize=(10, 6))
    importancia = model.feature_importances_
    indices = np.argsort(importancia)[::-1]
    nombres = data['features']
    
    plt.bar(range(len(nombres)), importancia[indices])
    plt.xticks(range(len(nombres)), [nombres[i] for i in indices], rotation=45, ha='right')
    plt.title('Importancia de Características - Calidad de Rutas')
    plt.tight_layout()
    
    # Guardar visualización de importancia
    vis_importancia_path = save_figure(plt.gcf(), 'importancia_caracteristicas.png')
    
    # Visualizar predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(data['y_test'], y_pred, alpha=0.5)
    plt.plot([data['y_test'].min(), data['y_test'].max()], 
             [data['y_test'].min(), data['y_test'].max()], 
             'r--', lw=2)
    plt.title('Predicción vs Real - Calidad de Rutas')
    plt.xlabel('Calidad real')
    plt.ylabel('Calidad predicha')
    plt.grid(True)
    
    # Guardar visualización de predicciones
    vis_prediccion_path = save_figure(plt.gcf(), 'prediccion_calidad.png')
    
    return {
        'metrics': {
            'mse': mse,
            'mae': mae,
            'r2': r2
        },
        'visualization_paths': {
            'importancia': vis_importancia_path,
            'prediccion': vis_prediccion_path
        }
    }