"""
Módulo para preprocesamiento y exploración de datos.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import DATA_DIR, VISUALIZATIONS_DIR, TEST_SIZE, RANDOM_SEED
from src.utils import ensure_dir_exists, save_figure

def load_datasets():
    """
    Carga los datasets desde archivos CSV.
    
    Returns:
        Diccionario con los dataframes cargados
    """
    df_tiempos = pd.read_csv(os.path.join(DATA_DIR, 'dataset_tiempos_viaje.csv'))
    df_congestion = pd.read_csv(os.path.join(DATA_DIR, 'dataset_congestion.csv'))
    df_rutas = pd.read_csv(os.path.join(DATA_DIR, 'dataset_rutas_optimas.csv'))
    
    return {
        'tiempos': df_tiempos,
        'congestion': df_congestion,
        'rutas': df_rutas
    }

def explore_datasets(datasets):
    """
    Explora y visualiza los datasets.
    
    Args:
        datasets: Diccionario con los dataframes
        
    Returns:
        Diccionario con estadísticas y rutas a visualizaciones
    """
    ensure_dir_exists(VISUALIZATIONS_DIR)
    results = {'stats': {}, 'visualizations': {}}
    
    # Estadísticas básicas
    for name, df in datasets.items():
        results['stats'][name] = {
            'shape': df.shape,
            'describe': df.describe().to_dict()
        }
    
    # Visualización de distribuciones
    plt.figure(figsize=(15, 5))
    
    # Distribución de tiempos de viaje
    plt.subplot(1, 3, 1)
    sns.histplot(datasets['tiempos']['tiempo_viaje'], kde=True)
    plt.title('Distribución de Tiempos de Viaje')
    plt.xlabel('Tiempo (minutos)')
    
    # Distribución de niveles de congestión
    plt.subplot(1, 3, 2)
    sns.histplot(datasets['congestion']['nivel_congestion'], kde=True)
    plt.title('Distribución de Niveles de Congestión')
    plt.xlabel('Nivel de congestión (0-1)')
    
    # Distribución de calidad de rutas
    plt.subplot(1, 3, 3)
    sns.histplot(datasets['rutas']['calidad_ruta'], kde=True)
    plt.title('Distribución de Calidad de Rutas')
    plt.xlabel('Calidad (0-10)')
    
    plt.tight_layout()
    results['visualizations']['distribuciones'] = save_figure(plt.gcf(), 'distribucion_datasets.png')
    
    # Análisis de correlación
    plt.figure(figsize=(15, 10))
    
    # Correlación en dataset de tiempos
    plt.subplot(2, 2, 1)
    corr_tiempos = datasets['tiempos'][['dia_semana', 'hora_dia', 'es_hora_pico', 'es_fin_semana', 
                              'demanda_origen', 'demanda_destino', 'tiempo_base', 'tiempo_viaje']].corr()
    sns.heatmap(corr_tiempos, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
    plt.title('Correlación - Dataset Tiempos de Viaje')
    
    # Correlación en dataset de congestión
    plt.subplot(2, 2, 2)
    corr_congestion = datasets['congestion'][['dia_semana', 'hora_dia', 'es_hora_pico', 'es_fin_semana', 
                                    'categoria_demanda', 'tiene_evento_especial', 'nivel_congestion']].corr()
    sns.heatmap(corr_congestion, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
    plt.title('Correlación - Dataset Congestión')
    
    # Correlación en dataset de rutas
    plt.subplot(2, 2, 3)
    corr_rutas = datasets['rutas'][['dia_semana', 'hora_dia', 'es_hora_pico', 'es_fin_semana',
                          'num_estaciones', 'num_transbordos', 'tiempo_total', 
                          'congestion_promedio', 'calidad_ruta']].corr()
    sns.heatmap(corr_rutas, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
    plt.title('Correlación - Dataset Rutas Óptimas')
    
    plt.tight_layout()
    results['visualizations']['correlaciones'] = save_figure(plt.gcf(), 'correlacion_datasets.png')
    
    return results

def prepare_tiempo_viaje_data(df_tiempos):
    """
    Prepara los datos para el modelo de tiempos de viaje.
    
    Args:
        df_tiempos: DataFrame con datos de tiempos
        
    Returns:
        Diccionario con datos preparados y transformadores
    """
    # Separar características y objetivo
    X_tiempos = df_tiempos.drop(['tiempo_viaje'], axis=1)
    y_tiempos = df_tiempos['tiempo_viaje']
    
    # Identificar columnas categóricas y numéricas
    cat_cols_tiempos = ['origen', 'destino', 'clima']
    num_cols_tiempos = ['dia_semana', 'hora_dia', 'es_hora_pico', 'es_fin_semana', 
                       'demanda_origen', 'demanda_destino', 'tiempo_base']
    
    # Crear transformadores para preprocesamiento
    preprocessor_tiempos = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols_tiempos),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_tiempos)
        ])
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_tiempos, y_tiempos, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor_tiempos,
        'cat_cols': cat_cols_tiempos,
        'num_cols': num_cols_tiempos
    }

def prepare_congestion_data(df_congestion):
    """
    Prepara los datos para el modelo de congestión.
    
    Args:
        df_congestion: DataFrame con datos de congestión
        
    Returns:
        Diccionario con datos preparados y transformadores
    """
    # Separar características y objetivo
    X_congestion = df_congestion.drop(['nivel_congestion'], axis=1)
    y_congestion = df_congestion['nivel_congestion']
    
    # Identificar columnas categóricas y numéricas
    cat_cols_congestion = ['estacion']
    num_cols_congestion = ['dia_semana', 'hora_dia', 'es_hora_pico', 'es_fin_semana', 
                          'categoria_demanda', 'tiene_evento_especial']
    
    # Crear transformadores para preprocesamiento
    preprocessor_congestion = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols_congestion),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_congestion)
        ])
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_congestion, y_congestion, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor_congestion,
        'cat_cols': cat_cols_congestion,
        'num_cols': num_cols_congestion
    }

def prepare_rutas_data(df_rutas):
    """
    Prepara los datos para el modelo de calidad de rutas.
    
    Args:
        df_rutas: DataFrame con datos de rutas
        
    Returns:
        Diccionario con datos preparados y transformadores
    """
    # Seleccionar características relevantes
    features = ['dia_semana', 'hora_dia', 'es_hora_pico', 'es_fin_semana',
               'num_estaciones', 'num_transbordos', 'tiempo_total', 
               'congestion_promedio', 'num_alternativas', 'num_estaciones_alternativas']
    
    X_rutas = df_rutas[features]
    y_rutas = df_rutas['calidad_ruta']
    
    # Aplicar escalado
    scaler_rutas = StandardScaler()
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_rutas, y_rutas, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler_rutas,
        'features': features
    }