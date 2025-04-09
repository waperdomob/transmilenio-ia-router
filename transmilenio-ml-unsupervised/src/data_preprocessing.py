"""
Módulo para preprocesamiento de datos con enfoque no supervisado.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from config import DATA_DIR, VISUALIZATIONS_DIR, N_COMPONENTS_PCA
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
    
    # Visualización de distribuciones y patrones
    plt.figure(figsize=(15, 10))
    
    # Análisis de patrones por hora del día
    plt.subplot(2, 2, 1)
    hourly_congestion = datasets['congestion'].groupby('hora_dia')['nivel_congestion'].mean()
    sns.lineplot(x=hourly_congestion.index, y=hourly_congestion.values)
    plt.title('Patrón de congestión por hora del día')
    plt.xlabel('Hora')
    plt.ylabel('Nivel promedio de congestión')
    
    # Análisis de patrones por día de la semana
    plt.subplot(2, 2, 2)
    day_congestion = datasets['congestion'].groupby('dia_semana')['nivel_congestion'].mean()
    sns.barplot(x=day_congestion.index, y=day_congestion.values)
    plt.title('Congestión por día de la semana')
    plt.xlabel('Día (0=Lunes, 6=Domingo)')
    plt.ylabel('Nivel promedio de congestión')
    
    # Tiempos de viaje por categoría de demanda de origen
    plt.subplot(2, 2, 3)
    sns.boxplot(x='demanda_origen', y='tiempo_viaje', data=datasets['tiempos'])
    plt.title('Tiempos de viaje por categoría de demanda')
    plt.xlabel('Categoría de demanda (0=Baja, 2=Alta)')
    plt.ylabel('Tiempo de viaje (minutos)')
    
    # Calidad de ruta vs. congestión y transbordos
    plt.subplot(2, 2, 4)
    plt.scatter(
        datasets['rutas']['congestion_promedio'], 
        datasets['rutas']['num_transbordos'],
        c=datasets['rutas']['calidad_ruta'], 
        cmap='viridis', 
        alpha=0.6
    )
    plt.colorbar(label='Calidad de ruta')
    plt.title('Relación entre congestión, transbordos y calidad')
    plt.xlabel('Congestión promedio')
    plt.ylabel('Número de transbordos')
    
    plt.tight_layout()
    results['visualizations']['patrones'] = save_figure(plt.gcf(), 'patrones_datos.png')
    
    return results

def prepare_station_clustering_data(df_congestion):
    """
    Prepara los datos para clustering de estaciones.
    
    Args:
        df_congestion: DataFrame con datos de congestión
        
    Returns:
        Diccionario con datos preparados para clustering
    """
    # Para cada estación, calcular métricas agregadas
    station_features = df_congestion.groupby('estacion').agg({
        'nivel_congestion': ['mean', 'std', 'max'],
        'es_hora_pico': 'mean',  # Proporción de muestras en hora pico
        'categoria_demanda': 'first'  # Categoría de demanda conocida
    })
    
    # Aplanar los índices multinivel
    station_features.columns = ['_'.join(col).strip() for col in station_features.columns.values]
    station_features = station_features.reset_index()
    
    # Crear métricas adicionales
    # Diferencia entre congestión en hora pico y no pico
    pico_congestion = df_congestion[df_congestion['es_hora_pico'] == 1].groupby('estacion')['nivel_congestion'].mean()
    no_pico_congestion = df_congestion[df_congestion['es_hora_pico'] == 0].groupby('estacion')['nivel_congestion'].mean()
    
    station_features['congestion_diff_pico'] = station_features['estacion'].map(
        lambda x: pico_congestion.get(x, 0) - no_pico_congestion.get(x, 0)
    )
    
    # Diferencia entre congestión fin de semana y semana
    weekend_congestion = df_congestion[df_congestion['es_fin_semana'] == 1].groupby('estacion')['nivel_congestion'].mean()
    weekday_congestion = df_congestion[df_congestion['es_fin_semana'] == 0].groupby('estacion')['nivel_congestion'].mean()
    
    station_features['congestion_diff_weekend'] = station_features['estacion'].map(
        lambda x: weekend_congestion.get(x, 0) - weekday_congestion.get(x, 0)
    )
    
    # Preparar datos para clustering
    X_stations = station_features.drop(['estacion', 'categoria_demanda_first'], axis=1)
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stations)
    
    return {
        'X': X_scaled,
        'feature_names': X_stations.columns.tolist(),
        'station_names': station_features['estacion'].values,
        'original_features': station_features
    }

def prepare_segment_clustering_data(df_tiempos):
    """
    Prepara los datos para clustering de segmentos de ruta.
    
    Args:
        df_tiempos: DataFrame con datos de tiempos de viaje
        
    Returns:
        Diccionario con datos preparados para clustering
    """
    # Identificar segmentos únicos (par origen-destino)
    segments = df_tiempos[['origen', 'destino']].drop_duplicates()
    segments['segmento'] = segments['origen'] + ' -> ' + segments['destino']
    
    # Para cada segmento, calcular métricas agregadas
    segment_features = df_tiempos.groupby(['origen', 'destino']).agg({
        'tiempo_viaje': ['mean', 'std', 'max', 'min'],
        'tiempo_base': 'first',
        'demanda_origen': 'first',
        'demanda_destino': 'first'
    })
    
    # Aplanar los índices multinivel
    segment_features.columns = ['_'.join(col).strip() for col in segment_features.columns.values]
    segment_features = segment_features.reset_index()
    
    # Calcular ratio de tiempo real vs tiempo base
    segment_features['tiempo_ratio'] = segment_features['tiempo_viaje_mean'] / segment_features['tiempo_base_first']
    
    # Calcular variabilidad del tiempo
    segment_features['tiempo_variabilidad'] = segment_features['tiempo_viaje_std'] / segment_features['tiempo_viaje_mean']
    
    # Crear segmento combinado
    segment_features['segmento'] = segment_features['origen'] + ' -> ' + segment_features['destino']
    
    # Preparar datos para clustering
    X_segments = segment_features.drop(['origen', 'destino', 'segmento', 'tiempo_base_first'], axis=1)
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_segments)
    
    return {
        'X': X_scaled,
        'feature_names': X_segments.columns.tolist(),
        'segment_names': segment_features['segmento'].values,
        'original_features': segment_features
    }

def prepare_anomaly_detection_data(df_tiempos, df_congestion):
    """
    Prepara los datos para detección de anomalías.
    
    Args:
        df_tiempos: DataFrame con datos de tiempos de viaje
        df_congestion: DataFrame con datos de congestión
        
    Returns:
        Diccionario con datos preparados para detección de anomalías
    """
    # Para tiempos de viaje
    # Calcular ratio de tiempo real vs tiempo base
    df_tiempos['tiempo_ratio'] = df_tiempos['tiempo_viaje'] / df_tiempos['tiempo_base']
    
    # Seleccionar características para detección de anomalías
    X_tiempos = df_tiempos[['tiempo_ratio', 'es_hora_pico', 'es_fin_semana', 'demanda_origen', 'demanda_destino']]
    
    # Para congestión
    # Seleccionar características para detección de anomalías
    X_congestion = df_congestion[['nivel_congestion', 'es_hora_pico', 'es_fin_semana', 'categoria_demanda', 'tiene_evento_especial']]
    
    # Escalar características
    scaler_tiempos = StandardScaler()
    X_tiempos_scaled = scaler_tiempos.fit_transform(X_tiempos)
    
    scaler_congestion = StandardScaler()
    X_congestion_scaled = scaler_congestion.fit_transform(X_congestion)
    
    return {
        'tiempos': {
            'X': X_tiempos_scaled,
            'feature_names': X_tiempos.columns.tolist(),
            'original_df': df_tiempos,
            'scaler': scaler_tiempos
        },
        'congestion': {
            'X': X_congestion_scaled,
            'feature_names': X_congestion.columns.tolist(),
            'original_df': df_congestion,
            'scaler': scaler_congestion
        }
    }

def prepare_pca_data(df_rutas):
    """
    Prepara los datos para reducción de dimensionalidad con PCA.
    
    Args:
        df_rutas: DataFrame con datos de rutas
        
    Returns:
        Diccionario con datos preparados para PCA
    """

    
    # Seleccionar características numéricas relevantes
    features = [
        'num_estaciones', 'num_transbordos', 'tiempo_total', 
        'congestion_promedio', 'num_alternativas', 'num_estaciones_alternativas'
    ]
    
    X = df_rutas[features]
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA(n_components=N_COMPONENTS_PCA)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calcular varianza explicada
    explained_variance = pca.explained_variance_ratio_
    
    return {
        'X_original': X_scaled,
        'X_pca': X_pca,
        'feature_names': features,
        'explained_variance': explained_variance,
        'pca': pca,
        'scaler': scaler,
        'original_df': df_rutas
    }