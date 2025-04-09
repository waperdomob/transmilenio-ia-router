"""
Módulo para implementar modelos de aprendizaje no supervisado.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from config import (
    MODELS_DIR, VISUALIZATIONS_DIR, N_CLUSTERS_STATIONS, 
    N_CLUSTERS_SEGMENTS, ANOMALY_CONTAMINATION, RANDOM_SEED
)
from src.utils import ensure_dir_exists, save_figure

def cluster_stations(data):
    """
    Aplica clustering a las estaciones usando K-Means.
    
    Args:
        data: Diccionario con datos preparados
        
    Returns:
        Diccionario con resultados del clustering
    """
    print("Aplicando clustering a estaciones...")
    
    # Aplicar K-Means
    kmeans = KMeans(
        n_clusters=N_CLUSTERS_STATIONS,
        random_state=RANDOM_SEED,
        n_init=10
    )
    clusters = kmeans.fit_predict(data['X'])
    
    # Guardar modelo
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'kmeans_stations.joblib')
    joblib.dump(kmeans, model_path)
    
    # Preparar resultados
    result_df = data['original_features'].copy()
    result_df['cluster'] = clusters
    
    # Visualizar clusters
    plt.figure(figsize=(12, 8))
    
    # Reducir dimensionalidad para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data['X'])
    
    # Graficar clusters en espacio PCA
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.8)
    plt.title(f'Clusters de estaciones (K-Means, k={N_CLUSTERS_STATIONS})')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    
    # Características promedio por cluster
    plt.subplot(1, 2, 2)
    numeric_columns = result_df.select_dtypes(include=['number']).columns
    cluster_means = result_df.groupby('cluster')[numeric_columns].mean()
    sns.heatmap(
        cluster_means[['nivel_congestion_mean', 'nivel_congestion_max', 'es_hora_pico_mean', 
                       'congestion_diff_pico', 'congestion_diff_weekend']],
        annot=True, cmap='YlGnBu', fmt='.2f'
    )
    plt.title('Características promedio por cluster')
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'clusters_estaciones.png')
    
    # Exportar mapeo de estaciones a clusters
    station_clusters = {
        station: cluster for station, cluster in zip(data['station_names'], clusters)
    }
    
    return {
        'clusters': clusters,
        'model': kmeans,
        'station_clusters': station_clusters,
        'result_df': result_df,
        'visualization_path': vis_path
    }

def cluster_segments(data):
    """
    Aplica clustering a segmentos de ruta usando DBSCAN.
    
    Args:
        data: Diccionario con datos preparados
        
    Returns:
        Diccionario con resultados del clustering
    """
    print("Aplicando clustering a segmentos de ruta...")
    
    # Aplicar DBSCAN para clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(data['X'])
    
    # Para visualización más clara, convertir -1 (ruido) a un número más alto
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    clusters_viz = np.array(clusters)
    if -1 in clusters:
        clusters_viz[clusters_viz == -1] = n_clusters
    
    # Guardar modelo
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'dbscan_segments.joblib')
    joblib.dump(dbscan, model_path)
    
    # Preparar resultados
    result_df = data['original_features'].copy()
    result_df['cluster'] = clusters
    
    # Visualizar clusters
    plt.figure(figsize=(12, 8))
    
    # Usar t-SNE para visualizar en 2D
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    X_tsne = tsne.fit_transform(data['X'])
    
    # Graficar clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters_viz, cmap='tab10', alpha=0.8)
    plt.title(f'Clusters de segmentos (DBSCAN, clusters={n_clusters})')
    plt.xlabel('t-SNE Dimensión 1')
    plt.ylabel('t-SNE Dimensión 2')
    
    # Características por cluster
    plt.subplot(1, 2, 2)
    valid_clusters = result_df[result_df['cluster'] != -1]
    if not valid_clusters.empty:
        cluster_means = valid_clusters.groupby('cluster').mean()
        sns.heatmap(
            cluster_means[['tiempo_viaje_mean', 'tiempo_viaje_std', 'tiempo_ratio', 'tiempo_variabilidad']],
            annot=True, cmap='YlGnBu', fmt='.2f'
        )
        plt.title('Características promedio por cluster')
    else:
        plt.text(0.5, 0.5, "No hay clusters válidos para mostrar", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'clusters_segmentos.png')
    
    # Exportar mapeo de segmentos a clusters
    segment_clusters = {
        segment: cluster for segment, cluster in zip(data['segment_names'], clusters)
    }
    
    return {
        'clusters': clusters,
        'n_clusters': n_clusters,
        'model': dbscan,
        'segment_clusters': segment_clusters,
        'result_df': result_df,
        'visualization_path': vis_path
    }

def detect_anomalies(data):
    """
    Aplica detección de anomalías a tiempos de viaje y congestión.
    
    Args:
        data: Diccionario con datos preparados
        
    Returns:
        Diccionario con resultados de la detección de anomalías
    """
    print("Aplicando detección de anomalías...")
    
    results = {}
    
    # Para tiempos de viaje
    print("Procesando anomalías en tiempos de viaje...")
    
    # Usar Isolation Forest para detección de anomalías
    iso_forest = IsolationForest(
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_SEED
    )
    
    # Entrenar y predecir
    anomalies_tiempos = iso_forest.fit_predict(data['tiempos']['X'])
    
    # Convertir a formato binario (1: normal, -1: anomalía)
    anomalies_tiempos_binary = (anomalies_tiempos == -1).astype(int)
    
    # Guardar modelo
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'isoforest_tiempos.joblib')
    joblib.dump(iso_forest, model_path)
    
    # Preparar resultados
    result_df_tiempos = data['tiempos']['original_df'].copy()
    result_df_tiempos['es_anomalia'] = anomalies_tiempos_binary
    
    # Visualizar anomalías en tiempos de viaje
    plt.figure(figsize=(12, 6))
    
    # Gráfico de dispersión de anomalías
    plt.subplot(1, 2, 1)
    plt.scatter(
        result_df_tiempos[result_df_tiempos['es_anomalia'] == 0]['tiempo_base'],
        result_df_tiempos[result_df_tiempos['es_anomalia'] == 0]['tiempo_viaje'],
        color='blue', alpha=0.5, label='Normal'
    )
    plt.scatter(
        result_df_tiempos[result_df_tiempos['es_anomalia'] == 1]['tiempo_base'],
        result_df_tiempos[result_df_tiempos['es_anomalia'] == 1]['tiempo_viaje'],
        color='red', alpha=0.7, label='Anomalía'
    )
    plt.xlabel('Tiempo base (min)')
    plt.ylabel('Tiempo real (min)')
    plt.title('Anomalías en tiempos de viaje')
    plt.legend()
    
    # Distribución de tiempos anómalos vs normales
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        result_df_tiempos[result_df_tiempos['es_anomalia'] == 0]['tiempo_ratio'],
        label='Normal', color='blue'
    )
    sns.kdeplot(
        result_df_tiempos[result_df_tiempos['es_anomalia'] == 1]['tiempo_ratio'],
        label='Anomalía', color='red'
    )
    plt.xlabel('Ratio tiempo real / tiempo base')
    plt.ylabel('Densidad')
    plt.title('Distribución de ratios de tiempo')
    plt.legend()
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path_tiempos = save_figure(plt.gcf(), 'anomalias_tiempos.png')
    
    # Almacenar resultados
    results['tiempos'] = {
        'anomalies': anomalies_tiempos_binary,
        'model': iso_forest,
        'result_df': result_df_tiempos,
        'visualization_path': vis_path_tiempos
    }
    
    # Para congestión
    print("Procesando anomalías en congestión...")
    
    # Usar Local Outlier Factor para detección de anomalías
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=ANOMALY_CONTAMINATION,
        novelty=False
    )
    
    # Predecir
    anomalies_congestion = lof.fit_predict(data['congestion']['X'])
    
    # Convertir a formato binario (1: normal, -1: anomalía)
    anomalies_congestion_binary = (anomalies_congestion == -1).astype(int)
    
    # No podemos guardar el modelo LOF en modo de ajuste, pero creamos uno nuevo para predicción
    lof_novelty = LocalOutlierFactor(
        n_neighbors=20,
        contamination=ANOMALY_CONTAMINATION,
        novelty=True
    )
    lof_novelty.fit(data['congestion']['X'])
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, 'lof_congestion.joblib')
    joblib.dump(lof_novelty, model_path)
    
    # Preparar resultados
    result_df_congestion = data['congestion']['original_df'].copy()
    result_df_congestion['es_anomalia'] = anomalies_congestion_binary
    
    # Visualizar anomalías en congestión
    plt.figure(figsize=(12, 6))
    
    # Gráfico de dispersión de anomalías
    plt.subplot(1, 2, 1)
    plt.scatter(
        result_df_congestion[result_df_congestion['es_anomalia'] == 0]['hora_dia'],
        result_df_congestion[result_df_congestion['es_anomalia'] == 0]['nivel_congestion'],
        color='blue', alpha=0.5, label='Normal'
    )
    plt.scatter(
        result_df_congestion[result_df_congestion['es_anomalia'] == 1]['hora_dia'],
        result_df_congestion[result_df_congestion['es_anomalia'] == 1]['nivel_congestion'],
        color='red', alpha=0.7, label='Anomalía'
    )
    plt.xlabel('Hora del día')
    plt.ylabel('Nivel de congestión')
    plt.title('Anomalías en niveles de congestión')
    plt.legend()
    
    # Distribución de congestión por tipo de estación
    # Distribución de congestión por tipo de estación
    plt.subplot(1, 2, 2)
    sns.boxplot(
        x='categoria_demanda',
        y='nivel_congestion',
        hue='es_anomalia',
        data=result_df_congestion,
        palette={0: 'blue', 1: 'red'}
    )
    plt.xlabel('Categoría de demanda (0: Baja, 2: Alta)')
    plt.ylabel('Nivel de congestión')
    plt.title('Anomalías por categoría de estación')
    plt.legend(title='Es anomalía')
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path_congestion = save_figure(plt.gcf(), 'anomalias_congestion.png')
    
    # Almacenar resultados
    results['congestion'] = {
        'anomalies': anomalies_congestion_binary,
        'model': lof_novelty,
        'result_df': result_df_congestion,
        'visualization_path': vis_path_congestion
    }
    
    return results

def apply_pca_analysis(data):
    """
    Aplica PCA para analizar y visualizar la estructura de las rutas.
    
    Args:
        data: Diccionario con datos preparados para PCA
        
    Returns:
        Diccionario con resultados del análisis PCA
    """
    print("Aplicando análisis PCA a datos de rutas...")
    
    # PCA ya aplicado en la preparación de datos
    pca = data['pca']
    X_pca = data['X_pca']
    explained_variance = data['explained_variance']
    
    # Guardar modelo
    ensure_dir_exists(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'pca_rutas.joblib')
    joblib.dump(pca, model_path)
    
    # Visualizar resultados PCA
    plt.figure(figsize=(15, 10))
    
    # Gráfico de varianza explicada
    plt.subplot(2, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'r-')
    plt.xlabel('Componentes principales')
    plt.ylabel('Proporción de varianza explicada')
    plt.title('Varianza explicada por componente')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid(True)
    
    # Gráfico de dispersión 2D
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=data['original_df']['calidad_ruta'],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Calidad de ruta')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} var. explicada)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} var. explicada)')
    plt.title('Proyección PCA de rutas')
    plt.grid(True)
    
    # Loadings (contribuciones de características)
    plt.subplot(2, 2, 3)
    loadings = pca.components_
    
    # Crear un mapa de calor con las contribuciones
    sns.heatmap(
        loadings[:3, :],  # Primeras 3 componentes
        annot=True,
        cmap='coolwarm',
        yticklabels=[f'PC{i+1}' for i in range(3)],
        xticklabels=data['feature_names'],
        fmt='.2f'
    )
    plt.title('Contribución de características a las componentes principales')
    
    # Proyección 3D si hay suficientes componentes
    if X_pca.shape[1] >= 3:
        ax = plt.subplot(2, 2, 4, projection='3d')
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=data['original_df']['calidad_ruta'],
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='Calidad de ruta')
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
        ax.set_title('Proyección 3D PCA')
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'pca_rutas.png')
    
    # Asociar cada ruta con su representación PCA
    rutas_pca = data['original_df'].copy()
    for i in range(X_pca.shape[1]):
        rutas_pca[f'PC{i+1}'] = X_pca[:, i]
    
    return {
        'pca': pca,
        'X_pca': X_pca,
        'feature_names': data['feature_names'],
        'explained_variance': explained_variance,
        'loadings': loadings,
        'rutas_pca': rutas_pca,
        'visualization_path': vis_path
    }