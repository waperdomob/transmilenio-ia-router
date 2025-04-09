"""
Módulo para analizar los patrones descubiertos por los modelos no supervisados.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from config import VISUALIZATIONS_DIR
from src.utils import save_figure

def analyze_station_clusters(results):
    """
    Analiza los patrones encontrados en los clusters de estaciones.
    
    Args:
        results: Resultados del clustering de estaciones
        
    Returns:
        Diccionario con análisis y visualizaciones
    """
    print("Analizando patrones en clusters de estaciones...")
    
    # Obtener resultados
    df = results['result_df']
    clusters = results['clusters']
    n_clusters = len(set(clusters))
    
    # Analizar características por cluster
    analysis = {}
    
    # Características promedio por cluster
    numeric_columns = df.select_dtypes(include=['number']).columns
    cluster_means = df.groupby('cluster')[numeric_columns].mean()
    analysis['cluster_means'] = cluster_means
    
    # Conteo de estaciones por cluster
    cluster_counts = df.groupby('cluster').size()
    analysis['cluster_counts'] = cluster_counts
    
    # Interpretar los clusters
    cluster_descriptions = {}
    
    for cluster in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster]
        
        # Determinar características principales
        congestion_mean = cluster_data['nivel_congestion_mean'].mean()
        congestion_std = cluster_data['nivel_congestion_std'].mean()
        pico_diff = cluster_data['congestion_diff_pico'].mean()
        weekend_diff = cluster_data['congestion_diff_weekend'].mean()
        
        # Crear descripción
        if congestion_mean > 0.7:
            base_desc = "Alta congestión"
        elif congestion_mean > 0.4:
            base_desc = "Congestión media"
        else:
            base_desc = "Baja congestión"
            
        if pico_diff > 0.3:
            pico_desc = " con gran diferencia en hora pico"
        elif pico_diff > 0.1:
            pico_desc = " con diferencia moderada en hora pico"
        else:
            pico_desc = " con comportamiento estable en hora pico"
            
        if weekend_diff < -0.2:
            weekend_desc = " y mucho menor congestión en fin de semana"
        elif weekend_diff < -0.05:
            weekend_desc = " y menor congestión en fin de semana"
        else:
            weekend_desc = " y congestión similar en fin de semana"
            
        cluster_descriptions[cluster] = base_desc + pico_desc + weekend_desc
    
    analysis['cluster_descriptions'] = cluster_descriptions
    
    # Visualización avanzada
    plt.figure(figsize=(15, 10))
    
    # Mapa de calor de características por cluster
    plt.subplot(2, 2, 1)
    key_features = ['nivel_congestion_mean', 'nivel_congestion_max', 'nivel_congestion_std', 
                    'es_hora_pico_mean', 'congestion_diff_pico', 'congestion_diff_weekend']
    
    sns.heatmap(
        cluster_means[key_features],
        annot=True,
        cmap='YlGnBu',
        fmt='.2f'
    )
    plt.title('Características promedio por cluster')
    
    # Distribución de estaciones por cluster
    plt.subplot(2, 2, 2)
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Número de estaciones')
    plt.title('Distribución de estaciones por cluster')
    plt.xticks(range(n_clusters))
    
    # Visualizar la diferencia de congestión en hora pico vs valle
    plt.subplot(2, 2, 3)
    for cluster in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(
            cluster_data['nivel_congestion_mean'],
            cluster_data['congestion_diff_pico'],
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    plt.xlabel('Nivel de congestión promedio')
    plt.ylabel('Diferencia de congestión en hora pico')
    plt.title('Patrones de congestión por cluster')
    plt.legend()
    plt.grid(True)
    
    # Comparación de congestión entre día de semana y fin de semana
    plt.subplot(2, 2, 4)
    for cluster in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(
            cluster_data['congestion_diff_pico'],
            cluster_data['congestion_diff_weekend'],
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    plt.xlabel('Diferencia en hora pico')
    plt.ylabel('Diferencia en fin de semana')
    plt.title('Patrones de variabilidad por cluster')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'analisis_clusters_estaciones.png')
    
    analysis['visualization_path'] = vis_path
    return analysis

def analyze_segment_clusters(results):
    """
    Analiza los patrones encontrados en los clusters de segmentos.
    
    Args:
        results: Resultados del clustering de segmentos
        
    Returns:
        Diccionario con análisis y visualizaciones
    """
    print("Analizando patrones en clusters de segmentos...")
    
    # Obtener resultados
    df = results['result_df']
    clusters = results['clusters']
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    # Analizar características por cluster
    analysis = {}
    
    # Filtrar clusters válidos (excluyendo ruido)
    valid_df = df[df['cluster'] != -1].copy()
    
    # Características promedio por cluster
    if not valid_df.empty:
        cluster_means = valid_df.groupby('cluster').mean()
        analysis['cluster_means'] = cluster_means
        
        # Conteo de segmentos por cluster
        cluster_counts = valid_df.groupby('cluster').size()
        analysis['cluster_counts'] = cluster_counts
        
        # Interpretar los clusters
        cluster_descriptions = {}
        
        for cluster in set(valid_df['cluster']):
            cluster_data = valid_df[valid_df['cluster'] == cluster]
            
            # Determinar características principales
            tiempo_medio = cluster_data['tiempo_viaje_mean'].mean()
            tiempo_ratio = cluster_data['tiempo_ratio'].mean()
            variabilidad = cluster_data['tiempo_variabilidad'].mean()
            
            # Crear descripción
            if tiempo_ratio > 1.5:
                tiempo_desc = "Tiempo mucho mayor al esperado"
            elif tiempo_ratio > 1.2:
                tiempo_desc = "Tiempo mayor al esperado"
            elif tiempo_ratio < 0.8:
                tiempo_desc = "Tiempo menor al esperado"
            else:
                tiempo_desc = "Tiempo cercano al esperado"
                
            if variabilidad > 0.3:
                var_desc = " con alta variabilidad"
            elif variabilidad > 0.15:
                var_desc = " con variabilidad media"
            else:
                var_desc = " con baja variabilidad"
                
            cluster_descriptions[cluster] = tiempo_desc + var_desc
        
        analysis['cluster_descriptions'] = cluster_descriptions
    
    # Contar segmentos considerados como ruido
    noise_count = (df['cluster'] == -1).sum()
    analysis['noise_count'] = noise_count
    
    # Visualización avanzada
    plt.figure(figsize=(15, 10))
    
    # Mapa de calor de características por cluster (si hay clusters válidos)
    plt.subplot(2, 2, 1)
    if not valid_df.empty:
        key_features = ['tiempo_viaje_mean', 'tiempo_viaje_max', 'tiempo_viaje_std', 
                        'tiempo_ratio', 'tiempo_variabilidad']
        
        sns.heatmap(
            cluster_means[key_features],
            annot=True,
            cmap='YlGnBu',
            fmt='.2f'
        )
        plt.title('Características promedio por cluster')
    else:
        plt.text(0.5, 0.5, "No hay clusters válidos para mostrar", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Características promedio por cluster')
    
    # Distribución de segmentos por cluster
    plt.subplot(2, 2, 2)
    if not valid_df.empty:
        plt.bar(cluster_counts.index, cluster_counts.values)
        plt.xlabel('Cluster')
        plt.ylabel('Número de segmentos')
        plt.title('Distribución de segmentos por cluster')
        plt.xticks(sorted(set(valid_df['cluster'])))
    else:
        plt.text(0.5, 0.5, "No hay clusters válidos para mostrar", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Distribución de segmentos por cluster')
    
    # Visualizar relación entre tiempo base y tiempo real
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(
        df['tiempo_base_first'],
        df['tiempo_viaje_mean'],
        c=df['cluster'],
        cmap='tab10',
        alpha=0.7
    )
    
    # Agregar línea de referencia (tiempo real = tiempo base)
    min_tiempo = min(df['tiempo_base_first'].min(), df['tiempo_viaje_mean'].min())
    max_tiempo = max(df['tiempo_base_first'].max(), df['tiempo_viaje_mean'].max())
    plt.plot([min_tiempo, max_tiempo], [min_tiempo, max_tiempo], 'k--', alpha=0.5)
    
    plt.xlabel('Tiempo base (min)')
    plt.ylabel('Tiempo real promedio (min)')
    plt.title('Relación entre tiempo base y tiempo real por cluster')
    plt.grid(True)
    
    # Leyenda solo si hay clusters válidos
    if n_clusters > 0:
        handles, labels = scatter.legend_elements()
        plt.legend(handles, [f'Cluster {i}' if i != -1 else 'Ruido' 
                            for i in sorted(set(df['cluster']))])
    
    # Visualizar variabilidad vs ratio de tiempo
    plt.subplot(2, 2, 4)
    plt.scatter(
        df['tiempo_ratio'],
        df['tiempo_variabilidad'],
        c=df['cluster'],
        cmap='tab10',
        alpha=0.7
    )
    plt.xlabel('Ratio tiempo real / tiempo base')
    plt.ylabel('Variabilidad del tiempo')
    plt.title('Patrones de variabilidad y eficiencia por cluster')
    plt.axhline(y=df['tiempo_variabilidad'].mean(), color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True)
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'analisis_clusters_segmentos.png')
    
    analysis['visualization_path'] = vis_path
    return analysis

def analyze_anomalies(results):
    """
    Analiza las anomalías detectadas en tiempos de viaje y congestión.
    
    Args:
        results: Resultados de la detección de anomalías
        
    Returns:
        Diccionario con análisis y visualizaciones
    """
    print("Analizando anomalías detectadas...")
    
    analysis = {}
    
    # Analizar anomalías en tiempos de viaje
    df_tiempos = results['tiempos']['result_df']
    anomalias_tiempos = df_tiempos[df_tiempos['es_anomalia'] == 1]
    
    # Estadísticas básicas
    analysis['tiempos'] = {
        'total_anomalias': len(anomalias_tiempos),
        'porcentaje_anomalias': len(anomalias_tiempos) / len(df_tiempos) * 100,
        'estadisticas': anomalias_tiempos.describe().to_dict()
    }
    
    # Identificar patrones en las anomalías
    if not anomalias_tiempos.empty:
        # Distribución por hora del día
        hora_counts = anomalias_tiempos.groupby('hora_dia').size()
        analysis['tiempos']['distribucion_hora'] = hora_counts.to_dict()
        
        # Distribución por clima
        clima_counts = anomalias_tiempos.groupby('clima').size()
        analysis['tiempos']['distribucion_clima'] = clima_counts.to_dict()
        
        # Ratio promedio de tiempo
        ratio_promedio = anomalias_tiempos['tiempo_viaje'] / anomalias_tiempos['tiempo_base']
        analysis['tiempos']['ratio_promedio'] = ratio_promedio.mean()
        
        # Estaciones más afectadas
        origen_counts = anomalias_tiempos.groupby('origen').size().sort_values(ascending=False)
        destino_counts = anomalias_tiempos.groupby('destino').size().sort_values(ascending=False)
        
        analysis['tiempos']['origenes_mas_afectados'] = origen_counts.head(5).to_dict()
        analysis['tiempos']['destinos_mas_afectados'] = destino_counts.head(5).to_dict()
    
    # Analizar anomalías en congestión
    df_congestion = results['congestion']['result_df']
    anomalias_congestion = df_congestion[df_congestion['es_anomalia'] == 1]
    
    # Estadísticas básicas
    analysis['congestion'] = {
        'total_anomalias': len(anomalias_congestion),
        'porcentaje_anomalias': len(anomalias_congestion) / len(df_congestion) * 100,
        'estadisticas': anomalias_congestion.describe().to_dict()
    }
    
    # Identificar patrones en las anomalías
    if not anomalias_congestion.empty:
        # Distribución por hora del día
        hora_counts = anomalias_congestion.groupby('hora_dia').size()
        analysis['congestion']['distribucion_hora'] = hora_counts.to_dict()
        
        # Distribución por categoría de demanda
        demanda_counts = anomalias_congestion.groupby('categoria_demanda').size()
        analysis['congestion']['distribucion_demanda'] = demanda_counts.to_dict()
        
        # Congestión promedio
        analysis['congestion']['congestion_promedio'] = anomalias_congestion['nivel_congestion'].mean()
        
        # Estaciones más afectadas
        estacion_counts = anomalias_congestion.groupby('estacion').size().sort_values(ascending=False)
        analysis['congestion']['estaciones_mas_afectadas'] = estacion_counts.head(5).to_dict()
        
        # Relación con eventos especiales
        eventos_ratio = anomalias_congestion['tiene_evento_especial'].mean()
        analysis['congestion']['ratio_eventos_especiales'] = eventos_ratio
    
    # Visualización avanzada
    plt.figure(figsize=(15, 12))
    
    # Patrones temporales de anomalías en tiempos
    plt.subplot(3, 2, 1)
    if not anomalias_tiempos.empty:
        sns.countplot(x='hora_dia', data=anomalias_tiempos)
        plt.title('Distribución de anomalías en tiempos por hora')
        plt.xlabel('Hora del día')
        plt.ylabel('Número de anomalías')
    else:
        plt.text(0.5, 0.5, "No hay anomalías detectadas", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Distribución de anomalías en tiempos por hora')
    
    # Patrones temporales de anomalías en congestión
    plt.subplot(3, 2, 2)
    if not anomalias_congestion.empty:
        sns.countplot(x='hora_dia', data=anomalias_congestion)
        plt.title('Distribución de anomalías en congestión por hora')
        plt.xlabel('Hora del día')
        plt.ylabel('Número de anomalías')
    else:
        plt.text(0.5, 0.5, "No hay anomalías detectadas", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Distribución de anomalías en congestión por hora')
    
    # Ratio de tiempo para anomalías vs normales
    plt.subplot(3, 2, 3)
    if not df_tiempos.empty:
        sns.boxplot(x='es_anomalia', y='tiempo_viaje', data=df_tiempos)
        plt.title('Tiempo de viaje: Normal vs Anomalía')
        plt.xlabel('Es anomalía')
        plt.ylabel('Tiempo de viaje (min)')
    else:
        plt.text(0.5, 0.5, "No hay datos disponibles", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Tiempo de viaje: Normal vs Anomalía')
    
    # Nivel de congestión para anomalías vs normales
    plt.subplot(3, 2, 4)
    if not df_congestion.empty:
        sns.boxplot(x='es_anomalia', y='nivel_congestion', data=df_congestion)
        plt.title('Nivel de congestión: Normal vs Anomalía')
        plt.xlabel('Es anomalía')
        plt.ylabel('Nivel de congestión')
    else:
        plt.text(0.5, 0.5, "No hay datos disponibles", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Nivel de congestión: Normal vs Anomalía')
    
    # Relación entre clima y anomalías en tiempos
    plt.subplot(3, 2, 5)
    if not df_tiempos.empty:
        clima_anomalia = pd.crosstab(df_tiempos['clima'], df_tiempos['es_anomalia'])
        clima_anomalia_pct = clima_anomalia.div(clima_anomalia.sum(axis=1), axis=0) * 100
        clima_anomalia_pct[1].plot(kind='bar')
        plt.title('Porcentaje de anomalías por condición climática')
        plt.xlabel('Condición climática')
        plt.ylabel('Porcentaje de anomalías')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "No hay datos disponibles", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Porcentaje de anomalías por condición climática')
    
    # Relación entre categoría de demanda y anomalías en congestión
    plt.subplot(3, 2, 6)
    if not df_congestion.empty:
        demanda_anomalia = pd.crosstab(df_congestion['categoria_demanda'], df_congestion['es_anomalia'])
        demanda_anomalia_pct = demanda_anomalia.div(demanda_anomalia.sum(axis=1), axis=0) * 100
        demanda_anomalia_pct[1].plot(kind='bar')
        plt.title('Porcentaje de anomalías por categoría de demanda')
        plt.xlabel('Categoría de demanda')
        plt.ylabel('Porcentaje de anomalías')
    else:
        plt.text(0.5, 0.5, "No hay datos disponibles", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Porcentaje de anomalías por categoría de demanda')
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'analisis_anomalias.png')
    
    analysis['visualization_path'] = vis_path
    return analysis

def analyze_pca_results(results):
    """
    Analiza los resultados del PCA para extraer patrones.
    
    Args:
        results: Resultados del análisis PCA
        
    Returns:
        Diccionario con análisis y visualizaciones
    """
    print("Analizando resultados de PCA...")
    # Extraer componentes principales
    X_pca = results['X_pca']
    loadings = results['loadings']
    explained_variance = results['explained_variance']
    feature_names = results['feature_names']
    rutas_pca = results['rutas_pca']
    
    analysis = {}
    
    # Varianza explicada
    analysis['explained_variance'] = explained_variance
    analysis['cumulative_variance'] = np.cumsum(explained_variance)
    
    # Interpretar componentes principales
    pc_interpretations = {}
    
    for i, pc_loadings in enumerate(loadings[:3]):
        # Encontrar características con mayor contribución
        sorted_idx = np.argsort(np.abs(pc_loadings))[::-1]
        top_features_idx = sorted_idx[:3]  # Top 3 características
        
        # Crear descripción
        interpretation = f"PC{i+1} ({explained_variance[i]:.2%} var. explicada) - "
        
        # Describir contribuciones
        contrib_desc = []
        for idx in top_features_idx:
            if idx < len(feature_names):
                feature = feature_names[idx]
                loading = pc_loadings[idx]
                direction = "alta" if loading > 0 else "baja"
                contrib_desc.append(f"{feature} {direction} ({loading:.2f})")
        
        interpretation += ", ".join(contrib_desc)
        pc_interpretations[i] = interpretation
    
    analysis['pc_interpretations'] = pc_interpretations
    
    # Identificar clusters naturales en el espacio PCA
    from sklearn.cluster import KMeans
    
    # Aplicar KMeans en el espacio PCA
    kmeans = KMeans(n_clusters=4, random_state=42)
    pca_clusters = kmeans.fit_predict(X_pca[:, :3])  # Usar las primeras 3 componentes
    
    # Agregar clusters al dataframe
    rutas_pca['pca_cluster'] = pca_clusters
    
    # Características promedio por cluster
    numeric_columns = rutas_pca.select_dtypes(include=['number']).columns
    pca_cluster_means = rutas_pca.groupby('pca_cluster')[numeric_columns].mean()
    analysis['pca_cluster_means'] = pca_cluster_means
    
    # Visualización avanzada
    plt.figure(figsize=(15, 12))
    
    # Visualizar clusters en espacio PCA 2D
    plt.subplot(2, 2, 1)
    for cluster in range(len(set(pca_clusters))):
        mask = pca_clusters == cluster
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
    plt.title('Clusters en espacio PCA (2D)')
    plt.legend()
    plt.grid(True)
    
    # Visualizar clusters en espacio PCA 3D
    ax = plt.subplot(2, 2, 2, projection='3d')
    for cluster in range(len(set(pca_clusters))):
        mask = pca_clusters == cluster
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
    ax.set_title('Clusters en espacio PCA (3D)')
    
    # Mapa de calor de características por cluster PCA
    plt.subplot(2, 2, 3)
    # Asegurarse de que todas estas características estén en las columnas numéricas
    key_features = ['num_estaciones', 'num_transbordos', 'tiempo_total', 
                    'congestion_promedio', 'calidad_ruta']
    key_features = [col for col in key_features if col in pca_cluster_means.columns]
    
    sns.heatmap(
        pca_cluster_means[key_features],
        annot=True,
        cmap='YlGnBu',
        fmt='.2f'
    )
    plt.title('Características promedio por cluster PCA')
    
    # Proyección t-SNE para visualización alternativa
    plt.subplot(2, 2, 4)
    
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Colorear por calidad de ruta
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=rutas_pca['calidad_ruta'],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Calidad de ruta')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Proyección t-SNE de rutas coloreada por calidad')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Guardar visualización
    vis_path = save_figure(plt.gcf(), 'analisis_pca.png')
    
    analysis['visualization_path'] = vis_path
    return analysis