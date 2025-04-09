"""
Script para ejecutar modelos de aprendizaje no supervisado.
"""
from src.data_preprocessing import (
    load_datasets, explore_datasets, prepare_station_clustering_data,
    prepare_segment_clustering_data, prepare_anomaly_detection_data,
    prepare_pca_data
)
from src.unsupervised_learning import (
    cluster_stations, cluster_segments, detect_anomalies, apply_pca_analysis
)
from src.pattern_analysis import (
    analyze_station_clusters, analyze_segment_clusters, analyze_anomalies,
    analyze_pca_results
)
from src.route_recommender import UnsupervisedRouteRecommender

if __name__ == "__main__":
    print("=== Entrenamiento de modelos no supervisados para el Sistema de Rutas Transmilenio ===")
    
    # Cargar datasets
    print("\nCargando datasets...")
    datasets = load_datasets()
    
    # Explorar datos
    print("\nExplorando datos...")
    explore_results = explore_datasets(datasets)
    
    # Preparar datos para modelos no supervisados
    print("\nPreparando datos para modelos no supervisados...")
    station_data = prepare_station_clustering_data(datasets['congestion'])
    segment_data = prepare_segment_clustering_data(datasets['tiempos'])
    anomaly_data = prepare_anomaly_detection_data(datasets['tiempos'], datasets['congestion'])
    pca_data = prepare_pca_data(datasets['rutas'])
    
    # Ejecutar modelos no supervisados
    print("\nEjecutando modelos no supervisados...")
    station_clusters = cluster_stations(station_data)
    segment_clusters = cluster_segments(segment_data)
    anomalies = detect_anomalies(anomaly_data)
    pca_results = apply_pca_analysis(pca_data)
    
    # Analizar patrones
    print("\nAnalizando patrones descubiertos...")
    station_analysis = analyze_station_clusters(station_clusters)
    segment_analysis = analyze_segment_clusters(segment_clusters)
    anomaly_analysis = analyze_anomalies(anomalies)
    pca_analysis = analyze_pca_results(pca_results)
    
    # Imprimir resumen de patrones descubiertos
    print("\n=== Resumen de patrones descubiertos ===")
    
    print("\nClusters de estaciones:")
    for cluster, description in station_analysis['cluster_descriptions'].items():
        count = station_analysis['cluster_counts'][cluster]
        print(f"Cluster {cluster} ({count} estaciones): {description}")
    
    print("\nClusters de segmentos:")
    if 'cluster_descriptions' in segment_analysis:
        for cluster, description in segment_analysis['cluster_descriptions'].items():
            count = segment_analysis['cluster_counts'][cluster]
            print(f"Cluster {cluster} ({count} segmentos): {description}")
    else:
        print("No se encontraron clusters de segmentos válidos.")
    
    print("\nResumen de anomalías:")
    print(f"Tiempos de viaje: {anomaly_analysis['tiempos']['total_anomalias']} anomalías "
          f"({anomaly_analysis['tiempos']['porcentaje_anomalias']:.2f}%)")
    print(f"Congestión: {anomaly_analysis['congestion']['total_anomalias']} anomalías "
          f"({anomaly_analysis['congestion']['porcentaje_anomalias']:.2f}%)")
    
    print("\nComponentes principales:")
    variance_explained = sum(pca_analysis['explained_variance'][:3])
    print(f"Las primeras 3 componentes explican el {variance_explained:.2%} de la varianza total")
    for pc, interpretation in pca_analysis['pc_interpretations'].items():
        print(f"{interpretation}")
    
    print("\nModelos entrenados y análisis completados con éxito.")
    print("Se han guardado las visualizaciones en el directorio 'visualizations'.")