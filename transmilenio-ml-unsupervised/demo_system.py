"""
Script para demostrar el funcionamiento del sistema de recomendación basado en patrones.
"""
from src.route_recommender import UnsupervisedRouteRecommender
from src.unsupervised_learning import (
    cluster_stations, cluster_segments, detect_anomalies
)
from src.data_preprocessing import (
    load_datasets, prepare_station_clustering_data, 
    prepare_segment_clustering_data, prepare_anomaly_detection_data
)

if __name__ == "__main__":
    print("=== Demostración del Sistema Inteligente de Rutas para Transmilenio (No Supervisado) ===")
    
    # Inicializar el recomendador de rutas
    print("\nCargando modelos y patrones descubiertos...")
    route_recommender = UnsupervisedRouteRecommender()
    
    # Cargar datasets para obtener patrones
    datasets = load_datasets()
    
    # Preparar datos 
    station_data = prepare_station_clustering_data(datasets['congestion'])
    segment_data = prepare_segment_clustering_data(datasets['tiempos'])
    anomaly_data = prepare_anomaly_detection_data(datasets['tiempos'], datasets['congestion'])
    
    # Aplicar modelos
    station_clusters = cluster_stations(station_data)
    segment_clusters = cluster_segments(segment_data)
    anomalies = detect_anomalies(anomaly_data)
    
    # Extraer mapeos
    station_mapping = station_clusters['station_clusters']
    segment_mapping = segment_clusters['segment_clusters']
    
    # Extraer segmentos anómalos
    tiempos_anomalias = anomalies['tiempos']['result_df']
    anomalous_segments = set()
    for _, row in tiempos_anomalias[tiempos_anomalias['es_anomalia'] == 1].iterrows():
        segment_key = f"{row['origen']} -> {row['destino']}"
        anomalous_segments.add(segment_key)
    
    # Inicializar patrones en el recomendador
    route_recommender.initialize_patterns(
        station_mapping, 
        segment_mapping, 
        anomalous_segments
    )
    
    # Datos de ejemplo para la demostración
    demo_origen = "Portal Norte"
    demo_destino = "Ricaurte"
    
    # Probar diferentes escenarios
    scenarios = [
        {"day": 2, "hour": 8, "description": "Día de semana, hora pico mañana"},
        {"day": 3, "hour": 14, "description": "Día de semana, hora valle"},
        {"day": 5, "hour": 10, "description": "Fin de semana, mañana"}
    ]
    
    for scenario in scenarios:
        day = scenario["day"]
        hour = scenario["hour"]
        desc = scenario["description"]
        
        print(f"\n\n=== Escenario: {desc} ===")
        
        # Encontrar la mejor ruta
        resultado = route_recommender.find_optimal_route(
            demo_origen, 
            demo_destino, 
            hour,
            day
        )
        
        # Visualizar la ruta
        vis_path = route_recommender.visualize_route(resultado)
        
        # Mostrar resumen
        if 'error' in resultado:
            print(f"\nError al buscar la ruta: {resultado['error']}")
        else:
            print("\n=== Resumen del resultado ===")
            print(f"Origen: {resultado['origin']}")
            print(f"Destino: {resultado['destination']}")
            print(f"Día: {resultado['day']} (0=Lunes, 6=Domingo)")
            print(f"Hora: {resultado['hour']}:00")
            
            best_route = resultado['best_route']
            print(f"\nMejor ruta encontrada:")
            print(f"{' -> '.join(best_route['path'])}")
            print(f"Tiempo total estimado: {best_route['total_time']:.2f} minutos")
            print(f"Número de transbordos: {best_route['num_transfers']}")
            print(f"Calidad de la ruta: {best_route['quality_score']:.2f}/10")
            
            print("\nDetalles de segmentos:")
            for i, segment in enumerate(resultado['route_details'], 1):
                anomaly_info = "⚠️ ANOMALÍA" if segment['is_anomaly'] else ""
                cluster_info = f"(Cluster {segment['segment_cluster']})" if segment['segment_cluster'] is not None else ""
                time_ratio = segment['time'] / segment['base_time']
                time_info = f"{segment['time']:.1f} min ({time_ratio:.2f}x el tiempo base)"
                
                print(f"{i}. {segment['from']} → {segment['to']}: {time_info} {cluster_info} {anomaly_info}")
            
            print("\nRutas alternativas:")
            for i, alt_route in enumerate(resultado['alternative_routes'][:2], 1):
                print(f"Alternativa {i}: Calidad {alt_route['quality_score']:.2f}/10, "
                      f"Tiempo: {alt_route['total_time']:.2f} min, "
                      f"Transbordos: {alt_route['num_transfers']}")
            
            if vis_path:
                print(f"\nVisualizacion guardada en: {vis_path}")
    
    print("\nDemostración completada con éxito.")