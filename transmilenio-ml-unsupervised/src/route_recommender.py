"""
Módulo para recomendar rutas basándose en patrones descubiertos.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import joblib
from sklearn.preprocessing import StandardScaler
from config import (
    MODELS_DIR, ESTACIONES, ESTACIONES_ALTA_DEMANDA,
    ESTACIONES_MEDIA_DEMANDA
)
from src.utils import create_transmilenio_graph, plot_graph_with_route

class UnsupervisedRouteRecommender:
    """
    Clase para recomendar rutas basándose en patrones descubiertos.
    """
    
    def __init__(self):
        """Inicializa el recomendador de rutas cargando modelos entrenados."""
        # Cargar modelos
        self.load_models()
        
        # Crear grafo de la red
        self.G = create_transmilenio_graph()
        
        # Inicializar mapeo de estaciones y segmentos a clusters
        self.station_clusters = {}
        self.segment_clusters = {}
        self.anomaly_segments = set()
        
    def load_models(self):
        """Carga los modelos entrenados desde archivos."""
        # Modelo de clustering de estaciones
        modelo_kmeans_path = os.path.join(MODELS_DIR, 'kmeans_stations.joblib')
        self.modelo_kmeans = joblib.load(modelo_kmeans_path)
        
        # Modelo de clustering de segmentos
        modelo_dbscan_path = os.path.join(MODELS_DIR, 'dbscan_segments.joblib')
        self.modelo_dbscan = joblib.load(modelo_dbscan_path)
        
        # Modelo de detección de anomalías para tiempos
        modelo_isoforest_path = os.path.join(MODELS_DIR, 'isoforest_tiempos.joblib')
        self.modelo_isoforest = joblib.load(modelo_isoforest_path)
        
        # Modelo de detección de anomalías para congestión
        modelo_lof_path = os.path.join(MODELS_DIR, 'lof_congestion.joblib')
        self.modelo_lof = joblib.load(modelo_lof_path)
        
        # Modelo PCA para rutas
        modelo_pca_path = os.path.join(MODELS_DIR, 'pca_rutas.joblib')
        self.modelo_pca = joblib.load(modelo_pca_path)
    
    def initialize_patterns(self, station_clusters, segment_clusters, anomaly_segments=None):
        """
        Inicializa los patrones descubiertos.
        
        Args:
            station_clusters: Diccionario mapeando estaciones a clusters
            segment_clusters: Diccionario mapeando segmentos a clusters
            anomaly_segments: Conjunto de segmentos detectados como anomalías
        """
        self.station_clusters = station_clusters
        self.segment_clusters = segment_clusters
        self.anomaly_segments = anomaly_segments or set()
    
    def get_station_cluster(self, station):
        """
        Obtiene el cluster al que pertenece una estación.
        
        Args:
            station: Nombre de la estación
            
        Returns:
            Número de cluster o None si no se encuentra
        """
        return self.station_clusters.get(station)
    
    def get_segment_cluster(self, origin, destination):
        """
        Obtiene el cluster al que pertenece un segmento.
        
        Args:
            origin: Estación de origen
            destination: Estación de destino
            
        Returns:
            Número de cluster o None si no se encuentra
        """
        segment_key = f"{origin} -> {destination}"
        return self.segment_clusters.get(segment_key)
    
    def is_segment_anomaly(self, origin, destination):
        """
        Verifica si un segmento está marcado como anomalía.
        
        Args:
            origin: Estación de origen
            destination: Estación de destino
            
        Returns:
            True si es anomalía, False en caso contrario
        """
        segment_key = f"{origin} -> {destination}"
        return segment_key in self.anomaly_segments
    
    def predict_travel_time(self, origin, destination, hour, day, is_weekend):
        """
        Predice el tiempo de viaje entre dos estaciones basado en patrones.
        
        Args:
            origin: Estación de origen
            destination: Estación de destino
            hour: Hora del día (0-23)
            day: Día de la semana (0-6)
            is_weekend: Indicador de fin de semana (0 o 1)
            
        Returns:
            Tiempo estimado de viaje en minutos
        """
        # Buscar tiempo base
        tiempo_base = None
        for u, v, data in self.G.edges(data=True):
            if u == origin and v == destination:
                tiempo_base = data['tiempo_base']
                break
        
        if tiempo_base is None:
            print(f"No hay conexión directa entre {origin} y {destination}")
            return None
        
        # Preparar datos
        is_peak_hour = 1 if (6 <= hour <= 9 or 16 <= hour <= 19) else 0
        
        # Determinar demanda
        if origin in ESTACIONES_ALTA_DEMANDA:
            demanda_origen = 2
        elif origin in ESTACIONES_MEDIA_DEMANDA:
            demanda_origen = 1
        else:
            demanda_origen = 0
            
        if destination in ESTACIONES_ALTA_DEMANDA:
            demanda_destino = 2
        elif destination in ESTACIONES_MEDIA_DEMANDA:
            demanda_destino = 1
        else:
            demanda_destino = 0
        
        # Obtener cluster del segmento
        segment_cluster = self.get_segment_cluster(origin, destination)
        
        # Ajustar tiempo según patrón del cluster
        if segment_cluster is not None and segment_cluster != -1:
            # Factores específicos por cluster (estos valores deberían venir del análisis)
            cluster_factors = {
                0: 1.0,  # Neutral - tiempo conforme a lo esperado
                1: 1.3,  # Lento - tiempo mayor al esperado
                2: 0.9,  # Rápido - tiempo menor al esperado
                3: 1.2,  # Variable - tiempo algo mayor y variable
                4: 1.1   # Moderado - ligeramente por encima de lo esperado
            }
            cluster_factor = cluster_factors.get(segment_cluster, 1.0)
        else:
            cluster_factor = 1.0
        
        # Ajustar por hora pico
        peak_factor = 1.3 if is_peak_hour else 1.0
        
        # Ajustar por fin de semana
        weekend_factor = 0.8 if is_weekend else 1.0
        
        # Verificar si es anomalía
        if self.is_segment_anomaly(origin, destination):
            anomaly_factor = 1.5  # Penalizar segmentos anómalos
        else:
            anomaly_factor = 1.0
        
        # Calcular tiempo final
        adjusted_time = tiempo_base * cluster_factor * peak_factor * weekend_factor * anomaly_factor
        
        return adjusted_time
    
    def find_optimal_route(self, origin, destination, hour, day, is_weekend=None):
        """
        Encuentra la ruta óptima entre dos estaciones usando los patrones descubiertos.
        
        Args:
            origin: Estación de origen
            destination: Estación de destino
            hour: Hora del día (0-23)
            day: Día de la semana (0-6)
            is_weekend: Indicador de fin de semana (si es None, se determina automáticamente)
            
        Returns:
            Diccionario con información de la mejor ruta
        """
        if is_weekend is None:
            is_weekend = 1 if day >= 5 else 0
        
        print(f"\nBuscando ruta óptima de {origin} a {destination} - día {day}, hora {hour}")
        
        # Crear grafo temporal con pesos ajustados según patrones
        G_temp = nx.DiGraph()
        
        for u, v, data in self.G.edges(data=True):
            # Predecir tiempo de viaje
            travel_time = self.predict_travel_time(u, v, hour, day, is_weekend)
            if travel_time is not None:
                G_temp.add_edge(u, v, weight=travel_time, original_time=data['tiempo_base'])
        
        # Buscar rutas alternativas
        try:
            # Encontrar camino más corto
            shortest_path = nx.shortest_path(G_temp, origin, destination, weight='weight')
            shortest_time = sum(G_temp[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
            
            # Calcular más rutas alternativas
            k_paths = list(nx.shortest_simple_paths(G_temp, origin, destination, weight='weight'))[:5]
            
            # Evaluar cada ruta
            route_evaluations = []
            
            for i, path in enumerate(k_paths):
                # Calcular tiempo total
                total_time = sum(G_temp[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                
                # Calcular número de transbordos (simplificación)
                num_transfers = len(path) - 2
                
                # Calcular número de segmentos anómalos
                anomalous_segments = sum(1 for u, v in zip(path[:-1], path[1:]) 
                                        if self.is_segment_anomaly(u, v))
                
                # Calcular número de estaciones en clusters problemáticos
                problematic_stations = 0
                for station in path:
                    station_cluster = self.get_station_cluster(station)
                    if station_cluster in [1, 3]:  # Clusters más congestionados
                        problematic_stations += 1
                
                # Calcular una puntuación de calidad
                quality_score = 10.0
                quality_score -= (total_time / shortest_time - 1) * 5  # Penalizar por tiempo adicional
                quality_score -= num_transfers * 0.5                  # Penalizar transbordos
                quality_score -= anomalous_segments * 1.5             # Penalizar anomalías
                quality_score -= problematic_stations * 0.3           # Penalizar estaciones problemáticas
                
                quality_score = max(0, min(10, quality_score))  # Limitar entre 0 y 10
                
                # Guardar evaluación
                route_evaluations.append({
                    'route_id': i+1,
                    'path': path,
                    'total_time': total_time,
                    'num_transfers': num_transfers,
                    'anomalous_segments': anomalous_segments,
                    'problematic_stations': problematic_stations,
                    'quality_score': quality_score
                })
            
            # Ordenar por puntuación de calidad
            route_evaluations.sort(key=lambda x: x['quality_score'], reverse=True)
            best_route = route_evaluations[0]
            
            # Preparar resultado detallado
            result = {
                'origin': origin,
                'destination': destination,
                'hour': hour,
                'day': day,
                'is_weekend': is_weekend,
                'best_route': {
                    'path': best_route['path'],
                    'total_time': best_route['total_time'],
                    'num_transfers': best_route['num_transfers'],
                    'quality_score': best_route['quality_score']
                },
                'alternative_routes': route_evaluations[1:],
                'route_details': []
            }
            
            # Agregar detalles de cada segmento de la mejor ruta
            for i in range(len(best_route['path']) - 1):
                u = best_route['path'][i]
                v = best_route['path'][i + 1]
                
                segment_time = G_temp[u][v]['weight']
                base_time = G_temp[u][v]['original_time']
                segment_cluster = self.get_segment_cluster(u, v)
                is_anomaly = self.is_segment_anomaly(u, v)
                
                result['route_details'].append({
                    'from': u,
                    'to': v,
                    'time': segment_time,
                    'base_time': base_time,
                    'segment_cluster': segment_cluster,
                    'is_anomaly': is_anomaly
                })
            
            return result
            
        except Exception as e:
            print(f"Error al buscar ruta: {e}")
            return {
                'error': str(e),
                'origin': origin,
                'destination': destination
            }
    
    def visualize_route(self, route_result):
        """
        Genera una visualización de la ruta recomendada.
        
        Args:
            route_result: Resultado de find_optimal_route
            
        Returns:
            Ruta al archivo de visualización o None si hay error
        """
        if 'error' in route_result:
            print(f"No se puede visualizar: {route_result['error']}")
            return None
        
        # Extraer path de la mejor ruta
        path = route_result['best_route']['path']
        
        # Crear título
        title = (f"Ruta óptima: {route_result['origin']} → {route_result['destination']}\n"
                f"Tiempo: {route_result['best_route']['total_time']:.1f} min, "
                f"Calidad: {route_result['best_route']['quality_score']:.1f}/10")
        
        # Generar visualización
        return plot_graph_with_route(self.G, path, title=title, filename="ruta_recomendada.png")