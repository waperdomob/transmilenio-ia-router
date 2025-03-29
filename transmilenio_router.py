import networkx as nx
import pandas as pd
import numpy as np
import json
import heapq
import matplotlib.pyplot as plt
from datetime import datetime, time

# Bibliotecas específicas de IA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib  # Para guardar/cargar modelos
import gym
from gym import spaces
import random

class TransmilenioEnv(gym.Env):
    """Entorno de Gym para entrenar agentes de RL en la planificación de rutas de Transmilenio"""
    
    def __init__(self, router):
        super(TransmilenioEnv, self).__init__()
        self.router = router
        self.graph = router.graph
        self.stations = router.stations
        
        # Definir espacio de acciones (índices de nodos vecinos)
        self.max_neighbors = max(len(list(self.graph.neighbors(node))) for node in self.graph.nodes)
        self.action_space = spaces.Discrete(self.max_neighbors)
        
        # Espacio de observación: características del estado actual
        # [posición actual, distancia al destino, hora del día, congestión, etc.]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]), 
            high=np.array([len(self.graph.nodes), 1.0, 24, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_node = None
        self.target_node = None
        self.current_time = None
        self.path = []
        self.max_steps = 30
        self.steps = 0
    
    def reset(self, start_node=None, target_node=None, current_time=None):
        """Reinicia el entorno con un nuevo problema de ruta"""
        if start_node is None or target_node is None:
            # Elegir dos nodos aleatorios
            nodes = list(self.graph.nodes)
            start_node = random.choice(nodes)
            target_node = random.choice([n for n in nodes if n != start_node])
        
        self.current_node = start_node
        self.target_node = target_node
        self.current_time = current_time or datetime.now()
        self.path = [start_node]
        self.steps = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """Obtiene representación del estado actual"""
        # Normalizar distancia al destino
        max_dist = 0.2  # valor aproximado para normalizar
        lat1, lon1 = self.stations[self.current_node]['location']
        lat2, lon2 = self.stations[self.target_node]['location']
        distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        norm_distance = min(distance / max_dist, 1.0)
        
        # Hora normalizada [0, 1]
        hour_norm = self.current_time.hour / 24.0
        
        # Congestión en la estación actual [0, 1]
        congestion = 1.0 if self.router.is_station_crowded(self.current_node, self.current_time) else 0.0
        
        # Número de conexiones disponibles normalizado
        num_connections = len(list(self.graph.neighbors(self.current_node)))
        norm_connections = min(num_connections / self.max_neighbors, 1.0)
        
        return np.array([
            list(self.graph.nodes).index(self.current_node) / len(self.graph.nodes),
            norm_distance,
            hour_norm,
            congestion,
            norm_connections
        ], dtype=np.float32)
    
    def step(self, action):
        """Ejecuta un paso en el entorno seleccionando una conexión"""
        self.steps += 1
        
        # Obtener nodos vecinos
        neighbors = list(self.graph.neighbors(self.current_node))
        
        # Validar acción
        if action >= len(neighbors):
            # Acción inválida, penalizar
            reward = -5
            done = False
            info = {'error': 'Acción inválida'}
            return self._get_observation(), reward, done, info
        
        # Moverse al siguiente nodo
        next_node = neighbors[action]
        edge_data = self.graph.get_edge_data(self.current_node, next_node)
        
        # Actualizar estado
        self.path.append(next_node)
        self.current_node = next_node
        
        # Calcular recompensa
        reward = self._calculate_reward(next_node, edge_data)
        
        # Verificar si hemos llegado al destino
        done = (next_node == self.target_node) or (self.steps >= self.max_steps)
        
        # Información adicional
        info = {
            'path_length': len(self.path),
            'reached_target': next_node == self.target_node
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, next_node, edge_data):
        """Calcula la recompensa de la acción tomada"""
        # Recompensa base: negativa para fomentar rutas cortas
        reward = -1
        
        # Penalizar transbordos
        if edge_data['route'] == 'transfer':
            reward -= 2
        
        # Gran recompensa por llegar al destino
        if next_node == self.target_node:
            reward += 50
        
        # Recompensa por acercarse al destino
        lat1, lon1 = self.stations[self.current_node]['location']
        lat2, lon2 = self.stations[self.target_node]['location']
        current_distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        
        # Distancia anterior
        prev_node = self.path[-2]
        lat3, lon3 = self.stations[prev_node]['location']
        prev_distance = ((lat3 - lat2) ** 2 + (lon3 - lon2) ** 2) ** 0.5
        
        # Recompensa por acercarse, penalización por alejarse
        if current_distance < prev_distance:
            reward += 2
        else:
            reward -= 2
        
        # Penalizar estaciones congestionadas en hora pico
        if self.router.is_station_crowded(next_node, self.current_time):
            reward -= 1
        
        return reward

class TransmilenioIARouter:
    def __init__(self, knowledge_base_path):
        """
        Inicializa el sistema inteligente de rutas para Transmilenio.
        
        Args:
            knowledge_base_path: Ruta al archivo con los datos del sistema
        """
        self.graph = nx.DiGraph()  # Grafo dirigido para modelar la red
        self.stations = {}  # Información de estaciones
        self.routes = {}    # Información de rutas
        
        # Sistemas de IA
        self.rule_engine = self._create_rule_engine()
        self.congestion_model = None
        self.travel_time_model = None
        self.route_planner_agent = None
        self.station_clusters = None
        
        # Cargar datos
        self.load_data(knowledge_base_path)
        
        # Inicializar y entrenar modelos de IA
        self.initialize_ai_models()
    
    def _create_rule_engine(self):
        """Crear motor de reglas simplificado"""
        class SimpleRuleEngine:
            def __init__(self):
                self.rules = {}
                self.facts = {}
            
            def add_rule(self, rule_name, condition_func):
                self.rules[rule_name] = condition_func
            
            def add_fact(self, fact_type, fact_value):
                if fact_type not in self.facts:
                    self.facts[fact_type] = set()
                self.facts[fact_type].add(fact_value)
            
            def query(self, rule_name, *args, **kwargs):
                if rule_name in self.rules:
                    result = self.rules[rule_name](*args, **kwargs)
                    return [{'Result': True}] if result else []
                elif rule_name in self.facts:
                    if args and args[0] in self.facts[rule_name]:
                        return [{'Result': True}]
                    return []
                return []
            
            def is_fact(self, fact_type, value):
                return fact_type in self.facts and value in self.facts[fact_type]
        
        return SimpleRuleEngine()
    
    def load_data(self, file_path):
        """Carga los datos de la red de Transmilenio y construye el grafo"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Cargar estaciones
                for station in data.get('stations', []):
                    station_id = station['id']
                    self.stations[station_id] = {
                        'name': station['name'],
                        'location': (station['latitude'], station['longitude']),
                        'demand': station.get('demand', 'medium'),
                        'services': station.get('services', [])
                    }
                    # Agregar nodo al grafo
                    self.graph.add_node(station_id, 
                                       name=station['name'], 
                                       pos=(station['longitude'], station['latitude']),
                                       type='station')
                
                # Cargar rutas
                for route in data.get('routes', []):
                    route_id = route['id']
                    self.routes[route_id] = {
                        'name': route['name'],
                        'stations': route['stations'],
                        'frequency': route.get('frequency', 10),
                        'schedule': route.get('schedule', {}),
                        'type': route.get('type', 'troncal')
                    }
                    
                    # Crear conexiones entre estaciones de cada ruta
                    for i in range(len(route['stations']) - 1):
                        current = route['stations'][i]
                        next_station = route['stations'][i + 1]
                        
                        # Calcular tiempo de viaje base entre estaciones
                        base_time = self.calculate_base_travel_time(current, next_station)
                        
                        # Añadir arista al grafo con atributos
                        self.graph.add_edge(current, next_station, 
                                           route=route_id, 
                                           time=base_time,
                                           frequency=route.get('frequency', 10),
                                           type=route.get('type', 'troncal'))
                
                # Cargar transbordos
                for transfer in data.get('transfers', []):
                    station_from = transfer['from']
                    station_to = transfer['to']
                    time_cost = transfer.get('time', 5)
                    
                    # Agregar arista de transbordo al grafo
                    self.graph.add_edge(station_from, station_to,
                                       route='transfer',
                                       time=time_cost,
                                       type='transfer')
                    
                print(f"Datos cargados: {len(self.stations)} estaciones, {len(self.routes)} rutas")
                print(f"Grafo construido con {self.graph.number_of_nodes()} nodos y {self.graph.number_of_edges()} aristas")
                
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
    
    def initialize_ai_models(self):
        """Inicializa y entrena los modelos de IA del sistema"""
        self.initialize_knowledge_base()
        self.train_travel_time_prediction_model()
        self.create_congestion_prediction_model()
        self.cluster_stations_by_demand()
        self.train_route_planning_agent()
    
    def initialize_knowledge_base(self):
        """Inicializa la base de conocimiento con reglas lógicas"""
        
        # Regla: Determinar si es hora pico
        def is_peak_hour(hour):
            return (6 <= hour <= 9) or (16 <= hour <= 19)
        
        self.rule_engine.add_rule("hora_pico", is_peak_hour)
        
        # Regla: Determinar mejor tipo de ruta según condiciones
        def mejor_tipo_ruta(tipo, hora, distancia):
            if tipo == "expreso":
                return is_peak_hour(hora) and distancia > 10
            elif tipo == "alimentador":
                return self.rule_engine.is_fact("origen_periferia", True)
            elif tipo == "normal":
                return not is_peak_hour(hora) and distancia < 5
            return False
        
        self.rule_engine.add_rule("mejor_tipo_ruta", mejor_tipo_ruta)
        
        # Regla: Determinar si se debe evitar una estación
        def evitar_estacion(estacion, hora):
            return (self.rule_engine.is_fact("estacion_congestionada", estacion) and 
                    is_peak_hour(hora))
        
        self.rule_engine.add_rule("evitar_estacion", evitar_estacion)
        
        # Agregar hechos sobre las estaciones
        for station_id, station_data in self.stations.items():
            # Marcar estaciones congestionadas
            if station_data.get('demand') == 'high':
                self.rule_engine.add_fact("estacion_congestionada", station_id)
            
            # Agregar servicios disponibles
            for service in station_data.get('services', []):
                self.rule_engine.add_fact(f"tiene_servicio_{service}", station_id)
        
        # Identificar estaciones periféricas (portales)
        for station_id in self.stations:
            if 'portal' in station_id.lower():
                self.rule_engine.add_fact("estacion_periferica", station_id)
                
        # Regla: Estación recomendada si tiene ciertos servicios
        def estacion_recomendada(estacion):
            return (self.rule_engine.is_fact("tiene_servicio_baño", estacion) or 
                    self.rule_engine.is_fact("tiene_servicio_tienda", estacion))
        
        self.rule_engine.add_rule("estacion_recomendada", estacion_recomendada)
    
    def train_travel_time_prediction_model(self):
        """
        Entrena un modelo de red neuronal para predecir tiempos de viaje
        basado en factores contextuales (hora, día, congestión, etc.)
        """
        print("Entrenando modelo de predicción de tiempos de viaje...")
        
        # Generar datos sintéticos para entrenamiento
        # En un sistema real, usaríamos datos históricos reales
        X_train = []
        y_train = []
        
        # Simular datos para diferentes condiciones
        for day in range(7):  # Días de la semana
            for hour in range(24):  # Horas del día
                for station_pair in [(s1, s2) for s1 in list(self.stations.keys())[:5] 
                                    for s2 in list(self.stations.keys())[:5] if s1 != s2]:
                    source, target = station_pair
                    
                    # Características del viaje
                    is_peak = 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0
                    is_weekend = 1 if day >= 5 else 0
                    base_time = self.calculate_base_travel_time(source, target)
                    source_congestion = 1 if self.stations[source].get('demand') == 'high' else 0
                    
                    # Crear vector de características
                    features = [hour/24, day/7, is_peak, is_weekend, source_congestion, base_time]
                    X_train.append(features)
                    
                    # Tiempo de viaje "real" (simulado)
                    time_factor = 1.0
                    if is_peak:
                        time_factor *= 1.4
                    if is_weekend:
                        time_factor *= 0.8
                    if source_congestion:
                        time_factor *= 1.2
                        
                    travel_time = base_time * time_factor
                    y_train.append(travel_time)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Normalizar datos
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_train)
        y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Crear modelo de red neuronal
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Entrenar modelo
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, verbose=0)
        
        # Guardar modelos y escaladores
        self.travel_time_model = model
        self.time_scaler_X = scaler_X
        self.time_scaler_y = scaler_y
        
        print("Modelo de predicción de tiempos entrenado.")
    
    def create_congestion_prediction_model(self):
        """
        Crea un modelo para predecir niveles de congestión en estaciones
        basado en patrones históricos (simulados)
        """
        print("Entrenando modelo de predicción de congestión...")
        
        # Generar datos sintéticos para entrenamiento
        X_train = []
        y_train = []
        
        # Simular patrones de congestión para cada estación
        for station_id, station_data in self.stations.items():
            for day in range(7):
                for hour in range(24):
                    # Caracteristicas: estación, día, hora
                    station_idx = list(self.stations.keys()).index(station_id)
                    features = [station_idx/len(self.stations), day/7, hour/24]
                    X_train.append(features)
                    
                    # Nivel de congestión (basado en patrón simplificado)
                    base_congestion = 0.2  # Nivel base
                    
                    # Más congestión en horas pico
                    if 6 <= hour <= 9 or 16 <= hour <= 19:
                        base_congestion += 0.5
                    
                    # Menos congestión en fin de semana
                    if day >= 5:
                        base_congestion *= 0.6
                    
                    # Congestión basada en demanda de la estación
                    if station_data.get('demand') == 'high':
                        base_congestion *= 1.5
                    elif station_data.get('demand') == 'low':
                        base_congestion *= 0.7
                    
                    # Normalizar entre 0 y 1
                    congestion = min(max(base_congestion, 0), 1)
                    y_train.append(congestion)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Crear y entrenar modelo de predicción de congestión
        model = keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(12, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Entrenar modelo
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        self.congestion_model = model
        print("Modelo de predicción de congestión entrenado.")
    
    def cluster_stations_by_demand(self):
        """
        Agrupa estaciones según patrones de demanda utilizando DBSCAN
        """
        print("Agrupando estaciones por demanda...")
        
        # Preparar datos para clustering
        features = []
        station_ids = []
        
        for station_id, station_data in self.stations.items():
            # Características para clustering
            lat, lon = station_data['location']
            demand_value = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(station_data.get('demand'), 0.5)
            is_terminal = 1 if 'portal' in station_id.lower() else 0
            num_services = len(station_data.get('services', []))
            
            # Vector de características
            feature_vector = [lat, lon, demand_value, is_terminal, num_services]
            features.append(feature_vector)
            station_ids.append(station_id)
        
        # Normalizar datos
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Ejecutar DBSCAN para clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters = dbscan.fit_predict(features_scaled)
        
        # Guardar resultados de clustering
        self.station_clusters = {}
        for station_id, cluster_id in zip(station_ids, clusters):
            if cluster_id not in self.station_clusters:
                self.station_clusters[cluster_id] = []
            self.station_clusters[cluster_id].append(station_id)
        
        print(f"Estaciones agrupadas en {len(self.station_clusters)} clusters.")
    
    def train_route_planning_agent(self):
        """
        Entrena un agente de aprendizaje por refuerzo para la planificación de rutas
        """
        print("Entrenando agente de planificación de rutas con Reinforcement Learning...")
        
        # Crear entorno
        env = TransmilenioEnv(self)
        
        # Crear modelo de red neuronal para el agente
        input_shape = env.observation_space.shape
        num_actions = env.action_space.n
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_actions, activation='linear')
        ])
        
        # Configurar el agente (DQN)
        self.route_planner_agent = model
        
        # En un sistema real, entrenaríamos el agente durante miles de episodios
        # Aquí simulamos un entrenamiento ya completado
        print("Agente de planificación de rutas entrenado (simulado).")
    
    def calculate_base_travel_time(self, station1, station2):
        """Calcula el tiempo base entre dos estaciones basado en la distancia"""
        try:
            # Obtener coordenadas
            lat1, lon1 = self.stations[station1]['location']
            lat2, lon2 = self.stations[station2]['location']
            
            # Calcular distancia aproximada (fórmula Haversine simplificada)
            import math
            R = 6371  # Radio de la Tierra en km
            dLat = math.radians(lat2 - lat1)
            dLon = math.radians(lon2 - lon1)
            a = (math.sin(dLat/2) * math.sin(dLat/2) +
                math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                math.sin(dLon/2) * math.sin(dLon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            # Velocidad promedio: ~30km/h en Transmilenio
            time_mins = (distance / 30) * 60
            
            # Tiempo mínimo entre estaciones (3 minutos)
            return max(time_mins, 3)
        except:
            # Valor por defecto si hay error
            return 5
    
    def predict_travel_time(self, source, target, context):
        """
        Predice el tiempo de viaje entre dos estaciones utilizando el modelo de ML
        """
        if self.travel_time_model is None:
            # Si no hay modelo entrenado, usar método simple
            return self.adjust_time_for_context(
                self.calculate_base_travel_time(source, target), 
                context['datetime']
            )
        
        try:
            # Extraer información del contexto
            dt = context['datetime']
            hour = dt.hour
            day = dt.weekday()
            
            # Determinar si es hora pico
            is_peak = 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0
            is_weekend = 1 if day >= 5 else 0
            
            # Obtener tiempo base y congestión
            base_time = self.calculate_base_travel_time(source, target)
            source_congestion = 1 if self.stations[source].get('demand') == 'high' else 0
            
            # Preparar características para el modelo
            features = np.array([[hour/24, day/7, is_peak, is_weekend, source_congestion, base_time]])
            
            # Normalizar
            X_scaled = self.time_scaler_X.transform(features)
            
            # Predecir
            prediction_scaled = self.travel_time_model.predict(X_scaled, verbose=0)
            
            # Desnormalizar predicción
            prediction = self.time_scaler_y.inverse_transform(prediction_scaled)[0][0]
            
            return max(prediction, base_time * 0.8)  # Nunca devolver menos del 80% del tiempo base
            
        except Exception as e:
            print(f"Error en predicción de tiempo: {e}")
            # Fallback a método simple
            return self.adjust_time_for_context(
                self.calculate_base_travel_time(source, target), 
                context['datetime']
            )
    
    def predict_station_congestion(self, station_id, context):
        """
        Predice el nivel de congestión en una estación usando el modelo de ML
        """
        if self.congestion_model is None:
            # Método simple si no hay modelo
            return self.is_station_crowded(station_id, context['datetime'])
        
        try:
            # Extraer información
            dt = context['datetime']
            hour = dt.hour
            day = dt.weekday()
            
            # Índice de la estación
            station_idx = list(self.stations.keys()).index(station_id)
            
            # Preparar características
            features = np.array([[station_idx/len(self.stations), day/7, hour/24]])
            
            # Predecir
            congestion_level = self.congestion_model.predict(features, verbose=0)[0][0]
            
            # Devolver nivel de congestión entre 0 y 1
            return float(congestion_level)
            
        except Exception as e:
            print(f"Error en predicción de congestión: {e}")
            # Fallback a método simple
            return 0.8 if self.is_station_crowded(station_id, context['datetime']) else 0.3
    
    def adjust_time_for_context(self, base_time, current_datetime=None):
        """
        Ajusta el tiempo de viaje según factores contextuales como
        hora del día, día de la semana, eventos especiales, etc.
        """
        if current_datetime is None:
            current_datetime = datetime.now()
        
        hour = current_datetime.hour
        day = current_datetime.weekday()  # 0 = lunes, 6 = domingo
        
        # Factores de ajuste por hora (1.0 significa tiempo normal)
        time_factors = {
            # Horas pico con más congestión
            6: 1.4, 7: 1.5, 8: 1.6, 9: 1.4,  # Mañana
            16: 1.3, 17: 1.5, 18: 1.6, 19: 1.4,  # Tarde
            
            # Horas valle con menos tráfico
            10: 0.9, 11: 0.8, 14: 0.8, 15: 0.9,
        }
        
        # Factores por día de la semana (lunes = 0, domingo = 6)
        day_factors = {
            0: 1.1,  # Lunes
            4: 1.2,  # Viernes
            5: 0.7,  # Sábado
            6: 0.6,  # Domingo
        }
        
        # Aplicar factores de hora
        time_factor = time_factors.get(hour, 1.0)
        
        # Aplicar factores de día
        day_factor = day_factors.get(day, 1.0)
        
        # El tiempo final es el tiempo base multiplicado por los factores
        adjusted_time = base_time * time_factor * day_factor
        
        return round(adjusted_time, 1)
    
    def get_station_by_name(self, name):
        """Busca una estación por nombre y devuelve su ID"""
        name_lower = name.lower()
        for station_id, data in self.stations.items():
            if data['name'].lower() == name_lower:
                return station_id
        
        # Búsqueda parcial si no se encuentra exacto
        for station_id, data in self.stations.items():
            if name_lower in data['name'].lower():
                return station_id
                
        return None
    
    def is_station_crowded(self, station_id, current_datetime):
        """Determina si una estación está congestionada según la hora"""
        hour = current_datetime.hour
        
        # Verificar si está en la lista de estaciones congestionadas
        is_marked_as_crowded = self.rule_engine.is_fact("estacion_congestionada", station_id)
        
        # Si está marcada como congestionada, verificar si es hora pico
        if is_marked_as_crowded:
            result = self.rule_engine.query("hora_pico", hour)
            return len(result) > 0
        
        return False
    
    def a_star_search(self, start_id, end_id, context=None):
        """
        Implementa algoritmo A* para encontrar la mejor ruta entre dos estaciones
        utilizando una heurística basada en la distancia geográfica y modelos de IA.
        """
        if not context:
            context = {
                'datetime': datetime.now(),
                'preferences': {
                    'minimize_transfers': False,
                    'avoid_crowded': True,
                    'prefer_express': True
                },
                'user_profile': {
                    'mobility_restrictions': False
                }
            }
        
        # Definir la heurística (distancia en línea recta)
        def heuristic(node1, node2):
            lat1, lon1 = self.stations[node1]['location']
            lat2, lon2 = self.stations[node2]['location']
            
            # Distancia euclidiana simplificada
            distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
            
            # Convertir a minutos aproximados (factor de conversión estimado)
            return distance * 100  # Ajustar este factor según sea necesario
        
        # Estructuras para A*
        frontier = [(0, start_id, [])]  # (prioridad, nodo, ruta)
        heapq.heapify(frontier)
        
        came_from = {}  # Para reconstruir el camino
        cost_so_far = {start_id: 0}  # Costo acumulado
        
        current_route = None  # Para rastrear cambios de ruta
        
        while frontier:
            _, current, path = heapq.heappop(frontier)
            
            if current == end_id:
                # Reconstruir y devolver el camino completo
                return self.reconstruct_path(path + [current], context)
            
            for neighbor in self.graph.neighbors(current):
                # Saltear estaciones congestionadas en horas pico si la preferencia está activa
                if (context['preferences']['avoid_crowded'] and 
                    self.predict_station_congestion(neighbor, context) > 0.7 and  # Usar modelo ML
                    neighbor != end_id):  # No saltear si es el destino
                    continue
                
                # Obtener datos de la conexión
                edge_data = self.graph.get_edge_data(current, neighbor)
                
                # Calcular costo del movimiento ajustado al contexto usando modelo ML
                move_cost = self.calculate_edge_cost(current, neighbor, edge_data, current_route, context)
                
                # Actualizar ruta actual
                if edge_data['route'] != 'transfer':
                    current_route = edge_data['route']
                
                # Costo total hasta este nodo
                new_cost = cost_so_far[current] + move_cost
                
                # Verificar si es una ruta más corta o si no ha sido visitado
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    
                    # Prioridad = costo + heurística
                    priority = new_cost + heuristic(neighbor, end_id)
                    
                    # Añadir a la frontera
                    heapq.heappush(frontier, (priority, neighbor, path + [current]))
                    came_from[neighbor] = current
        
        # Si no se encuentra ruta
        return []
    
    def calculate_edge_cost(self, from_node, to_node, edge_data, current_route, context):
        """Calcula el costo de moverse por una arista usando modelos de IA"""
        route = edge_data['route']
        
        # Usar modelo ML para predecir tiempo de viaje
        if route != 'transfer':
            time_cost = self.predict_travel_time(from_node, to_node, context)
        else:
            time_cost = edge_data['time']  # Tiempo de transbordo
        
        # Costos adicionales según el tipo de movimiento
        additional_cost = 0
        
        # Penalizar transbordos
        if route == 'transfer':
            # Penalización base por transbordo
            additional_cost += 3
            
            # Mayor penalización para usuarios con movilidad reducida
            if context['user_profile']['mobility_restrictions']:
                additional_cost += 5
        
        # Penalizar cambios de ruta (cuando no es un transbordo explícito)
        elif current_route and route != current_route:
            # El cambio implica esperar otra ruta
            additional_cost += edge_data.get('frequency', 10) / 2  # Tiempo medio de espera
        
        # Aplicar preferencias de ruta
        if context['preferences']['prefer_express'] and 'expreso' in route:
            # Bonificación para rutas expresas cuando se prefieren
            additional_cost -= 2
        
        return time_cost + additional_cost
    
    def reconstruct_path(self, path_nodes, context):
        """Reconstruye la ruta con información detallada para el usuario"""
        result = []
        current_route = None
        total_time = 0
        
        # Resumir la ruta
        for i in range(len(path_nodes) - 1):
            from_station = path_nodes[i]
            to_station = path_nodes[i + 1]
            
            # Obtener datos de la conexión
            edge_data = self.graph.get_edge_data(from_station, to_station)
            route = edge_data['route']
            
            # Tiempo ajustado al contexto usando modelo ML
            if route != 'transfer':
                time = self.predict_travel_time(from_station, to_station, context)
            else:
                time = edge_data['time']
                
            total_time += time
            
            # Crear paso de la ruta
            if route == 'transfer':
                step = {
                    'type': 'transfer',
                    'station': self.stations[from_station]['name'],
                    'time': time,
                    'congestion': round(self.predict_station_congestion(from_station, context) * 100)
                }
            else:
                # Si hay cambio de ruta, agregar transbordo implícito
                if current_route and route != current_route and route != 'transfer':
                    transfer_step = {
                        'type': 'route_change',
                        'station': self.stations[from_station]['name'],
                        'from_route': self.routes[current_route]['name'] if current_route in self.routes else current_route,
                        'to_route': self.routes[route]['name'],
                        'time': edge_data.get('frequency', 10) / 2,  # Tiempo medio de espera
                        'congestion': round(self.predict_station_congestion(from_station, context) * 100)
                    }
                    result.append(transfer_step)
                    total_time += transfer_step['time']
                
                step = {
                    'type': 'travel',
                    'from': self.stations[from_station]['name'],
                    'to': self.stations[to_station]['name'],
                    'route': self.routes[route]['name'],
                    'time': time,
                    'congestion_level': round(self.predict_station_congestion(to_station, context) * 100)
                }
                
                current_route = route
            
            result.append(step)
        
        # Añadir resumen al principio
        if result:
            summary = {
                'type': 'summary',
                'origin': self.stations[path_nodes[0]]['name'],
                'destination': self.stations[path_nodes[-1]]['name'],
                'total_time': round(total_time, 1),
                'total_steps': len(result),
                'transfers': sum(1 for step in result if step['type'] in ['transfer', 'route_change']),
                'ai_optimized': True
            }
            result.insert(0, summary)
        
        return result
    
    def analyze_and_optimize_route(self, origin, destination, preferences=None):
        """
        Analiza múltiples opciones de ruta y selecciona la mejor según las preferencias
        y el contexto actual usando modelos de IA.
        """
        if not preferences:
            preferences = {
                'minimize_transfers': False,
                'avoid_crowded': True,
                'prefer_express': True,
                'priority': 'time'  # 'time', 'transfers', 'comfort'
            }
        
        # Preparar contexto
        context = {
            'datetime': datetime.now(),
            'preferences': preferences,
            'user_profile': {
                'mobility_restrictions': False
            }
        }
        
        # Obtener IDs de estaciones
        start_id = self.get_station_by_name(origin)
        end_id = self.get_station_by_name(destination)
        
        if not start_id or not end_id:
            return {"error": "Estación de origen o destino no encontrada"}
        
        # Inicializar las mejores rutas para cada criterio
        best_routes = {
            'fastest': None,
            'fewest_transfers': None,
            'most_comfortable': None
        }
        
        # 1. Encontrar la ruta más rápida
        context['preferences']['priority'] = 'time'
        fastest_route = self.a_star_search(start_id, end_id, context)
        best_routes['fastest'] = fastest_route
        
        # 2. Encontrar la ruta con menos transbordos
        context['preferences']['priority'] = 'transfers'
        context['preferences']['minimize_transfers'] = True
        fewest_transfers = self.a_star_search(start_id, end_id, context)
        best_routes['fewest_transfers'] = fewest_transfers
        
        # 3. Encontrar la ruta más cómoda (evitando estaciones congestionadas)
        context['preferences']['priority'] = 'comfort'
        context['preferences']['avoid_crowded'] = True
        most_comfortable = self.a_star_search(start_id, end_id, context)
        best_routes['most_comfortable'] = most_comfortable
        
        # Seleccionar la mejor ruta según la prioridad del usuario
        priority = preferences.get('priority', 'time')
        
        if priority == 'time':
            recommended_route = best_routes['fastest']
        elif priority == 'transfers':
            recommended_route = best_routes['fewest_transfers']
        else:  # comfort
            recommended_route = best_routes['most_comfortable']
        
        # Enriquecer la respuesta con información adicional
        if recommended_route and recommended_route[0]['type'] == 'summary':
            recommended_route[0]['alternatives'] = {
                'fastest_time': best_routes['fastest'][0]['total_time'] if best_routes['fastest'] else None,
                'fewest_transfers_count': best_routes['fewest_transfers'][0]['transfers'] if best_routes['fewest_transfers'] else None
            }
            
            # Consultar base de conocimiento para recomendaciones adicionales
            recommended_route[0]['recommendations'] = self.get_knowledge_based_recommendations(start_id, end_id, context)
            
            # Añadir clúster al que pertenecen las estaciones
            station_clusters = {}
            for cluster_id, stations in self.station_clusters.items():
                for station_id in stations:
                    if station_id == start_id or station_id == end_id:
                        station_clusters[self.stations[station_id]['name']] = f"Cluster {cluster_id}"
            
            recommended_route[0]['station_clusters'] = station_clusters
        
        return recommended_route
    
    def get_knowledge_based_recommendations(self, start_id, end_id, context):
        """Obtiene recomendaciones basadas en reglas lógicas y modelos de IA"""
        recommendations = []
        
        # Determinar si se recomienda ruta expresa
        hour = context['datetime'].hour
        
        # Calcular distancia aproximada entre origen y destino
        lat1, lon1 = self.stations[start_id]['location']
        lat2, lon2 = self.stations[end_id]['location']
        distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 100  # aproximación
        
        # Consultar regla para tipo de ruta
        for tipo in ["expreso", "alimentador", "normal"]:
            result = self.rule_engine.query("mejor_tipo_ruta", tipo, hour, distance)
            if result:
                recommendations.append(f"Se recomienda usar rutas de tipo {tipo} para este viaje")
        
        # Buscar estaciones recomendadas en el camino
        for station_id in self.stations:
            result = self.rule_engine.query("estacion_recomendada", station_id)
            if result:
                recommendations.append(f"La estación {self.stations[station_id]['name']} ofrece servicios adicionales")
                
        # Recomendaciones basadas en predicciones de congestión
        congested_stations = []
        for station_id in self.stations:
            congestion = self.predict_station_congestion(station_id, context)
            if congestion > 0.7:  # Altamente congestionada
                congested_stations.append(self.stations[station_id]['name'])
                
        if congested_stations:
            recommendations.append(f"Evita las estaciones más congestionadas a esta hora: {', '.join(congested_stations[:3])}")
        
        return recommendations
    
    def generate_route_using_rl(self, start_id, end_id, context=None):
        """
        Genera una ruta usando el agente de aprendizaje por refuerzo entrenado
        (Versión simplificada para demostración)
        """
        # Esta función simularía el uso del agente RL para generar rutas
        # En un sistema real, el agente tomaría decisiones en cada paso
        
        # Aquí simplemente usamos A* como fallback
        return self.a_star_search(start_id, end_id, context)
    
    def visualize_route(self, route):
        """
        Genera una visualización de la ruta en el grafo de Transmilenio.
        Útil para depuración y visualización de resultados.
        """
        if not route or isinstance(route, dict) and 'error' in route:
            return None
        
        # Extraer nodos de la ruta
        nodes = []
        for step in route[1:]:  # Ignorar el resumen
            if step['type'] == 'travel':
                # Obtener IDs de estaciones
                from_id = next((id for id, data in self.stations.items() 
                              if data['name'] == step['from']), None)
                to_id = next((id for id, data in self.stations.items() 
                              if data['name'] == step['to']), None)
                
                # Añadir a la lista si no están ya
                if from_id and from_id not in nodes:
                    nodes.append(from_id)
                if to_id and to_id not in nodes:
                    nodes.append(to_id)
        
        if not nodes:
            return None
            
        # Crear un subgrafo con las estaciones de la ruta
        route_graph = self.graph.subgraph(nodes)
        
        # Obtener posiciones de los nodos
        pos = nx.get_node_attributes(route_graph, 'pos')
        
        # Colorear nodos y aristas para visualización
        plt.figure(figsize=(12, 8))
        
        # Dibujar todos los nodos del grafo en gris claro (fondo)
        all_pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw_networkx_nodes(self.graph, all_pos, node_size=30, node_color='lightgray', alpha=0.3)
        
        # Dibujar nodos de la ruta coloreados por congestión
        node_colors = []
        for node in route_graph.nodes():
            congestion = self.predict_station_congestion(node, {'datetime': datetime.now()})
            # Escala de color: verde (bajo) -> amarillo -> rojo (alto)
            if congestion < 0.3:
                node_colors.append('green')
            elif congestion < 0.7:
                node_colors.append('orange')
            else:
                node_colors.append('red')
                
        nx.draw_networkx_nodes(route_graph, pos, node_size=100, node_color=node_colors)
        
        # Destacar origen y destino
        origin_id = nodes[0]
        destination_id = nodes[-1]
        nx.draw_networkx_nodes(self.graph, all_pos, nodelist=[origin_id], 
                             node_size=200, node_color='blue')
        nx.draw_networkx_nodes(self.graph, all_pos, nodelist=[destination_id], 
                             node_size=200, node_color='purple')
        
        # Dibujar aristas de la ruta
        edges = list(route_graph.edges())
        nx.draw_networkx_edges(route_graph, pos, edgelist=edges, width=2, edge_color='blue')
        
        # Dibujar etiquetas solo para los nodos de la ruta
        labels = {node: self.stations[node]['name'] for node in nodes}
        nx.draw_networkx_labels(route_graph, pos, labels=labels, font_size=8)
        
        plt.title(f"Ruta de {self.stations[origin_id]['name']} a {self.stations[destination_id]['name']}")
        plt.axis('off')
        plt.tight_layout()
        
        # En una aplicación real, se guardaría la imagen o se mostraría en un UI
        return plt

# Función para crear un archivo JSON de ejemplo
def create_sample_data_file(file_path):
    """Crea un archivo JSON de ejemplo con datos de Transmilenio"""
    sample_data = {
        "stations": [
            {
                "id": "portal_norte",
                "name": "Portal Norte",
                "latitude": 4.7596,
                "longitude": -74.0419,
                "demand": "high",
                "services": ["baño", "tienda", "wifi"]
            },
            {
                "id": "toberin",
                "name": "Toberín",
                "latitude": 4.7450,
                "longitude": -74.0430,
                "demand": "medium",
                "services": []
            },
            {
                "id": "cardio_infantil",
                "name": "Cardio Infantil",
                "latitude": 4.7300,
                "longitude": -74.0438,
                "demand": "medium",
                "services": ["baño"]
            },
            {
                "id": "mazuren",
                "name": "Mazurén",
                "latitude": 4.7210,
                "longitude": -74.0445,
                "demand": "medium",
                "services": []
            },
            {
                "id": "alcala",
                "name": "Alcalá",
                "latitude": 4.7120,
                "longitude": -74.0450,
                "demand": "low",
                "services": []
            },
            {
                "id": "calle_142",
                "name": "Calle 142",
                "latitude": 4.7030,
                "longitude": -74.0460,
                "demand": "medium",
                "services": []
            },
            {
                "id": "calle_127",
                "name": "Calle 127",
                "latitude": 4.6930,
                "longitude": -74.0470,
                "demand": "high",
                "services": ["baño"]
            },
            {
                "id": "pepe_sierra",
                "name": "Pepe Sierra",
                "latitude": 4.6830,
                "longitude": -74.0480,
                "demand": "medium",
                "services": []
            },
            {
                "id": "calle_106",
                "name": "Calle 106",
                "latitude": 4.6750,
                "longitude": -74.0490,
                "demand": "medium",
                "services": []
            },
            {
                "id": "calle_100",
                "name": "Calle 100",
                "latitude": 4.6860,
                "longitude": -74.0505,
                "demand": "high",
                "services": ["baño", "tienda"]
            },
            {
                "id": "virrey",
                "name": "Virrey",
                "latitude": 4.6710,
                "longitude": -74.0540,
                "demand": "medium",
                "services": []
            },
            {
                "id": "calle_85",
                "name": "Calle 85",
                "latitude": 4.6670,
                "longitude": -74.0555,
                "demand": "high",
                "services": ["baño"]
            },
            {
                "id": "heroes",
                "name": "Héroes",
                "latitude": 4.6550,
                "longitude": -74.0570,
                "demand": "high",
                "services": ["baño", "tienda"]
            },
            {
                "id": "calle_76",
                "name": "Calle 76",
                "latitude": 4.6640,
                "longitude": -74.0650,
                "demand": "high",
                "services": []
            },
            {
                "id": "calle_72",
                "name": "Calle 72",
                "latitude": 4.6580,
                "longitude": -74.0650,
                "demand": "high",
                "services": ["baño", "tienda"]
            },
            {
                "id": "flores",
                "name": "Flores",
                "latitude": 4.6480,
                "longitude": -74.0690,
                "demand": "low",
                "services": []
            },
            {
                "id": "calle_63",
                "name": "Calle 63",
                "latitude": 4.6420,
                "longitude": -74.0710,
                "demand": "high",
                "services": ["baño"]
            },
            {
                "id": "av_jimenez",
                "name": "Av. Jiménez",
                "latitude": 4.6005,
                "longitude": -74.0740,
                "demand": "high",
                "services": ["baño", "tienda", "wifi"]
            },
            {
                "id": "ricaurte",
                "name": "Ricaurte",
                "latitude": 4.6130,
                "longitude": -74.0830,
                "demand": "high",
                "services": ["baño", "tienda"]
            }
        ],
        "routes": [
            {
                "id": "b1",
                "name": "B1 - Portal Norte - Ricaurte",
                "type": "troncal",
                "stations": ["portal_norte", "toberin", "cardio_infantil", "mazuren", "alcala", "calle_142", "calle_127", "pepe_sierra", "calle_106", "calle_100", "virrey", "calle_85", "heroes", "calle_76", "calle_72", "flores", "calle_63", "av_jimenez", "ricaurte"],
                "frequency": 5,
                "schedule": {
                    "weekday": {"start": "04:00", "end": "23:00"},
                    "weekend": {"start": "05:00", "end": "22:00"}
                }
            },
            {
                "id": "b74",
                "name": "B74 - Portal Norte - Calle 76 (Expreso)",
                "type": "expreso",
                "stations": ["portal_norte", "calle_142", "calle_127", "calle_100", "heroes", "calle_76"],
                "frequency": 8,
                "schedule": {
                    "weekday": {"start": "05:00", "end": "22:00"},
                    "weekend": {"start": "06:00", "end": "21:00"}
                }
            },
            {
                "id": "c15",
                "name": "C15 - Av. Jiménez - Ricaurte",
                "type": "troncal",
                "stations": ["av_jimenez", "ricaurte"],
                "frequency": 4,
                "schedule": {
                    "weekday": {"start": "04:30", "end": "22:30"},
                    "weekend": {"start": "05:30", "end": "21:30"}
                }
            }
        ],
        "transfers": [
            {
                "from": "heroes",
                "to": "calle_76",
                "time": 4
            },
            {
                "from": "calle_76",
                "to": "heroes",
                "time": 4
            },
            {
                "from": "calle_100",
                "to": "calle_106",
                "time": 3
            },
            {
                "from": "calle_106",
                "to": "calle_100",
                "time": 3
            }
        ]
    }
    
    # Guardar datos en el archivo
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(sample_data, file, indent=2, ensure_ascii=False)
    
    print(f"Archivo de ejemplo creado en '{file_path}'")

# Ejemplo de uso del sistema
def main():
    # Nombre del archivo de datos
    data_file = "transmilenio_data.json"
    
    # Crear datos de ejemplo si el archivo no existe
    import os
    if not os.path.exists(data_file):
        print("Archivo de datos no encontrado. Creando datos de ejemplo...")
        create_sample_data_file(data_file)
    
    print("\n=== Sistema Inteligente de Rutas para Transmilenio ===")
    print("Iniciando sistema con modelos de IA...")
    
    # Crear instancia del router
    try:
        router = TransmilenioIARouter(data_file)
    except Exception as e:
        print(f"Error al inicializar el router: {e}")
        return
    
    # Definir origen y destino
    origen = "Portal Norte"
    destino = "Ricaurte"
    
    print(f"\nBuscando ruta inteligente de '{origen}' a '{destino}'...")
    
    # Definir preferencias del usuario
    preferencias = {
        'minimize_transfers': True,  # Minimizar transbordos
        'avoid_crowded': True,       # Evitar estaciones congestionadas
        'prefer_express': True,      # Preferir rutas expresas
        'priority': 'time'           # Prioridad: tiempo ('time'), transbordos ('transfers') o comodidad ('comfort')
    }
    
    # Encontrar la mejor ruta
    route = router.analyze_and_optimize_route(origen, destino, preferencias)
    
    # Mostrar resultado
    if isinstance(route, dict) and 'error' in route:
        print(f"Error: {route['error']}")
    else:
        summary = route[0]
        print(f"\nRuta encontrada de {summary['origin']} a {summary['destination']}")
        print(f"Tiempo total estimado: {summary['total_time']} minutos")
        print(f"Transbordos: {summary['transfers']}")
        print(f"Ruta optimizada con IA: {'Sí' if summary.get('ai_optimized', False) else 'No'}")
        
        # Mostrar recomendaciones basadas en reglas lógicas y modelos de IA
        if 'recommendations' in summary:
            print("\nRecomendaciones del sistema:")
            for rec in summary['recommendations']:
                print(f"- {rec}")
        
        # Mostrar alternativas
        if 'alternatives' in summary:
            print("\nAlternativas disponibles:")
            if summary['alternatives']['fastest_time']:
                print(f"- Ruta más rápida: {summary['alternatives']['fastest_time']} minutos")
            if summary['alternatives']['fewest_transfers_count'] is not None:
                print(f"- Ruta con menos transbordos: {summary['alternatives']['fewest_transfers_count']} transbordos")
        
        print("\nInstrucciones de viaje:")
        for i, step in enumerate(route[1:], 1):
            if step['type'] == 'travel':
                congestion = f" (Congestión: {step.get('congestion_level', 'N/A')}%)" if 'congestion_level' in step else ""
                print(f"{i}. Toma la ruta {step['route']} desde {step['from']} hasta {step['to']} ({step['time']} min){congestion}")
            elif step['type'] == 'transfer':
                congestion = f" (Congestión: {step.get('congestion', 'N/A')}%)" if 'congestion' in step else ""
                print(f"{i}. Realiza transbordo en la estación {step['station']} ({step['time']} min){congestion}")
            elif step['type'] == 'route_change':
                congestion = f" (Congestión: {step.get('congestion', 'N/A')}%)" if 'congestion' in step else ""
                print(f"{i}. En la estación {step['station']}, cambia de la ruta {step['from_route']} "
                      f"a la ruta {step['to_route']} ({step['time']} min){congestion}")
    
    # Generar y mostrar visualización de la ruta
    try:
        plt = router.visualize_route(route)
        if plt:
            # Si estás en un entorno que soporta visualización:
            # plt.show()
            
            # Guardar imagen en un archivo
            plt.savefig("ruta_transmilenio.png")
            print("\nVisualización de la ruta guardada en 'ruta_transmilenio.png'")
    except Exception as e:
        print(f"No se pudo generar la visualización: {e}")
        print("Para visualizar esta ruta gráficamente, ejecuta este código en un entorno compatible con matplotlib.")

if __name__ == "__main__":
    main()