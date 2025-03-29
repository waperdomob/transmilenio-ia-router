import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import heapq
import matplotlib.pyplot as plt
from datetime import datetime, time
import pyswip  # Biblioteca de Prolog para Python, para la base de conocimiento basada en reglas

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
        self.prolog = pyswip.Prolog()  # Motor de inferencia para reglas lógicas
        
        # Cargar datos
        self.load_data(knowledge_base_path)
        # Inicializar base de conocimiento en reglas lógicas
        self.initialize_knowledge_base()
        # Crear y entrenar modelo para predecir tiempos según día/hora
        self.train_time_prediction_model()
    
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
                        # En un sistema real, esto se podría calcular usando datos históricos
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
    
    def initialize_knowledge_base(self):
        """Inicializa la base de conocimiento en Prolog con reglas lógicas"""
        
        # Reglas para determinar el mejor tipo de ruta según condiciones
        rules = """
        % Definir prioridades de rutas según factores
        
        % Regla: Elegir rutas expresas en horas pico para viajes largos
        mejor_tipo_ruta(expreso, Hora, Distancia) :-
            hora_pico(Hora),
            Distancia > 10.
            
        % Regla: Elegir rutas alimentadoras para zonas periféricas
        mejor_tipo_ruta(alimentador, _, _) :-
            origen_periferia.
            
        % Regla: Para viajes cortos en horas valle, cualquier ruta funciona
        mejor_tipo_ruta(normal, Hora, Distancia) :-
            not(hora_pico(Hora)),
            Distancia < 5.
            
        % Definir horas pico
        hora_pico(Hora) :- Hora >= 6, Hora =< 9.  % Mañana
        hora_pico(Hora) :- Hora >= 16, Hora =< 19.  % Tarde
        
        % Reglas para evitar estaciones congestionadas en horas pico
        evitar_estacion(Estacion, Hora) :-
            estacion_congestionada(Estacion),
            hora_pico(Hora).
            
        % Reglas para preferir rutas con menos transbordos para personas especiales
        preferir_menos_transbordos :-
            usuario_movilidad_reducida.
            
        % Reglas para recomendar estaciones con servicios específicos
        estacion_recomendada(Estacion) :-
            tiene_servicio(Estacion, baño).
            
        estacion_recomendada(Estacion) :-
            tiene_servicio(Estacion, tienda).
        """
        
        # Cargar reglas en el motor de Prolog
        self.prolog.assertz(rules)
        
        # Agregar hechos sobre las estaciones y rutas
        for station_id, station_data in self.stations.items():
            # Marcar estaciones congestionadas
            if station_data.get('demand') == 'high':
                self.prolog.assertz(f"estacion_congestionada('{station_id}')")
            
            # Agregar servicios disponibles
            for service in station_data.get('services', []):
                self.prolog.assertz(f"tiene_servicio('{station_id}', {service})")
                
        # Identificar estaciones periféricas (portales y primeras estaciones)
        for route_id, route_data in self.routes.items():
            if 'portal' in route_data['stations'][0]:
                self.prolog.assertz(f"estacion_periferica('{route_data['stations'][0]}')")
    
    def train_time_prediction_model(self):
        """
        Entrena un modelo sencillo para predecir tiempos de viaje 
        según patrones de día/hora (por defecto usa un modelo base)
        """
        # En un sistema real, esto utilizaría datos históricos de tiempos de viaje
        # Aquí usamos un modelo simplificado basado en factores según hora del día
        
        # Factores de ajuste por hora (1.0 significa tiempo normal)
        self.time_factors = {
            # Horas pico con más congestión
            6: 1.4, 7: 1.5, 8: 1.6, 9: 1.4,  # Mañana
            16: 1.3, 17: 1.5, 18: 1.6, 19: 1.4,  # Tarde
            
            # Horas valle con menos tráfico
            10: 0.9, 11: 0.8, 14: 0.8, 15: 0.9,
            
            # Resto de horas: factor normal
        }
        
        # Factores por día de la semana (lunes = 0, domingo = 6)
        self.day_factors = {
            0: 1.1,  # Lunes
            4: 1.2,  # Viernes
            5: 0.7,  # Sábado
            6: 0.6,  # Domingo
        }
    
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
    
    def adjust_time_for_context(self, base_time, current_datetime=None):
        """
        Ajusta el tiempo de viaje según factores contextuales como
        hora del día, día de la semana, eventos especiales, etc.
        """
        if current_datetime is None:
            current_datetime = datetime.now()
        
        hour = current_datetime.hour
        day = current_datetime.weekday()  # 0 = lunes, 6 = domingo
        
        # Aplicar factores de hora
        time_factor = self.time_factors.get(hour, 1.0)
        
        # Aplicar factores de día
        day_factor = self.day_factors.get(day, 1.0)
        
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
    
    def a_star_search(self, start_id, end_id, context=None):
        """
        Implementa algoritmo A* para encontrar la mejor ruta entre dos estaciones
        utilizando una heurística basada en la distancia geográfica.
        
        Args:
            start_id: ID de la estación de origen
            end_id: ID de la estación de destino
            context: Diccionario con información contextual (hora, día, etc.)
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
                    self.is_station_crowded(neighbor, context['datetime']) and
                    neighbor != end_id):  # No saltear si es el destino
                    continue
                
                # Obtener datos de la conexión
                edge_data = self.graph.get_edge_data(current, neighbor)
                
                # Calcular costo del movimiento ajustado al contexto
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
        """Calcula el costo de moverse por una arista considerando el contexto"""
        route = edge_data['route']
        base_time = edge_data['time']
        
        # Ajustar por hora del día
        time_cost = self.adjust_time_for_context(base_time, context['datetime'])
        
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
    
    def is_station_crowded(self, station_id, current_datetime):
        """Determina si una estación está congestionada según la hora"""
        hour = current_datetime.hour
        
        # Verificar si está en la lista de estaciones congestionadas
        is_marked_as_crowded = False
        
        # Consultar la base de conocimiento
        for sol in self.prolog.query(f"estacion_congestionada('{station_id}')"):
            is_marked_as_crowded = True
            break
        
        # Si está marcada como congestionada, verificar si es hora pico
        if is_marked_as_crowded:
            is_peak_hour = False
            for sol in self.prolog.query(f"hora_pico({hour})"):
                is_peak_hour = True
                break
            
            return is_peak_hour
        
        return False
    
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
            
            # Tiempo ajustado al contexto
            time = self.adjust_time_for_context(edge_data['time'], context['datetime'])
            total_time += time
            
            # Crear paso de la ruta
            if route == 'transfer':
                step = {
                    'type': 'transfer',
                    'station': self.stations[from_station]['name'],
                    'time': time
                }
            else:
                # Si hay cambio de ruta, agregar transbordo implícito
                if current_route and route != current_route and route != 'transfer':
                    transfer_step = {
                        'type': 'route_change',
                        'station': self.stations[from_station]['name'],
                        'from_route': self.routes[current_route]['name'] if current_route in self.routes else current_route,
                        'to_route': self.routes[route]['name'],
                        'time': edge_data.get('frequency', 10) / 2  # Tiempo medio de espera
                    }
                    result.append(transfer_step)
                    total_time += transfer_step['time']
                
                step = {
                    'type': 'travel',
                    'from': self.stations[from_station]['name'],
                    'to': self.stations[to_station]['name'],
                    'route': self.routes[route]['name'],
                    'time': time
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
                'transfers': sum(1 for step in result if step['type'] in ['transfer', 'route_change'])
            }
            result.insert(0, summary)
        
        return result
    
    def analyze_and_optimize_route(self, origin, destination, preferences=None):
        """
        Analiza múltiples opciones de ruta y selecciona la mejor según las preferencias
        y el contexto actual.
        
        Args:
            origin: Nombre de la estación de origen
            destination: Nombre de la estación de destino
            preferences: Diccionario con preferencias del usuario
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
        
        return recommended_route
    
    def get_knowledge_based_recommendations(self, start_id, end_id, context):
        """Obtiene recomendaciones basadas en reglas lógicas"""
        recommendations = []
        
        # Determinar si se recomienda ruta expresa
        hour = context['datetime'].hour
        
        # Calcular distancia aproximada entre origen y destino
        lat1, lon1 = self.stations[start_id]['location']
        lat2, lon2 = self.stations[end_id]['location']
        distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 100  # aproximación
        
        # Consultar regla para tipo de ruta
        for sol in self.prolog.query(f"mejor_tipo_ruta(Tipo, {hour}, {distance})"):
            recommendations.append(f"Se recomienda usar rutas de tipo {sol['Tipo']} para este viaje")
        
        # Buscar estaciones recomendadas en el camino
        for station_id in self.stations:
            for sol in self.prolog.query(f"estacion_recomendada('{station_id}')"):
                recommendations.append(f"La estación {self.stations[station_id]['name']} ofrece servicios adicionales")
                break
        
        return recommendations
    
    def visualize_route(self, route):
        """
        Genera una visualización de la ruta en el grafo de Transmilenio.
        Útil para depuración y visualización de resultados.
        """
        if not route or 'error' in route:
            return None
        
        # Extraer nodos de la ruta
        nodes = []
        for step in route[1:]:  # Ignorar el resumen
            if step['type'] == 'travel':
                # Obtener IDs de estaciones
                from_id = next(id for id, data in self.stations.items() 
                              if data['name'] == step['from'])
                to_id = next(id for id, data in self.stations.items() 
                              if data['name'] == step['to'])
                
                # Añadir a la lista si no están ya
                if from_id not in nodes:
                    nodes.append(from_id)
                if to_id not in nodes:
                    nodes.append(to_id)
        
        # Crear un subgrafo con las estaciones de la ruta
        route_graph = self.graph.subgraph(nodes)
        
        # Obtener posiciones de los nodos
        pos = nx.get_node_attributes(route_graph, 'pos')
        
        # Colorear nodos y aristas para visualización
        plt.figure(figsize=(12, 8))
        
        # Dibujar todos los nodos del grafo en gris claro (fondo)
        all_pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw_networkx_nodes(self.graph, all_pos, node_size=30, node_color='lightgray', alpha=0.3)
        
        # Dibujar nodos de la ruta
        nx.draw_networkx_nodes(route_graph, pos, node_size=100, node_color='blue')
        
        # Destacar origen y destino
        origin_id = nodes[0]
        destination_id = nodes[-1]
        nx.draw_networkx_nodes(self.graph, all_pos, nodelist=[origin_id], 
                             node_size=200, node_color='green')
        nx.draw_networkx_nodes(self.graph, all_pos, nodelist=[destination_id], 
                             node_size=200, node_color='red')
        
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

# Ejemplo de uso del sistema
def main():
    # Crear instancia del router
    router = TransmilenioIARouter('transmilenio_data.json')
    
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
        
        # Mostrar recomendaciones basadas en reglas lógicas
        if 'recommendations' in summary:
            print("\nRecomendaciones adicionales:")
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
                print(f"{i}. Toma la ruta {step['route']} desde {step['from']} hasta {step['to']} ({step['time']} min)")
            elif step['type'] == 'transfer':
                print(f"{i}. Realiza transbordo en la estación {step['station']} ({step['time']} min)")
            elif step['type'] == 'route_change':
                print(f"{i}. En la estación {step['station']}, cambia de la ruta {step['from_route']} "
                      f"a la ruta {step['to_route']} ({step['time']} min)")
    
    # Generar y mostrar visualización de la ruta
    # En un entorno gráfico, esto mostraría la visualización
    # router.visualize_route(route)
    print("\nPara visualizar esta ruta gráficamente, ejecuta este código en un entorno que soporte matplotlib.")

if __