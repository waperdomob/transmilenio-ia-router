"""
Módulo para encontrar y evaluar rutas entre estaciones.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import tensorflow as tf
from config import MODELS_DIR, ESTACIONES, ESTACIONES_ALTA_DEMANDA, ESTACIONES_MEDIA_DEMANDA
from src.utils import create_transmilenio_graph, plot_graph_with_route

class RouteFinder:
    """
    Clase para encontrar y evaluar rutas en el sistema Transmilenio.
    """
    
    def __init__(self):
        """Inicializa el buscador de rutas cargando modelos entrenados."""
        # Cargar modelos
        self.load_models()
        
        # Crear grafo de la red
        self.G = create_transmilenio_graph()
    
    def load_models(self):
        """Carga los modelos entrenados desde archivos."""
        # Modelo de tiempos de viaje
        modelo_tiempos_path = os.path.join(MODELS_DIR, 'modelo_tiempos.joblib')
        self.modelo_tiempos = joblib.load(modelo_tiempos_path)
        
        # Modelo de congestión
        modelo_congestion_path = os.path.join(MODELS_DIR, 'modelo_congestion.keras')
        preprocessor_congestion_path = os.path.join(MODELS_DIR, 'preprocessor_congestion.joblib')
        self.modelo_congestion = tf.keras.models.load_model(modelo_congestion_path)
        self.preprocessor_congestion = joblib.load(preprocessor_congestion_path)
        
        # Modelo de calidad de rutas
        modelo_rutas_path = os.path.join(MODELS_DIR, 'modelo_rutas.joblib')
        scaler_rutas_path = os.path.join(MODELS_DIR, 'scaler_rutas.joblib')
        self.modelo_rutas = joblib.load(modelo_rutas_path)
        self.scaler_rutas = joblib.load(scaler_rutas_path)
    
    def predecir_tiempo_viaje(self, origen, destino, dia, hora, clima='Normal'):
        """
        Predice el tiempo de viaje entre dos estaciones.
        
        Args:
            origen: Estación de origen
            destino: Estación de destino
            dia: Día de la semana (0-6)
            hora: Hora del día (0-23)
            clima: Condición climática
            
        Returns:
            Tiempo estimado de viaje en minutos
        """
        # Buscar tiempo base
        tiempo_base = None
        for u, v, data in self.G.edges(data=True):
            if u == origen and v == destino:
                tiempo_base = data['tiempo_base']
                break
        
        if tiempo_base is None:
            print(f"No hay conexión directa entre {origen} y {destino}")
            return None
        
        # Preparar datos
        es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
        es_fin_semana = 1 if dia >= 5 else 0
        
        # Determinar demanda
        if origen in ESTACIONES_ALTA_DEMANDA:
            demanda_origen = 2
        elif origen in ESTACIONES_MEDIA_DEMANDA:
            demanda_origen = 1
        else:
            demanda_origen = 0
            
        if destino in ESTACIONES_ALTA_DEMANDA:
            demanda_destino = 2
        elif destino in ESTACIONES_MEDIA_DEMANDA:
            demanda_destino = 1
        else:
            demanda_destino = 0
        
        # Crear muestra
        muestra = pd.DataFrame({
            'origen': [origen],
            'destino': [destino],
            'dia_semana': [dia],
            'hora_dia': [hora],
            'clima': [clima],
            'es_hora_pico': [es_hora_pico],
            'es_fin_semana': [es_fin_semana],
            'demanda_origen': [demanda_origen],
            'demanda_destino': [demanda_destino],
            'tiempo_base': [tiempo_base]
        })
        
        # Predecir
        tiempo_predicho = self.modelo_tiempos.predict(muestra)[0]
        return tiempo_predicho
    
    def predecir_congestion(self, estacion, dia, hora):
        """
        Predice el nivel de congestión en una estación.
        
        Args:
            estacion: Nombre de la estación
            dia: Día de la semana (0-6)
            hora: Hora del día (0-23)
            
        Returns:
            Nivel de congestión (0-1)
        """
        # Preparar datos
        es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
        es_fin_semana = 1 if dia >= 5 else 0
        
        # Determinar categoría de demanda
        if estacion in ESTACIONES_ALTA_DEMANDA:
            categoria_demanda = 2
        elif estacion in ESTACIONES_MEDIA_DEMANDA:
            categoria_demanda = 1
        else:
            categoria_demanda = 0
        
        # Crear muestra
        muestra = pd.DataFrame({
            'estacion': [estacion],
            'dia_semana': [dia],
            'hora_dia': [hora],
            'es_hora_pico': [es_hora_pico],
            'es_fin_semana': [es_fin_semana],
            'categoria_demanda': [categoria_demanda],
            'tiene_evento_especial': [0]  # Asumimos que no hay evento especial
        })
        
        # Preprocesar
        muestra_prep = self.preprocessor_congestion.transform(muestra)
        
        # Predecir
        congestion_predicha = self.modelo_congestion.predict(muestra_prep)[0][0]
        return congestion_predicha
    
    def evaluar_calidad_ruta(self, origen, destino, ruta, dia, hora):
        """
        Evalúa la calidad de una ruta propuesta.
        
        Args:
            origen: Estación de origen
            destino: Estación de destino
            ruta: Lista de estaciones que forman la ruta
            dia: Día de la semana (0-6)
            hora: Hora del día (0-23)
            
        Returns:
            Tupla con (calidad_predicha, tiempo_total, congestion_promedio)
        """
        # Preparar datos
        es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
        es_fin_semana = 1 if dia >= 5 else 0
        
        # Calcular métricas de la ruta
        num_estaciones = len(ruta)
        num_transbordos = num_estaciones - 2  # Simplificación
        
        # Calcular tiempo total (suma de segmentos)
        tiempo_total = 0
        for i in range(len(ruta) - 1):
            tiempo_segmento = self.predecir_tiempo_viaje(ruta[i], ruta[i+1], dia, hora)
            if tiempo_segmento is not None:
                tiempo_total += tiempo_segmento
        
        # Calcular congestión promedio
        congestion_total = 0
        for estacion in ruta:
            congestion_estacion = self.predecir_congestion(estacion, dia, hora)
            congestion_total += congestion_estacion
        
        congestion_promedio = congestion_total / num_estaciones
        
        # Buscar rutas alternativas (simplificado)
        num_alternativas = 2
        num_estaciones_alternativas = num_estaciones + 3
        
        # Crear muestra
        muestra = pd.DataFrame({
            'dia_semana': [dia],
            'hora_dia': [hora],
            'es_hora_pico': [es_hora_pico],
            'es_fin_semana': [es_fin_semana],
            'num_estaciones': [num_estaciones],
            'num_transbordos': [num_transbordos],
            'tiempo_total': [tiempo_total],
            'congestion_promedio': [congestion_promedio],
            'num_alternativas': [num_alternativas],
            'num_estaciones_alternativas': [num_estaciones_alternativas]
        })
        
        # Escalar
        muestra_scaled = self.scaler_rutas.transform(muestra)
        
        # Predecir
        calidad_predicha = self.modelo_rutas.predict(muestra_scaled)[0]
        return calidad_predicha, tiempo_total, congestion_promedio
    
    def encontrar_mejor_ruta(self, origen, destino, dia, hora, visualizar=False):
        """
        Encuentra la mejor ruta entre dos estaciones usando modelos entrenados.
        
        Args:
            origen: Estación de origen
            destino: Estación de destino
            dia: Día de la semana (0-6)
            hora: Hora del día (0-23)
            visualizar: Si es True, genera una visualización de la ruta
            
        Returns:
            Diccionario con información de la mejor ruta
        """
        print(f"\nBuscando mejor ruta de {origen} a {destino} para día {dia} y hora {hora}...")
        
        # Aplicar algoritmo para encontrar caminos
        mejor_ruta = None
        mejor_calidad = -1
        resultados_rutas = []
        
        # Encontrar caminos posibles (simplificado)
        try:
            # Encontrar hasta 5 caminos más cortos
            rutas_posibles = list(nx.shortest_simple_paths(self.G, origen, destino, weight='tiempo_base'))[:5]
            
            print(f"Se encontraron {len(rutas_posibles)} rutas posibles")
            
            for i, ruta in enumerate(rutas_posibles):
                # Evaluar calidad
                calidad, tiempo, congestion = self.evaluar_calidad_ruta(origen, destino, ruta, dia, hora)
                
                resultado_ruta = {
                    'id': i+1,
                    'ruta': ruta,
                    'tiempo_estimado': round(tiempo, 2),
                    'congestion_promedio': round(congestion, 2),
                    'calidad_predicha': round(calidad, 2)
                }
                
                resultados_rutas.append(resultado_ruta)
                
                print(f"Ruta {i+1}: {' -> '.join(ruta)}")
                print(f"  Tiempo estimado: {tiempo:.2f} minutos")
                print(f"  Congestión promedio: {congestion:.2f}")
                print(f"  Calidad predicha: {calidad:.2f}/10")
                
                if calidad > mejor_calidad:
                    mejor_calidad = calidad
                    mejor_ruta = ruta
            
            print(f"\nMejor ruta recomendada:")
            print(f"{' -> '.join(mejor_ruta)}")
            print(f"Calidad: {mejor_calidad:.2f}/10")
            
            # Visualizar la ruta si se solicita
            vis_path = None
            if visualizar and mejor_ruta:
                titulo = f"Mejor ruta de {origen} a {destino}"
                vis_path = plot_graph_with_route(self.G, mejor_ruta, title=titulo, filename="mejor_ruta.png")
            
            return {
                'origen': origen,
                'destino': destino,
                'dia': dia,
                'hora': hora,
                'mejor_ruta': mejor_ruta,
                'calidad': mejor_calidad,
                'todas_rutas': resultados_rutas,
                'visualizacion': vis_path
            }
            
        except Exception as e:
            print(f"Error al buscar rutas: {e}")
            return {
                'error': str(e),
                'origen': origen,
                'destino': destino
            }