"""
Módulo para generar datasets sintéticos del sistema Transmilenio.
"""
import os
import pandas as pd
import numpy as np
import random
import networkx as nx
from config import (
    DATA_DIR, ESTACIONES, ESTACIONES_ALTA_DEMANDA, ESTACIONES_MEDIA_DEMANDA,
    ESTACIONES_BAJA_DEMANDA, DIAS_SEMANA, HORAS_DIA, CONDICIONES_CLIMA, NUM_ROUTE_SAMPLES
)
from src.utils import ensure_dir_exists, create_transmilenio_graph

def generate_tiempo_viaje_dataset(G):
    """
    Genera dataset de tiempos de viaje entre estaciones.
    
    Args:
        G: Grafo de NetworkX con la red de Transmilenio
        
    Returns:
        DataFrame con los datos generados
    """
    print("Generando dataset de tiempos de viaje...")
    
    data_tiempos = []
    
    # Para cada par de estaciones conectadas
    for origen, destino, datos in G.edges(data=True):
        tiempo_base = datos['tiempo_base']
        
        # Generar múltiples muestras con diferentes condiciones
        for dia in DIAS_SEMANA:
            for hora in HORAS_DIA:
                for clima in CONDICIONES_CLIMA:
                    # Factores que afectan el tiempo
                    es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
                    es_fin_semana = 1 if dia >= 5 else 0
                    
                    # Nivel de demanda de las estaciones
                    if origen in ESTACIONES_ALTA_DEMANDA:
                        demanda_origen = 2  # Alta
                    elif origen in ESTACIONES_MEDIA_DEMANDA:
                        demanda_origen = 1  # Media
                    else:
                        demanda_origen = 0  # Baja
                    
                    if destino in ESTACIONES_ALTA_DEMANDA:
                        demanda_destino = 2  # Alta
                    elif destino in ESTACIONES_MEDIA_DEMANDA:
                        demanda_destino = 1  # Media
                    else:
                        demanda_destino = 0  # Baja
                    
                    # Factor climático
                    if clima == "Normal":
                        factor_clima = 1.0
                    elif clima == "Lluvia":
                        factor_clima = 1.2
                    else:  # Lluvia fuerte
                        factor_clima = 1.5
                    
                    # Calcular tiempo con factores y algo de aleatoriedad
                    tiempo_ajustado = tiempo_base
                    
                    # Ajustar por hora pico
                    if es_hora_pico:
                        tiempo_ajustado *= 1.3 + (0.2 * random.random())
                    
                    # Ajustar por fin de semana
                    if es_fin_semana:
                        tiempo_ajustado *= 0.8 + (0.1 * random.random())
                    
                    # Ajustar por demanda
                    tiempo_ajustado *= 1.0 + (0.1 * demanda_origen) + (0.05 * demanda_destino)
                    
                    # Ajustar por clima
                    tiempo_ajustado *= factor_clima
                    
                    # Agregar algo de ruido aleatorio (±10%)
                    tiempo_final = tiempo_ajustado * (0.9 + 0.2 * random.random())
                    
                    # Agregar a los datos
                    data_tiempos.append({
                        'origen': origen,
                        'destino': destino,
                        'dia_semana': dia,
                        'hora_dia': hora,
                        'clima': clima,
                        'es_hora_pico': es_hora_pico,
                        'es_fin_semana': es_fin_semana,
                        'demanda_origen': demanda_origen,
                        'demanda_destino': demanda_destino,
                        'tiempo_base': tiempo_base,
                        'tiempo_viaje': round(tiempo_final, 2)
                    })
    
    # Crear DataFrame
    df_tiempos = pd.DataFrame(data_tiempos)
    return df_tiempos

def generate_congestion_dataset():
    """
    Genera dataset de niveles de congestión en estaciones.
    
    Returns:
        DataFrame con los datos generados
    """
    print("Generando dataset de congestión...")
    
    data_congestion = []
    
    # Para cada estación
    for estacion in ESTACIONES:
        # Categoría base de demanda
        if estacion in ESTACIONES_ALTA_DEMANDA:
            categoria_demanda = 2  # Alta
        elif estacion in ESTACIONES_MEDIA_DEMANDA:
            categoria_demanda = 1  # Media
        else:
            categoria_demanda = 0  # Baja
        
        # Generar datos para diferentes días y horas
        for dia in DIAS_SEMANA:
            for hora in HORAS_DIA:
                # Factores base
                es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
                es_fin_semana = 1 if dia >= 5 else 0
                
                # Factor base de congestión según categoría
                if categoria_demanda == 2:  # Alta demanda
                    base_congestion = 0.7
                elif categoria_demanda == 1:  # Media demanda
                    base_congestion = 0.5
                else:  # Baja demanda
                    base_congestion = 0.3
                
                # Ajustar por hora pico
                if es_hora_pico:
                    base_congestion += 0.2
                
                # Ajustar por fin de semana
                if es_fin_semana:
                    base_congestion -= 0.3
                
                # Ajustar por hora específica (mañana vs tarde)
                if 7 <= hora <= 9:  # Pico mañana
                    base_congestion += 0.1
                elif 17 <= hora <= 19:  # Pico tarde
                    base_congestion += 0.15
                
                # Eventos especiales aleatorios que afectan la congestión
                tiene_evento_especial = 1 if random.random() < 0.05 else 0
                if tiene_evento_especial:
                    base_congestion += 0.2
                
                # Asegurar que esté entre 0 y 1
                nivel_congestion = max(0, min(1, base_congestion + (0.1 * random.random() - 0.05)))
                
                # Agregar a los datos
                data_congestion.append({
                    'estacion': estacion,
                    'dia_semana': dia,
                    'hora_dia': hora,
                    'es_hora_pico': es_hora_pico,
                    'es_fin_semana': es_fin_semana,
                    'categoria_demanda': categoria_demanda,
                    'tiene_evento_especial': tiene_evento_especial,
                    'nivel_congestion': round(nivel_congestion, 3)
                })
    
    # Crear DataFrame
    df_congestion = pd.DataFrame(data_congestion)
    return df_congestion

def encontrar_rutas(grafo, origen, destino, hora, dia, clima):
    """
    Encuentra caminos óptimos entre dos estaciones.
    
    Args:
        grafo: Grafo de NetworkX
        origen: Estación de origen
        destino: Estación de destino
        hora: Hora del día
        dia: Día de la semana
        clima: Condición climática
        
    Returns:
        Tupla con (mejor_ruta, tiempo_total, rutas_alternativas)
    """
    # Calcular pesos de aristas según condiciones
    es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
    es_fin_semana = 1 if dia >= 5 else 0
    
    # Factor climático
    if clima == "Normal":
        factor_clima = 1.0
    elif clima == "Lluvia":
        factor_clima = 1.2
    else:  # Lluvia fuerte
        factor_clima = 1.5
    
    # Crear nuevo grafo con pesos ajustados
    G_temp = nx.DiGraph()
    
    for u, v, data in grafo.edges(data=True):
        tiempo_base = data['tiempo_base']
        
        # Calcular tiempo ajustado similar al dataset de tiempos
        tiempo_ajustado = tiempo_base
        
        # Ajustar por hora pico
        if es_hora_pico:
            tiempo_ajustado *= 1.3
        
        # Ajustar por fin de semana
        if es_fin_semana:
            tiempo_ajustado *= 0.8
        
        # Ajustar por demanda de origen
        if u in ESTACIONES_ALTA_DEMANDA:
            tiempo_ajustado *= 1.2
        elif u in ESTACIONES_MEDIA_DEMANDA:
            tiempo_ajustado *= 1.1
        
        # Ajustar por clima
        tiempo_ajustado *= factor_clima
        
        G_temp.add_edge(u, v, weight=tiempo_ajustado)
    
    # Encontrar camino más corto
    try:
        ruta_mas_corta = nx.shortest_path(G_temp, origen, destino, weight='weight')
        
        # Calcular tiempo total
        tiempo_total = sum(G_temp[u][v]['weight'] for u, v in zip(ruta_mas_corta[:-1], ruta_mas_corta[1:]))
        
        # Encontrar rutas alternativas (k=2)
        try:
            rutas_alternativas = list(nx.shortest_simple_paths(G_temp, origen, destino, weight='weight'))[:3]
        except:
            rutas_alternativas = [ruta_mas_corta]
        
        return ruta_mas_corta, tiempo_total, rutas_alternativas
    except:
        return None, float('inf'), []

def generate_rutas_dataset(G):
    """
    Genera dataset de rutas óptimas entre estaciones.
    
    Args:
        G: Grafo de NetworkX con la red de Transmilenio
        
    Returns:
        DataFrame con los datos generados
    """
    print("Generando dataset de rutas óptimas...")
    
    data_rutas = []
    
    # Generar rutas para pares aleatorios de estaciones
    pares_estaciones = []
    
    # Asegurar que se cubran todas las estaciones
    for estacion in ESTACIONES:
        destinos_posibles = [dest for dest in ESTACIONES if dest != estacion]
        pares_estaciones.append((estacion, random.choice(destinos_posibles)))
    
    # Agregar pares aleatorios adicionales
    while len(pares_estaciones) < NUM_ROUTE_SAMPLES:
        origen = random.choice(ESTACIONES)
        destino = random.choice([e for e in ESTACIONES if e != origen])
        pares_estaciones.append((origen, destino))
    
    # Limitar al número de muestras deseado
    pares_estaciones = pares_estaciones[:NUM_ROUTE_SAMPLES]
    
    for origen, destino in pares_estaciones:
        # Condiciones aleatorias
        dia = random.choice(DIAS_SEMANA)
        hora = random.choice(HORAS_DIA)
        clima = random.choice(CONDICIONES_CLIMA)
        
        # Encontrar rutas
        mejor_ruta, tiempo_total, alternativas = encontrar_rutas(G, origen, destino, hora, dia, clima)
        
        if mejor_ruta:
            num_transbordos = len(mejor_ruta) - 2  # Aproximación simplificada de transbordos
            
            # Características de la ruta
            es_hora_pico = 1 if (6 <= hora <= 9 or 16 <= hora <= 19) else 0
            es_fin_semana = 1 if dia >= 5 else 0
            
            # Nivel de congestión en la ruta (promedio de estaciones)
            congestiones = []
            for estacion in mejor_ruta:
                if estacion in ESTACIONES_ALTA_DEMANDA:
                    base = 0.7
                elif estacion in ESTACIONES_MEDIA_DEMANDA:
                    base = 0.5
                else:
                    base = 0.3
                    
                # Ajustar por factores
                if es_hora_pico:
                    base += 0.2
                if es_fin_semana:
                    base -= 0.3
                    
                congestiones.append(max(0, min(1, base)))
            
            congestion_promedio = sum(congestiones) / len(congestiones)
            
            # Evaluar calidad de la ruta (0 a 10)
            # Fórmula ponderada: prioriza tiempo, penaliza transbordos y congestión
            calidad_base = 10
            calidad_base -= (tiempo_total / 20)  # Penalizar por tiempo
            calidad_base -= (num_transbordos * 1.5)  # Penalizar por transbordos
            calidad_base -= (congestion_promedio * 3)  # Penalizar por congestión
            
            calidad_ruta = max(0, min(10, calidad_base))
            
            # Número de estaciones alternativas consideradas en rutas alternativas
            estaciones_alternativas = set()
            for ruta in alternativas:
                for estacion in ruta:
                    estaciones_alternativas.add(estacion)
            
            # Agregar a los datos
            data_rutas.append({
                'origen': origen,
                'destino': destino,
                'dia_semana': dia,
                'hora_dia': hora,
                'clima': clima,
                'es_hora_pico': es_hora_pico,
                'es_fin_semana': es_fin_semana,
                'mejor_ruta': '|'.join(mejor_ruta),  # Guardar como string separado por |
                'num_estaciones': len(mejor_ruta),
                'num_transbordos': num_transbordos,
                'tiempo_total': round(tiempo_total, 2),
                'congestion_promedio': round(congestion_promedio, 3),
                'num_alternativas': len(alternativas),
                'num_estaciones_alternativas': len(estaciones_alternativas),
                'calidad_ruta': round(calidad_ruta, 2)
            })
    
    # Crear DataFrame
    df_rutas = pd.DataFrame(data_rutas)
    return df_rutas

def generate_all_datasets():
    """
    Genera todos los datasets y los guarda en archivos CSV.
    
    Returns:
        Diccionario con los dataframes generados
    """
    # Asegurar que el directorio de datos exista
    ensure_dir_exists(DATA_DIR)
    
    # Crear grafo de la red
    G = create_transmilenio_graph()
    
    # Generar datasets
    df_tiempos = generate_tiempo_viaje_dataset(G)
    df_congestion = generate_congestion_dataset()
    df_rutas = generate_rutas_dataset(G)
    
    # Guardar datasets
    df_tiempos.to_csv(os.path.join(DATA_DIR, 'dataset_tiempos_viaje.csv'), index=False)
    df_congestion.to_csv(os.path.join(DATA_DIR, 'dataset_congestion.csv'), index=False)
    df_rutas.to_csv(os.path.join(DATA_DIR, 'dataset_rutas_optimas.csv'), index=False)
    
    print(f"Dataset de tiempos creado con {len(df_tiempos)} registros.")
    print(f"Dataset de congestión creado con {len(df_congestion)} registros.")
    print(f"Dataset de rutas óptimas creado con {len(df_rutas)} registros.")
    
    return {
        'tiempos': df_tiempos,
        'congestion': df_congestion,
        'rutas': df_rutas
    }