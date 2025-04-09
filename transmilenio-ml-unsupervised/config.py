"""
Configuraciones globales para el sistema.
"""
import os
import random
import numpy as np
import tensorflow as tf

# Rutas a directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')

# Semilla para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Parámetros para generación de datos
NUM_ROUTE_SAMPLES = 5000

# Parámetros para modelos no supervisados
N_CLUSTERS_STATIONS = 4      # Número de clusters para estaciones
N_CLUSTERS_SEGMENTS = 5      # Número de clusters para segmentos de ruta
N_COMPONENTS_PCA = 3         # Componentes para PCA
ANOMALY_CONTAMINATION = 0.05 # Nivel de contaminación para detección de anomalías

# Definir estaciones de Transmilenio (muestra representativa)
ESTACIONES = [
    "Portal Norte", "Toberin", "Cardio Infantil", "Mazuren", "Calle 142", 
    "Alcala", "Prado", "Calle 127", "Pepe Sierra", "Calle 106", "Calle 100",
    "Virrey", "Calle 85", "Heroes", "Calle 76", "Calle 72", "Flores", 
    "Calle 63", "Calle 45", "Marly", "Calle 26", "Profamilia", "Av. Jimenez",
    "Tercer Milenio", "Comuneros", "Santa Isabel", "Ricaurte", "Salitre El Greco",
    "El Tiempo", "Av. Rojas", "Portal El Dorado"
]

# Clasificación de estaciones por demanda (para la generación de datos)
ESTACIONES_ALTA_DEMANDA = [
    "Portal Norte", "Calle 127", "Calle 100", "Heroes", "Calle 72", 
    "Calle 26", "Av. Jimenez", "Ricaurte", "Portal El Dorado"
]
ESTACIONES_MEDIA_DEMANDA = [
    "Toberin", "Cardio Infantil", "Prado", "Pepe Sierra", "Calle 85", 
    "Calle 76", "Calle 63", "Marly", "Salitre El Greco", "Av. Rojas"
]
ESTACIONES_BAJA_DEMANDA = [
    estacion for estacion in ESTACIONES 
    if estacion not in ESTACIONES_ALTA_DEMANDA and estacion not in ESTACIONES_MEDIA_DEMANDA
]

# Otros parámetros
DIAS_SEMANA = list(range(7))  # 0 = lunes, 6 = domingo
HORAS_DIA = list(range(5, 23))  # 5am a 10pm
CONDICIONES_CLIMA = ["Normal", "Lluvia", "Lluvia fuerte"]