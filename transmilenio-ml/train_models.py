"""
Script para entrenar y evaluar los modelos de aprendizaje automático.
"""
from src.data_preprocessing import load_datasets, prepare_tiempo_viaje_data, prepare_congestion_data, prepare_rutas_data
from src.model_training import train_tiempo_viaje_model, train_congestion_model, train_rutas_model
from src.model_evaluation import evaluate_tiempo_viaje_model, evaluate_congestion_model, evaluate_rutas_model

if __name__ == "__main__":
    print("=== Entrenamiento de modelos para el Sistema de Rutas Transmilenio ===")
    
    # Cargar datasets
    print("\nCargando datasets...")
    datasets = load_datasets()
    
    # Preparar datos para cada modelo
    print("\nPreparando datos...")
    data_tiempos = prepare_tiempo_viaje_data(datasets['tiempos'])
    data_congestion = prepare_congestion_data(datasets['congestion'])
    data_rutas = prepare_rutas_data(datasets['rutas'])
    
    # Entrenar modelos
    print("\nEntrenando modelos...")
    modelo_tiempos = train_tiempo_viaje_model(data_tiempos)
    modelo_congestion, history_congestion = train_congestion_model(data_congestion)
    modelo_rutas = train_rutas_model(data_rutas)
    
    # Evaluar modelos
    print("\nEvaluando modelos...")
    eval_tiempos = evaluate_tiempo_viaje_model(modelo_tiempos, data_tiempos)
    eval_congestion = evaluate_congestion_model(modelo_congestion, data_congestion['preprocessor'], data_congestion)
    eval_rutas = evaluate_rutas_model(modelo_rutas, data_rutas['scaler'], data_rutas)
    
    print("\nEntrenamiento y evaluación completados con éxito.")
    print("Se han guardado los modelos y las visualizaciones de evaluación.")