"""
Script para generar los datasets sintéticos.
"""
from src.data_generation import generate_all_datasets
from src.data_preprocessing import explore_datasets

if __name__ == "__main__":
    print("=== Generación de datasets para el Sistema de Rutas Transmilenio ===")
    
    # Generar datasets
    datasets = generate_all_datasets()
    
    # Explorar y visualizar
    results = explore_datasets(datasets)
    
    print("\nGeneración de datasets completada con éxito.")
    print(f"Se generaron visualizaciones en los archivos:")
    for name, path in results['visualizations'].items():
        print(f"- {name}: {path}")