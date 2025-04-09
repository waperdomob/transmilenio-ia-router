"""
Script para demostrar el funcionamiento del sistema completo.
"""
from src.router_finder import RouteFinder

if __name__ == "__main__":
    print("=== Demostración del Sistema Inteligente de Rutas para Transmilenio ===")
    
    # Inicializar el buscador de rutas
    print("\nCargando modelos y datos del sistema...")
    route_finder = RouteFinder()
    
    # Datos de ejemplo para la demostración
    demo_origen = "Portal Norte"
    demo_destino = "Ricaurte"
    demo_dia = 2  # Miércoles
    demo_hora = 8  # 8:00 AM (hora pico)
    
    # Encontrar la mejor ruta
    resultado = route_finder.encontrar_mejor_ruta(
        demo_origen, 
        demo_destino, 
        demo_dia, 
        demo_hora,
        visualizar=True
    )
    
    # Mostrar resumen
    if 'error' in resultado:
        print(f"\nError al buscar la ruta: {resultado['error']}")
    else:
        print("\n=== Resumen del resultado ===")
        print(f"Origen: {resultado['origen']}")
        print(f"Destino: {resultado['destino']}")
        print(f"Día: {resultado['dia']} (0=Lunes, 6=Domingo)")
        print(f"Hora: {resultado['hora']}:00")
        print(f"\nMejor ruta encontrada:")
        print(f"{' -> '.join(resultado['mejor_ruta'])}")
        print(f"Calidad de la ruta: {resultado['calidad']:.2f}/10")
        
        print("\nComparativa de rutas:")
        for ruta_info in resultado['todas_rutas']:
            print(f"Ruta {ruta_info['id']}:")
            print(f"  Tiempo estimado: {ruta_info['tiempo_estimado']} minutos")
            print(f"  Congestión: {ruta_info['congestion_promedio']}")
            print(f"  Calidad: {ruta_info['calidad_predicha']}/10")
        
        if resultado['visualizacion']:
            print(f"\nVisualizacion guardada en: {resultado['visualizacion']}")
    
    # Demostración adicional: análisis de congestión
    print("\n=== Análisis de congestión ===")
    print("Niveles de congestión en estaciones clave:")
    
    estaciones_clave = ["Portal Norte", "Calle 100", "Héroes", "Calle 72", "Av. Jimenez", "Ricaurte"]
    for estacion in estaciones_clave:
        congestion = route_finder.predecir_congestion(estacion, demo_dia, demo_hora)
        nivel = "Alto" if congestion > 0.7 else "Medio" if congestion > 0.4 else "Bajo"
        print(f"- {estacion}: {congestion:.2f} (Nivel {nivel})")
    
    print("\nDemostración completada con éxito.")