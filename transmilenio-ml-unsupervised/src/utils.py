"""
Funciones de utilidad para el sistema.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from config import VISUALIZATIONS_DIR

def ensure_dir_exists(directory):
    """Asegura que un directorio exista."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_figure(fig, filename):
    """Guarda una figura en el directorio de visualizaciones."""
    ensure_dir_exists(VISUALIZATIONS_DIR)
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    return filepath

def create_transmilenio_graph():
    """Crea un grafo de la red de Transmilenio."""
    from config import ESTACIONES
    
    # Crear un grafo dirigido
    G = nx.DiGraph()
    
    # Agregar nodos (estaciones)
    for estacion in ESTACIONES:
        G.add_node(estacion)
    
    # Agregar conexiones principales (simplificadas)
    conexiones = [
        ("Portal Norte", "Toberin", 4), ("Toberin", "Cardio Infantil", 3), 
        ("Cardio Infantil", "Mazuren", 3), ("Mazuren", "Calle 142", 4),
        ("Calle 142", "Alcala", 3), ("Alcala", "Prado", 4), ("Prado", "Calle 127", 3),
        ("Calle 127", "Pepe Sierra", 4), ("Pepe Sierra", "Calle 106", 3), 
        ("Calle 106", "Calle 100", 4), ("Calle 100", "Virrey", 5), ("Virrey", "Calle 85", 3),
        ("Calle 85", "Heroes", 4), ("Heroes", "Calle 76", 3), ("Calle 76", "Calle 72", 3),
        ("Calle 72", "Flores", 3), ("Flores", "Calle 63", 3), ("Calle 63", "Calle 45", 4),
        ("Calle 45", "Marly", 3), ("Marly", "Calle 26", 5), ("Calle 26", "Profamilia", 4),
        ("Profamilia", "Av. Jimenez", 5), ("Av. Jimenez", "Tercer Milenio", 3),
        ("Tercer Milenio", "Comuneros", 3), ("Comuneros", "Santa Isabel", 4),
        ("Santa Isabel", "Ricaurte", 4), ("Calle 26", "Salitre El Greco", 5),
        ("Salitre El Greco", "El Tiempo", 3), ("El Tiempo", "Av. Rojas", 4),
        ("Av. Rojas", "Portal El Dorado", 6)
    ]
    
    for origen, destino, tiempo_base in conexiones:
        G.add_edge(origen, destino, tiempo_base=tiempo_base)
        # Agregar conexión en sentido contrario
        G.add_edge(destino, origen, tiempo_base=tiempo_base)
    
    return G

def plot_graph_with_route(G, route, title="Ruta en Transmilenio", filename="ruta.png"):
    """
    Visualiza una ruta en el grafo de Transmilenio.
    
    Args:
        G: Grafo de NetworkX
        route: Lista de estaciones que forman la ruta
        title: Título del gráfico
        filename: Nombre del archivo para guardar
        
    Returns:
        Ruta del archivo guardado
    """
    plt.figure(figsize=(12, 10))
    
    # Crear posiciones para el grafo (aproximación simplificada)
    pos = {}
    from config import ESTACIONES
    for i, estacion in enumerate(ESTACIONES):
        # Posición aproximada norte-sur
        pos[estacion] = (i % 3, -i / 3)
    
    # Dibujar todos los nodos y aristas
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.3, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=False)
    
    # Resaltar nodos y aristas de la ruta
    path_edges = list(zip(route[:-1], route[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=route, node_size=300, node_color='green')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='green', arrows=True)
    
    # Añadir etiquetas
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    
    # Guardar y cerrar
    return save_figure(plt.gcf(), filename)