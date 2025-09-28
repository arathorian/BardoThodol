#!/usr/bin/env python3
"""
Script de prueba para el módulo de visualización reconstruido.
"""

import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from visualization.plotter import BardoVisualizer, quick_visualize
    from qiskit import QuantumCircuit
    print("✅ Módulo de visualización importado correctamente")
except ImportError as e:
    print(f"❌ Error importando el módulo de visualización: {e}")
    sys.exit(1)

def test_visualization():
    """Prueba todas las funcionalidades de visualización."""
    print("🧪 Probando visualizaciones del Bardo...")

    # Datos de ejemplo
    counts = {'00': 256, '01': 245, '10': 268, '11': 255}
    states = ['Chonyid Bardo', 'Sidpa Bardo', 'Luz Clara', 'Intermedio']
    probabilities = [0.25, 0.35, 0.25, 0.15]

    # Crear visualizador
    visualizer = BardoVisualizer()

    # Probar cada método
    print("1. Probando visualización de resultados cuánticos...")
    fig1 = visualizer.plot_quantum_results(counts)

    print("2. Probando gráfico de estados del Bardo...")
    fig2 = visualizer.plot_bardo_states(states, probabilities)

    print("3. Probando visualización rápida...")
    fig3 = quick_visualize(counts, "Prueba Rápida")

    print("4. Probando guardado de visualizaciones...")
    visualizer.save_all_visualizations(counts, states, probabilities, QuantumCircuit(2))

    print("✅ Todas las pruebas completadas exitosamente")

    # Mostrar gráficos
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    test_visualization()
