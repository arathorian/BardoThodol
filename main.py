#!/usr/bin/env python3
"""
Simulación principal del Bardo Thödol.
Módulo principal que coordina la simulación cuántica de los estados del Bardo.
"""

import sys
import os
import matplotlib.pyplot as plt  # Añadido para mostrar gráficos

# Configuración del path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

try:
    from core.quantum_state import QuantumStateManager
    from core.simulator import BardoSimulator
    from visualization.plotter import BardoVisualizer
    print("✅ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

def main():
    """Simulación completa del Bardo Thödol con visualizaciones avanzadas."""

    print("🌌 SIMULACIÓN AVANZADA DEL BARDO THÖDOL")
    print("=" * 50)

    # Inicializar componentes
    state_manager = QuantumStateManager()
    simulator = BardoSimulator()
    visualizer = BardoVisualizer()

    # Crear y ejecutar circuito
    print("Creando circuito cuántico...")
    circuit = state_manager.create_bardo_circuit()
    circuit.measure_all()

    print("Ejecutando simulación (1024 shots)...")
    results = simulator.simulate(circuit)

    # Mostrar resultados
    total = sum(results.values())
    print(f"\n🎯 RESULTADOS (Total: {total} ejecuciones)")
    print("-" * 40)

    for state, count in sorted(results.items()):
        percentage = (count / total) * 100
        print(f"Estado {state}: {count:4d} veces ({percentage:5.1f}%)")

    # Generar visualizaciones avanzadas
    print("\n📊 Generando visualizaciones avanzadas...")

    # Dashboard completo
    dashboard = visualizer.create_dashboard(
        results,
        state_manager.states,
        state_manager.probabilities.tolist(),
        circuit,
        "Dashboard del Bardo Thödol - Simulación Completa"
    )

    # Guardar todas las visualizaciones
    visualizer.save_all_visualizations(
        results,
        state_manager.states,
        state_manager.probabilities.tolist(),
        circuit
    )

    # Mostrar gráficos
    plt.show()

    return results

if __name__ == "__main__":
    results = main()
    print(f"\n✅ Simulación completada. Visualizaciones guardadas en 'results/'")
