# Simulacion_avanzada.py - Versión corregida
import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from quantum_system import KarmicDynamics, BardoQuantumSystem, QuantumMetrics
    from visualization import QuantumVisualizer
    print("✓ Módulos importados correctamente")
except ImportError as e:
    print(f"✗ Error de importación: {e}")
    print("Asegúrate de que los archivos estén en la carpeta 'src/'")
    sys.exit(1)

# El resto del código de Simulacion_avanzada.py aquí...
def main():
    print("Ejecutando simulación avanzada del Bardo Thödol...")

    # Tu código existente...
import numpy as np
from src.quantum_system import KarmicDynamics

# Configurar dinámica kármica personalizada
karma_dynamics = KarmicDynamics(
    dimensions=3,
    decoherence_rates=[0.01, 0.02, 0.015],
    attention_parameters={'type': 'adaptive', 'sensitivity': 0.8}
)

# Simular evolución temporal
evolution_data = karma_dynamics.compute_evolution(
    initial_state='void',
    time_range=(0, 4*np.pi),
    metrics=['coherence', 'purity', 'entanglement_entropy']
)
    bardo_system = BardoQuantumSystem()
    results = bardo_system.simulate_bardo_transition()

    # Visualizar resultados
    visualizer = QuantumVisualizer()
    visualizer.create_comprehensive_visualization(results)

    print("Simulación avanzada completada!")

if __name__ == "__main__":
    main()