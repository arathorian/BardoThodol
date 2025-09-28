#!/usr/bin/env python3
"""
Simulación principal del Bardo Thödol.
"""
import sys
import os
import matplotlib.pyplot as plt

# Configuración robusta del path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

# Añadir src al path de manera absoluta
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"🔍 Configurando entorno...")
print(f"📁 Directorio actual: {current_dir}")
print(f"📁 Path src: {src_path}")
print(f"📁 Existe src: {os.path.exists(src_path)}")

if os.path.exists(src_path):
    print("📁 Contenido de src:")
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isdir(item_path):
            print(f"  📂 {item}/")
            for subitem in os.listdir(item_path):
                if subitem.endswith('.py'):
                    print(f"    📄 {subitem}")
        elif item.endswith('.py'):
            print(f"  📄 {item}")

# Importaciones con verificación individual
try:
    from core.quantum_state import QuantumStateManager
    print("✅ QuantumStateManager importado correctamente")
except ImportError as e:
    print(f"❌ Error importando QuantumStateManager: {e}")
    # Crear una versión mínima temporal
    print("🔄 Creando versión mínima temporal...")
    from qiskit import QuantumCircuit
    import numpy as np

    class QuantumStateManager:
        def __init__(self, state_data=None):
            self.states = ["chonyid_bardo", "sidpa_bardo", "clear_light", "intermediate"]
            self.probabilities = np.array([0.3, 0.4, 0.2, 0.1])

        def create_bardo_circuit(self, num_qubits=2):
            qc = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                qc.h(i)
            return qc

try:
    from core.simulator import BardoSimulator
    print("✅ BardoSimulator importado correctamente")
except ImportError as e:
    print(f"❌ Error importando BardoSimulator: {e}")
    print("🔄 Creando versión mínima temporal...")
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import Aer

    class BardoSimulator:
        def __init__(self, backend='aer_simulator'):
            self.backend = Aer.get_backend(backend)

        def simulate(self, circuit, shots=1024):
            compiled = transpile(circuit, self.backend)
            job = self.backend.run(compiled, shots=shots)
            return job.result().get_counts()

try:
    from visualization.plotter import BardoVisualizer
    print("✅ BardoVisualizer importado correctamente")
except ImportError as e:
    print(f"❌ Error importando BardoVisualizer: {e}")
    print("🔄 Creando versión mínima temporal...")

    class BardoVisualizer:
        def plot_quantum_results(self, counts, title="Resultados"):
            from qiskit.visualization import plot_histogram
            return plot_histogram(counts, title=title)

        def plot_bardo_states(self, states, probabilities):
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(states, probabilities)
            return fig

def main():
    """Función principal de la simulación."""
    print("\n🌌 INICIANDO SIMULACIÓN DEL BARDO THÖDOL")
    print("=" * 50)

    try:
        # Inicializar componentes
        state_manager = QuantumStateManager()
        simulator = BardoSimulator()
        visualizer = BardoVisualizer()

        # Crear circuito
        print("🎛️  Creando circuito cuántico...")
        circuit = state_manager.create_bardo_circuit()
        circuit.measure_all()

        # Simular
        print("⚡ Ejecutando simulación (1024 shots)...")
        results = simulator.simulate(circuit)

        # Mostrar resultados
        total = sum(results.values())
        print(f"\n📊 RESULTADOS (Total: {total} ejecuciones)")
        print("-" * 40)

        for state, count in sorted(results.items()):
            percentage = (count / total) * 100
            print(f"🔮 Estado {state}: {count:4d} veces ({percentage:5.1f}%)")

        # Visualización
        print("\n🎨 Generando visualizaciones...")
        try:
            fig1 = visualizer.plot_quantum_results(results, "Simulación Bardo Thödol")
            fig2 = visualizer.plot_bardo_states(state_manager.states, state_manager.probabilities)
            plt.show()
        except Exception as e:
            print(f"⚠️  Visualización no disponible: {e}")

        return results

    except Exception as e:
        print(f"💥 Error durante la simulación: {e}")
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Simulación completada exitosamente!")
        print(f"📈 Estados obtenidos: {len(results)}")
    else:
        print(f"\n❌ La simulación falló")
