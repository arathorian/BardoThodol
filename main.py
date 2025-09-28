#!/usr/bin/env python3
"""
Simulación principal del Bardo Thödol.
"""
import sys
import os
import json
import matplotlib.pyplot as plt

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

def guardar_resultados(results, estados, probabilidades, filename="resultados_simulacion.json"):
    """Guarda los resultados de la simulación en un archivo JSON."""
    datos = {
        "metadata": {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "total_ejecuciones": sum(results.values()),
            "estados_posibles": len(results)
        },
        "resultados_cuanticos": results,
        "estados_bardo": estados,
        "probabilidades": probabilidades.tolist() if hasattr(probabilidades, 'tolist') else probabilidades
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    print(f"💾 Resultados guardados en: {filename}")
    return filename

def main():
    """Simulación completa del Bardo Thödol con visualizaciones avanzadas."""

    print("🌌 SIMULACIÓN AVANZADA DEL BARDO THÖDOL")
    print("=" * 50)

    # Inicializar componentes
    state_manager = QuantumStateManager()
    simulator = BardoSimulator()
    visualizer = BardoVisualizer()

    # Crear y ejecutar circuito
    print("🎛️  Creando circuito cuántico...")
    circuit = state_manager.create_bardo_circuit()
    circuit.measure_all()

    print("⚡ Ejecutando simulación (1024 shots)...")
    results = simulator.simulate(circuit)

    # Mostrar resultados
    total = sum(results.values())
    print(f"\n📊 RESULTADOS (Total: {total} ejecuciones)")
    print("-" * 40)

    for state, count in sorted(results.items()):
        percentage = (count / total) * 100
        print(f"🔮 Estado {state}: {count:4d} veces ({percentage:5.1f}%)")

    # Guardar resultados
    archivo_resultados = guardar_resultados(
        results,
        state_manager.states,
        state_manager.probabilities
    )

    # Generar visualizaciones avanzadas
    print("\n🎨 Generando visualizaciones...")

    try:
        # Dashboard completo
        dashboard = visualizer.create_dashboard(
            results,
            state_manager.states,
            state_manager.probabilities.tolist(),
            circuit,
            "Dashboard del Bardo Thödol - Simulación Completa"
        )

        # Guardar visualizaciones
        visualizer.save_all_visualizations(
            results,
            state_manager.states,
            state_manager.probabilities.tolist(),
            circuit
        )

        # Mostrar gráficos
        plt.show()

    except Exception as e:
        print(f"⚠️  Error en visualizaciones: {e}")

    # Análisis avanzado de resultados
    try:
        from analysis.analyzer import BardoAnalyzer
        analyzer = BardoAnalyzer(archivo_resultados)
        print(analyzer.generar_reporte())
        analyzer.guardar_reporte("reporte_detallado.txt")
    except ImportError:
        print("⚠️  Módulo de análisis no disponible")
    except Exception as e:
        print(f"⚠️  Error en análisis: {e}")

    return results, archivo_resultados

if __name__ == "__main__":
    try:
        resultados, archivo = main()
        print(f"\n✅ Simulación completada exitosamente!")
        print(f"📈 Estados obtenidos: {len(resultados)}")
        print(f"💾 Archivo de resultados: {archivo}")
        print(f"📊 Resumen: {sum(resultados.values())} ejecuciones, {len(resultados)} estados únicos")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        sys.exit(1)
