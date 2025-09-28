#!/usr/bin/env python3
"""
Verificación completa del proyecto BardoThodol.
"""
import os
import sys

def verificar_proyecto():
    print("🔍 VERIFICACIÓN COMPLETA DEL PROYECTO")
    print("=" * 50)

    # Verificar estructura de directorios
    requeridos = [
        ('src', True),
        ('src/__init__.py', True),
        ('src/core', True),
        ('src/core/__init__.py', True),
        ('src/core/quantum_state.py', True),
        ('src/core/simulator.py', True),
        ('src/visualization', True),
        ('src/visualization/__init__.py', True),
        ('src/visualization/plotter.py', True)
    ]

    problemas = []
    for path, obligatorio in requeridos:
        existe = os.path.exists(path)
        estado = "✅" if existe else "❌"
        print(f"{estado} {path}")

        if not existe and obligatorio:
            problemas.append(f"Falta: {path}")

    # Verificar imports
    print("\n🔗 VERIFICANDO IMPORTS...")
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)

    try:
        from core.quantum_state import QuantumStateManager
        print("✅ core.quantum_state → QuantumStateManager")
    except ImportError as e:
        problemas.append(f"Import error quantum_state: {e}")
        print("❌ core.quantum_state")

    try:
        from core.simulator import BardoSimulator
        print("✅ core.simulator → BardoSimulator")
    except ImportError as e:
        problemas.append(f"Import error simulator: {e}")
        print("❌ core.simulator")

    try:
        from visualization.plotter import BardoVisualizer
        print("✅ visualization.plotter → BardoVisualizer")
    except ImportError as e:
        problemas.append(f"Import error plotter: {e}")
        print("❌ visualization.plotter")

    # Resultado final
    if problemas:
        print(f"\n❌ Se encontraron {len(problemas)} problemas:")
        for problema in problemas:
            print(f"  • {problema}")
        return False
    else:
        print("\n🎉 ¡Proyecto verificado correctamente!")
        return True

if __name__ == "__main__":
    verificar_proyecto()
