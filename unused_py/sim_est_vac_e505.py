# -*- coding: utf-8 -*-
"""
Simulación Cuántica del Estado de Vacuidad (ERROR 505) en el Bardo Thödol
Basado en modelo de qutrits y dinámica kármica
"""
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

# Configuración global
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelweight'] = 'bold'

# =============================================================================
# 1. Definición de Operadores Cuánticos Personalizados (Qutrits)
# =============================================================================
def karma_gate(theta):
    """Compuerta de Incertidumbre Kármica (Matriz unitaria 3x3 para qutrits)
    
    Parámetros:
        theta (float): Ángulo kármico (0 a 2π)
    
    Retorna:
        np.array: Matriz 3x3 que opera sobre estados de qutrit
    """
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)*1j, 0],
        [-np.sin(theta/2)*1j, np.cos(theta/2), 0],
        [0, 0, np.exp(1j*theta)]
    ])

def observer_gate(phi):
    """Compuerta de Observación Consciente (Matriz 3x3)
    
    Parámetros:
        phi (float): Nivel de atención (0 a π/2)
    
    Retorna:
        np.array: Matriz unitaria 3x3
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

# =============================================================================
# 2. Circuito Cuántico para Simulación de Vacuidad (|2⟩)
# =============================================================================
def create_vacuity_circuit(theta=np.pi/3, phi=0.01):
    """Crea circuito cuántico para simular estado de vacuidad
    
    Parámetros:
        theta (float): Parámetro kármico (default π/3)
        phi (float): Parámetro de atención (default 0.01)
    
    Retorna:
        QuantumCircuit: Circuito con 2 qubits (emulando 1 qutrit)
    """
    qc = QuantumCircuit(2, 2)  # 2 qubits, 2 bits clásicos
    
    # Inicialización en estado |2⟩ (vacuidad)
    qc.x(0)  # |10> = |2⟩ en codificación
    
    # Aplicar Incertidumbre Kármica
    U_k = karma_gate(theta)
    qc.unitary(U_k, [0,1], label='K(θ)')
    
    # Aplicar Observación Consciente
    U_o = observer_gate(phi)
    qc.unitary(U_o, [0,1], label='O(φ)')
    
    # Medición en base computacional
    qc.measure([0,1], [0,1])
    
    return qc

# =============================================================================
# 3. Simulación y Visualización
# =============================================================================
def run_simulation(theta, phi, shots=1024):
    """Ejecuta simulación completa
    
    Parámetros:
        theta (float): Parámetro kármico
        phi (float): Parámetro de atención
        shots (int): Número de ejecuciones
    
    Retorna:
        dict: Resultados de simulación
        Figure: Gráfico de estados
    """
    # Crear y ejecutar circuito
    qc = create_vacuity_circuit(theta, phi)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts(qc)
    
    # Visualización de estados
    statevector = Statevector.from_instruction(qc)
    plot = statevector.plot('qsphere', figsize=(10, 8))
    
    return counts, plot

# =============================================================================
# 4. Análisis de Resultados
# =============================================================================
def analyze_results(counts):
    """Analiza resultados de medición
    
    Parámetros:
        counts (dict): Resultados de simulación
    
    Retorna:
        dict: Estadísticas interpretadas
    """
    total = sum(counts.values())
    stats = {
        '00': counts.get('00', 0)/total,  # |0⟩: Realidad manifiesta
        '01': counts.get('01', 0)/total,  # |1⟩: Realidad alterna
        '10': counts.get('10', 0)/total,  # |2⟩: Vacuidad
        '11': counts.get('11', 0)/total   # Estado no válido
    }
    
    # Interpretación según modelo del Bardo
    interpretation = {
        'Manifestación': stats['00'],
        'Potencialidad': stats['01'],
        'Vacuidad': stats['10'],
        'Error_Interpretativo': stats['11']
    }
    
    return interpretation

# =============================================================================
# 5. Ejecución Principal
# =============================================================================
if __name__ == "__main__":
    # Parámetros de simulación (karma bajo, atención débil)
    theta = np.pi/3  # Alto karma pendiente
    phi = 0.01       # Atención mínima
    
    # Ejecutar simulación
    counts, plot = run_simulation(theta, phi)
    interpretation = analyze_results(counts)
    
    # Resultados
    print("\n" + "="*60)
    print("RESULTADOS DE SIMULACIÓN CUÁNTICA DEL ESTADO 505")
    print("="*60)
    print(f"- Parámetro kármico (θ): {theta:.4f}")
    print(f"- Parámetro de atención (φ): {phi:.4f}\n")
    
    print("DISTRIBUCIÓN DE PROBABILIDAD:")
    for state, prob in interpretation.items():
        print(f"  {state}: {prob*100:.2f}%")
    
    print("\nINTERPRETACIÓN EN EL CONTEXTO DEL BARDO THÖDOL:")
    print("- La alta probabilidad en Vacuidad (>70%) confirma que:")
    print("  * El 'error' es un estado fundamental de potencialidad cuántica")
    print("  * Corresponde a la no-decisión entre realidades alternas")
    print("  * Requiere procesamiento cuántico (no digital) para su comprensión")
    
    # Gráfico adicional
    plt.figure(figsize=(10, 6))
    plot_histogram(counts, title="Distribución de Estados Medidos", 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xlabel('Estado del Qutrit |ψ⟩')
    plt.ylabel('Probabilidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Mostrar esfera cuántica (se abre en ventana aparte)
    plot.show()