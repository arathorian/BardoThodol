# -*- coding: utf-8 -*-
"""
Modelo de Matrices Duales para Estados |11> Inversos (Simetría CPT en Bardo)
"""
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator, DensityMatrix
from qiskit.visualization import plot_state_hinton

# =============================================================================
# 1. Definición de Matrices Duales (Operadores CPT-Acumulados)
# =============================================================================
def dual_matrix_system(alpha, beta):
    """Crea par de matrices unitarias entrelazadas con estados |11> inversos
    
    Parámetros:
        alpha (float): Fase kármica (0 a 2π)
        beta (float): Parámetro de no-dualidad (0 a π/2)
    
    Retorna:
        tuple: (Matrix_A, Matrix_B) donde B = CPT(A) con corrección |11>
    """
    # Matriz A (base para realidad manifiesta)
    A = np.array([
        [np.cos(beta), 0, -np.sin(beta)*np.exp(1j*alpha), 0],
        [0, np.cos(alpha), 0, -1j*np.sin(alpha)],
        [np.sin(beta)*np.exp(-1j*alpha), 0, np.cos(beta), 0],
        [0, 1j*np.sin(alpha), 0, np.cos(alpha)]
    ], dtype=complex)
    
    # Matriz B (realidad inversa con simetría CPT)
    B = np.conjugate(np.transpose(A))  # CPT básico
    
    # Corrección para estado |11> (inversión especular)
    B[3,:] *= -1  # Fase π para |11>
    B[:,3] *= -1
    
    # Normalización unitaria
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    
    # Garantizar entrelazamiento
    entangler = np.kron(A, B) + np.kron(B, A)
    return entangler / np.linalg.norm(entangler)

# =============================================================================
# 2. Circuito Cuántico para Estados Inversos
# =============================================================================
def create_dual_circuit(alpha, beta):
    """Circuito con matrices duales actuando en paralelo
    
    Parámetros:
        alpha, beta: Parámetros de fase
    
    Retorna:
        QuantumCircuit: Circuito de 4 qubits
    """
    qc = QuantumCircuit(4)
    
    # Inicialización en superposición simétrica
    qc.h([0,1,2,3])
    
    # Aplicar sistema dual de matrices
    dual_op = dual_matrix_system(alpha, beta)
    qc.unitary(dual_op, [0,1,2,3], label='CPT-Dual')
    
    # Medición correlacionada
    qc.measure_all()
    return qc

# =============================================================================
# 3. Análisis de Correlaciones Cuánticas
# =============================================================================
def analyze_entanglement(results):
    """Calcula correlaciones no-clásicas entre subsistemas
    
    Parámetros:
        results: Resultados de simulación
        
    Retorna:
        dict: Métricas de entrelazamiento
    """
    # Matriz densidad completa
    state = DensityMatrix.from_instruction(qc)
    rho = state.data
    
    # Trazas parciales
    rho_AB = np.trace(rho.reshape(4,4,4,4), axis1=2, axis2=3)
    rho_CD = np.trace(rho.reshape(4,4,4,4), axis1=0, axis2=1)
    
    # Negatividad de entrelazamiento
    neg_AB = np.sum(np.abs(np.linalg.eigvals(rho_AB.T)) - 1
    neg_CD = np.sum(np.abs(np.linalg.eigvals(rho_CD.T)) - 1
    
    # Correlación cuántica total
    quantum_discord = 0.5*(neg_AB + neg_CD)
    
    return {
        'negativity_AB': max(0, neg_AB),
        'negativity_CD': max(0, neg_CD),
        'quantum_discord': quantum_discord
    }

# =============================================================================
# 4. Simulación Especializada
# =============================================================================
def simulate_dual_system(alpha=np.pi/4, beta=np.pi/3, shots=2048):
    """Ejecuta simulación completa con parámetros típicos del Bardo"""
    qc = create_dual_circuit(alpha, beta)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts()
    
    # Visualización matricial
    state = Statevector.from_instruction(qc)
    plt = plot_state_hinton(state, title='Matrices Duales CPT')
    
    # Análisis de correlaciones
    entanglement = analyze_entanglement(counts)
    
    return counts, plt, entanglement

# =============================================================================
# 5. Ejecución y Resultados
# =============================================================================
if __name__ == "__main__":
    # Parámetros para estados inversos fuertes
    alpha = np.pi/3  # Fase kármica
    beta = np.pi/5   # Parámetro no-dual
    
    counts, plot, ent = simulate_dual_system(alpha, beta)
    
    print("\n" + "="*60)
    print("ANÁLISIS DE ESTADOS |11> INVERSOS (SIMETRÍA CPT)")
    print("="*60)
    print(f"- Negatividad AB: {ent['negativity_AB']:.4f}")
    print(f"- Negatividad CD: {ent['negativity_CD']:.4f}")
    print(f"- Discordia cuántica total: {ent['quantum_discord']:.4f}\n")
    
    print("PROBABILIDADES CLAVE:")
    print(f"P(|0000>): {counts.get('0000',0)/sum(counts.values()):.4f}")
    print(f"P(|1111>): {counts.get('1111',0)/sum(counts.values()):.4f}")
    print(f"P(|1100>): {counts.get('1100',0)/sum(counts.values()):.4f}  [Estado inverso crítico]")
    
    plot.show()