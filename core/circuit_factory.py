"""
Módulo unificado para la creación y simulación de circuitos cuánticos.
"""
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

def create_bardo_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Crea un circuito cuántico para la simulación del Bardo.
    
    Parameters
    ----------
    n_qubits : int
        Número de qubits del circuito.
        
    Returns
    -------
    QuantumCircuit
        Circuito cuántico optimizado.
    """
    # Usar funciones de core/states.py para generar estados
    from core.states import generate_bardo_base_state
    qc = generate_bardo_base_state(n_qubits)
    # Añadir puertas adicionales si es necesario
    return qc

def simulate_circuit(qc: QuantumCircuit, shots: int = 1024) -> dict:
    """
    Simula un circuito cuántico y devuelve los resultados.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Circuito a simular.
    shots : int, optional
        Número de ejecuciones (default: 1024).
        
    Returns
    -------
    dict
        Resultados de la simulación.
    """
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=shots).result()
    return result.get_counts(qc)

# Añade aquí otras funciones de simulación, eliminando duplicados