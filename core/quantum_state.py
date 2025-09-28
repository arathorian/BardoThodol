import numpy as np
from qiskit import QuantumCircuit
from typing import Dict, List

class QuantumStateManager:
    """Gestor de estados cuánticos para las simulaciones del Bardo."""

    def __init__(self, state_data: Dict = None):
        if state_data is None:
            state_data = {
                "chonyid_bardo": 0.3,
                "sidpa_bardo": 0.4,
                "clear_light": 0.2,
                "intermediate": 0.1
            }
        self.states = list(state_data.keys())
        self.probabilities = self._normalize_probabilities(list(state_data.values()))

    def _normalize_probabilities(self, probs: List[float]) -> np.ndarray:
        """Normaliza las probabilidades para que sumen 1."""
        total = sum(probs)
        return np.array(probs) / total

    def create_bardo_circuit(self, num_qubits: int = 2) -> QuantumCircuit:
        """Crea un circuito cuántico representando los estados del Bardo."""
        qc = QuantumCircuit(num_qubits)

        # Aplicar compuertas Hadamard para crear superposición
        for i in range(num_qubits):
            qc.h(i)

        return qc
