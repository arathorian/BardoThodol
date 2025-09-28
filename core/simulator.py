from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from typing import Dict

class BardoSimulator:
    """Simulador para las experiencias del Bardo."""

    def __init__(self, backend: str = 'aer_simulator'):
        self.backend = Aer.get_backend(backend)

    def simulate(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """Ejecuta la simulación cuántica."""
        compiled = transpile(circuit, self.backend)
        job = self.backend.run(compiled, shots=shots)
        return job.result().get_counts()
