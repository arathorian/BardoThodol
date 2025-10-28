"""
Análisis del ERROR 505 en el Modelo del Bardo
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class Error505Analyzer:
    """
    Analizador del ERROR 505 (estado de vacuidad dominante).
    """

    def __init__(self, qutrit_system):
        self.qutrit_system = qutrit_system

    def compute_vacuity_probability(self, qc: QuantumCircuit) -> float:
        """
        Calcula la probabilidad del estado de vacuidad (|2⟩) en el circuito.

        Args:
            qc: Circuito cuántico.

        Returns:
            Probabilidad de vacuidad.
        """
        statevector = Statevector.from_instruction(qc)
        probs = statevector.probabilities()

        # En nuestra codificación, el estado |2⟩ corresponde a |10> en los dos qubits del qutrit.
        # Para un solo qutrit (2 qubits), los estados son:
        # |00> -> índice 0, |01> -> índice 1, |10> -> índice 2, |11> -> índice 3
        # Nos interesa el estado |10> (índice 2) para el primer qutrit.
        # Para múltiples qutrits, hay que sumar las probabilidades de los estados donde cada qutrit está en |2>.
        # Por simplicidad, asumimos un solo qutrit por ahora.
        if self.qutrit_system.num_qutrits == 1:
            return probs[2]  # |10> para el primer qutrit
        else:
            # Para múltiples qutrits, necesitamos una lógica más compleja
            # Por ejemplo, para 2 qutrits, el estado |2⟩|2⟩ es |1010> que en little-endian es el índice 10.
            # Pero Qiskit usa little-endian, así que el estado |q0 q1 q2 q3> = |1010> es el índice 10.
            # Sin embargo, es más seguro calcularlo basado en la definición.
            vacuity_prob = 0.0
            for i, prob in enumerate(probs):
                # Convertir el índice a binario y verificar los qubits
                bin_rep = format(i, f'0{self.qutrit_system.num_qubits}b')
                # Revisar cada qutrit (cada dos qubits)
                for j in range(0, self.qutrit_system.num_qubits, 2):
                    if bin_rep[j] == '1' and bin_rep[j+1] == '0':
                        vacuity_prob += prob
            return vacuity_prob

    def analyze_error_505(self, qc: QuantumCircuit, threshold: float = 0.5) -> Dict:
        """
        Analiza si el circuito está en estado de ERROR 505.

        Args:
            qc: Circuito cuántico.
            threshold: Umbral para considerar dominancia de vacuidad.

        Returns:
            Diccionario con el análisis.
        """
        vacuity_prob = self.compute_vacuity_probability(qc)
        is_error_505 = vacuity_prob > threshold

        analysis = {
            'vacuity_probability': vacuity_prob,
            'is_error_505': is_error_505,
            'threshold': threshold,
            'interpretation': self.interpret_error_505(vacuity_prob, threshold)
        }

        return analysis

    def interpret_error_505(self, vacuity_prob: float, threshold: float) -> str:
        """
        Interpreta el resultado del ERROR 505.

        Args:
            vacuity_prob: Probabilidad de vacuidad.
            threshold: Umbral.

        Returns:
            Interpretación en texto.
        """
        if vacuity_prob > threshold:
            return "Dominancia de vacuidad: ERROR 505 activo. Estado de máxima potencialidad cuántica."
        else:
            return "Estado de realidad manifiesta o en transición. No es ERROR 505."

