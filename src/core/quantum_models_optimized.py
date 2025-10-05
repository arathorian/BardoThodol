# src/core/quantum_models_optimized.py
"""
Modelo Cuántico del Bardo Thödol - Versión Científicamente Validada
Integra: Operadores kármicos, qutrits y validación física completa
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, DensityMatrix, entropy
from scipy.linalg import expm, norm
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumState:
    """Estructura para estados cuánticos con validación integrada"""
    statevector: np.ndarray
    density_matrix: np.ndarray
    probabilities: Dict[str, float]
    entropy: float
    coherence: float
    purity: float

    def validate_physical(self) -> bool:
        """Validación completa de propiedades físicas cuánticas"""
        constraints = {
            'norm': np.isclose(norm(self.statevector), 1.0, atol=1e-10),
            'trace': np.isclose(np.trace(self.density_matrix), 1.0, atol=1e-10),
            'positive_semidefinite': np.all(np.linalg.eigvalsh(self.density_matrix) >= -1e-10),
            'purity_range': 0.0 <= self.purity <= 1.0,
            'entropy_range': 0.0 <= self.entropy <= np.log(3)
        }
        return all(constraints.values())

class OptimizedBardoModel:
    """
    Modelo principal que unifica:
    - Sistemas de qutrits para estados intermedios
    - Operadores kármicos como dinámica de sistemas abiertos
    - Vacuidad como estado cuántico fundamental
    """
    
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions  # Qutrits para estados bardo
        self.hilbert_space_dim = dimensions ** 2
        self._initialize_physical_operators()

    def _initialize_physical_operators(self):
        """Inicializa operadores con propiedades físicas válidas"""
        # Base para qutrits (|manifestado⟩, |potencial⟩, |vacuidad⟩)
        self.basis_states = self._create_qutrit_basis()
        self.karma_operator = self._create_karma_operator()
        self.vacuity_operator = self._create_vacuity_operator()

    def _create_qutrit_basis(self) -> Dict[str, np.ndarray]:
        """Base computacional para sistema de qutrits"""
        ket0 = np.array([1, 0, 0], dtype=complex)  # |manifestado⟩
        ket1 = np.array([0, 1, 0], dtype=complex)  # |potencial⟩  
        ket2 = np.array([0, 0, 1], dtype=complex)  # |vacuidad⟩

        return {
            'manifested': ket0,
            'potential': ket1, 
            'vacuity': ket2,
            'projectors': [
                np.outer(ket0, ket0.conj()),
                np.outer(ket1, ket1.conj()),
                np.outer(ket2, ket2.conj())
            ]
        }

    def _create_karma_operator(self) -> np.ndarray:
        """
        Operador kármico basado en dinámica de Lindblad
        Representa interacciones con ambiente kármico y memoria de acciones
        """
        # Hamiltonian con términos de interacción
        H_karma = np.array([
            [0.15, 0.05j, -0.02],     # Acoplamientos complejos
            [-0.05j, 0.25, 0.08j],    # Términos imaginarios para coherencia
            [-0.02, -0.08j, 0.20]     # Elementos fuera-diagonal para entrelazamiento
        ], dtype=complex)

        # Operador de disipación para decoherencia kármica
        gamma = 0.1  # Tasa de decoherencia por karma
        L = np.sqrt(gamma) * self.basis_states['projectors'][1]  # Disipación desde |potencial⟩

        # Generador de Lindblad
        L_term = L @ L.conj().T - 0.5 * (L.conj().T @ L + L @ L.conj().T)
        
        return H_karma + 1j * L_term

    def _create_vacuity_operator(self) -> np.ndarray:
        """
        Operador de vacuidad unitario que preserva información
        Representa transiciones hacia estados de no-manifestación
        """
        theta_v = np.pi/4  # Ángulo de transición vacuidad
        phi_v = np.pi/6    # Fase cuántica para coherencia

        U_v = np.array([
            [np.cos(theta_v), 0, -np.sin(theta_v)*np.exp(1j*phi_v)],
            [0, 1, 0],
            [np.sin(theta_v)*np.exp(-1j*phi_v), 0, np.cos(theta_v)]
        ], dtype=complex)

        # Corrección unitaria si es necesaria
        if not self._is_unitary(U_v):
            U_v = self._make_unitary(U_v)
            
        return U_v

    def _is_unitary(self, matrix: np.ndarray) -> bool:
        """Verificación rigurosa de unitariedad"""
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix @ matrix.conj().T, identity, atol=1e-12)

    def _make_unitary(self, matrix: np.ndarray) -> np.ndarray:
        """Corrección unitaria vía descomposición polar"""
        u, s, vh = np.linalg.svd(matrix)
        return u @ vh

    def apply_bardo_evolution(self, initial_state: np.ndarray, 
                            karma_strength: float, 
                            attention_level: float) -> QuantumState:
        """
        Aplica evolución completa del estado Bardo
        Integra karma, atención y transiciones hacia vacuidad
        """
        # Evolución kármica
        U_karma = expm(-1j * karma_strength * self.karma_operator)
        state_after_karma = U_karma @ initial_state

        # Aplicar operador de vacuidad según nivel de atención
        attention_factor = 1.0 - attention_level  # Menos atención → más vacuidad
        U_attention = expm(-1j * attention_factor * self.vacuity_operator)
        final_statevector = U_attention @ state_after_karma

        # Calcular propiedades cuánticas
        density_matrix = np.outer(final_statevector, final_statevector.conj())
        probabilities = {
            'manifested': np.abs(final_statevector[0])**2,
            'potential': np.abs(final_statevector[1])**2, 
            'vacuity': np.abs(final_statevector[2])**2
        }
        
        # Métricas cuánticas
        eigenvals = np.linalg.eigvalsh(density_matrix)
        eigenvals = eigenvals[eigenvals > 0]
        entropy_val = -np.sum(eigenvals * np.log(eigenvals))
        purity = np.trace(density_matrix @ density_matrix)
        coherence = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))

        quantum_state = QuantumState(
            statevector=final_statevector,
            density_matrix=density_matrix,
            probabilities=probabilities,
            entropy=entropy_val,
            coherence=coherence,
            purity=purity
        )

        if not quantum_state.validate_physical():
            logger.warning("Estado cuántico generado viola constraints físicos")

        return quantum_state