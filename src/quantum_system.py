# src/quantum_system.py
"""
Módulo de sistemas cuánticos para simulación del Bardo Thödol
Contiene las clases principales para manejar estados cuánticos y operadores kármicos
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

class BardoQuantumSystem:
    """
    Sistema principal de simulación cuántica del Bardo Thödol
    Implementa estados de qutrit, operadores kármicos y dinámica temporal
    """

    def __init__(self, dimensions=3, **parameters):
        self.dim = dimensions
        self.set_parameters(parameters)
        self.initialize_quantum_system()
        self.metrics = QuantumMetrics()

    def set_parameters(self, params):
        """Configura parámetros del sistema según main.tex"""
        self.karma_params = params.get('karma_params', {
            'clarity': 0.8,
            'attachment': 0.3,
            'compassion': 0.9,
            'wisdom': 0.7
        })
        self.time_parameters = params.get('time_params', {
            'total_time': 4*np.pi,
            'steps': 1000
        })
        self.bardo_stages = params.get('bardo_stages', [
            {'name': 'Chikhai Bardo', 'time': 0.5, 'state': 'void'},
            {'name': 'Chönyid Bardo', 'time': 1.5, 'state': 'superposition'},
            {'name': 'Sidpa Bardo', 'time': 3.0, 'state': 'manifestation'}
        ])

    def initialize_quantum_system(self):
        """Inicializa el sistema cuántico base según Definición 1 en main.tex"""
        # Estados fundamentales como definidos en main.tex
        self.states = {
            'samsara': qt.basis(3, 0),      # |0⟩ Realidad manifiesta
            'karmic': qt.basis(3, 1),       # |1⟩ Potencial kármico
            'void': qt.basis(3, 2)          # |2⟩ Vacuidad fundamental
        }

        # Operadores base
        self.operators = self._create_operators()
        self.current_state = self.states['void']  # Estado inicial en vacuidad

    def _create_operators(self):
        """Crea operadores cuánticos según ecuaciones en main.tex"""
        operators = {}

        # Operadores de proyección P_i = |i⟩⟨i|
        for i in range(3):
            operators[f'P_{i}'] = qt.projection(3, i, i)

        # Operador kármico basado en ecuación (4) de main.tex
        operators['karma'] = self._construct_karma_operator()

        # Hamiltoniano base según ecuación (4)
        operators['hamiltonian'] = self._construct_hamiltonian()

        return operators

    def _construct_karma_operator(self):
        """Construye operador kármico con parámetros personalizados"""
        k = np.zeros((3, 3), dtype=complex)

        # Acoplamientos kármicos basados en parámetros
        k[0,1] = k[1,0] = self.karma_params['attachment']  # |0⟩↔|1⟩
        k[1,2] = k[2,1] = self.karma_params['clarity']     # |1⟩↔|2⟩
        k[2,0] = k[0,2] = self.karma_params['compassion']  # |2⟩↔|0⟩

        # Elementos diagonales representan potenciales intrínsecos
        k[0,0] = 0.7  # Estabilidad samsárica
        k[1,1] = 0.6  # Potencial kármico
        k[2,2] = 0.8  # Claridad vacuidad

        return qt.Qobj(k)

    def _construct_hamiltonian(self):
        """Construye Hamiltoniano según ecuación (4) de main.tex"""
        H = np.zeros((3, 3), dtype=complex)

        # Acoplamientos entre estados
        H[0,1] = H[1,0] = 0.3  # Acoplamiento |0⟩↔|1⟩
        H[1,2] = H[2,1] = 0.4  # Acoplamiento |1⟩↔|2⟩
        H[2,0] = H[0,2] = 0.2  # Acoplamiento |2⟩↔|0⟩

        return qt.Qobj(H)

    def _attention_evolution(self, time, function_type='logistic'):
        """Evolución del parámetro de atención según tiempo"""
        if function_type == 'logistic':
            return 1 / (1 + np.exp(-2 * (time - np.pi)))
        elif function_type == 'sinusoidal':
            return 0.5 * (1 + np.sin(time))
        else:  # linear
            return np.clip(time / (2 * np.pi), 0, 1)

    def simulate_bardo_transition(self, time_steps=1000, attention_function='logistic'):
        """
        Simula la transición completa a través de los estados del Bardo
        Alineado con la sección de Metodología en main.tex
        """
        times = np.linspace(0, self.time_parameters['total_time'], time_steps)

        results = {
            'probabilities': [],
            'coherence': [],
            'purity': [],
            'entropy': [],
            'states': [],
            'times': times
        }

        current_state = self.current_state

        print("Iniciando simulación cuántica del Bardo Thödol...")
        print(f"Parámetros kármicos: {self.karma_params}")

        for i, t in enumerate(times):
            # Factor de atención dependiente del tiempo
            attention = self._attention_evolution(t, attention_function)

            # Hamiltoniano efectivo con componente kármico
            H_eff = self.operators['hamiltonian'] + attention * self.operators['karma']

            # Evolución unitaria
            U = (-1j * t * H_eff).expm()
            evolved_state = U * current_state

            # Cálculo de métricas cuánticas
            probs = [qt.expect(self.operators[f'P_{j}'], evolved_state)
                    for j in range(self.dim)]

            coherence = self.metrics.calculate_coherence(evolved_state)
            purity = self.metrics.calculate_purity(evolved_state)
            entropy = self.metrics.entanglement_entropy(evolved_state, [0])

            # Almacenar resultados
            results['probabilities'].append(probs)
            results['coherence'].append(coherence)
            results['purity'].append(purity)
            results['entropy'].append(entropy)
            results['states'].append(evolved_state)

            current_state = evolved_state

            # Progress bar
            if i % (time_steps // 10) == 0:
                progress = (i / time_steps) * 100
                print(f"Progreso: {progress:.0f}%")

        print("Simulación completada exitosamente!")
        return results

class KarmicDynamics:
    """
    Dinámica de evolución de estados kármicos
    Implementa operadores de transición entre estados del Bardo
    """

    def __init__(self, dimensions=3):
        self.dim = dimensions
        self.hamiltonian = self._construct_hamiltonian()
        self.decoherence_rates = [0.01, 0.02, 0.015]

    def _construct_hamiltonian(self):
        """Hamiltoniano que gobierna transiciones entre estados del Bardo"""
        H = np.zeros((3,3), dtype=complex)
        H[0,1] = H[1,0] = 0.3  # Acoplamiento |0⟩↔|1⟩
        H[1,2] = H[2,1] = 0.4  # Acoplamiento |1⟩↔|2⟩
        H[2,0] = H[0,2] = 0.2  # Acoplamiento |2⟩↔|0⟩
        return qt.Qobj(H)

    def evolve_state(self, state, time, attention_factor=1.0):
        """Evolución temporal del estado con parámetro de atención"""
        U = (-1j * time * attention_factor * self.hamiltonian).expm()
        evolved_state = U * state
        return evolved_state

class QuantumMetrics:
    """
    Sistema de validación de coherencia cuántica
    Alineado con la sección de Marco de Validación en main.tex
    """

    @staticmethod
    def calculate_purity(state):
        """Calcula la pureza del estado cuántico - Ecuación relacionada"""
        if not isinstance(state, qt.Qobj):
            state = qt.Qobj(state)
        density_matrix = state * state.dag()
        return (density_matrix ** 2).tr().real

    @staticmethod
    def entanglement_entropy(state, subsystem):
        """Entropía de entrelazamiento para subsistemas"""
        if not isinstance(state, qt.Qobj):
            state = qt.Qobj(state)

        # Partial trace sobre subsistema complementario
        rho_subsystem = state.ptrace(subsystem)
        eigenvalues = rho_subsystem.eigenenergies()

        # Evitar log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log(eigenvalues))

    @staticmethod
    def calculate_coherence(state):
        """Calcula la coherencia cuántica usando norma l1 fuera de diagonal"""
        if not isinstance(state, qt.Qobj):
            state = qt.Qobj(state)

        rho = state * state.dag() if state.type == 'ket' else state
        rho_array = rho.full()

        coherence = 0
        for i in range(len(rho_array)):
            for j in range(len(rho_array)):
                if i != j:
                    coherence += abs(rho_array[i, j])

        return coherence

    @staticmethod
    def classical_fidelity(state1, state2):
        """Fidelidad entre estados cuánticos"""
        return qt.fidelity(state1, state2)
