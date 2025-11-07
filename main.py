#!/usr/bin/env python3
"""
Bardo Thödol Quantum Simulation Project
Main execution file - Aligned with main.tex structure

Author: Horacio Héctor Hamann
Repository: https://github.com/arathorian/BardoThodol
Date: July 2025
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración científica
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8)
})

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
        self.visualizer = QuantumVisualizer()

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
    """Construye operador kármico ALINEADO con ecuación (4)"""
    k = np.zeros((3, 3), dtype=complex)

    # Usar parámetros kármicos definidos, no valores fijos
    k[0,1] = k[1,0] = self.karma_params['attachment']
    k[1,2] = k[2,1] = self.karma_params['clarity']
    k[2,0] = k[0,2] = self.karma_params['compassion']

    # Elementos diagonales desde parámetros
    k[0,0] = self.karma_params.get('samsara_stability', 0.7)
    k[1,1] = self.karma_params.get('karmic_potential', 0.6)
    k[2,2] = self.karma_params.get('void_clarity', 0.8)

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

class QuantumVisualizer:
    """
    Sistema completo de visualización científica
    Genera todas las figuras mencionadas en main.tex
    """

    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.state_names = ['$|0\\rangle$ Samsara', '$|1\\rangle$ Kármico', '$|2\\rangle$ Vacuidad']

    def create_comprehensive_visualization(self, results, bardo_stages=None, save_path=None):
        """Crea visualizaciones completas para publicación"""

        if bardo_stages is None:
            bardo_stages = [
                {'name': 'Chikhai Bardo', 'time': 0.5},
                {'name': 'Chönyid Bardo', 'time': 1.5},
                {'name': 'Sidpa Bardo', 'time': 3.0}
            ]

        # Crear figura principal
        fig = plt.figure(figsize=(20, 16))

        # 1. Evolución temporal de probabilidades (Figura 1 en main.tex)
        ax1 = fig.add_subplot(2, 3, 1)
        self._plot_temporal_evolution(ax1, results, bardo_stages)

        # 2. Coherencia cuántica y pureza (Figura relacionada)
        ax2 = fig.add_subplot(2, 3, 2)
        self._plot_quantum_metrics(ax2, results, bardo_stages)

        # 3. Esfera de Bloch para qutrits (Figura 2 en main.tex)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        self._plot_bloch_sphere(ax3, results)

        # 4. Matriz de densidad final
        ax4 = fig.add_subplot(2, 3, 4)
        self._plot_density_matrix(ax4, results)

        # 5. Diagrama de fases del Bardo
        ax5 = fig.add_subplot(2, 3, 5)
        self._plot_phase_diagram(ax5, results, bardo_stages)

        # 6. Análisis espectral
        ax6 = fig.add_subplot(2, 3, 6)
        self._plot_spectral_analysis(ax6, results)

        plt.suptitle('Simulación Cuántica Completa del Bardo Thödol\n'
                    'Análisis Multidimensional de Estados de Consciencia',
                    fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Figuras guardadas en: {save_path}")

        return fig

    def _plot_temporal_evolution(self, ax, results, bardo_stages):
        """Evolución temporal de probabilidades por estado"""
        times = results['times']
        probabilities = np.array(results['probabilities'])

        for i in range(3):
            ax.plot(times, probabilities[:, i],
                   color=self.colors[i], linewidth=2.5, label=self.state_names[i])

        # Marcadores de transición entre Bardos
        for stage in bardo_stages:
            ax.axvline(x=stage['time'], color='gray',
                      linestyle='--', alpha=0.7)
            ax.text(stage['time'], 0.02, stage['name'],
                   rotation=90, va='bottom', ha='center', fontsize=9)

        ax.set_xlabel('Tiempo (unidades arbitrarias)')
        ax.set_ylabel('Probabilidad')
        ax.set_title('Evolución de Probabilidades por Estado')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_quantum_metrics(self, ax, results, bardo_stages):
        """Métricas cuánticas: coherencia y pureza"""
        times = results['times']

        # Coherencia
        ax.plot(times, results['coherence'],
               color=self.colors[0], linewidth=2, label='Coherencia')

        # Pureza
        ax.plot(times, results['purity'],
               color=self.colors[1], linewidth=2, label='Pureza')

        # Entropía
        ax.plot(times, results['entropy'],
               color=self.colors[2], linewidth=2, label='Entropía')

        for stage in bardo_stages:
            ax.axvline(x=stage['time'], color='gray',
                      linestyle='--', alpha=0.7)

        ax.set_xlabel('Tiempo (unidades arbitrarias)')
        ax.set_ylabel('Métrica Cuántica')
        ax.set_title('Evolución de Métricas Cuánticas')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bloch_sphere(self, ax, results):
        """Representación en esfera de Bloch para qutrits"""
        # Para qutrits, usamos una representación simplificada
        states = results['states']
        n_states = len(states)

        # Tomar muestras equidistantes
        step = max(1, n_states // 50)
        sample_states = states[::step]

        # Calcular componentes para visualización
        x_vals, y_vals, z_vals = [], [], []
        colors = []

        for i, state in enumerate(sample_states):
            # Proyecciones aproximadas para visualización
            probs = [abs(state[j][0][0])**2 for j in range(3)]

            x = probs[0] - probs[1]  # Diferencia |0⟩ - |1⟩
            y = probs[1] - probs[2]  # Diferencia |1⟩ - |2⟩
            z = probs[2] - probs[0]  # Diferencia |2⟩ - |0⟩

            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            colors.append(i / len(sample_states))

        # Plot
        scatter = ax.scatter(x_vals, y_vals, z_vals,
                           c=colors, cmap='viridis', alpha=0.7)

        # Esfera de referencia
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

        ax.set_xlabel('X: |0⟩ - |1⟩')
        ax.set_ylabel('Y: |1⟩ - |2⟩')
        ax.set_zlabel('Z: |2⟩ - |0⟩')
        ax.set_title('Espacio de Estados del Bardo\n(Representación Qutrit)')

    def _plot_density_matrix(self, ax, results):
        """Visualización de la matriz de densidad final"""
        final_state = results['states'][-1]
        rho_final = final_state * final_state.dag()
        rho_array = rho_final.full()

        im = ax.imshow(np.abs(rho_array), cmap='Blues', aspect='equal')

        # Anotar valores
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{abs(rho_array[i, j]):.2f}',
                              ha="center", va="center", color="black")

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['$|0⟩$', '$|1⟩$', '$|2⟩$'])
        ax.set_yticklabels(['$⟨0|$', '$⟨1|$', '$⟨2|$'])
        ax.set_title('Matriz de Densidad Final')
        plt.colorbar(im, ax=ax)

    def _plot_phase_diagram(self, ax, results, bardo_stages):
        """Diagrama de fases del proceso del Bardo"""
        probabilities = np.array(results['probabilities'])

        # Coordenadas barycéntricas para triple diagrama
        x = probabilities[:, 0] - probabilities[:, 2]  # |0⟩ - |2⟩
        y = (2 * probabilities[:, 1] - probabilities[:, 0] - probabilities[:, 2]) / np.sqrt(3)

        scatter = ax.scatter(x, y, c=results['times'], cmap='plasma', alpha=0.7)

        # Triángulo de referencia
        triangle_x = [1, 0, -1, 1]
        triangle_y = [0, np.sqrt(3), 0, 0]
        ax.plot(triangle_x, triangle_y, 'k--', alpha=0.5)

        # Etiquetas de vértices
        ax.text(1, 0, '$|0⟩$\nSamsara', ha='center', va='top')
        ax.text(-1, 0, '$|2⟩$\nVacuidad', ha='center', va='top')
        ax.text(0, np.sqrt(3), '$|1⟩$\nKármico', ha='center', va='bottom')

        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.set_title('Diagrama de Fases del Bardo')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.colorbar(scatter, ax=ax, label='Tiempo')

    def _plot_spectral_analysis(self, ax, results):
        """Análisis espectral de la evolución"""
        probabilities = np.array(results['probabilities'])

        # Transformada de Fourier para análisis frecuencial
        for i in range(3):
            signal = probabilities[:, i]
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))

            # Solo frecuencias positivas
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])

            ax.plot(positive_freqs, positive_fft,
                   color=self.colors[i], linewidth=2, label=self.state_names[i])

        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Amplitud FFT')
        ax.set_title('Análisis Espectral')
        ax.legend()
        ax.grid(True, alpha=0.3)

def main():
    """
    Función principal - Ejecuta la simulación completa del Bardo Thödol
    Alineada con la estructura de main.tex
    """
    print("=" * 70)
    print("SIMULACIÓN CUÁNTICA DEL BARDO THÖDOL")
    print("Autor: Horacio Héctor Hamann")
    print("Repositorio: https://github.com/arathorian/BardoThodol")
    print("=" * 70)

    # Configuración de parámetros según main.tex
    simulation_params = {
        'karma_params': {
            'clarity': 0.85,
            'attachment': 0.25,
            'compassion': 0.95,
            'wisdom': 0.75
        },
        'time_params': {
            'total_time': 4 * np.pi,
            'steps': 2000
        },
        'bardo_stages': [
            {'name': 'Chikhai Bardo', 'time': 0.8, 'state': 'void'},
            {'name': 'Chönyid Bardo', 'time': 2.2, 'state': 'superposition'},
            {'name': 'Sidpa Bardo', 'time': 3.5, 'state': 'manifestation'}
        ]
    }

    # Inicializar sistema cuántico
    print("\n1. Inicializando sistema cuántico del Bardo...")
    bardo_system = BardoQuantumSystem(**simulation_params)

    # Ejecutar simulación completa
    print("\n2. Ejecutando simulación de transiciones del Bardo...")
    results = bardo_system.simulate_bardo_transition(
        time_steps=simulation_params['time_params']['steps'],
        attention_function='logistic'
    )

    # Generar visualizaciones completas
    print("\n3. Generando visualizaciones científicas...")
    fig = bardo_system.visualizer.create_comprehensive_visualization(
        results,
        bardo_stages=simulation_params['bardo_stages'],
        save_path='bardo_simulation_complete.png'
    )

    # Análisis de resultados
    print("\n4. Análisis de resultados:")
    final_probs = np.array(results['probabilities'][-1])
    max_coherence = max(results['coherence'])
    avg_purity = np.mean(results['purity'])

    print(f"   - Probabilidades finales: |0⟩={final_probs[0]:.3f}, |1⟩={final_probs[1]:.3f}, |2⟩={final_probs[2]:.3f}")
    print(f"   - Coherencia máxima: {max_coherence:.3f}")
    print(f"   - Pureza promedio: {avg_purity:.3f}")
    print(f"   - Estados simulados: {len(results['states'])}")

    # Validación científica
    print("\n5. Validación científica:")
    validation = QuantumMetrics()
    initial_state = results['states'][0]
    final_state = results['states'][-1]

    fidelity = validation.classical_fidelity(initial_state, final_state)
    print(f"   - Fidelidad inicial-final: {fidelity:.3f}")
    print(f"   - Pureza inicial: {validation.calculate_purity(initial_state):.3f}")
    print(f"   - Pureza final: {validation.calculate_purity(final_state):.3f}")

    print("\n" + "=" * 70)
    print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("Resultados alineados con el marco teórico de main.tex")
    print("=" * 70)

    # Mostrar figura
    plt.show()

if __name__ == "__main__":
    main()