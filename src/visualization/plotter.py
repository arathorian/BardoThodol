"""
Módulo de visualización para las simulaciones cuánticas del Bardo Thödol.
Proporciona herramientas para visualizar resultados de simulaciones cuánticas
y estados del Bardo de manera clara y efectiva.
"""

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
import os

class BardoVisualizer:
    """
    Visualizador avanzado para simulaciones del Bardo Thödol.
    Combina visualizaciones cuánticas estándar con representaciones
    específicas de los estados del Bardo.
    """

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Inicializa el visualizador con el estilo especificado.

        Args:
            style (str): Estilo de matplotlib a utilizar
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.fig_size = (12, 8)

    def plot_quantum_results(self,
                           counts: Dict[str, int],
                           title: str = "Resultados de la Simulación Cuántica del Bardo",
                           filename: Optional[str] = None) -> plt.Figure:
        """
        Visualiza resultados cuánticos usando plot_histogram de Qiskit.

        Args:
            counts (Dict[str, int]): Resultados de la simulación
            title (str): Título del gráfico
            filename (str, optional): Path para guardar la figura

        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        fig = plot_histogram(counts, title=title, figsize=self.fig_size)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado como: {filename}")

        return fig

    def plot_bardo_states(self,
                         states: List[str],
                         probabilities: List[float],
                         title: str = "Distribución de Estados del Bardo Thödol",
                         filename: Optional[str] = None) -> plt.Figure:
        """
        Visualiza las probabilidades de los estados del Bardo en un gráfico de barras.

        Args:
            states (List[str]): Nombres de los estados del Bardo
            probabilities (List[float]): Probabilidades de cada estado
            title (str): Título del gráfico
            filename (str, optional): Path para guardar la figura

        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Colores específicos para los estados del Bardo
        colors = ['#8B0000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        bars = ax.bar(states, probabilities,
                     color=colors[:len(states)],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.2)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Probabilidad', fontsize=12, fontweight='bold')
        ax.set_xlabel('Estados del Bardo', fontsize=12, fontweight='bold')

        # Añadir valores en las barras
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.2%}',
                   ha='center',
                   va='bottom',
                   fontweight='bold',
                   fontsize=10)

        # Mejorar el estilo
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Añadir anotación descriptiva
        ax.text(0.02, 0.98, f'Total de estados: {len(states)}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 Gráfico de estados guardado como: {filename}")

        return fig

    def plot_statevector_evolution(self,
                                 circuit: QuantumCircuit,
                                 title: str = "Evolución del Estado Cuántico",
                                 filename: Optional[str] = None) -> plt.Figure:
        """
        Visualiza la evolución del vector de estado en la esfera de Bloch.

        Args:
            circuit (QuantumCircuit): Circuito cuántico a visualizar
            title (str): Título del gráfico
            filename (str, optional): Path para guardar la figura

        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        try:
            # Calcular el estado final
            state = Statevector.from_instruction(circuit)
            fig = plot_bloch_multivector(state, title=title)

            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"🌐 Esfera de Bloch guardada como: {filename}")

            return fig
        except Exception as e:
            print(f"⚠️ No se pudo visualizar la esfera de Bloch: {e}")
            # Crear una figura vacía como fallback
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Visualización no disponible',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig

    def plot_comparative_analysis(self,
                                results_before: Dict[str, int],
                                results_after: Dict[str, int],
                                title: str = "Análisis Comparativo del Bardo",
                                filename: Optional[str] = None) -> plt.Figure:
        """
        Compara resultados de simulaciones antes y después de alguna transformación.

        Args:
            results_before (Dict[str, int]): Resultados antes de la transformación
            results_after (Dict[str, int]): Resultados después de la transformación
            title (str): Título del gráfico
            filename (str, optional): Path para guardar la figura

        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Normalizar resultados
        total_before = sum(results_before.values())
        total_after = sum(results_after.values())

        prob_before = {k: v/total_before for k, v in results_before.items()}
        prob_after = {k: v/total_after for k, v in results_after.items()}

        # Gráfico antes
        states = list(prob_before.keys())
        probabilities_before = [prob_before[state] for state in states]

        bars1 = ax1.bar(states, probabilities_before, color='lightblue', alpha=0.7)
        ax1.set_title('Antes de la Transformación', fontweight='bold')
        ax1.set_ylabel('Probabilidad')
        ax1.tick_params(axis='x', rotation=45)

        # Gráfico después
        probabilities_after = [prob_after.get(state, 0) for state in states]
        bars2 = ax2.bar(states, probabilities_after, color='lightcoral', alpha=0.7)
        ax2.set_title('Después de la Transformación', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        # Añadir valores
        for bars, ax in zip([bars1, bars2], [ax1, ax2]):
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # Solo mostrar valores significativos
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.1%}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📋 Análisis comparativo guardado como: {filename}")

        return fig

    def create_dashboard(self,
                        counts: Dict[str, int],
                        states: List[str],
                        probabilities: List[float],
                        circuit: QuantumCircuit,
                        title: str = "Dashboard Completo del Bardo Thödol",
                        filename: Optional[str] = None) -> plt.Figure:
        """
        Crea un dashboard completo con múltiples visualizaciones.

        Args:
            counts (Dict[str, int]): Resultados de la simulación
            states (List[str]): Estados del Bardo
            probabilities (List[float]): Probabilidades de cada estado
            circuit (QuantumCircuit): Circuito cuántico utilizado
            title (str): Título del dashboard
            filename (str, optional): Path para guardar la figura

        Returns:
            matplotlib.figure.Figure: Figura del dashboard
        """
        fig = plt.figure(figsize=(20, 12))

        # Definir el layout del dashboard
        gs = fig.add_gridspec(2, 2)

        # 1. Histograma cuántico
        ax1 = fig.add_subplot(gs[0, 0])
        plot_histogram(counts, ax=ax1)
        ax1.set_title('Distribución Cuántica', fontweight='bold')

        # 2. Estados del Bardo
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax2.bar(states, probabilities, color=colors[:len(states)])
        ax2.set_title('Estados del Bardo', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom')

        # 3. Información textual
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')

        info_text = f"""
        INFORMACIÓN DE LA SIMULACIÓN:

        • Total de ejecuciones: {sum(counts.values()):,}
        • Estados cuánticos posibles: {len(counts)}
        • Estados del Bardo modelados: {len(states)}
        • Circuito: {circuit.num_qubits} qubits, {circuit.depth()} profundidad

        DISTRIBUCIÓN PRINCIPAL:
        """

        # Añadir los estados más probables
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for i, (state, count) in enumerate(sorted_counts[:4]):
            percentage = (count / sum(counts.values())) * 100
            info_text += f"\n  {i+1}. Estado {state}: {count} veces ({percentage:.1f}%)"

        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.suptitle(title, fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 Dashboard guardado como: {filename}")

        return fig

    def save_all_visualizations(self,
                              counts: Dict[str, int],
                              states: List[str],
                              probabilities: List[float],
                              circuit: QuantumCircuit,
                              base_filename: str = "bardo_visualization") -> None:
        """
        Guarda todas las visualizaciones en archivos.

        Args:
            counts (Dict[str, int]): Resultados de la simulación
            states (List[str]): Estados del Bardo
            probabilities (List[float]): Probabilidades
            circuit (QuantumCircuit): Circuito cuántico
            base_filename (str): Nombre base para los archivos
        """
        # Crear directorio de resultados si no existe
        os.makedirs('results', exist_ok=True)

        # Generar todas las visualizaciones
        self.plot_quantum_results(counts, filename=f"results/{base_filename}_histogram.png")
        self.plot_bardo_states(states, probabilities, filename=f"results/{base_filename}_states.png")
        self.plot_statevector_evolution(circuit, filename=f"results/{base_filename}_bloch.png")
        self.create_dashboard(counts, states, probabilities, circuit,
                            filename=f"results/{base_filename}_dashboard.png")

        print(f"✅ Todas las visualizaciones guardadas en la carpeta 'results/'")

# Función de conveniencia para uso rápido
def quick_visualize(counts: Dict[str, int],
                   title: str = "Resultados del Bardo") -> plt.Figure:
    """
    Función rápida para visualizar resultados sin crear una instancia.

    Args:
        counts (Dict[str, int]): Resultados de la simulación
        title (str): Título del gráfico

    Returns:
        matplotlib.figure.Figure: Figura generada
    """
    visualizer = BardoVisualizer()
    return visualizer.plot_quantum_results(counts, title)

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo básico de uso
    sample_counts = {'00': 256, '01': 245, '10': 268, '11': 255}
    sample_states = ['Chonyid Bardo', 'Sidpa Bardo', 'Luz Clara', 'Intermedio']
    sample_probs = [0.3, 0.4, 0.2, 0.1]

    visualizer = BardoVisualizer()

    # Visualización individual
    visualizer.plot_quantum_results(sample_counts)
    visualizer.plot_bardo_states(sample_states, sample_probs)

    # Dashboard completo
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    visualizer.create_dashboard(sample_counts, sample_states, sample_probs, qc)
    plt.show()
