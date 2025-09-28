import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import seaborn as sns
from typing import Dict

class BardoVisualizer:
    """Visualizador para los resultados de la simulación del Bardo."""

    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_quantum_results(self, counts: Dict[str, int], title: str = "Resultados Cuánticos"):
        """Visualiza los resultados cuánticos usando plot_histogram de Qiskit."""
        return plot_histogram(counts, title=title)

    def plot_bardo_states(self, states: list, probabilities: list):
        """Visualiza las probabilidades de los estados del Bardo."""
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(states, probabilities, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])

        ax.set_title('Estados del Bardo Thödol', fontsize=16, fontweight='bold')
        ax.set_ylabel('Probabilidad', fontsize=12)
        ax.set_xlabel('Estados', fontsize=12)

        # Añadir valores en las barras
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.2%}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
