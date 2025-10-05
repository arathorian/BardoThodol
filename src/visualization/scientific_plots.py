# src/visualization/scientific_plots.py
"""
Visualizaciones científicas avanzadas para el modelo del Bardo
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ScientificVisualizer:
    """Generador de visualizaciones científicas de alta calidad"""
    
    def __init__(self):
        self.setup_scientific_style()
    
    def setup_scientific_style(self):
        """Configura estilo de publicación científica"""
        plt.style.use('default')
        sns.set_style("whitegrid")
        
        self.colors = {
            'manifested': '#2E86AB',
            'potential': '#A23B72', 
            'vacuity': '#F18F01',
            'error_505': '#C73E1D',
            'karma': '#3C91E6',
            'attention': '#47E5BC'
        }
    
    def create_comprehensive_dashboard(self, simulation_results: Dict) -> go.Figure:
        """Crea dashboard interactivo completo con Plotly"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Evolución de Probabilidades', 'Espacio de Fases Cuántico',
                'Distribución de Vacuidad', 'Correlaciones Kármicas',
                'Entropía vs Pureza', 'Estados en Esfera de Bloch',
                'Análisis de ERROR 505', 'Validación Científica', 'Estadísticas'
            ),
            specs=[
                [{"type": "xy"}, {"type": "scene"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "scene"}],
                [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
            ]
        )
        
        # 1. Evolución de probabilidades
        self._add_probability_evolution(fig, simulation_results, row=1, col=1)
        
        # 2. Espacio de fases cuántico
        self._add_quantum_phase_space(fig, simulation_results, row=1, col=2)
        
        # 3. Distribución de vacuidad
        self._add_vacuity_distribution(fig, simulation_results, row=1, col=3)
        
        # 4. Correlaciones kármicas
        self._add_karmic_correlations(fig, simulation_results, row=2, col=1)
        
        # 5. Entropía vs Pureza
        self._add_entropy_purity_plot(fig, simulation_results, row=2, col=2)
        
        # 6. Esfera de Bloch para qutrits
        self._add_qutrit_bloch_sphere(fig, simulation_results, row=2, col=3)
        
        # 7. Análisis de ERROR 505
        self._add_error_505_analysis(fig, simulation_results, row=3, col=1)
        
        # 8. Validación científica
        self._add_scientific_validation(fig, simulation_results, row=3, col=2)
        
        # 9. Estadísticas
        self._add_statistical_summary(fig, simulation_results, row=3, col=3)
        
        fig.update_layout(
            height=1200,
            title_text="Dashboard Científico del Modelo Cuántico del Bardo",
            showlegend=False
        )
        
        return fig
    
    def create_publication_quality_plots(self, results: Dict) -> plt.Figure:
        """Crea figuras de calidad para publicación científica"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig)
        
        # 1. Diagrama de transiciones de estado
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_state_transitions(ax1, results)
        
        # 2. Mapa de calor de la matriz densidad
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_density_matrix_heatmap(ax2, results)
        
        # 3. Evolución temporal de métricas
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_metric_evolution(ax3, results)
        
        # 4. Análisis de componentes principales
        ax4 = fig.add_subplot(gs[2, 0:2])
        self._plot_quantum_pca(ax4, results)
        
        # 5. Distribución de probabilidades
        ax5 = fig.add_subplot(gs[2, 2:])
        self._plot_probability_distributions(ax5, results)
        
        # 6. Validación de constraints físicos
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_physical_constraints(ax6, results)
        
        # 7. Análisis de ERROR 505
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_error_505_analysis(ax7, results)
        
        # 8. Resumen estadístico
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_statistical_summary(ax8, results)
        
        plt.tight_layout()
        return fig
    
    def _plot_state_transitions(self, ax, results: Dict):
        """Diagrama de transiciones entre estados del Bardo"""
        states = ['Manifestado', 'Potencial', 'Vacuidad']
        transition_probs = self._calculate_transition_probabilities(results)
        
        # Crear gráfico de red
        pos = {'Manifestado': [0, 1], 'Potencial': [1, 0], 'Vacuidad': [0, -1]}
        
        # Dibujar nodos
        for state, (x, y) in pos.items():
            ax.scatter(x, y, s=500, c=self.colors[state.lower()], alpha=0.7)
            ax.text(x, y, state, ha='center', va='center', fontweight='bold')
        
        # Dibujar transiciones
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i != j and transition_probs[i, j] > 0.1:
                    x1, y1 = pos[state1]
                    x2, y2 = pos[state2]
                    
                    # Línea con ancho proporcional a la probabilidad
                    width = transition_probs[i, j] * 5
                    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle="->", 
                                             lw=width,
                                             alpha=0.7,
                                             color='gray'))
                    
                    # Etiqueta de probabilidad
                    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                    ax.text(mid_x, mid_y, f'{transition_probs[i,j]:.2f}',
                           ha='center', va='center', fontsize=8)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title('Diagrama de Transiciones de Estado', fontweight='bold')
        ax.axis('off')
    
    def _calculate_transition_probabilities(self, results: Dict) -> np.ndarray:
        """Calcula probabilidades de transición entre estados"""
        # Matriz de transición 3x3
        transition_matrix = np.zeros((3, 3))
        states = results['states']
        
        for i in range(1, len(states)):
            prev_probs = states[i-1]['probabilities'][:3]
            curr_probs = states[i]['probabilities'][:3]
            
            # Estimación simple de transiciones
            for j in range(3):
                if prev_probs[j] > 0.1:  # Estado anterior significativo
                    for k in range(3):
                        if curr_probs[k] > 0.1:  # Estado actual significativo
                            transition_matrix[j, k] += curr_probs[k]
        
        # Normalizar
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Evitar división por cero
        return transition_matrix / row_sums[:, np.newaxis]