# scripts/generate_paper_figures.py
"""
Script de Generación de Figuras para el Paper Científico
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.simulations.advanced_simulator import AdvancedBardoSimulator
from src.visualization.publication_figures import PublicationFigureGenerator
import matplotlib.pyplot as plt

def generate_all_paper_figures():
    """Genera todas las figuras para el paper científico"""
    print("Iniciando generación de figuras para publicación...")

    # Ejecutar simulación comprehensiva
    simulator = AdvancedBardoSimulator()
    results = simulator.run_comprehensive_simulation(num_samples=500)

    # Generar figuras
    figure_generator = PublicationFigureGenerator()

    # Figura principal comprehensiva
    main_figure = figure_generator.create_comprehensive_figure(results)
    main_figure.savefig('paper/figures/comprehensive_analysis.png',
                       dpi=300, bbox_inches='tight')
    print("Figura principal guardada: comprehensive_analysis.png")

    # Figuras individuales para secciones específicas
    individual_figures = generate_individual_figures(results, figure_generator)

    print("Generación de figuras completada")
    return main_figure, individual_figures

def generate_individual_figures(results: Dict, generator: PublicationFigureGenerator):
    """Genera figuras individuales para diferentes secciones del paper"""
    individual_figs = {}

    # Figura para la sección de métodos
    fig_methods = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig_methods)

    ax1 = fig_methods.add_subplot(gs[0, 0])
    generator._plot_probability_evolution(ax1, results)

    ax2 = fig_methods.add_subplot(gs[0, 1])
    generator._plot_quantum_phase_space(ax2, results)

    ax3 = fig_methods.add_subplot(gs[1, 0])
    generator._plot_karmic_correlations(ax3, results)

    ax4 = fig_methods.add_subplot(gs[1, 1])
    generator._plot_error_505_distribution(ax4, results)

    plt.tight_layout()
    fig_methods.savefig('paper/figures/methods_analysis.png', dpi=300, bbox_inches='tight')
    individual_figs['methods'] = fig_methods

    # Figura para la sección de resultados
    fig_results = plt.figure(figsize=(12, 6))
    gs_results = GridSpec(1, 2, figure=fig_results)

    ax5 = fig_results.add_subplot(gs_results[0, 0])
    generator._plot_entropy_analysis(ax5, results)

    ax6 = fig_results.add_subplot(gs_results[0, 1])
    generator._plot_scientific_validation(ax6, results)

    plt.tight_layout()
    fig_results.savefig('paper/figures/results_analysis.png', dpi=300, bbox_inches='tight')
    individual_figs['results'] = fig_results

    return individual_figs

if __name__ == "__main__":
    generate_all_paper_figures()
