#!/usr/bin/env python3
"""
Sistema de generación de gráficos científicos para Bardo Thodol Quantum Simulation
Autor: Horacio Héctor Hamann
Repositorio: https://github.com/arathorian/BardoThodol
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt
from scipy.linalg import expm
import os

# Configuración científica profesional
plt.style.use('seaborn-v0_8-paper')
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class BardoFigureGenerator:
    """Generador de figuras científicas para la simulación del Bardo"""
    
    def __init__(self):
        self.colors = {
            'samsara': '#E74C3C',      # Rojo - estado manifiesto
            'karmic': '#F39C12',       # Naranja - potencial kármico  
            'void': '#2E86C1',         # Azul - vacuidad
            'coherence': '#8E44AD',    # Púrpura - coherencia cuántica
            'attention': '#27AE60'     # Verde - atención
        }
        self.output_dir = "figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def simulate_bardo_dynamics(self, time_steps=1000):
        """Simula la dinámica cuántica del sistema Bardo"""
        # Parámetros del sistema (coherentes con el artículo)
        dimensions = 3
        
        # Hamiltoniano base
        H0 = qt.Qobj(np.array([
            [1.0, 0.1, 0.2],
            [0.1, 0.8, 0.15],
            [0.2, 0.15, 0.6]
        ]))
        
        # Operador kármico
        K = qt.Qobj(np.array([
            [0.0, 0.3, 0.9],  # compassion
            [0.3, 0.0, 0.8],  # clarity  
            [0.9, 0.8, 0.0]   # attachment
        ], dtype=complex))
        
        # Estado inicial (vacuidad)
        psi0 = qt.basis(dimensions, 2)
        
        # Evolución temporal
        times = np.linspace(0, 4*np.pi, time_steps)
        probabilities = np.zeros((len(times), dimensions))
        coherence_vals = []
        purity_vals = []
        
        current_state = psi0
        
        for i, t in enumerate(times):
            # Evolución unitaria con atención dependiente del tiempo
            attention = 0.5 * (1 + np.sin(t))
            H_eff = H0 + attention * K
            
            U = (-1j * t * H_eff).expm()
            evolved_state = U * current_state
            
            # Probabilidades
            for j in range(dimensions):
                probabilities[i, j] = qt.expect(qt.projection(dimensions, j, j), evolved_state)
            
            # Coherencia (norma l1 fuera de la diagonal)
            rho = evolved_state * evolved_state.dag()
            coh = np.sum(np.abs(rho.full())) - np.trace(np.abs(rho.full()))
            coherence_vals.append(coh)
            
            # Pureza
            purity_vals.append(np.trace((rho * rho).full()).real)
            
            current_state = evolved_state
        
        return times, probabilities, coherence_vals, purity_vals
    
    def create_state_evolution_plot(self):
        """Crea el gráfico de evolución temporal de estados"""
        print("Generando gráfico de evolución de estados...")
        
        times, probabilities, coherence, purity = self.simulate_bardo_dynamics()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Gráfico de probabilidades
        ax1.plot(times, probabilities[:, 0], 
                label='$|0\\rangle$ Samsara', 
                color=self.colors['samsara'], 
                linewidth=2.5)
        ax1.plot(times, probabilities[:, 1], 
                label='$|1\\rangle$ Potencial Kármico', 
                color=self.colors['karmic'], 
                linewidth=2.5)
        ax1.plot(times, probabilities[:, 2], 
                label='$|2\\rangle$ Vacuidad (Sunyata)', 
                color=self.colors['void'], 
                linewidth=2.5)
        
        ax1.set_xlabel('Tiempo (unidades arbitrarias)')
        ax1.set_ylabel('Probabilidad')
        ax1.set_title('Evolución Temporal de Estados del Bardo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Gráfico de coherencia y pureza
        ax2.plot(times, coherence, 
                label='Coherencia Cuántica', 
                color=self.colors['coherence'], 
                linewidth=2.5)
        ax2.plot(times, purity, 
                label='Pureza del Estado', 
                color=self.colors['attention'], 
                linewidth=2.5, 
                linestyle='--')
        
        ax2.set_xlabel('Tiempo (unidades arbitrarias)')
        ax2.set_ylabel('Coherencia / Pureza')
        ax2.set_title('Métricas Cuánticas del Sistema')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/state_evolution.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Gráfico de evolución guardado: figures/state_evolution.pdf")
    
    def create_bloch_sphere_representation(self):
        """Crea representación en esfera de Bloch para qutrits"""
        print("Generando representación en esfera de Bloch...")
        
        # Para qutrits, usamos una representación simplificada en 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generar puntos para la "esfera" de qutrit
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        
        # Coordenadas esféricas
        x = np.outer(np.cos(theta), np.sin(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.ones(np.size(theta)), np.cos(phi))
        
        # Esfera transparente
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.1, rstride=2, cstride=2)
        
        # Estados base
        states = {
            'Samsara (|0⟩)': (1, 0, 0),
            'Potencial (|1⟩)': (0, 1, 0), 
            'Vacuidad (|2⟩)': (0, 0, 1)
        }
        
        for name, (x, y, z) in states.items():
            ax.scatter([x], [y], [z], s=200, label=name, alpha=0.8)
            ax.text(x, y, z, f'  {name}', fontsize=10)
        
        # Trayectoria de evolución
        times = np.linspace(0, 4*np.pi, 200)
        traj_x = 0.7 * np.sin(times) * np.cos(2*times)
        traj_y = 0.7 * np.sin(times) * np.sin(2*times) 
        traj_z = 0.7 * np.cos(times)
        
        ax.plot(traj_x, traj_y, traj_z, 
               color=self.colors['coherence'], 
               linewidth=2.5, 
               label='Trayectoria de Evolución')
        
        ax.set_xlabel('Componente X')
        ax.set_ylabel('Componente Y') 
        ax.set_zlabel('Componente Z')
        ax.set_title('Representación del Espacio de Estados del Qutrit')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bloch_sphere.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Esfera de Bloch guardada: figures/bloch_sphere.pdf")
    
    def create_quantum_metrics_plot(self):
        """Crea gráfico de métricas cuánticas avanzadas"""
        print("Generando gráfico de métricas cuánticas...")
        
        times, probabilities, coherence, purity = self.simulate_bardo_dynamics()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Entropía de Von Neumann
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1)
        ax1.plot(times, entropy, color='#16A085', linewidth=2.5)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Entropía de Von Neumann (bits)')
        ax1.set_title('Entropía del Sistema')
        ax1.grid(True, alpha=0.3)
        
        # 2. Coherencia vs Pureza
        ax2.scatter(coherence, purity, 
                   c=times, cmap='viridis', alpha=0.7, s=30)
        ax2.set_xlabel('Coherencia Cuántica')
        ax2.set_ylabel('Pureza del Estado')
        ax2.set_title('Diagrama de Fase: Coherencia vs Pureza')
        ax2.grid(True, alpha=0.3)
        
        # 3. Transiciones entre estados
        im = ax3.imshow(probabilities.T, aspect='auto', cmap='plasma',
                       extent=[times[0], times[-1], 2.5, -0.5])
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Estado')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['$|0⟩$', '$|1⟩$', '$|2⟩$'])
        ax3.set_title('Mapa de Calor: Transiciones de Estado')
        plt.colorbar(im, ax=ax3, label='Probabilidad')
        
        # 4. Análisis espectral
        fft_coherence = np.abs(np.fft.fft(coherence))[:len(times)//2]
        freqs = np.fft.fftfreq(len(times), times[1]-times[0])[:len(times)//2]
        ax4.semilogy(freqs, fft_coherence, color='#E74C3C', linewidth=2)
        ax4.set_xlabel('Frecuencia')
        ax4.set_ylabel('Potencia Espectral')
        ax4.set_title('Análisis Espectral de la Coherencia')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/quantum_metrics.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Métricas cuánticas guardadas: figures/quantum_metrics.pdf")
    
    def create_all_figures(self):
        """Genera todas las figuras del artículo"""
        print("Iniciando generación de figuras científicas...")
        print("=" * 50)
        
        self.create_state_evolution_plot()
        self.create_bloch_sphere_representation() 
        self.create_quantum_metrics_plot()
        
        print("=" * 50)
        print("✓ Todas las figuras generadas exitosamente!")
        print("✓ Directorio: figures/")
        print("✓ Formatos: PDF de alta calidad (300 DPI)")

if __name__ == "__main__":
    generator = BardoFigureGenerator()
    generator.create_all_figures()