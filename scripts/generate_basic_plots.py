# scripts/generate_basic_plots.py
import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_directories():
    """Crear directorios necesarios"""
    dirs = ['graphics/generated', 'data', 'scripts/temp']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directorios creados/verificados")

def generate_qutrit_space():
    """Generar gráfico del espacio de qutrit"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Crear malla de puntos para el espacio de qutrit
        theta = np.linspace(0, np.pi, 30)
        phi = np.linspace(0, 2*np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)

        # Coordenadas esféricas para qutrit
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Plot superficie
        surf = ax.plot_surface(x, y, z, alpha=0.6, cmap='viridis')

        # Etiquetas de estados
        ax.text(1.1, 0, 0, r'$|0\rangle$', fontsize=16, color='red')
        ax.text(0, 1.1, 0, r'$|1\rangle$', fontsize=16, color='blue')
        ax.text(0, 0, 1.1, r'$|2\rangle$', fontsize=16, color='green')

        ax.set_xlabel('Estado |0⟩ (Samsara)')
        ax.set_ylabel('Estado |1⟩ (Potencial)')
        ax.set_zlabel('Estado |2⟩ (Vacuidad)')
        ax.set_title('Espacio de Hilbert del Qutrit Bardo-Thödol')

        plt.savefig('graphics/generated/qutrit_space.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico qutrit_space.png generado")
    except Exception as e:
        print(f"✗ Error en generate_qutrit_space: {e}")

def generate_state_evolution():
    """Generar evolución temporal de estados"""
    try:
        t = np.linspace(0, 10, 100)

        # Simular evolución de probabilidades
        prob_0 = 0.5 + 0.3 * np.sin(t) * np.exp(-0.2*t)
        prob_1 = 0.3 + 0.2 * np.cos(2*t) * np.exp(-0.1*t)
        prob_2 = 0.2 + 0.5 * np.sin(0.5*t) * np.exp(-0.05*t)
        attention = 0.5 + 0.3 * np.cos(0.8*t) * np.exp(-0.15*t)

        # Normalizar
        total = prob_0 + prob_1 + prob_2
        prob_0 /= total
        prob_1 /= total
        prob_2 /= total

        # Guardar datos para LaTeX
        with open('data/state_evolution.dat', 'w') as f:
            f.write("time prob0 prob1 prob2 attention\n")
            for i in range(len(t)):
                f.write(f"{t[i]:.3f} {prob_0[i]:.3f} {prob_1[i]:.3f} {prob_2[i]:.3f} {attention[i]:.3f}\n")

        # Generar gráfico
        plt.figure(figsize=(12, 6))
        plt.plot(t, prob_0, 'r-', linewidth=2, label='Estado |0⟩ (Samsara)')
        plt.plot(t, prob_1, 'b-', linewidth=2, label='Estado |1⟩ (Potencial)')
        plt.plot(t, prob_2, 'g-', linewidth=2, label='Estado |2⟩ (Vacuidad)')
        plt.plot(t, attention, 'k--', linewidth=1, label='Atención')

        plt.xlabel('Tiempo (iteraciones)')
        plt.ylabel('Probabilidad')
        plt.title('Evolución Temporal de Estados Bardo-Thödol')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig('graphics/generated/state_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico state_evolution.png generado")

    except Exception as e:
        print(f"✗ Error en generate_state_evolution: {e}")

def generate_error505_analysis():
    """Generar análisis del ERROR 505"""
    try:
        # Parámetros
        kappa_range = np.linspace(0, 1, 20)
        alpha_range = np.linspace(0, 1, 20)

        # Calcular probabilidad del estado |2⟩
        K, A = np.meshgrid(kappa_range, alpha_range)
        prob_2 = 0.8 * (1 - A) * (K**0.5)  # Modelo simplificado

        # Guardar datos para superficie 3D
        with open('data/error505_surface.dat', 'w') as f:
            for i in range(len(kappa_range)):
                for j in range(len(alpha_range)):
                    f.write(f"{kappa_range[i]:.3f} {alpha_range[j]:.3f} {prob_2[j,i]:.3f}\n")
                f.write("\n")  # Separador para pgfplots

        # Generar gráfico
        plt.figure(figsize=(10, 8))
        plt.contourf(K, A, prob_2, levels=20, cmap='viridis')
        plt.colorbar(label='Probabilidad Estado |2⟩')
        plt.xlabel('Carga Kármica (κ)')
        plt.ylabel('Atención (α)')
        plt.title('ERROR 505 como Estado de Vacuidad - Probabilidad |2⟩')

        plt.savefig('graphics/generated/error505_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráfico error505_analysis.png generado")

    except Exception as e:
        print(f"✗ Error en generate_error505_analysis: {e}")

def main():
    """Función principal"""
    print("Iniciando generación de gráficos para BardoThodol...")
    ensure_directories()

    generate_qutrit_space()
    generate_state_evolution()
    generate_error505_analysis()

    print("✅ Todos los gráficos generados exitosamente!")
    print("📍 Gráficos guardados en: graphics/generated/")
    print("📍 Datos guardados en: data/")

if __name__ == "__main__":
    main()
