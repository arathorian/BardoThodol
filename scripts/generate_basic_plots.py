import matplotlib.pyplot as plt
import numpy as np

# Configuración del estilo científico
plt.style.use('seaborn-v0_8-whitegrid')
SCIENCE_COLORS = {
    'quantum_blue': '#00529B',
    'bardo_gold': '#BD9331', 
    'state_green': '#007F00',
    'transition_orange': '#E69F00'
}

def create_science_plot():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Datos de ejemplo para transiciones de estado
    t = np.linspace(0, 10, 100)
    states = {
        'Estado Chikhai': np.exp(-0.5*t) * np.sin(2*t),
        'Estado Chönyid': np.exp(-0.3*t) * np.cos(1.5*t),
        'Estado Sidpa': np.exp(-0.2*t) * np.sin(3*t)
    }
    
    for i, (label, values) in enumerate(states.items()):
        color = list(SCIENCE_COLORS.values())[i]
        ax.plot(t, values, label=label, color=color, linewidth=2.5)
    
    # Estilo científico
    ax.set_xlabel('Tiempo (días post-mortem)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitud de Probabilidad', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/quantum_states.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/quantum_states.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    create_science_plot()