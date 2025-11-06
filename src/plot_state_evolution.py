def plot_state_evolution(time_points, probabilities, bardo_stages):
    """Evolución temporal de probabilidades por estado"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    states = ['|0⟩ Samsara', '|1⟩ Kármico', '|2⟩ Vacuidad']
    
    for i in range(3):
        ax.plot(time_points, probabilities[:, i], 
                color=colors[i], linewidth=2.5, label=states[i])
    
    # Marcadores de transición entre Bardos
    for stage in bardo_stages:
        ax.axvline(x=stage['time'], color='gray', 
                  linestyle='--', alpha=0.7)
        ax.text(stage['time'], 0.95, stage['name'], 
               rotation=90, va='top')
    
    ax.set_xlabel('Tiempo (unidades arbitrarias)')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Evolución Cuántica de Estados de Consciencia en el Bardo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig