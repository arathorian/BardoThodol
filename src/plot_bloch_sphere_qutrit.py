def plot_bloch_sphere_qutrit(states, titles=None):
    """Visualización de estados en esfera de Bloch extendida"""
    fig = plt.figure(figsize=(15, 5))
    
    # Proyecciones en los tres planos principales
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Calcular componentes para cada proyección
        x = [qt.expect(qt.sigmax(), state) for state in states]
        y = [qt.expect(qt.sigmay(), state) for state in states] 
        z = [qt.expect(qt.sigmaz(), state) for state in states]
        
        ax.scatter(x, y, z, c=range(len(states)), cmap='viridis')
        ax.set_xlabel('X |0⟩↔|1⟩')
        ax.set_ylabel('Y |1⟩↔|2⟩')
        ax.set_zlabel('Z |2⟩↔|0⟩')
        
        if titles:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    return fig