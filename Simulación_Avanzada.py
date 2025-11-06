import numpy as np
from src.quantum_system import KarmicDynamics

# Configurar dinámica kármica personalizada
karma_dynamics = KarmicDynamics(
    dimensions=3,
    decoherence_rates=[0.01, 0.02, 0.015],
    attention_parameters={'type': 'adaptive', 'sensitivity': 0.8}
)

# Simular evolución temporal
evolution_data = karma_dynamics.compute_evolution(
    initial_state='void',
    time_range=(0, 4*np.pi),
    metrics=['coherence', 'purity', 'entanglement_entropy']
)