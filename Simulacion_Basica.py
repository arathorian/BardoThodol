from src.main import BardoQuantumSystem, QuantumVisualizer

# Inicializar sistema con parámetros kármicos
bardo_system = BardoQuantumSystem(
    karma_params={
        'clarity': 0.85,
        'attachment': 0.25,
        'compassion': 0.95,
        'wisdom': 0.75
    }
)

# Ejecutar simulación completa
results = bardo_system.simulate_full_bardo(
    time_steps=2000,
    attention_evolution='logistic'
)

# Generar visualizaciones
visualizer = QuantumVisualizer()
figures = visualizer.create_publication_quality_plots(results)
