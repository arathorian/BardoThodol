from src.validation import QuantumMetrics

metrics = QuantumMetrics()
state = bardo_system.current_state

print(f"Pureza: {metrics.calculate_purity(state):.3f}")
print(f"Coherencia: {metrics.calculate_coherence(state):.3f}")
print(f"Entropía: {metrics.entanglement_entropy(state, [0]):.3f}")
