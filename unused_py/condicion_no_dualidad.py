# Condición de no-dualidad
assert np.allclose(A @ B.conj().T, np.eye(4))  # Unitariedad
assert np.isclose(np.linalg.det(A), np.exp(1j*phase))  # Fase global