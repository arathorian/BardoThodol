import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parámetros del sistema
dim = 3
times = np.linspace(0, 4*np.pi, 1000)

# Operadores y estado inicial
H = 0.5 * rand_herm(dim)  # Hamiltoniano aleatorio
psi0 = basis(dim, 2)       # Estado inicial |2>

# Evolución temporal
result = sesolve(H, psi0, times)

# Calcular probabilidades
probs = np.zeros((len(times), dim))
for i, t in enumerate(times):
    state = result.states[i]
    for j in range(dim):
        probs[i, j] = expect(projection(dim, j, j), state)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(times, probs[:, 0], label='$|0\\rangle$ Samsara', linewidth=2)
plt.plot(times, probs[:, 1], label='$|1\\rangle$ Kármico', linewidth=2)
plt.plot(times, probs[:, 2], label='$|2\\rangle$ Vacuidad', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Probabilidad')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Evolución temporal de probabilidades por estado')
plt.savefig('state_evolution.pdf', bbox_inches='tight')
plt.close()