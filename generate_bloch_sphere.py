import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del sistema
dim = 3
times = np.linspace(0, 4*np.pi, 1000)

# Operadores y estado inicial
H = 0.5 * rand_herm(dim)  # Hamiltoniano aleatorio
psi0 = basis(dim, 2)       # Estado inicial |2>

# Evolución temporal
result = sesolve(H, psi0, times)

# Extraer el estado en el subespacio (0,1) (trazando sobre el estado |2>)
# Para cada estado, tomamos las amplitudes para |0> y |1> y normalizamos
# para obtener un estado de qubit en la esfera de Bloch.
bloch_vectors = np.zeros((len(times), 3))

for i, t in enumerate(times):
    state = result.states[i]
    # Proyectamos al subespacio {|0>, |1>}
    a0 = state[0,0]
    a1 = state[1,0]
    norm = np.sqrt(abs(a0)**2 + abs(a1)**2)
    if norm > 0:
        a0 /= norm
        a1 /= norm
    else:
        a0 = 0
        a1 = 0
    # Convertir a vector de Bloch
    bloch_vectors[i, 0] = 2 * np.real(a0 * np.conj(a1))
    bloch_vectors[i, 1] = 2 * np.imag(a0 * np.conj(a1))
    bloch_vectors[i, 2] = abs(a0)**2 - abs(a1)**2

# Graficar en la esfera de Bloch
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dibujar la esfera
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='y', alpha=0.1)

# Dibujar la evolución
ax.plot(bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2], 'b-', linewidth=2)
ax.scatter(bloch_vectors[0, 0], bloch_vectors[0, 1], bloch_vectors[0, 2], color='g', s=100, label='Inicio')
ax.scatter(bloch_vectors[-1, 0], bloch_vectors[-1, 1], bloch_vectors[-1, 2], color='r', s=100, label='Fin')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Evolución del estado en la esfera de Bloch (subespacio |0>,|1>)')
ax.legend()
plt.savefig('bloch_sphere.pdf', bbox_inches='tight')
plt.close()