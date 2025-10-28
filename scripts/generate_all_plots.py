import numpy as np
import matplotlib.pyplot as plt
import os

# Crear directorios si no existen
os.makedirs('data', exist_ok=True)
os.makedirs('graphics/generated', exist_ok=True)

# 1. Generar data/qutrit_surface.dat y graphics/generated/qutrit_space.png
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

with open('data/qutrit_surface.dat', 'w') as f:
    for i in range(len(theta)):
        for j in range(len(phi)):
            f.write(f'{x[i,j]} {y[i,j]} {z[i,j]}\n')
        f.write('\n')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, alpha=0.6, cmap='viridis')
ax.set_xlabel('Estado |0⟩ (Samsara)')
ax.set_ylabel('Estado |1⟩ (Potencial)')
ax.set_zlabel('Estado |2⟩ (Vacuidad)')
ax.set_title('Espacio de Hilbert del Qutrit Bardo-Thödol')
plt.savefig('graphics/generated/qutrit_space.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Generar data/state_evolution.dat y graphics/generated/state_evolution.png
t = np.linspace(0, 10, 100)
prob0 = 0.5 + 0.3 * np.sin(t) * np.exp(-0.2*t)
prob1 = 0.3 + 0.2 * np.cos(2*t) * np.exp(-0.1*t)
prob2 = 0.2 + 0.5 * np.sin(0.5*t) * np.exp(-0.05*t)
attention = 0.5 + 0.5 * np.sin(0.8*t) * np.exp(-0.1*t)

with open('data/state_evolution.dat', 'w') as f:
    f.write('time prob0 prob1 prob2 attention\n')
    for i in range(len(t)):
        f.write(f'{t[i]} {prob0[i]} {prob1[i]} {prob2[i]} {attention[i]}\n')

plt.figure(figsize=(10, 6))
plt.plot(t, prob0, label='Estado |0⟩')
plt.plot(t, prob1, label='Estado |1⟩')
plt.plot(t, prob2, label='Estado |2⟩')
plt.plot(t, attention, label='Atención', linestyle='--')
plt.xlabel('Tiempo (iteraciones)')
plt.ylabel('Probabilidad')
plt.legend()
plt.title('Evolución Temporal de Estados Bardo-Thödol')
plt.savefig('graphics/generated/state_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Generar data/error505_surface.dat
kappa_vals = np.linspace(0, 1, 20)
attention_vals = np.linspace(0, 1, 20)

with open('data/error505_surface.dat', 'w') as f:
    for kappa in kappa_vals:
        for attention in attention_vals:
            prob2 = 0.5 * (1 - attention) * (1 + kappa)
            prob2 = max(0, min(1, prob2))
            f.write(f'{kappa} {attention} {prob2}\n')
        f.write('\n')

# 4. Generar data/quantum_metrics.dat
n_points = 100
coherence = np.random.uniform(0, 1, n_points)
entropy = np.random.uniform(0, 1, n_points)
state2 = np.random.uniform(0, 1, n_points)

with open('data/quantum_metrics.dat', 'w') as f:
    f.write('coherence entropy state2\n')
    for i in range(n_points):
        f.write(f'{coherence[i]} {entropy[i]} {state2[i]}\n')

# 5. Generar data/probability_distribution0.dat, etc.
n_samples = 1000
prob0_dist = np.random.beta(2, 5, n_samples)
prob1_dist = np.random.beta(2, 2, n_samples)
prob2_dist = np.random.beta(5, 2, n_samples)

with open('data/probability_distribution0.dat', 'w') as f:
    for p in prob0_dist:
        f.write(f'{p}\n')

with open('data/probability_distribution1.dat', 'w') as f:
    for p in prob1_dist:
        f.write(f'{p}\n')

with open('data/probability_distribution2.dat', 'w') as f:
    for p in prob2_dist:
        f.write(f'{p}\n')

# 6. Generar data/attention_transition.dat
attention_vals = np.linspace(0, 1, 50)
transition_rate = 0.1 + 0.5 * (1 - attention_vals)**2
error = 0.05 * np.ones_like(attention_vals)

with open('data/attention_transition.dat', 'w') as f:
    f.write('attention transition_rate error\n')
    for i in range(len(attention_vals)):
        f.write(f'{attention_vals[i]} {transition_rate[i]} {error[i]}\n')

print("Todos los datos y gráficos generados exitosamente.")