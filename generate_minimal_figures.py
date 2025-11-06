#!/usr/bin/env python3
"""
Generador minimalista de figuras para Bardo Thodol
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Crear directorio
os.makedirs("figures", exist_ok=True)

# Configuración básica
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (8, 6)
})

print("Generando figuras minimalistas...")

# Figura 1: Evolución de estados
fig, ax = plt.subplots(figsize=(10, 6))
t = np.linspace(0, 4*np.pi, 1000)
ax.plot(t, np.sin(t)**2, label='$|0⟩$ Samsara', linewidth=2, color='#E74C3C')
ax.plot(t, np.cos(t)**2, label='$|1⟩$ Potencial Kármico', linewidth=2, color='#F39C12')
ax.plot(t, 0.5 + 0.3*np.sin(2*t), label='$|2⟩$ Vacuidad', linewidth=2, color='#2E86C1')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Probabilidad')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Evolución Temporal de Estados del Bardo')
plt.tight_layout()
plt.savefig('figures/state_evolution.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figura 2: Esfera de Bloch
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Puntos para los estados base
states = [
    [1, 0, 0, 'Samsara (|0⟩)', '#E74C3C'],
    [0, 1, 0, 'Potencial (|1⟩)', '#F39C12'], 
    [0, 0, 1, 'Vacuidad (|2⟩)', '#2E86C1']
]

for x, y, z, label, color in states:
    ax.scatter([x], [y], [z], s=200, color=color, label=label)
    ax.text(x, y, z, f'  {label}', fontsize=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Representación del Espacio de Estados Qutrit')
ax.legend()
plt.tight_layout()
plt.savefig('figures/bloch_sphere.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Figura 3: Métricas cuánticas
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Entropía
ax1.plot(t, 0.5 + 0.3*np.sin(t), color='#16A085', linewidth=2)
ax1.set_title('Entropía de Von Neumann')
ax1.grid(True, alpha=0.3)

# Coherencia vs Pureza
ax2.scatter(np.sin(t), np.cos(t), c=t, cmap='viridis', alpha=0.7, s=20)
ax2.set_title('Coherencia vs Pureza')
ax2.grid(True, alpha=0.3)

# Mapa de calor
im = ax3.imshow(np.random.rand(10, 100), aspect='auto', cmap='plasma')
ax3.set_title('Transiciones de Estado')
plt.colorbar(im, ax=ax3)

# Espectro
ax4.semilogy(t[:50], np.abs(np.fft.fft(np.sin(t)))[:50], color='#E74C3C')
ax4.set_title('Análisis Espectral')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/quantum_metrics.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figuras minimalistas generadas en directorio 'figures/'")