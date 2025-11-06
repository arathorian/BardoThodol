# -*- coding: utf-8 -*-
"""
Modelo Cuántico Dinámico del Bardo con Variables Kármicas Ajustables
"""
import numpy as np
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit.visualization import plot_histogram, plot_state_city
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =============================================================================
# 1. Parámetros Kármicos Dinámicos (Controlables con Sliders)
# =============================================================================
# Valores iniciales
karma_init = 0.7  # Nivel kármico (0=puro, 1=pesado)
atencion_init = 0.3  # Atención consciente (0=distraído, 1=enfocado)
tiempo_init = 2  # Instantes temporales en el Bardo (1 a 5)

# =============================================================================
# 2. Sistema de Matrices Adaptativas (dependientes de karma/atención)
# =============================================================================
def crear_matrices_adaptativas(k, a):
    """Genera matrices duales con parámetros kármicos en tiempo real"""
    # Ajuste no-lineal de parámetros
    alpha = np.pi * (1 - k) * (1 + 0.5*np.sin(a*np.pi))
    beta = (np.pi/3) * k * np.exp(-a)
    
    # Matriz A (realidad manifiesta)
    A = np.array([
        [np.cos(beta), 0, -np.sin(beta)*np.exp(1j*alpha), 0],
        [0, np.cos(alpha), 0, -1j*np.sin(alpha)],
        [np.sin(beta)*np.exp(-1j*alpha), 0, np.cos(beta), 0],
        [0, 1j*np.sin(alpha), 0, np.cos(alpha)]
    ], dtype=complex)
    
    # Matriz B (realidad inversa CPT)
    B = np.conjugate(np.transpose(A))
    B[3,:] *= -1  # Fase π para |11>
    B[:,3] *= -1
    
    # Acoplamiento kármico
    entanglement_factor = 0.5 + 2*k - a  # [0.5, 2.5]
    dual_matrix = entanglement_factor * np.kron(A, B) + (1 - entanglement_factor) * np.kron(B, A)
    
    return dual_matrix / np.linalg.norm(dual_matrix)

# =============================================================================
# 3. Circuito Cuántico Dinámico (Evolución Temporal)
# =============================================================================
def crear_circuito_dinamico(k, a, steps=3):
    """Crea circuito con evolución temporal escalonada"""
    qc = QuantumCircuit(4, 4)
    
    # Inicialización en estado de vacuidad |2> para todos los qutrits
    # |00> = |0>, |01> = |1>, |10> = |2> (vacuidad)
    for i in range(0, 4, 2):
        qc.x(i)  # |10> = |2>
    
    for paso in range(steps):
        # Aplicar matriz dual con parámetros actualizados
        matriz_dual = crear_matrices_adaptativas(k * (1 - paso/steps), a * (1 + paso/steps))
        qc.unitary(matriz_dual, range(4), label=f'Dual_{paso}')
        
        # Operador de decoherencia kármica
        for i in range(4):
            qc.rz(k * np.pi/4, i)  # Rotación dependiente de karma
    
    qc.measure(range(4), range(4))
    return qc

# =============================================================================
# 4. Simulación con Visualización Interactiva
# =============================================================================
def simular_sistema_karmico(k, a, steps):
    """Ejecuta simulación completa y retorna resultados"""
    qc = crear_circuito_dinamico(k, a, steps)
    simulador = Aer.get_backend('qasm_simulator')
    transpilado = transpile(qc, simulador)
    resultado = simulador.run(transpilado, shots=5000).result()
    conteos = resultado.get_counts()
    
    # Calcular estado final
    estado = Statevector.from_instruction(transpilado)
    densidad = DensityMatrix(estado)
    
    return conteos, densidad

# =============================================================================
# 5. Visualización Interactiva
# =============================================================================
# Configuración de la figura
fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# Espacio para sliders
plt.subplots_adjust(bottom=0.3)
ax_karma = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_atencion = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_tiempo = plt.axes([0.2, 0.1, 0.65, 0.03])

# Crear sliders
slider_karma = Slider(ax_karma, 'Karma', 0.0, 1.0, valinit=karma_init)
slider_atencion = Slider(ax_atencion, 'Atención', 0.0, 1.0, valinit=atencion_init)
slider_tiempo = Slider(ax_tiempo, 'Tiempo en Bardo', 1, 5, valinit=tiempo_init, valstep=1)

# Función de actualización
def actualizar(val):
    k = slider_karma.val
    a = slider_atencion.val
    t = int(slider_tiempo.val)
    
    # Limpiar ejes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.cla()
    
    # Ejecutar simulación
    conteos, densidad = simular_sistema_karmico(k, a, t)
    
    # Visualización 1: Histograma de estados
    plot_histogram(conteos, ax=ax1, title=f"Distribución de Estados (K={k:.2f}, A={a:.2f})")
    ax1.set_ylim(0, 1)
    
    # Visualización 2: Matriz de densidad (3D)
    plot_state_city(densidad, ax=ax2, title="Matriz de Densidad")
    ax2.set_zlim(0, 0.5)
    
    # Visualización 3: Estados críticos
    estados_criticos = {est: prob for est, prob in conteos.items() if est.endswith('00') or est.startswith('11')}
    plot_histogram(estados_criticos, ax=ax3, color='purple', title="Estados Críticos |11xx> y |xx00>")
    
    # Visualización 4: Discordia cuántica
    subsistema = partial_trace(densidad, [0, 1])  # Traza parcial primeros 2 qubits
    entropia = -np.sum(np.diag(subsistema.data) * np.log2(np.diag(subsistema.data) + 1e-10))
    ax4.bar(['Discordia'], [entropia], color='orange')
    ax4.set_ylim(0, 2)
    ax4.set_title(f"Entropía Cuántica: {entropia:.4f}")
    
    fig.canvas.draw_idle()

# Registrar actualizaciones
slider_karma.on_changed(actualizar)
slider_atencion.on_changed(actualizar)
slider_tiempo.on_changed(actualizar)

# Simulación inicial
actualizar(None)

plt.tight_layout()
plt.show()