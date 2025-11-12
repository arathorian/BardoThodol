# 🌀 Bardo Thödol Quantum Simulation Project
*Simulación Cuántica de Estados de Consciencia con Transparencia Epistemológica*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![QuTiP](https://img.shields.io/badge/QuTiP-4.7+-green.svg)](https://qutip.org/)
[![Debian 12](https://img.shields.io/badge/Debian-12-FF69B4.svg)](https://www.debian.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Epistemology: Reflexive](https://img.shields.io/badge/Epistemology-Reflexive-orange.svg)]()

---

## 🎯 Visión Interdisciplinaria

Este proyecto establece un puente entre la sabiduría contemplativa tibetana y la computación cuántica moderna, proponiendo que el **Bardo Thödol** (Libro Tibetano de los Muertos) puede ser interpretado como un algoritmo ancestral susceptible de modelado mediante sistemas cuánticos.

> **Marco Metodológico**: Aplicamos el método Madhyamaka de las **Dos Verdades** (saṃvṛti-satya/paramārtha-satya) al modelado computacional, documentando explícitamente las paradojas irresolibles inherentes al proyecto.

---

## ⚠️ ADVERTENCIA EPISTEMOLÓGICA FUNDAMENTAL

Este modelo es **upāya** (medio hábil), no descripción ontológica:

```
┌─────────────────────────────────────────────────────────────┐
│  "Las palabras son como un dedo apuntando a la luna.       │
│   El dedo no es la luna."                                   │
│                                    — Laṅkāvatāra Sūtra      │
│                                                              │
│  Este modelo computacional es el DEDO.                      │
│  La experiencia directa del Bardo es la LUNA.               │
└─────────────────────────────────────────────────────────────┘
```

### Paradojas Documentadas

| # | Paradoja | Brecha Irreducible | Valor Pedagógico |
|---|----------|-------------------|------------------|
| **1** | **Cuantificación Kármica** | Parámetros numéricos reifican flujo impermanente | Explorar dependencias sin afirmar identidad |
| **2** | **Reificación de Vacuidad** | Vector \|2⟩ cosifica śūnyatā | Demostrar necesidad de lógicas no-binarias |
| **3** | **Temporalidad Artificial** | Tiempo matemático vs experiencia atemporal | Mostrar dinámica como proceso |
| **4** | **Dualismo Observacional** | Mantiene sujeto-objeto ausente en rigpa | Analogía útil para decoherencia |

---

## 🧠 Marco Teórico Fundamental

### Sistema de Estados Cuánticos (Qutrit)

**NIVEL CONVENCIONAL** (saṃvṛti-satya):

| Estado | Representación | Interpretación | Operador |
|--------|----------------|---------------|----------|
| \|0⟩ | `[1, 0, 0]ᵀ` | **Samsara** - Manifestación | `P₀ = |0⟩⟨0|` |
| \|1⟩ | `[0, 1, 0]ᵀ` | **Potencial Kármico** | `P₁ = |1⟩⟨1|` |
| \|2⟩ | `[0, 0, 1]ᵀ` | **Señala hacia Śūnyatā** | `P₂ = |2⟩⟨2|` |

**NIVEL ÚLTIMO** (paramārtha-satya):
- Los tres estados NO son realidades separadas
- Interpenetran sin frontera fija
- La separación es convención pedagógica

### Hamiltoniano Kármico

```math
\hat{H}_K = \sum_{i≠j} k_{ij}(|i⟩⟨j| + |j⟩⟨i|) + \sum_i \epsilon_i |i⟩⟨i|
```

⚠️ **Sujeto a Paradoja #1**: Los parámetros `k_ij` no SON el karma, lo MODELAN convencionalmente.

---

## 📂 Estructura del Proyecto

```
BardoThodol/
├── main.py                    # Sistema principal con reflexividad epistemológica
├── main.tex                   # Paper académico con metamodelado explícito
├── references.bib             # Bibliografía interdisciplinaria
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias (QuTiP, NumPy, etc.)
├── figures/                   # Visualizaciones generadas
│   ├── state_evolution.png
│   ├── bloch_sphere_qutrit.png
│   └── quantum_coherence.png
├── src/
│   ├── quantum_system.py      # Clases BardoQuantumSystem
│   ├── quantum_metrics.py     # Métricas cuánticas centralizadas
│   ├── quantum_analytics.py   # Análisis unificado (sin duplicación)
│   └── visualization.py       # Visualizaciones científicas
├── papers/
│   ├── main.tex               # Documento LaTeX principal
│   ├── main.pdf               # PDF generado
│   └── figures/               # Figuras para el paper
├── simulations/
│   ├── bardo_transitions/     # Datos de simulaciones
│   └── epistemic_notes/       # Notas sobre limitaciones del modelo
├── docs/
│   └── epistemology.md        # Discusión epistemológica extendida
└── tests/
    └── test_paradoxes.py      # Tests que verifican coherencia de paradojas
```

---

## 💻 Instalación y Uso

### Requisitos Previos

```bash
# Debian 12 / Ubuntu
sudo apt update
sudo apt install python3.11 python3-pip texlive-full

# Python dependencies
pip install -r requirements.txt
```

### Ejecución Básica

```python
from src.quantum_system import BardoQuantumSystem

# Crear sistema con parámetros kármicos
system = BardoQuantumSystem(karma_params={
    'clarity': 0.85,      # ⚠️ Convención numérica, no realidad
    'attachment': 0.25,
    'compassion': 0.92
})

# Ejecutar simulación completa
results, times, analysis = system.run_complete_simulation()

# Revisar advertencias epistemológicas
print("Limitaciones del modelo:")
for key, warning in analysis['epistemic_warnings'].items():
    print(f"  • {warning}")

# Resultados convencionales
print(f"\nEstado final: {analysis['final_state_classification']['dominant_state']}")
print(f"Nota: {analysis['final_state_classification']['note']}")
```

### Simulación Avanzada con Transparencia

```python
from src.quantum_system import BardoQuantumSystem
from src.visualization import QuantumVisualizer

# Sistema con karma temporal (no estático)
def karma_evolutivo(t):
    """Karma como función del tiempo - reconoce impermanencia"""
    return {
        'clarity': 0.9 - 0.1 * np.exp(-t/5),
        'attachment': 0.5 * np.exp(-t/3),
        'compassion': 0.85 + 0.1 * np.tanh(t/4)
    }

system = BardoQuantumSystem(
    karma_function=karma_evolutivo,  # NO estático
    attention_function=lambda t: 0.9
)

results, times, analysis = system.run_complete_simulation()

# Visualización con notas epistemológicas
viz = QuantumVisualizer()
fig = viz.create_comprehensive_visualization(
    results, times,
    include_epistemic_notes=True  # Incluye advertencias en gráficos
)
fig.savefig('bardo_simulation_with_notes.png', dpi=300)
```

---

## 📊 Resultados y Visualizaciones

### 1. Evolución Temporal (Nivel Convencional)

![State Evolution](figures/state_evolution.png)

**Nota epistemológica**: Estas trayectorias son formalmente válidas pero no describen experiencia contemplativa directa.

### 2. Métricas Cuánticas

| Métrica | Chikhai Bardo | Chönyid Bardo | Sidpa Bardo |
|---------|---------------|---------------|-------------|
| Coherencia | 0.95 ± 0.02 | 0.87 ± 0.04 | 0.45 ± 0.07 |
| Pureza | 0.98 ± 0.01 | 0.92 ± 0.03 | 0.78 ± 0.06 |
| Entropía | 0.12 ± 0.03 | 0.28 ± 0.05 | 0.65 ± 0.08 |

⚠️ La coherencia cuántica es **análoga** (no idéntica) a la interpenetración no-dual.

---

## 🎯 Características Principales

### ✅ Implementado

- [x] Sistema de qutrits con estados |0⟩, |1⟩, |2⟩
- [x] Operadores kármicos parametrizables **con advertencias explícitas**
- [x] Evolución temporal unitaria
- [x] `QuantumMetrics` centralizada (sin duplicación)
- [x] `QuantumAnalytics` unificada (evita redundancia)
- [x] Visualizaciones científicas con notas epistemológicas
- [x] Paper académico en LaTeX con metamodelado reflexivo
- [x] **Documentación explícita de paradojas irresolibles**
- [x] Tests de consistencia epistémica

### 🚧 En Desarrollo

- [ ] Integración con hardware cuántico real (IBM Q) *con advertencias*
- [ ] Validación con datos de meditación avanzada *como correlato, no identidad*
- [ ] Extensión a sistemas de 5 y 7 estados *manteniendo transparencia*
- [ ] Interfaz web interactiva con explicaciones epistemológicas

---

## 📚 Base Teórica y Referencias

### Fundamentos Filosóficos

- **Bardo Thödol**: Texto base de la tradición Nyingma tibetana
- **Filosofía Madhyamaka**: Doctrina de las Dos Verdades (Nāgārjuna)
- **Prajñāpāramitā**: Advertencias sobre reificación de vacuidad

### Fundamentos Científicos

- **Computación Cuántica**: Qutrits y sistemas de múltiples estados
- **Teoría de la Información**: Métricas de coherencia y entropía
- **Neurofenomenología**: Varela, Thompson & Rosch (1991)

### Epistemología Reflexiva

- **Metamodelado**: Modelos que incorporan crítica a sí mismos
- **Upāya-kauśalya**: Medios hábiles sin afirmación ontológica

---

## 🔬 Validación Científica

### Métricas Implementadas

```python
class QuantumMetrics:
    @staticmethod
    def coherence(state):
        """Coherencia cuántica - ANALOGÍA de no-dualidad"""
        # ...
    
    @staticmethod
    def von_neumann_entropy(state):
        """Entropía - cuantifica indeterminación formal"""
        # ...
```

### Limitaciones Documentadas

1. **Brecha fenomenológica**: No captura experiencia directa (pratyakṣa)
2. **Reduccionismo paramétrico**: Karma cuantificado contradice pratītyasamutpāda
3. **Temporalidad artificial**: Tiempo matemático ≠ experiencia atemporal
4. **Dualismo observacional**: Mantiene separación ausente en rigpa
5. **Cosificación de vacuidad**: |2⟩ contradice niḥsvabhāva

---

## 🎓 Origen Conceptual

El proyecto surgió de una crítica fundamental: clasificar estados del Bardo como "ERROR 505" revelaba la **insuficiencia de marcos binarios** para representar:

- Estados de superposición no colapsados
- No-dualidad de śūnyatā
- Potencialidad kármica latente

La solución requirió:
1. **Qutrits** en lugar de bits (tres estados base)
2. **Metamodelado** en lugar de modelado ingenuo
3. **Transparencia epistemológica** sobre límites del formalismo

---

## 👨‍💻 Autor y Contribuciones

### Autor Principal

**Horacio Héctor Hamann**

- 📧 GitHub: [https://github.com/arathorian/BardoTodol](https://github.com/arathorian/BardoTodol)
- 🔬 Áreas: Computación Cuántica, Filosofía de la Mente, Epistemología del Modelado

### Línea Temporal

- **Enero 2025**: Investigación teórica inicial
- **Marzo 2025**: Desarrollo del framework cuántico
- **Mayo 2025**: Implementación de simulaciones
- **Julio 2025**: Publicación inicial
- **Noviembre 2025**: Integración de metamodelado epistemológico

### Cómo Contribuir

1. Fork el proyecto
2. Crea rama: `git checkout -b feature/AmazingFeature`
3. **Documenta paradojas** si añades nuevos elementos formales
4. Commit: `git commit -m 'Add feature with epistemic transparency'`
5. Push: `git push origin feature/AmazingFeature`
6. Abre Pull Request **incluyendo reflexión epistemológica**

---

## 📖 Compilación del Paper

```bash
cd papers/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

El PDF generado incluye:
- 4 paradojas formalizadas como teoremas
- Tabla de Dos Verdades
- Advertencias epistemológicas distribuidas
- Código corregido sin inconsistencias
- Conclusión con marco metodológico explícito

---

## 🌟 Principio Rector

```
Este proyecto NO afirma que:
  ❌ El Bardo Thodol "es" un algoritmo cuántico
  ❌ La conciencia "es" un sistema de qutrits
  ❌ El karma "es" un operador matemático

Este proyecto PROPONE que:
  ✅ El formalismo cuántico puede usarse como upāya
  ✅ Las matemáticas señalan estructuras sin describir esencias
  ✅ El modelado reflexivo explicita sus propias limitaciones
```

---

## 📜 Licencia

MIT License - Ver `LICENSE` para detalles.

**Nota adicional**: El conocimiento contemplativo del Bardo Thödol pertenece a la tradición Nyingma tibetana. Este proyecto es una interpretación computacional respetuosa que no reemplaza ni pretende ser equivalente a la práctica tradicional.

---

## 🙏 Agradecimientos

- Tradición Nyingma por preservar el Bardo Thödol
- Nāgārjuna por el método Madhyamaka
- Francisco Varela por neurofenomenología
- Comunidad QuTiP por herramientas cuánticas

---

**Recuerda**: El mapa no es el territorio. El modelo no es la experiencia. El dedo no es la luna.

🌙 ☝️
