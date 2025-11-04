# 🌀 Bardo Thödol Quantum Simulation Project
*Simulación Cuántica de Estados de Consciencia Basada en el Bardo Thödol*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![QuTiP](https://img.shields.io/badge/QuTiP-4.7+-green.svg)](https://qutip.org/)
[![Debian 12](https://img.shields.io/badge/Debian-12-FF69B4.svg)](https://www.debian.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Visión Interdisciplinaria

Este proyecto establece un puente innovador entre la sabiduría contemplativa tibetana y la computación cuántica moderna, proponiendo que el **Bardo Thödol** (Libro Tibetano de los Muertos) puede ser interpretado como un algoritmo ancestral que describe la dinámica de estados de consciencia, susceptible de modelado mediante sistemas cuánticos de múltiples estados.

> **Hipótesis Central**: Los estados post-mortem descritos en el Bardo Thödol pueden ser simulados mediante sistemas de qutrits, donde la vacuidad (śūnyatā) corresponde a estados de superposición cuántica no colapsados.

---

## 🧠 Marco Teórico Fundamental

### Sistema de Estados Cuánticos (Qutrit)

| Estado | Representación | Interpretación Filosófica | Operador |
|--------|----------------|---------------------------|----------|
| \|0⟩ | `[1, 0, 0]ᵀ` | **Samsara** - Realidad manifiesta | `P₀ = |0⟩⟨0|` |
| \|1⟩ | `[0, 1, 0]ᵀ` | **Potencial Kármico** - Estados latentes | `P₁ = |1⟩⟨1|` |
| \|2⟩ | `[0, 0, 1]ᵀ` | **Śūnyatā** - Vacuidad fundamental | `P₂ = |2⟩⟨2|` |

### Hamiltoniano Kármico

```math
\hat{H}_K = \sum_{i≠j} k_{ij}(|i⟩⟨j| + |j⟩⟨i|) + \sum_i \epsilon_i |i⟩⟨i|