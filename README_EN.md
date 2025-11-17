# 🌀 Bardo Thodol Quantum Simulation Project
*Quantum Simulation of Consciousness States with Epistemological Transparency*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![QuTiP](https://img.shields.io/badge/QuTiP-4.7+-green.svg)](https://qutip.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 🎯 Interdisciplinary Vision

This project establishes a bridge between Tibetan contemplative wisdom and modern quantum computing, proposing that the **Bardo Thodol** (Tibetan Book of the Dead) can be interpreted as an ancestral algorithm susceptible to modeling via quantum systems.

> **Methodological Framework**: We apply the Madhyamaka method of **Two Truths** (saṃvṛti-satya/paramārtha-satya) to computational modeling, explicitly documenting irreducible paradoxes inherent to the project.

---

## ⚠️ FUNDAMENTAL EPISTEMOLOGICAL WARNING

This model is **upāya** (skillful means), not ontological description:

```
┌─────────────────────────────────────────────────────────────┐
│  "Words are like a finger pointing at the moon.              │
│   The finger is not the moon."                               │
│                                    — Laṅkāvatāra Sūtra       │
│                                                              │
│  This computational model is the FINGER.                     │
│  Direct experience of the Bardo is the MOON.                 │
└─────────────────────────────────────────────────────────────┘
```

### Documented Paradoxes

| # | Paradox | Irreducible Gap | Pedagogical Value |
|---|---------|----------------|-------------------|
| **1** | **Karmic Quantification** | Numerical parameters reify impermanent flux | Explore dependencies without affirming identity |
| **2** | **Emptiness Reification** | Vector \|2⟩ reifies śūnyatā | Demonstrate need for non-binary logics |
| **3** | **Artificial Temporality** | Mathematical time vs atemporal experience | Show dynamics as process |
| **4** | **Observational Dualism** | Maintains subject-object absent in rigpa | Useful analogy for decoherence |

---

## 🧠 Fundamental Theoretical Framework

### Quantum State System (Qutrit)

**CONVENTIONAL LEVEL** (saṃvṛti-satya):

| State | Representation | Interpretation | Operator |
|-------|----------------|---------------|----------|
| \|0⟩ | `[1, 0, 0]ᵀ` | **Samsara** - Manifestation | `P₀ = |0⟩⟨0|` |
| \|1⟩ | `[0, 1, 0]ᵀ` | **Karmic Potential** | `P₁ = |1⟩⟨1|` |
| \|2⟩ | `[0, 0, 1]ᵀ` | **Points toward Śūnyatā** | `P₂ = |2⟩⟨2|` |

**ULTIMATE LEVEL** (paramārtha-satya):
- Three states are NOT separate realities
- They interpenetrate without fixed boundary
- Separation is pedagogical convention

### Karmic Hamiltonian

```math
\hat{H}_K = \sum_{i≠j} k_{ij}(|i⟩⟨j| + |j⟩⟨i|) + \sum_i \epsilon_i |i⟩⟨i|
```

⚠️ **Subject to Paradox #1**: Parameters `k_ij` are not karma, they MODEL it conventionally.

---

## 📂 Project Structure

```
BardoThodol/
├── main.py                    # Main system with epistemological reflexivity
├── main_eng.tex               # Academic paper with explicit meta-modeling
├── references.bib             # Interdisciplinary bibliography
├── README.md                  # This file
├── requirements.txt           # Dependencies (QuTiP, NumPy, etc.)
├── figures/                   # Generated visualizations
│   ├── state_evolution.png
│   ├── bloch_sphere_qutrit.png
│   └── quantum_coherence.png
├── src/
│   ├── quantum_system.py      # BardoQuantumSystem classes
│   ├── quantum_metrics.py     # Centralized quantum metrics
│   ├── quantum_analytics.py   # Unified analysis (no duplication)
│   └── visualization.py       # Scientific visualizations
└── simulations/
    ├── bardo_transitions/     # Simulation data
    └── epistemic_notes/       # Notes on model limitations
```

---

## 💻 Installation and Usage

### Prerequisites

```bash
# Debian 12 / Ubuntu
sudo apt update
sudo apt install python3.11 python3-pip texlive-full

# Python dependencies
pip install -r requirements.txt
```

### Basic Execution

```python
from src.quantum_system import BardoQuantumSystem

# Create system with karmic parameters
system = BardoQuantumSystem(karma_params={
    'clarity': 0.85,      # ⚠️ Numerical convention, not reality
    'attachment': 0.25,
    'compassion': 0.92
})

# Run complete simulation
results, times, analysis = system.run_complete_simulation()

# Review epistemological warnings
print("Model limitations:")
for key, warning in analysis['epistemic_warnings'].items():
    print(f"  • {warning}")

# Conventional results
print(f"\nFinal state: {analysis['final_state_classification']['dominant_state']}")
print(f"Note: {analysis['final_state_classification']['note']}")
```

### Advanced Simulation with Transparency

```python
from src.quantum_system import BardoQuantumSystem
from src.visualization import QuantumVisualizer

# System with temporal karma (not static)
def evolutionary_karma(t):
    """Karma as time function - recognizes impermanence"""
    return {
        'clarity': 0.9 - 0.1 * np.exp(-t/5),
        'attachment': 0.5 * np.exp(-t/3),
        'compassion': 0.85 + 0.1 * np.tanh(t/4)
    }

system = BardoQuantumSystem(
    karma_function=evolutionary_karma,  # NOT static
    attention_function=lambda t: 0.9
)

results, times, analysis = system.run_complete_simulation()

# Visualization with epistemological notes
viz = QuantumVisualizer()
fig = viz.create_comprehensive_visualization(
    results, times,
    include_epistemic_notes=True  # Include warnings in graphs
)
fig.savefig('bardo_simulation_with_notes.png', dpi=300)
```

---

## 📊 Results and Visualizations

### 1. Temporal Evolution (Conventional Level)

**Epistemological note**: These trajectories are formally valid but do not describe direct contemplative experience.

### 2. Quantum Metrics

| Metric | Chikhai Bardo | Chönyid Bardo | Sidpa Bardo |
|--------|---------------|---------------|-------------|
| Coherence | 0.95 ± 0.02 | 0.87 ± 0.04 | 0.45 ± 0.07 |
| Purity | 0.98 ± 0.01 | 0.92 ± 0.03 | 0.78 ± 0.06 |
| Entropy | 0.12 ± 0.03 | 0.28 ± 0.05 | 0.65 ± 0.08 |

⚠️ Quantum coherence is **analogous** (not identical) to non-dual interpenetration.

---

## 🎯 Key Features

### ✅ Implemented

- [x] Qutrit system with states |0⟩, |1⟩, |2⟩
- [x] Parametrizable karmic operators **with explicit warnings**
- [x] Unitary temporal evolution
- [x] Centralized `QuantumMetrics` (no duplication)
- [x] Unified `QuantumAnalytics` (avoids redundancy)
- [x] Scientific visualizations with epistemological notes
- [x] Academic paper in LaTeX with reflexive meta-modeling
- [x] **Explicit documentation of irreducible paradoxes**
- [x] Epistemic consistency tests

### 🚧 In Development

- [ ] Integration with real quantum hardware (IBM Q) *with warnings*
- [ ] Validation with advanced meditation data *as correlate, not identity*
- [ ] Extension to 5 and 7-state systems *maintaining transparency*
- [ ] Interactive web interface with epistemological explanations

---

## 📚 Theoretical Basis and References

### Philosophical Foundations

- **Bardo Thodol**: Base text from Tibetan Nyingma tradition
- **Madhyamaka Philosophy**: Two Truths doctrine (Nāgārjuna)
- **Prajñāpāramitā**: Warnings against emptiness reification

### Scientific Foundations

- **Quantum Computing**: Qutrits and multi-state systems
- **Information Theory**: Coherence and entropy metrics
- **Neurophenomenology**: Varela, Thompson & Rosch (1991)

### Reflexive Epistemology

- **Meta-modeling**: Models incorporating self-criticism
- **Upāya-kauśalya**: Skillful means without ontological affirmation

---

## 🔬 Scientific Validation

### Implemented Metrics

```python
class QuantumMetrics:
    @staticmethod
    def coherence(state):
        """Quantum coherence - ANALOGY of non-duality"""
        # ...

    @staticmethod
    def von_neumann_entropy(state):
        """Entropy - quantifies formal indeterminacy"""
        # ...
```

### Documented Limitations

1. **Phenomenological gap**: Does not capture direct experience (pratyakṣa)
2. **Parametric reductionism**: Quantified karma contradicts pratītyasamutpāda
3. **Artificial temporality**: Mathematical time ≠ atemporal experience
4. **Observational dualism**: Maintains separation absent in rigpa
5. **Emptiness reification**: |2⟩ contradicts niḥsvabhāva

---

## 🎓 Conceptual Origin

Project arose from fundamental critique: classifying Bardo states as "ERROR 505" revealed **binary framework insufficiency** for representing:

- Uncollapsed superposition states
- Non-duality of śūnyatā
- Latent karmic potentiality

Solution required:
1. **Qutrits** instead of bits (three basis states)
2. **Meta-modeling** instead of naive modeling
3. **Epistemological transparency** on formalism limits

---

## 👨‍💻 Author and Contributions

### Principal Author

**Horacio Hector Hamann**

- 📧 GitHub: [https://github.com/arathorian/BardoThodol](https://github.com/arathorian/BardoThodol)
- 🔬 Areas: Quantum Computing, Philosophy of Mind, Modeling Epistemology

### Timeline

- **January 2025**: Initial theoretical research
- **March 2025**: Quantum framework development
- **May 2025**: Simulation implementation
- **July 2025**: Initial publication
- **November 2025**: Epistemological meta-modeling integration

### How to Contribute

1. Fork the project
2. Create branch: `git checkout -b feature/AmazingFeature`
3. **Document paradoxes** if adding new formal elements
4. Commit: `git commit -m 'Add feature with epistemic transparency'`
5. Push: `git push origin feature/AmazingFeature`
6. Open Pull Request **including epistemological reflection**

---

## 📖 Paper Compilation

```bash
cd papers/
pdflatex main_eng.tex
bibtex main_eng
pdflatex main_eng.tex
pdflatex main_eng.tex
```

Generated PDF includes:
- 4 paradoxes formalized as theorems
- Two Truths table
- Distributed epistemological warnings
- Corrected code without inconsistencies
- Conclusion with explicit methodological framework

---

## 🌟 Guiding Principle

```
This project does NOT affirm that:
  ❌ Bardo Thodol "is" a quantum algorithm
  ❌ Consciousness "is" a qutrit system
  ❌ Karma "is" a mathematical operator

This project PROPOSES that:
  ✅ Quantum formalism can be used as upāya
  ✅ Mathematics point to structures without describing essences
  ✅ Reflexive modeling explicates its own limitations
```

---

## 📜 License

This project is licensed under MIT License - see [LICENSE](LICENSE) file for details.

**Additional note**: Contemplative knowledge of Bardo Thodol belongs to Tibetan Nyingma tradition. This project is a respectful computational interpretation not replacing or claiming equivalence to traditional practice.

---

## 🙏 Acknowledgments

- Nyingma tradition for preserving Bardo Thodol
- Nāgārjuna for Madhyamaka method
- Francisco Varela for neurophenomenology
- QuTiP community for quantum tools

---

## 📚 How to Cite This Work

If you use this code or paper in your research, please cite:

### Paper
```bibtex
@article{hamann2025bardo,
  title={Quantum Simulation of Consciousness States in the Bardo Thodol: A Computational Approach via Qutrit Theory and Karmic Dynamics},
  author={Hamann, Horacio Hector},
  year={2025},
  journal={Preprint},
  url={https://github.com/arathorian/BardoThodol},
  doi={10.5281/zenodo.XXXXXXX}
}
```

### Software
```bibtex
@software{hamann2025bardosoftware,
  title={Bardo Thodol Quantum Simulation},
  author={Hamann, Horacio Hector},
  year={2025},
  version={1.0.0},
  url={https://github.com/arathorian/BardoThodol},
  doi={10.5281/zenodo.XXXXXXX}
}
```

---

## 📦 Zenodo Deposit Contents

This repository contains:

- **Source Code**: Complete Python implementation with epistemological reflexivity
- **Academic Paper**: LaTeX manuscript with meta-modeling framework
- **Documentation**: Comprehensive README with paradox documentation
- **Simulation Data**: Example results and metrics
- **Bibliography**: Interdisciplinary references (BibTeX format)

---

## 🔗 Related Resources

- **GitHub Repository**: [https://github.com/arathorian/BardoThodol](https://github.com/arathorian/BardoThodol)
- **QuTiP Documentation**: [https://qutip.org/docs/latest/](https://qutip.org/docs/latest/)
- **Madhyamaka Resources**: [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/madhyamaka/)

---

**Remember**: The map is not the territory. The model is not the experience. The finger is not the moon.

🌙 ☝️