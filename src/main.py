"""
Simulación Cuántica del Bardo Thödol
=====================================

Sistema de modelado con transparencia epistemológica explícita.
Aplica el método Madhyamaka de las Dos Verdades al modelado computacional.

Autor: Horacio Héctor Hamann
Proyecto: https://github.com/arathorian/BardoTodol
Fecha: Noviembre 2025
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# =============================================================================
# ADVERTENCIAS EPISTEMOLÓGICAS GLOBALES
# =============================================================================

EPISTEMIC_WARNINGS = {
    'karma_quantification': (
        'PARADOJA #1: Los parámetros numéricos del karma reifican '
        'lo que el Abhidharma describe como flujo impermanente (anitya). '
        'Valor pedagógico: explorar dependencias sin afirmar identidad numérica.'
    ),
    'sunyata_reification': (
        'PARADOJA #2: Representar śūnyatā como vector |2⟩ contradice '
        'su naturaleza de niḥsvabhāva (ausencia de ser inherente). '
        'Valor pedagógico: señalar hacia necesidad de lógicas no-binarias.'
    ),
    'temporal_parameter': (
        'PARADOJA #3: El tiempo t es parámetro matemático, no refleja '
        'experiencia atemporal de samādhi. Kāla es construcción mental. '
        'Valor pedagógico: mostrar dinámica como proceso secuencial.'
    ),
    'measurement_duality': (
        'PARADOJA #4: El formalismo mantiene separación observador-observado '
        'ausente en rigpa (conciencia no-dual). '
        'Valor pedagógico: analogía útil para decoherencia.'
    )
}


def print_epistemic_banner():
    """Imprime advertencia epistemológica al iniciar"""
    print("="*70)
    print(" BARDO THÖDOL QUANTUM SIMULATION".center(70))
    print(" Con Transparencia Epistemológica Explícita".center(70))
    print("="*70)
    print("\n⚠️  ADVERTENCIA METODOLÓGICA:")
    print("   Este modelo es UPĀYA (medio hábil), no descripción ontológica.")
    print("   El dedo que señala la luna no es la luna misma.\n")
    print("="*70 + "\n")


# =============================================================================
# MÉTRICAS CUÁNTICAS CENTRALIZADAS
# =============================================================================

class QuantumMetrics:
    """
    Clase centralizada para cálculo de métricas cuánticas.
    Evita duplicación de código entre BardoQuantumSystem y QuantumVisualizer.
    """
    
    @staticmethod
    def coherence(state: qt.Qobj) -> float:
        """
        Calcula coherencia cuántica (norma l1 de elementos fuera de diagonal).
        
        NOTA EPISTEMOLÓGICA: Esta métrica es ANÁLOGA (no idéntica) a la
        interpenetración no-dual fenomenológica.
        """
        if state.type == 'ket':
            rho = state * state.dag()
        else:
            rho = state
        
        rho_array = rho.full()
        n = rho_array.shape[0]
        
        coh = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    coh += abs(rho_array[i, j])
        
        return float(coh)
    
    @staticmethod
    def purity(state: qt.Qobj) -> float:
        """
        Calcula pureza del estado: Tr(ρ²).
        
        NOTA EPISTEMOLÓGICA: Pureza = 1 indica estado puro, no necesariamente
        claridad contemplativa (prajñā).
        """
        if state.type == 'ket':
            return 1.0
        else:
            rho = state
            return float(np.real((rho * rho).tr()))
    
    @staticmethod
    def von_neumann_entropy(state: qt.Qobj) -> float:
        """
        Calcula entropía de Von Neumann: -Tr(ρ log₂ ρ).
        
        NOTA EPISTEMOLÓGICA: Cuantifica indeterminación formal,
        no la "confusión" mental contemplativa.
        """
        if state.type == 'ket':
            rho = state * state.dag()
        else:
            rho = state
        
        eigvals = rho.eigenvalues()
        entropy = 0.0
        
        for v in eigvals:
            if v > 1e-12:  # Umbral numérico para estabilidad
                entropy -= v * np.log2(v)
        
        return float(entropy)


# =============================================================================
# ANÁLISIS CUÁNTICO CENTRALIZADO
# =============================================================================

class QuantumAnalytics:
    """
    Sistema centralizado de análisis para evitar duplicación de código.
    Implementa métodos compartidos por BardoQuantumSystem y QuantumVisualizer.
    """
    
    @staticmethod
    def analyze_transitions(
        probabilities: np.ndarray,
        threshold: float = 0.1
    ) -> List[Dict]:
        """
        Analiza transiciones significativas entre estados.
        
        Args:
            probabilities: Array (N_steps, 3) de probabilidades
            threshold: Umbral para detectar transición significativa
        
        Returns:
            Lista de diccionarios con información de transiciones
        """
        probs = np.array(probabilities)
        transitions = []
        
        for i in range(1, len(probs)):
            changes = np.abs(probs[i] - probs[i-1])
            max_change = np.max(changes)
            
            if max_change > threshold:
                transitions.append({
                    'time_index': i,
                    'magnitude': float(max_change),
                    'from_state': int(np.argmax(probs[i-1])),
                    'to_state': int(np.argmax(probs[i])),
                    'change_vector': changes.tolist()
                })
        
        return transitions
    
    @staticmethod
    def find_dominant_state(probabilities: np.ndarray) -> Dict:
        """
        Analiza estado dominante a lo largo del tiempo.
        
        NOTA EPISTEMOLÓGICA: "Dominante" es convención matemática,
        no indica realidad ontológica separada.
        """
        probs = np.array(probabilities)
        dominant_states = np.argmax(probs, axis=1)
        total_steps = len(dominant_states)
        
        return {
            'dominant_states': dominant_states.tolist(),
            'time_in_samsara': int(np.sum(dominant_states == 0)),
            'time_in_karmic': int(np.sum(dominant_states == 1)),
            'time_in_void': int(np.sum(dominant_states == 2)),
            'dominance_ratio': {
                'samsara': float(np.sum(dominant_states == 0) / total_steps),
                'karmic': float(np.sum(dominant_states == 1) / total_steps),
                'void': float(np.sum(dominant_states == 2) / total_steps)
            }
        }
    
    @staticmethod
    def calculate_stability(probabilities: np.ndarray) -> Dict:
        """Calcula métricas de estabilidad del sistema"""
        probs = np.array(probabilities)
        variances = np.var(probs, axis=0)
        gradients = np.gradient(probs, axis=0)
        gradient_norms = np.linalg.norm(gradients, axis=1)
        
        return {
            'variance_per_state': variances.tolist(),
            'overall_stability': float(1.0 - np.mean(variances)),
            'stationary_indices': np.where(gradient_norms < 0.01)[0].tolist(),
            'max_gradient': float(np.max(gradient_norms))
        }


# =============================================================================
# SISTEMA CUÁNTICO PRINCIPAL
# =============================================================================

@dataclass
class KarmaParameters:
    """
    Parámetros kármicos con validación.
    
    ⚠️ SUJETO A PARADOJA #1: Estos valores reifican karma como magnitud,
    contradiciendo su naturaleza de proceso interdependiente (pratītyasamutpāda).
    """
    clarity: float = 0.8
    attachment: float = 0.3
    compassion: float = 0.9
    wisdom: float = 0.7
    
    def __post_init__(self):
        """Valida que parámetros estén en rango [0,1]"""
        for name, value in [
            ('clarity', self.clarity),
            ('attachment', self.attachment),
            ('compassion', self.compassion),
            ('wisdom', self.wisdom)
        ]:
            if not 0 <= value <= 1:
                raise ValueError(
                    f"Parámetro '{name}' debe estar en [0,1], recibido: {value}"
                )


class BardoQuantumSystem:
    """
    Sistema cuántico del Bardo Thödol con reflexividad epistemológica.
    
    Este sistema:
    - Modela formalmente transiciones entre estados (nivel convencional)
    - Documenta explícitamente sus limitaciones (nivel último)
    - Se usa como herramienta heurística (nivel pedagógico/upāya)
    """
    
    def __init__(
        self,
        karma_params: Optional[KarmaParameters] = None,
        karma_function: Optional[Callable[[float], Dict[str, float]]] = None,
        attention_function: Optional[Callable[[float], float]] = None
    ):
        """
        Inicializa el sistema cuántico.
        
        Args:
            karma_params: Parámetros kármicos estáticos (sujeto a Paradoja #1)
            karma_function: Función t → karma(t) para karma temporal
            attention_function: Función t → atención(t)
        """
        self.karma_params = karma_params or KarmaParameters()
        self.karma_function = karma_function
        self.attention_function = attention_function or self._default_attention
        
        self.dim = 3
        self.metrics = QuantumMetrics()
        self.analytics = QuantumAnalytics()
        
        # Crear operadores cuánticos
        self.operators = self._create_operators()
        self.current_state = qt.basis(self.dim, 2)  # Iniciar en |2⟩ (vacuidad)
        
        # Documentar limitaciones del modelo
        self.model_limitations = EPISTEMIC_WARNINGS.copy()
    
    def _create_operators(self) -> Dict[str, qt.Qobj]:
        """
        Crea operadores cuánticos fundamentales.
        
        Returns:
            Diccionario con operadores P0, P1, P2, S01, S12, S20, H0, K
        """
        # Estados base
        kets = [qt.basis(3, i) for i in range(3)]
        
        # Proyectores: P_i = |i⟩⟨i|
        P = {f'P{i}': kets[i] * kets[i].dag() for i in range(3)}
        
        # Operadores de transición
        S01 = kets[0] * kets[1].dag()
        S12 = kets[1] * kets[2].dag()
        S20 = kets[2] * kets[0].dag()
        
        # Hamiltoniano base (energías de estados)
        H0 = 0.1 * P['P0'] + 0.2 * P['P1'] + 0.3 * P['P2']
        
        # Operador kármico (⚠️ sujeto a Paradoja #1)
        K = (self.karma_params.attachment * (S01 + S01.dag()) +
             self.karma_params.clarity * (S12 + S12.dag()) +
             self.karma_params.compassion * (S20 + S20.dag()))
        
        # Actualizar diccionario con operadores de transición y Hamiltonianos
        P.update({
            'S01': S01, 'S12': S12, 'S20': S20,
            'H0': H0, 'K': K
        })
        
        return P
    
    def _default_attention(self, t: float) -> float:
        """
        Función de atención por defecto (logística).
        
        ⚠️ SUJETO A PARADOJA #3: Modela atención como función del tiempo,
        pero en samādhi profundo no hay experiencia temporal lineal.
        """
        return 1.0 / (1.0 + np.exp(-0.5 * (t - 2*np.pi)))
    
    def _get_karma_at_time(self, t: float) -> Dict[str, float]:
        """Obtiene parámetros kármicos en tiempo t"""
        if self.karma_function:
            return self.karma_function(t)
        else:
            return {
                'clarity': self.karma_params.clarity,
                'attachment': self.karma_params.attachment,
                'compassion': self.karma_params.compassion
            }
    
    def simulate_bardo_transition(
        self,
        time_steps: int = 1000,
        time_span: float = 4*np.pi
    ) -> Tuple[Dict, np.ndarray]:
        """
        Simula transición completa a través de estados del Bardo.
        
        NIVEL CONVENCIONAL (saṃvṛti-satya):
        Evolución unitaria formalmente válida en espacio de Hilbert.
        
        NIVEL ÚLTIMO (paramārtha-satya):
        No describe experiencia contemplativa directa (pratyakṣa).
        
        Args:
            time_steps: Número de pasos temporales
            time_span: Duración total de la simulación
        
        Returns:
            (results, times) donde results contiene probabilidades,
            coherencia, pureza, entropía y estados
        """
        times = np.linspace(0, time_span, time_steps)
        results = {
            'probabilities': [],
            'coherence': [],
            'purity': [],
            'entropy': [],
            'states': []
        }
        
        current_state = self.current_state
        
        for t in times:
            # Factor de atención en tiempo t
            attention = self.attention_function(t)
            
            # Obtener karma en tiempo t (si es función temporal)
            karma_t = self._get_karma_at_time(t)
            
            # Reconstruir operador kármico si es necesario
            if self.karma_function:
                kets = [qt.basis(3, i) for i in range(3)]
                S01 = kets[0] * kets[1].dag()
                S12 = kets[1] * kets[2].dag()
                S20 = kets[2] * kets[0].dag()
                K_t = (karma_t['attachment'] * (S01 + S01.dag()) +
                       karma_t['clarity'] * (S12 + S12.dag()) +
                       karma_t['compassion'] * (S20 + S20.dag()))
            else:
                K_t = self.operators['K']
            
            # Hamiltoniano efectivo
            H_eff = self.operators['H0'] + attention * K_t
            
            # Evolución unitaria: U(dt) = exp(-iH·dt)
            dt = times[1] - times[0] if len(times) > 1 else 0.01
            U = (-1j * dt * H_eff).expm()
            current_state = U * current_state
            
            # ✅ CORRECCIÓN: usar self.operators[f'P{i}'] en lugar de qt.projection()
            probs = [
                float(qt.expect(self.operators[f'P{i}'], current_state))
                for i in range(self.dim)
            ]
            
            # Calcular métricas cuánticas
            coherence = self.metrics.coherence(current_state)
            purity = self.metrics.purity(current_state)
            entropy = self.metrics.von_neumann_entropy(current_state)
            
            # Almacenar resultados
            results['probabilities'].append(probs)
            results['coherence'].append(coherence)
            results['purity'].append(purity)
            results['entropy'].append(entropy)
            results['states'].append(current_state)
        
        return results, times
    
    def run_complete_simulation(self) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Ejecuta simulación completa con análisis comprehensivo.
        
        Returns:
            (results, times, analysis_report)
        """
        results, times = self.simulate_bardo_transition()
        probs_array = np.array(results['probabilities'])
        
        # Análisis usando QuantumAnalytics centralizado
        analysis_report = {
            'final_state_classification': self._classify_final_state(
                results['states'][-1]
            ),
            'transitions': self.analytics.analyze_transitions(probs_array),
            'dominant_state_analysis': self.analytics.find_dominant_state(
                probs_array
            ),
            'stability': self.analytics.calculate_stability(probs_array),
            'quantum_metrics': {
                'final_entropy': results['entropy'][-1],
                'avg_coherence': float(np.mean(results['coherence'])),
                'avg_purity': float(np.mean(results['purity'])),
                'max_coherence': float(np.max(results['coherence'])),
                'min_purity': float(np.min(results['purity']))
            },
            'epistemic_warnings': self.model_limitations
        }
        
        return results, times, analysis_report
    
    def _classify_final_state(self, state: qt.Qobj) -> Dict:
        """
        Clasifica el estado final según probabilidades.
        
        NOTA EPISTEMOLÓGICA: Clasificación en nivel convencional (saṃvṛti).
        No indica realidad ontológica separada.
        """
        probs = [
            float(qt.expect(self.operators[f'P{i}'], state))
            for i in range(3)
        ]
        
        state_names = ['Samsara', 'Kármico', 'Vacuidad']
        dominant_idx = int(np.argmax(probs))
        
        return {
            'dominant_state': state_names[dominant_idx],
            'probabilities': probs,
            'certainty': float(max(probs)),
            'superposition_degree': float(1.0 - max(probs)),
            'epistemic_note': (
                'Clasificación en nivel convencional (saṃvṛti-satya). '
                'No describe experiencia contemplativa directa.'
            )
        }


# =============================================================================
# VISUALIZACIÓN CIENTÍFICA
# =============================================================================

class QuantumVisualizer:
    """
    Sistema de visualización científica con notas epistemológicas.
    Usa QuantumAnalytics centralizado para evitar duplicación.
    """
    
    def __init__(self, style: str = 'seaborn'):
        self.analytics = QuantumAnalytics()
        self.metrics = QuantumMetrics()
        plt.style.use(style)
    
    def create_comprehensive_visualization(
        self,
        results: Dict,
        times: np.ndarray,
        include_epistemic_notes: bool = True
    ) -> plt.Figure:
        """
        Crea visualización completa con notas epistemológicas opcionales.
        
        Args:
            results: Diccionario con resultados de simulación
            times: Array de tiempos
            include_epistemic_notes: Si True, incluye advertencias en figura
        
        Returns:
            Figura de matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Evolución de probabilidades
        ax1 = axes[0, 0]
        probs = np.array(results['probabilities'])
        ax1.plot(times, probs[:, 0], label='|0⟩ Samsara', linewidth=2)
        ax1.plot(times, probs[:, 1], label='|1⟩ Kármico', linewidth=2)
        ax1.plot(times, probs[:, 2], label='|2⟩ Vacuidad', linewidth=2)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Probabilidad')
        ax1.legend()
        ax1.grid(True, alpha=0.3)  # ✅ ESTANDARIZADO
        ax1.set_title('Evolución Temporal de Estados')
        
        # 2. Coherencia cuántica
        ax2 = axes[0, 1]
        ax2.plot(times, results['coherence'], color='purple', linewidth=2)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Coherencia Cuántica')
        ax2.grid(True, alpha=0.3)  # ✅ ESTANDARIZADO
        ax2.set_title('Coherencia del Sistema')
        
        # 3. Entropía de Von Neumann
        ax3 = axes[1, 0]
        ax3.plot(times, results['entropy'], color='brown', linewidth=2)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Entropía de Von Neumann')
        ax3.grid(True, alpha=0.3)  # ✅ ESTANDARIZADO
        ax3.set_title('Evolución de la Entropía')
        
        # 4. Pureza del estado
        ax4 = axes[1, 1]
        ax4.plot(times, results['purity'], color='green', linewidth=2)
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Pureza del Estado')
        ax4.grid(True, alpha=0.3)  # ✅ ESTANDARIZADO
        ax4.set_title('Pureza Cuántica')
        
        # Agregar nota epistemológica si se solicita
        if include_epistemic_notes:
            fig.text(
                0.5, 0.02,
                'Nivel Convencional (saṃvṛti-satya): Métricas formalmente válidas\n'
                'No describen experiencia contemplativa directa',
                ha='center', fontsize=9, style='italic',
                color='red', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            )
        
        plt.tight_layout()
        return fig
    
    def generate_analysis_report(
        self,
        results: Dict,
        include_warnings: bool = True
    ) -> Dict:
        """
        Genera reporte de análisis usando QuantumAnalytics centralizado.
        
        Args:
            results: Resultados de simulación
            include_warnings: Si True, incluye advertencias epistemológicas
        
        Returns:
            Diccionario con análisis completo
        """
        probs_array = np.array(results['probabilities'])
        
        report = {
            'final_probabilities': results['probabilities'][-1],
            'max_coherence': float(np.max(results['coherence'])),
            'min_purity': float(np.min(results['purity'])),
            'state_transitions': self.analytics.analyze_transitions(probs_array),
            'dominant_state_analysis': self.analytics.find_dominant_state(probs_array),
            'entropy_analysis': {
                'initial': self.metrics.von_neumann_entropy(results['states'][0]),
                'final': self.metrics.von_neumann_entropy(results['states'][-1]),
                'average': float(np.mean([
                    self.metrics.von_neumann_entropy(s) for s in results['states']
                ]))
            }
        }
        
        if include_warnings:
            report['epistemic_warnings'] = EPISTEMIC_WARNINGS
        
        return report


# =============================================================================
# FUNCIÓN PRINCIPAL DE DEMOSTRACIÓN
# =============================================================================

def main():
    """Función principal de demostración del sistema"""
    
    # Imprimir banner epistemológico
    print_epistemic_banner()
    
    # Configurar parámetros kármicos
    karma = KarmaParameters(
        clarity=0.85,
        attachment=0.25,
        compassion=0.92,
        wisdom=0.75
    )
    
    print("📊 Configuración de Parámetros Kármicos:")
    print(f"   Clarity (claridad): {karma.clarity}")
    print(f"   Attachment (apego): {karma.attachment}")
    print(f"   Compassion (compasión): {karma.compassion}")
    print(f"   Wisdom (sabiduría): {karma.wisdom}")
    print(f"\n⚠️  {EPISTEMIC_WARNINGS['karma_quantification']}\n")
    
    # Crear sistema cuántico
    print("🔬 Inicializando sistema cuántico del Bardo...")
    system = BardoQuantumSystem(karma_params=karma)
    
    # Ejecutar simulación completa
    print("⏳ Ejecutando simulación completa...\n")
    results, times, analysis = system.run_complete_simulation()
    
    # Mostrar resultados
    print("="*70)
    print(" RESULTADOS DE LA SIMULACIÓN".center(70))
    print("="*70 + "\n")
    
    print("📈 Estado Final del Sistema:")
    final_class = analysis['final_state_classification']
    print(f"   Estado dominante: {final_class['dominant_state']}")
    print(f"   Certeza: {final_class['certainty']:.3f}")
    print(f"   Grado de superposición: {final_class['superposition_degree']:.3f}")
    print(f"   Nota: {final_class['epistemic_note']}\n")
    
    print("🔄 Análisis de Transiciones:")
    print(f"   Transiciones detectadas: {len(analysis['transitions'])}")
    print(f"   Estabilidad global: {analysis['stability']['overall_stability']:.3f}\n")
    
    print("📊 Métricas Cuánticas Promedio:")
    metrics = analysis['quantum_metrics']
    print(f"   Coherencia promedio: {metrics['avg_coherence']:.3f}")
    print(f"   Pureza promedio: {metrics['avg_purity']:.3f}")
    print(f"   Entropía final: {metrics['final_entropy']:.3f}\n")
    
    print("🌍 Distribución Temporal de Estados:")
    dom_analysis = analysis['dominant_state_analysis']
    for state, ratio in dom_analysis['dominance_ratio'].items():
        print(f"   {state.capitalize()}: {ratio:.1%}")
    
    print("\n" + "="*70)
    print(" ADVERTENCIAS EPISTEMOLÓGICAS DOCUMENTADAS".center(70))
    print("="*70 + "\n")
    
    for i, (key, warning) in enumerate(analysis['epistemic_warnings'].items(), 1):
        print(f"{i}. {warning}\n")
    
    # Crear visualización
    print("📊 Generando visualizaciones...\n")
    viz = QuantumVisualizer()
    fig = viz.create_comprehensive_visualization(
        results, times,
        include_epistemic_notes=True
    )
    
    output_filename = 'bardo_simulation_results.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Visualización guardada en: {output_filename}")
    
    print("\n" + "="*70)
    print(" REFLEXIÓN FINAL".center(70))
    print("="*70)
    print("\n   Este modelo es el DEDO apuntando a la luna.")
    print("   La experiencia directa del Bardo es la LUNA.")
    print("   No confundir uno con otro es la sabiduría.\n")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
