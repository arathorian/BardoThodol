# src/core/quantum_validator.py
"""
Sistema de Validación Científica para el Modelo del Bardo
Incorpora criterios de física cuántica, neurociencia y filosofía budista
"""

import numpy as np
from scipy import stats
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ScientificValidator:
    """Validador con criterios interdisciplinarios rigurosos"""

    def __init__(self):
        self.tolerance = 1e-10
        self.interdisciplinary_constraints = self._define_constraints()

    def _define_constraints(self) -> Dict:
        """Define constraints físicos, cognitivos y filosóficos"""
        return {
            # Constraints de física cuántica
            'quantum_norm': (0.999, 1.001),
            'density_trace': (0.999, 1.001),
            'positive_semidefinite': True,
            
            # Constraints de teoría de información
            'entropy_range': (0.0, np.log(3)),  # Máximo para qutrits
            'purity_range': (1/3, 1.0),  # Mínimo para estado completamente mixto
            
            # Constraints filosóficos del Bardo
            'max_vacuity_prob': 0.95,  # La vacuidad no puede dominar completamente
            'min_manifestation_prob': 0.01,  # Siempre hay algo de manifestación
            'karma_conservation': True  # La información kármica se preserva
        }

    def validate_complete_state(self, quantum_state: 'QuantumState', 
                              initial_conditions: Dict) -> Dict[str, bool]:
        """Validación integral desde múltiples perspectivas"""
        validation_results = {}

        # Validación física cuántica
        validation_results['quantum_physics'] = self._validate_quantum_physics(quantum_state)
        
        # Validación dinámica kármica
        validation_results['karmic_dynamics'] = self._validate_karmic_dynamics(
            quantum_state, initial_conditions)
            
        # Validación filosófica budista
        validation_results['buddhist_philosophy'] = self._validate_buddhist_philosophy(
            quantum_state)

        validation_results['completely_valid'] = all(validation_results.values())
        
        return validation_results

    def _validate_quantum_physics(self, state: 'QuantumState') -> bool:
        """Verifica que el estado obedezca las leyes cuánticas"""
        checks = []
        
        # Conservación de norma
        state_norm = np.linalg.norm(state.statevector)
        min_norm, max_norm = self.interdisciplinary_constraints['quantum_norm']
        checks.append(min_norm <= state_norm <= max_norm)
        
        # Traza de matriz densidad
        trace = np.trace(state.density_matrix)
        min_trace, max_trace = self.interdisciplinary_constraints['density_trace']
        checks.append(min_trace <= trace <= max_trace)
        
        # Semidefinida positiva
        if self.interdisciplinary_constraints['positive_semidefinite']:
            eigenvalues = np.linalg.eigvalsh(state.density_matrix)
            checks.append(np.all(eigenvalues >= -self.tolerance))
            
        return all(checks)

    def _validate_karmic_dynamics(self, state: 'QuantumState', 
                                initial_conditions: Dict) -> bool:
        """Valida la dinámica kármica desde perspectiva de sistemas complejos"""
        # Verificar que la evolución preserve información
        entropy_change = abs(state.entropy - initial_conditions.get('initial_entropy', 0))
        entropy_conserved = entropy_change < 0.1  # Cambio pequeño en entropía
        
        # Verificar bounds kármicos razonables
        karma = initial_conditions.get('karma', 0.5)
        vacuity_prob = state.probabilities['vacuity']
        karma_effect_consistent = (vacuity_prob <= 0.5 + 0.3 * karma)  # Efecto kármico acotado
        
        return entropy_conserved and karma_effect_consistent

    def _validate_buddhist_philosophy(self, state: 'QuantumState') -> bool:
        """Valida consistencia con principios budistas"""
        probs = state.probabilities
        
        # La vacuidad no puede ser total (siempre hay dependencia originada)
        vacuity_valid = probs['vacuity'] <= self.interdisciplinary_constraints['max_vacuity_prob']
        
        # La manifestación nunca es completamente cero
        manifestation_valid = probs['manifested'] >= self.interdisciplinary_constraints['min_manifestation_prob']
        
        # Las probabilidades deben sumar 1 (unidad de la realidad)
        total_prob_valid = np.isclose(sum(probs.values()), 1.0, atol=1e-10)
        
        return vacuity_valid and manifestation_valid and total_prob_valid

class StatisticalAnalyzer:
    """Analizador estadístico para validación empírica"""
    
    def analyze_simulation_results(self, results: List[Dict]) -> Dict:
        """Análisis estadístico completo de resultados de simulación"""
        metrics_df = self._results_to_dataframe(results)
        
        analysis = {
            'descriptive_statistics': self._compute_descriptive_stats(metrics_df),
            'correlation_analysis': self._compute_correlations(metrics_df),
            'hypothesis_tests': self._perform_hypothesis_tests(metrics_df),
            'normality_assessment': self._assess_normality(metrics_df)
        }
        
        return analysis

    def _perform_hypothesis_tests(self, df) -> Dict:
        """Tests de hipótesis específicas del modelo Bardo"""
        tests = {}
        
        # Test: El karma afecta la probabilidad de vacuidad
        high_karma = df[df['karma'] > 0.7]['vacuity_prob']
        low_karma = df[df['karma'] < 0.3]['vacuity_prob']
        
        if len(high_karma) > 0 and len(low_karma) > 0:
            t_stat, p_value = stats.ttest_ind(high_karma, low_karma)
            tests['karma_effect'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'El karma influye significativamente en la vacuidad' 
                if p_value < 0.05 else 'No hay efecto kármico detectable'
            }
            
        return tests