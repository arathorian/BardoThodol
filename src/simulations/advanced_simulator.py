# src/simulations/advanced_simulator.py
"""
Simulador Avanzado del Bardo Quantum Model
Integra: Simulaciones cuánticas, análisis estadístico y validación científica
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from ..core.quantum_models_optimized import OptimizedBardoModel, QuantumState
from ..core.quantum_validator import ScientificValidator, StatisticalAnalyzer

logger = logging.getLogger(__name__)

class AdvancedBardoSimulator:
    """
    Simulador principal que unifica:
    - Evolución cuántica de estados Bardo
    - Análisis estadístico riguroso
    - Validación científica en tiempo real
    """
    
    def __init__(self, model_dimensions: int = 3):
        self.quantum_model = OptimizedBardoModel(dimensions=model_dimensions)
        self.validator = ScientificValidator()
        self.analyzer = StatisticalAnalyzer()
        
        # Estados iniciales basados en textos budistas
        self.initial_states = {
            'clear_light': np.array([0.1, 0.1, 0.8], dtype=complex),  # Estado de claridad
            'karmic_clouded': np.array([0.6, 0.3, 0.1], dtype=complex),  # Estado kármico
            'balanced': np.array([0.33, 0.33, 0.34], dtype=complex)  # Estado equilibrado
        }

    def run_comprehensive_study(self, num_simulations: int = 1000) -> Dict:
        """
        Ejecuta estudio completo con validación integral
        Incluye análisis de ERROR 505 como estado de vacuidad cuántica
        """
        all_results = []
        validation_reports = []
        
        for i in range(num_simulations):
            # Parámetros aleatorios dentro de rangos válidos
            karma = np.random.beta(2, 2)  # Distribución centrada en 0.5
            attention = np.random.beta(3, 1.5)  # Sesgado hacia alta atención
            initial_state_type = np.random.choice(list(self.initial_states.keys()))
            
            result = self.run_single_experiment(
                initial_state=self.initial_states[initial_state_type],
                karma_strength=karma,
                attention_level=attention,
                experiment_id=i
            )
            
            if result['validation_passed']:
                all_results.append(result)
                validation_reports.append(result['validation_report'])
            
            # Log cada 100 simulaciones
            if (i + 1) % 100 == 0:
                logger.info(f"Completadas {i + 1}/{num_simulations} simulaciones válidas")

        # Análisis estadístico completo
        statistical_analysis = self.analyzer.analyze_simulation_results(all_results)
        
        # Detección de estados ERROR 505 (vacuidad dominante)
        error_505_analysis = self.analyze_error_505_states(all_results)
        
        return {
            'successful_simulations': all_results,
            'validation_summary': self._summarize_validations(validation_reports),
            'statistical_analysis': statistical_analysis,
            'error_505_analysis': error_505_analysis,
            'quantum_insights': self._extract_quantum_insights(all_results)
        }

    def run_single_experiment(self, initial_state: np.ndarray,
                            karma_strength: float,
                            attention_level: float,
                            experiment_id: int) -> Dict:
        """Ejecuta un experimento individual con validación completa"""
        # Evolución cuántica
        quantum_state = self.quantum_model.apply_bardo_evolution(
            initial_state, karma_strength, attention_level)
        
        # Validación científica
        initial_conditions = {
            'initial_entropy': -np.sum(np.abs(initial_state)**2 * np.log(np.abs(initial_state)**2 + 1e-12)),
            'karma': karma_strength,
            'attention': attention_level
        }
        
        validation_report = self.validator.validate_complete_state(
            quantum_state, initial_conditions)
        
        # Detectar estado ERROR 505 (vacuidad > 90%)
        is_error_505 = quantum_state.probabilities['vacuity'] > 0.9
        
        result = {
            'experiment_id': experiment_id,
            'initial_conditions': initial_conditions,
            'quantum_state': quantum_state,
            'probabilities': quantum_state.probabilities,
            'quantum_metrics': {
                'entropy': quantum_state.entropy,
                'purity': quantum_state.purity,
                'coherence': quantum_state.coherence
            },
            'validation_report': validation_report,
            'validation_passed': validation_report['completely_valid'],
            'is_error_505': is_error_505,
            'bardo_interpretation': self._provide_bardo_interpretation(quantum_state)
        }
        
        return result

    def analyze_error_505_states(self, results: List[Dict]) -> Dict:
        """Analiza estados de ERROR 505 desde perspectiva cuántica y filosófica"""
        error_states = [r for r in results if r['is_error_505']]
        
        if not error_states:
            return {'count': 0, 'analysis': 'No se detectaron estados ERROR 505'}
        
        analysis = {
            'count': len(error_states),
            'percentage': len(error_states) / len(results) * 100,
            'common_characteristics': self._find_common_patterns(error_states),
            'quantum_properties': {
                'avg_entropy': np.mean([s['quantum_metrics']['entropy'] for s in error_states]),
                'avg_coherence': np.mean([s['quantum_metrics']['coherence'] for s in error_states]),
                'avg_purity': np.mean([s['quantum_metrics']['purity'] for s in error_states])
            },
            'karmic_correlations': self._analyze_karmic_correlations(error_states),
            'philosophical_interpretation': (
                "ERROR 505 representa estado de vacuidad cuántica donde "
                "la manifestación colapsa hacia no-manifestación. Corresponde "
                "a experiencias de claro-luz en el Bardo thödol."
            )
        }
        
        return analysis

    def _provide_bardo_interpretation(self, quantum_state: QuantumState) -> Dict:
        """Provee interpretación filosófica basada en el estado cuántico"""
        probs = quantum_state.probabilities
        
        if probs['vacuity'] > 0.7:
            stage = "Bardo de la Realidad Suprema (Dharma)"
            description = "Estado de claro-luz y vacuidad predominante"
        elif probs['potential'] > probs['manifested']:
            stage = "Bardo del Devenir (Sidpa)" 
            description = "Estado de potencialidad kármica predominante"
        else:
            stage = "Bardo del Momento de la Muerte (Chikhai)"
            description = "Estado de manifestación y transición"
            
        return {
            'bardo_stage': stage,
            'interpretation': description,
            'recommended_practice': self._suggest_practice(quantum_state)
        }

    def _suggest_practice(self, quantum_state: QuantumState) -> str:
        """Sugiere prácticas basadas en el estado cuántico actual"""
        probs = quantum_state.probabilities
        
        if probs['vacuity'] > 0.6:
            return "Meditación en la naturaleza de la mente - reconocer la vacuidad"
        elif probs['potential'] > 0.5:
            return "Purificación kármica - trabajar con semillas latentes"
        else:
            return "Estabilización de la atención - cultivar mindfulness"