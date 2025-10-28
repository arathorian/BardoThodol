# scripts/prepare_reproducibility_data.py
"""
Preparación de Datasets para Reproducibilidad Científica
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def generate_reproducibility_dataset():
    """Genera dataset completo para reproducibilidad científica"""

    dataset = {
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'model_version': '1.0.0',
            'quantum_framework': 'qiskit-1.0.0',
            'python_version': '3.10+',
            'citation': 'Bardo Quantum Model v1.0.0',
            'license': 'MIT'
        },
        'parameters': {
            'dimensions': 3,
            'num_qutrits': 2,
            'karma_range': [0.0, 1.0],
            'attention_range': [0.0, 1.0],
            'num_samples': 1000
        },
        'simulation_data': {},
        'validation_metrics': {},
        'statistical_analysis': {}
    }

    # Ejecutar simulación para datos
    from src.simulations.advanced_simulator import AdvancedBardoSimulator
    from src.core.error505_analysis import Error505Analyzer

    simulator = AdvancedBardoSimulator()
    results = simulator.run_comprehensive_simulation(num_samples=100)

    # Procesar datos para reproducibilidad
    dataset['simulation_data'] = process_simulation_data(results)
    dataset['validation_metrics'] = process_validation_metrics(results)
    dataset['statistical_analysis'] = process_statistical_analysis(results)

    # Guardar dataset
    with open('data/processed/reproducibility_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2, cls=NumpyEncoder)

    # Guardar versión CSV para análisis externo
    save_csv_version(dataset)

    return dataset

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizado para numpy arrays en JSON"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return {'real': obj.real, 'imag': obj.imag}
        return super().default(obj)

def process_simulation_data(results: Dict) -> Dict:
    """Procesa datos de simulación para reproducibilidad"""
    processed = {
        'states': [],
        'metrics': results.get('metrics', []),
        'timestamps': [s['timestamp'].isoformat() if hasattr(s['timestamp'], 'isoformat')
                      else str(s['timestamp']) for s in results['states']]
    }

    for state in results['states']:
        processed_state = {
            'karma': state['karma'],
            'attention': state['attention'],
            'probabilities': state['probabilities'].tolist() if hasattr(state['probabilities'], 'tolist')
                           else state['probabilities'],
            'metrics': state['metrics']
        }

        # Convertir arrays numpy a listas
        if hasattr(state['statevector'], 'tolist'):
            processed_state['statevector'] = {
                'real': state['statevector'].real.tolist(),
                'imag': state['statevector'].imag.tolist()
            }

        processed['states'].append(processed_state)

    return processed

def save_csv_version(dataset: Dict):
    """Guarda versión CSV del dataset para análisis externo"""
    # Datos de estados
    states_data = []
    for state in dataset['simulation_data']['states']:
        row = {
            'karma': state['karma'],
            'attention': state['attention'],
            'manifested_prob': state['probabilities'][0],
            'potential_prob': state['probabilities'][1],
            'vacuity_prob': state['probabilities'][2],
            'entropy': state['metrics']['entropy'],
            'purity': state['metrics']['purity'],
            'coherence': state['metrics']['coherence']
        }
        states_data.append(row)

    df_states = pd.DataFrame(states_data)
    df_states.to_csv('data/processed/quantum_states_dataset.csv', index=False)

    # Métricas de validación
    if dataset['validation_metrics']:
        df_validation = pd.DataFrame([dataset['validation_metrics']])
        df_validation.to_csv('data/processed/validation_metrics.csv', index=False)
