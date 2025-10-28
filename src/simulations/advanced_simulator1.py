"""
Advanced Bardo Simulator with Scientific Validation
Complete simulation framework for quantum Bardo states
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path

from ..core.quantum_models_optimized import OptimizedBardoModel, EfficientQutritSystem, QuantumState
from ..core.quantum_validator import ScientificValidator, BardoPhysicsValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedBardoSimulator:
    """Advanced simulator with quality metrics and validation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.model = OptimizedBardoModel()
        self.qutrit_system = EfficientQutritSystem()
        self.validator = ScientificValidator()
        self.physics_validator = BardoPhysicsValidator()
        
        # Configuration
        self.config = config or self._default_config()
        
        # Setup plotting style
        self.setup_plotting_style()
        
        # Results storage
        self.simulation_history = []
        
    def _default_config(self) -> Dict:
        """Default simulation configuration"""
        return {
            'num_samples': 1000,
            'karma_range': (0.0, 1.0),
            'attention_range': (0.0, 1.0),
            'time_steps': 10,
            'validation_strict': True,
            'save_results': True,
            'output_dir': 'results/simulations'
        }
    
    def setup_plotting_style(self):
        """Setup scientific plotting style"""
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        self.plot_params = {
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'figure.dpi': 300
        }
        
        plt.rcParams.update(self.plot_params)
        
        # Custom colors for Bardo states
        self.colors = {
            'manifested': '#2E86AB',
            'potential': '#A23B72', 
            'vacuity': '#F18F01',
            'error_505': '#C73E1D',
            'karma': '#3C91E6',
            'attention': '#47E5BC'
        }
    
    def run_comprehensive_simulation(self, num_samples: Optional[int] = None) -> Dict:
        """Run comprehensive simulation with statistical validation"""
        num_samples = num_samples or self.config['num_samples']
        
        logger.info(f"Starting comprehensive simulation with {num_samples} samples")
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_samples': num_samples,
                'config': self.config
            },
            'states': [],
            'metrics': [],
            'validations': [],
            'statistical_tests': {},
            'performance_metrics': {}
        }
        
        start_time = datetime.now()
        
        for i in range(num_samples):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{num_samples} samples")
            
            # Generate random parameters within valid ranges
            karma = np.random.uniform(*self.config['karma_range'])
            attention = np.random.uniform(*self.config['attention_range'])
            
            # Run single simulation
            sim_result = self.run_single_simulation(karma, attention)
            
            # Validate result
            validation = self.validator.validate_quantum_state(
                sim_result['statevector'],
                sim_result['density_matrix']
            )
            
            if not self.config['validation_strict'] or validation['complete']:
                results['states'].append(sim_result)
                results['metrics'].append(sim_result['metrics'])
                results['validations'].append(validation)
        
        # Statistical analysis
        results['statistical_tests'] = self.perform_statistical_analysis(results)
        
        # Performance metrics
        end_time = datetime.now()
        results['performance_metrics'] = {
            'total_time': (end_time - start_time).total_seconds(),
            'samples_per_second': num_samples / (end_time - start_time).total_seconds(),
            'success_rate': len(results['states']) / num_samples
        }
        
        # Save results if configured
        if self.config['save_results']:
            self.save_simulation_results(results)
        
        logger.info(f"Simulation completed: {len(results['states'])} valid states")
        return results
    
    def run_single_simulation(self, karma: float, attention: float, 
                            time_steps: Optional[int] = None) -> Dict:
        """Run single simulation with complete metrics"""
        time_steps = time_steps or self.config['time_steps']
        
        # Create initial state (vacuity)
        state = self.model.create_initial_state("vacuity")
        state_history = [state.copy()]
        
        # Evolve through time steps
        for step in range(time_steps):
            state = self.model.apply_karmic_evolution(
                state, karma, attention, time_step=1.0
            )
            state_history.append(state.copy())
        
        # Calculate metrics for final state
        final_metrics = self.model.calculate_state_metrics(state)
        
        # Create comprehensive result
        result = {
            'parameters': {
                'karma': karma,
                'attention': attention,
                'time_steps': time_steps
            },
            'statevector': state,
            'density_matrix': final_metrics.density_matrix,
            'probabilities': final_metrics.probabilities,
            'state_history': [s.tolist() for s in state_history],
            'metrics': {
                'entropy': final_metrics.entropy,
                'purity': np.trace(final_metrics.density_matrix @ final_metrics.density_matrix),
                'coherence': final_metrics.coherence,
                'fidelity_vacuity': abs(state[2])**2,
                'manifestation_prob': final_metrics.probabilities['manifested'],
                'potential_prob': final_metrics.probabilities['potential'],
                'vacuity_prob': final_metrics.probabilities['vacuity']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def perform_statistical_analysis(self, results: Dict) -> Dict:
        """Perform comprehensive statistical analysis"""
        if not results['metrics']:
            return {}
            
        metrics_df = pd.DataFrame(results['metrics'])
        
        analysis = {
            'descriptive_statistics': self._calculate_descriptive_stats(metrics_df),
            'correlations': self._calculate_correlations(metrics_df),
            'normality_tests': self._perform_normality_tests(metrics_df),
            'significance_tests': self._perform_significance_tests(results, metrics_df),
            'cluster_analysis': self._perform_cluster_analysis(metrics_df)
        }
        
        return analysis
    
    def _calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics"""
        return df.describe().to_dict()
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate correlation matrix"""
        corr_matrix = df.corr()
        
        # Find strong correlations (|r| > 0.5)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _perform_normality_tests(self, df: pd.DataFrame) -> Dict:
        """Perform normality tests for all metrics"""
        normality_results = {}
        
        for column in df.columns:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(df[column])
            
            # D'Agostino test
            dagostino_stat, dagostino_p = stats.normaltest(df[column])
            
            normality_results[column] = {
                'shapiro_wilk': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                },
                'dagostino': {
                    'statistic': dagostino_stat,
                    'p_value': dagostino_p,
                    'is_normal': dagostino_p > 0.05
                }
            }
        
        return normality_results
    
    def _perform_significance_tests(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Perform significance tests between groups"""
        significance_results = {}
        
        # Test karma effect (high vs low karma)
        high_karma_mask = np.array([s['parameters']['karma'] > 0.7 for s in results['states']])
        low_karma_mask = np.array([s['parameters']['karma'] < 0.3 for s in results['states']])
        
        if np.any(high_karma_mask) and np.any(low_karma_mask):
            high_karma_vacuity = df['vacuity_prob'][high_karma_mask]
            low_karma_vacuity = df['vacuity_prob'][low_karma_mask]
            
            # T-test
            t_stat, t_p = stats.ttest_ind(high_karma_vacuity, low_karma_vacuity)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(high_karma_vacuity, low_karma_vacuity)
            
            significance_results['karma_effect'] = {
                't_test': {
                    'statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < 0.05
                },
                'mann_whitney': {
                    'statistic': u_stat,
                    'p_value': u_p,
                    'significant': u_p < 0.05
                },
                'effect_size': high_karma_vacuity.mean() - low_karma_vacuity.mean()
            }
        
        return significance_results
    
    def _perform_cluster_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform basic cluster analysis"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        for k in range(1, 6):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Use 3 clusters for Bardo states
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        return {
            'inertias': inertias,
            'clusters': clusters.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def save_simulation_results(self, results: Dict, filename: Optional[str] = None):
        """Save simulation results to file"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
        
        filepath = output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_results_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def _make_results_serializable(self, results: Dict) -> Dict:
        """Convert numpy types to Python types for JSON serialization"""
        import copy
        
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        return convert(copy.deepcopy(results))
    
    def load_simulation_results(self, filepath: str) -> Dict:
        """Load simulation results from file"""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from: {filepath}")
        return results

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = AdvancedBardoSimulator()
    
    # Run comprehensive simulation
    results = simulator.run_comprehensive_simulation(num_samples=100)
    
    # Print summary
    print(f"Simulation completed with {len(results['states'])} valid states")
    print(f"Success rate: {results['performance_metrics']['success_rate']:.2%}")
    print(f"Average vacuity probability: {np.mean([m['vacuity_prob'] for m in results['metrics']]):.3f}")