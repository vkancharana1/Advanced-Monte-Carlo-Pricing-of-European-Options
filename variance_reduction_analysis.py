"""
Comprehensive analysis of variance reduction techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class VarianceReductionAnalysis:
    """
    Analyze and compare variance reduction techniques
    """
    
    def __init__(self, mc_pricer):
        self.mc_pricer = mc_pricer
    
    def compare_variance_reduction(self, n_simulations_range, n_steps=252, n_trials=50):
        """
        Compare variance reduction techniques across different simulation counts
        """
        methods = ['basic', 'antithetic', 'control_variates', 'moment_matching']
        results = {method: {'prices': [], 'std_errors': [], 'variances': []} for method in methods}
        
        for n_simulations in n_simulations_range:
            print(f"Testing with {n_simulations} simulations...")
            
            # Run multiple trials for each method
            for method in methods:
                prices_trial = []
                std_errors_trial = []
                
                for _ in range(n_trials):
                    if method == 'basic':
                        price, std_error = self.mc_pricer.price_basic_monte_carlo(n_simulations, n_steps)
                    elif method == 'antithetic':
                        price, std_error = self.mc_pricer.price_antithetic_variates(n_simulations, n_steps)
                    elif method == 'control_variates':
                        price, std_error, _ = self.mc_pricer.price_control_variates(n_simulations, n_steps)
                    elif method == 'moment_matching':
                        price, std_error = self.mc_pricer.price_moment_matching(n_simulations, n_steps)
                    
                    prices_trial.append(price)
                    std_errors_trial.append(std_error)
                
                results[method]['prices'].append(np.mean(prices_trial))
                results[method]['std_errors'].append(np.mean(std_errors_trial))
                results[method]['variances'].append(np.var(prices_trial))
        
        return results
    
    def plot_variance_comparison(self, results, n_simulations_range):
        """
        Plot comparison of variance reduction techniques
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = list(results.keys())
        colors = ['blue', 'red', 'green', 'orange']
        
        # Plot standard errors
        for i, method in enumerate(methods):
            ax1.plot(n_simulations_range, results[method]['std_errors'], 
                    color=colors[i], linewidth=2, label=method.replace('_', ' ').title())
        
        ax1.set_xlabel('Number of Simulations')
        ax1.set_ylabel('Standard Error')
        ax1.set_title('Standard Error Comparison by Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # Plot variances
        for i, method in enumerate(methods):
            ax2.plot(n_simulations_range, results[method]['variances'], 
                    color=colors[i], linewidth=2, label=method.replace('_', ' ').title())
        
        ax2.set_xlabel('Number of Simulations')
        ax2.set_ylabel('Variance')
        ax2.set_title('Variance Comparison by Method')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def calculate_efficiency_gain(self, results, baseline_method='basic'):
        """
        Calculate efficiency gain of variance reduction techniques
        """
        baseline_variances = np.array(results[baseline_method]['variances'])
        efficiency_gains = {}
        
        for method in results.keys():
            if method != baseline_method:
                method_variances = np.array(results[method]['variances'])
                # Efficiency gain = variance_baseline / variance_method
                efficiency_gains[method] = baseline_variances / method_variances
        
        return efficiency_gains
    
    def plot_efficiency_gains(self, efficiency_gains, n_simulations_range):
        """
        Plot efficiency gains of variance reduction techniques
        """
        plt.figure(figsize=(10, 6))
        
        for method, gains in efficiency_gains.items():
            plt.plot(n_simulations_range, gains, linewidth=2, 
                    label=method.replace('_', ' ').title())
        
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
        plt.xlabel('Number of Simulations')
        plt.ylabel('Efficiency Gain')
        plt.title('Variance Reduction Efficiency Gains\n(Higher is Better)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()