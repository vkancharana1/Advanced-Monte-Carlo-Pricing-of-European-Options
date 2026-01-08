"""
Utility functions for plotting and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(mc_prices, mc_errors, analytical_price, method_names):
    """
    Plot comparison of Monte Carlo results with analytical price
    """
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(method_names))
    
    # Create bar plot
    bars = plt.bar(x_pos, mc_prices, yerr=mc_errors, 
                   capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    
    # Add analytical price line
    plt.axhline(y=analytical_price, color='red', linestyle='--', 
                linewidth=2, label=f'Analytical Price: {analytical_price:.4f}')
    
    # Customize plot
    plt.xlabel('Pricing Method')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo vs Analytical Option Pricing')
    plt.xticks(x_pos, method_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (price, error) in enumerate(zip(mc_prices, mc_errors)):
        plt.text(i, price + error + 0.1, f'{price:.4f} ± {error:.4f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_convergence(mc_pricer, analytical_price, n_steps, max_simulations=5000, step_size=100):
    """
    Plot convergence of Monte Carlo estimate
    """
    print("\nRunning convergence analysis...")
    simulation_sizes, prices, errors = mc_pricer.convergence_analysis(
        n_steps, max_simulations, step_size
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Price convergence
    ax1.plot(simulation_sizes, prices, 'b-', alpha=0.7, linewidth=2, label='Monte Carlo Price')
    ax1.axhline(y=analytical_price, color='r', linestyle='--', 
                linewidth=2, label=f'Analytical Price: {analytical_price:.4f}')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Option Price')
    ax1.set_title('Monte Carlo Price Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error convergence
    ax2.loglog(simulation_sizes, errors, 'g-', alpha=0.7, linewidth=2, label='Absolute Error')
    
    # Add reference line for 1/sqrt(N) convergence
    ref_error = errors[0] * np.sqrt(simulation_sizes[0]) / np.sqrt(simulation_sizes)
    ax2.loglog(simulation_sizes, ref_error, 'r--', alpha=0.7, 
               label='O(1/√N) reference')
    
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Monte Carlo Error Convergence (Log-Log)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_payoff_distribution(mc_pricer, n_simulations=10000, n_steps=252):
    """
    Plot the distribution of option payoffs
    """
    paths = mc_pricer.simulate_gbm_paths(n_simulations, n_steps)
    ST = paths[:, -1]
    payoffs = mc_pricer.calculate_payoff(ST)
    discounted_payoffs = np.exp(-mc_pricer.r * mc_pricer.T) * payoffs
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Histogram of payoffs
    plt.subplot(1, 2, 1)
    plt.hist(payoffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(payoffs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(payoffs):.2f}')
    plt.xlabel('Payoff at Maturity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Option Payoffs at Maturity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of discounted payoffs
    plt.subplot(1, 2, 2)
    plt.hist(discounted_payoffs, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=np.mean(discounted_payoffs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(discounted_payoffs):.4f}')
    plt.xlabel('Discounted Payoff (Present Value)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Discounted Option Payoffs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()