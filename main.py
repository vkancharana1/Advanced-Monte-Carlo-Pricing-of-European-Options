"""
Monte Carlo Pricing of European Options
Enhanced with advanced variance reduction and Greeks
"""
import numpy as np
from monte_carlo_pricer import MonteCarloPricer
from black_scholes import BlackScholes
from utils import plot_results, plot_convergence, plot_payoff_distribution
from variance_reduction_analysis import VarianceReductionAnalysis

def main():
    print("=== Advanced Monte Carlo Pricing of European Options ===\n")
    
    # Parameters
    S0 = 100.0      # Initial stock price
    K = 105.0       # Strike price
    T = 1.0         # Time to maturity (1 year)
    r = 0.05        # Risk-free rate
    sigma = 0.2     # Volatility
    option_type = 'call'  # 'call' or 'put'
    
    # Monte Carlo parameters
    n_simulations = 10000
    n_steps = 252   # Daily steps for 1 year
    
    # Initialize pricer
    mc_pricer = MonteCarloPricer(S0, K, T, r, sigma, option_type)
    
    # Calculate analytical Black-Scholes price and Greeks
    bs = BlackScholes()
    analytical_price = bs.price(S0, K, T, r, sigma, option_type)
    analytical_greeks = bs.greeks(S0, K, T, r, sigma, option_type)
    
    print("Running Monte Carlo simulations with various techniques...")
    
    # Price using different methods
    price_basic, std_error_basic = mc_pricer.price_basic_monte_carlo(n_simulations, n_steps)
    price_antithetic, std_error_antithetic = mc_pricer.price_antithetic_variates(n_simulations, n_steps)
    price_control, std_error_control, theta = mc_pricer.price_control_variates(n_simulations, n_steps)
    price_moment, std_error_moment = mc_pricer.price_moment_matching(n_simulations, n_steps)
    
    # Calculate Monte Carlo Greeks
    print("Calculating Monte Carlo Greeks...")
    mc_greeks = mc_pricer.calculate_greeks_monte_carlo(5000, n_steps)
    
    # Display results
    print(f"\n=== PRICING RESULTS ===")
    print(f"Option Type: {option_type.upper()}")
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    
    print(f"\n--- Analytical Black-Scholes ---")
    print(f"Price: {analytical_price:.4f}")
    
    print(f"\n--- Monte Carlo Prices ---")
    print(f"Basic MC: {price_basic:.4f} ± {std_error_basic:.4f}")
    print(f"Antithetic Variates: {price_antithetic:.4f} ± {std_error_antithetic:.4f}")
    print(f"Control Variates: {price_control:.4f} ± {std_error_control:.4f} (theta: {theta:.4f})")
    print(f"Moment Matching: {price_moment:.4f} ± {std_error_moment:.4f}")
    
    # Calculate errors
    errors = {
        'Basic': abs(price_basic - analytical_price),
        'Antithetic': abs(price_antithetic - analytical_price),
        'Control Variates': abs(price_control - analytical_price),
        'Moment Matching': abs(price_moment - analytical_price)
    }
    
    print(f"\n--- Pricing Errors vs Analytical ---")
    for method, error in errors.items():
        print(f"{method}: {error:.4f}")
    
    print(f"\n=== GREEKS COMPARISON ===")
    print(f"{'Greek':<10} {'Analytical':<12} {'Monte Carlo':<12} {'Error':<10}")
    print("-" * 50)
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        analytical_val = analytical_greeks[greek]
        mc_val = mc_greeks[greek]
        error = abs(analytical_val - mc_val)
        print(f"{greek:<10} {analytical_val:>10.4f} {mc_val:>10.4f} {error:>10.4f}")
    
    # Plotting
    print(f"\nGenerating plots...")
    
    # Basic convergence plot
    plot_convergence(mc_pricer, analytical_price, n_steps)
    
    # Payoff distribution
    plot_payoff_distribution(mc_pricer)
    
    # Sample paths
    mc_pricer.plot_sample_paths(10, n_steps)
    
    # Variance reduction analysis
    print(f"\n=== VARIANCE REDUCTION ANALYSIS ===")
    n_simulations_range = [100, 500, 1000, 2000, 5000, 10000]
    vr_analysis = VarianceReductionAnalysis(mc_pricer)
    results = vr_analysis.compare_variance_reduction(n_simulations_range, n_steps, n_trials=20)
    
    # Plot variance comparison
    vr_analysis.plot_variance_comparison(results, n_simulations_range)
    
    # Calculate and plot efficiency gains
    efficiency_gains = vr_analysis.calculate_efficiency_gain(results)
    vr_analysis.plot_efficiency_gains(efficiency_gains, n_simulations_range)
    
    print(f"\n=== EFFICIENCY GAINS SUMMARY ===")
    for method, gains in efficiency_gains.items():
        avg_gain = np.mean(gains)
        max_gain = np.max(gains)
        print(f"{method.replace('_', ' ').title():<20} Avg Gain: {avg_gain:.2f}x, Max Gain: {max_gain:.2f}x")

if __name__ == "__main__":
    main()