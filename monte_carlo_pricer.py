"""
Monte Carlo pricer for European options using stochastic calculus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class MonteCarloPricer:
    """
    Monte Carlo pricer for European options under Black-Scholes framework
    """
    
    def __init__(self, S0, K, T, r, sigma, option_type='call'):
        """
        Initialize the pricer with option parameters
        
        Parameters:
        S0: initial stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        option_type: 'call' or 'put'
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def simulate_gbm_paths(self, n_simulations, n_steps, use_antithetic=False):
        """
        Simulate Geometric Brownian Motion paths under risk-neutral measure
        
        Parameters:
        n_simulations: number of paths to simulate
        n_steps: number of time steps
        use_antithetic: whether to use antithetic variates
        
        Returns:
        paths: array of simulated paths (n_simulations x n_steps+1)
        """
        dt = self.T / n_steps
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = self.S0
        
        # Generate random shocks
        if use_antithetic:
            # For antithetic, we'll generate half the required paths
            # and use their negatives for the other half
            n_half = n_simulations // 2
            Z = np.random.standard_normal((n_half, n_steps))
            Z = np.vstack([Z, -Z])  # Stack original and antithetic
            if n_simulations % 2 != 0:
                # If odd number, add one more
                Z = np.vstack([Z, np.random.standard_normal((1, n_steps))])
        else:
            Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Simulate paths using exact solution of GBM SDE
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt + 
                self.sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        return paths
    
    def calculate_payoff(self, ST):
        """
        Calculate option payoff at maturity
        
        Parameters:
        ST: stock price at maturity
        
        Returns:
        payoff: option payoff
        """
        if self.option_type == 'call':
            return np.maximum(ST - self.K, 0)
        else:  # put
            return np.maximum(self.K - ST, 0)
    
    def price_basic_monte_carlo(self, n_simulations, n_steps):
        """
        Price option using basic Monte Carlo simulation
        
        Returns:
        price: estimated option price
        std_error: standard error of the estimate
        """
        # Simulate paths
        paths = self.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=False)
        
        # Get terminal prices
        ST = paths[:, -1]
        
        # Calculate payoffs
        payoffs = self.calculate_payoff(ST)
        
        # Discount to present value
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error
    
    def price_antithetic_variates(self, n_simulations, n_steps):
        """
        Price option using Monte Carlo with antithetic variates for variance reduction
        
        Returns:
        price: estimated option price
        std_error: standard error of the estimate
        """
        # Simulate paths with antithetic variates
        paths = self.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=True)
        
        # Get terminal prices
        ST = paths[:, -1]
        
        # Calculate payoffs
        payoffs = self.calculate_payoff(ST)
        
        # Discount to present value
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error
    
    def plot_sample_paths(self, n_paths, n_steps):
        """
        Plot sample simulated paths
        """
        paths = self.simulate_gbm_paths(n_paths, n_steps, use_antithetic=False)
        time = np.linspace(0, self.T, n_steps + 1)
        
        plt.figure(figsize=(10, 6))
        for i in range(n_paths):
            plt.plot(time, paths[i], alpha=0.7, linewidth=1)
        
        plt.axhline(y=self.K, color='r', linestyle='--', label=f'Strike Price (K={self.K})')
        plt.xlabel('Time (Years)')
        plt.ylabel('Stock Price')
        plt.title(f'Sample Geometric Brownian Motion Paths\n'
                 f'S0={self.S0}, μ={self.r}, σ={self.sigma}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def convergence_analysis(self, n_steps, max_simulations=10000, step_size=100):
        """
        Analyze convergence of Monte Carlo estimate
        
        Returns:
        simulation_sizes: array of simulation counts
        prices: corresponding price estimates
        errors: errors vs analytical price
        """
        from black_scholes import BlackScholes
        
        bs = BlackScholes()
        analytical_price = bs.price(self.S0, self.K, self.T, self.r, self.sigma, self.option_type)
        
        simulation_sizes = np.arange(step_size, max_simulations + step_size, step_size)
        prices = []
        errors = []
        
        for n in simulation_sizes:
            price, _ = self.price_basic_monte_carlo(n, n_steps)
            prices.append(price)
            errors.append(abs(price - analytical_price))
        
        return simulation_sizes, prices, errors
    
    def price_control_variates(self, n_simulations, n_steps):
        """
        Price option using control variates for variance reduction
        Uses the underlying stock price as control variate
        """
        # Simulate paths
        paths = self.simulate_gbm_paths(n_simulations, n_steps)
        ST = paths[:, -1]
        
        # Calculate option payoffs
        option_payoffs = self.calculate_payoff(ST)
        discounted_option_payoffs = np.exp(-self.r * self.T) * option_payoffs
        
        # Control variate: stock price (known expectation in risk-neutral world)
        stock_payoffs = ST
        expected_stock = self.S0 * np.exp(self.r * self.T)  # E[ST] in risk-neutral measure
        
        # Calculate optimal control coefficient
        covariance = np.cov(option_payoffs, stock_payoffs)[0, 1]
        variance_stock = np.var(stock_payoffs)
        theta = -covariance / variance_stock
        
        # Apply control variate
        controlled_payoffs = discounted_option_payoffs + theta * (stock_payoffs - expected_stock) * np.exp(-self.r * self.T)
        
        price = np.mean(controlled_payoffs)
        std_error = np.std(controlled_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error, theta
    
    def price_moment_matching(self, n_simulations, n_steps):
        """
        Price option using moment matching for variance reduction
        Ensures simulated paths match theoretical moments
        """
        # Simulate paths
        paths = self.simulate_gbm_paths(n_simulations, n_steps)
        ST = paths[:, -1]
        
        # Calculate theoretical mean and variance of log(ST)
        theoretical_mean_log = np.log(self.S0) + (self.r - 0.5 * self.sigma**2) * self.T
        theoretical_var_log = self.sigma**2 * self.T
        
        # Calculate sample mean and variance of log(ST)
        sample_mean_log = np.mean(np.log(ST))
        sample_var_log = np.var(np.log(ST))
        
        # Adjust moments to match theoretical
        adjusted_log_ST = (np.log(ST) - sample_mean_log) * np.sqrt(theoretical_var_log / sample_var_log) + theoretical_mean_log
        adjusted_ST = np.exp(adjusted_log_ST)
        
        # Calculate payoffs
        payoffs = self.calculate_payoff(adjusted_ST)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return price, std_error
    
    def calculate_greeks_monte_carlo(self, n_simulations, n_steps, bump_size=0.01):
        """
        Calculate option Greeks using Monte Carlo with path recycling
        """
        # Base case
        paths_base = self.simulate_gbm_paths(n_simulations, n_steps)
        ST_base = paths_base[:, -1]
        payoffs_base = self.calculate_payoff(ST_base)
        price_base = np.mean(np.exp(-self.r * self.T) * payoffs_base)
        
        # Delta: bump initial price
        S0_bump = self.S0 * (1 + bump_size)
        mc_delta = MonteCarloPricer(S0_bump, self.K, self.T, self.r, self.sigma, self.option_type)
        paths_delta = mc_delta.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=False)
        ST_delta = paths_delta[:, -1]
        payoffs_delta = mc_delta.calculate_payoff(ST_delta)
        price_delta = np.mean(np.exp(-self.r * self.T) * payoffs_delta)
        
        delta = (price_delta - price_base) / (self.S0 * bump_size)
        
        # Gamma: second derivative
        S0_bump_down = self.S0 * (1 - bump_size)
        mc_gamma_down = MonteCarloPricer(S0_bump_down, self.K, self.T, self.r, self.sigma, self.option_type)
        paths_gamma_down = mc_gamma_down.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=False)
        ST_gamma_down = paths_gamma_down[:, -1]
        payoffs_gamma_down = mc_gamma_down.calculate_payoff(ST_gamma_down)
        price_gamma_down = np.mean(np.exp(-self.r * self.T) * payoffs_gamma_down)
        
        gamma = (price_delta - 2 * price_base + price_gamma_down) / (self.S0 * bump_size)**2
        
        # Vega: bump volatility
        sigma_bump = self.sigma + bump_size
        mc_vega = MonteCarloPricer(self.S0, self.K, self.T, self.r, sigma_bump, self.option_type)
        paths_vega = mc_vega.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=False)
        ST_vega = paths_vega[:, -1]
        payoffs_vega = mc_vega.calculate_payoff(ST_vega)
        price_vega = np.mean(np.exp(-self.r * self.T) * payoffs_vega)
        
        vega = (price_vega - price_base) / (100 * bump_size)  # per 1% change in vol
        
        # Theta: bump time (decrease time to maturity)
        T_bump = max(self.T - bump_size, 0.001)  # avoid negative time
        mc_theta = MonteCarloPricer(self.S0, self.K, T_bump, self.r, self.sigma, self.option_type)
        paths_theta = mc_theta.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=False)
        ST_theta = paths_theta[:, -1]
        payoffs_theta = mc_theta.calculate_payoff(ST_theta)
        price_theta = np.mean(np.exp(-self.r * T_bump) * payoffs_theta)
        
        theta = (price_theta - price_base) / (365 * bump_size)  # per day
        
        # Rho: bump interest rate
        r_bump = self.r + bump_size
        mc_rho = MonteCarloPricer(self.S0, self.K, self.T, r_bump, self.sigma, self.option_type)
        paths_rho = mc_rho.simulate_gbm_paths(n_simulations, n_steps, use_antithetic=False)
        ST_rho = paths_rho[:, -1]
        payoffs_rho = mc_rho.calculate_payoff(ST_rho)
        price_rho = np.mean(np.exp(-r_bump * self.T) * payoffs_rho)
        
        rho = (price_rho - price_base) / (100 * bump_size)  # per 1% change in rate
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
        
        return greeks