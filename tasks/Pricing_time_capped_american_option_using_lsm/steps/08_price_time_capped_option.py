"""
Full LSMC pricing algorithm (integration step).

Implement the complete Least Squares Monte Carlo algorithm for pricing time-capped
American options. This function integrates all previous steps to compute the final
option price.

The algorithm proceeds as follows:
1. Generate Monte Carlo paths for the underlying asset (step 1)
2. Determine time-cap occurrence for each path (step 2)
3. Calculate immediate exercise payoffs at all time steps (step 3)
4. Perform backward induction from maturity (t_n) to time 0:
   - At maturity (t_n): value = payoff (exercise if in-the-money)
   - For each earlier time step t_i (i = n-1, ..., 0):
     a. Evaluate basis functions at current asset prices (step 4)
     b. Estimate continuation values via regression with indicator (step 5)
     c. Determine exercise decision (step 6)
     d. Update value function (step 7)
5. Compute final option price as the discounted average of value function at time 0

The discount factor is exp(-r * dt) where dt = T / n_steps and r is the risk-free rate.
For simplicity, we may use r = 0 or pass r as a parameter.

This is the final integration step that uses outputs from all previous steps.

Constraints:
- All input parameters must satisfy the constraints from previous steps
- The function should return a single float value: the option price

The function should return the option price as a float.
"""

import numpy as np
from typing import Tuple
import importlib.util
from pathlib import Path

# Import functions from previous steps using importlib.util (since module names start with numbers)
_steps_dir = Path(__file__).parent

def _load_module(module_name, file_name):
    """Load a module from a file path."""
    file_path = _steps_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load step modules
_step1 = _load_module('step1', '01_generate_levy_paths.py')
_step2 = _load_module('step2', '02_determine_time_cap.py')
_step3 = _load_module('step3', '03_calculate_payoffs.py')
_step4 = _load_module('step4', '04_evaluate_basis_functions.py')
_step5 = _load_module('step5', '05_estimate_continuation_values.py')
_step6 = _load_module('step6', '06_determine_exercise_decision.py')
_step7 = _load_module('step7', '07_update_value_function.py')

# Get functions from modules
generate_levy_paths = _step1.generate_levy_paths
determine_time_cap = _step2.determine_time_cap
calculate_payoffs = _step3.calculate_payoffs
evaluate_basis_functions = _step4.evaluate_basis_functions
estimate_continuation_values = _step5.estimate_continuation_values
determine_exercise_decision = _step6.determine_exercise_decision
update_value_function = _step7.update_value_function


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def price_time_capped_option(S0: float, mu: float, sigma: float, lambda_param: float,
                              rho: float, T: float, n_steps: int, n_paths: int,
                              seed: int, drawdown_threshold: float, initial_max: float,
                              K: float, option_type: str, n_basis: int, r: float = 0.0) -> float:
    '''
    Price a time-capped American option using the Least Squares Monte Carlo method.
    
    Parameters
    ----------
    S0 : float
        Initial stock price (must be > 0).
    mu : float
        Drift parameter for Lévy process.
    sigma : float
        Volatility parameter (must be >= 0).
    lambda_param : float
        Jump intensity for Poisson process (must be >= 0).
    rho : float
        Parameter for exponential jump distribution (must be > 0).
    T : float
        Time to maturity in years (must be > 0).
    n_steps : int
        Number of time steps for discretization (must be >= 1).
    n_paths : int
        Number of Monte Carlo paths to simulate (must be >= 1).
    seed : int
        Random seed for reproducibility.
    drawdown_threshold : float
        Drawdown threshold C where 0 < C <= 1.
    initial_max : float
        Initial maximum (historical maximum until option issue date, must be > 0).
    K : float
        Strike price (must be > 0).
    option_type : str
        Type of option: 'put' or 'call' (case-insensitive).
    n_basis : int
        Number of basis functions for regression (must be >= 1).
    r : float
        Risk-free interest rate (default: 0.0).
    
    Returns
    -------
    option_price : float
        The estimated option price.
    '''
    # Step 1: Generate Monte Carlo paths
    paths = generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
    
    # Step 2: Determine time-cap occurrence
    time_cap_indices = determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
    
    # Step 3: Calculate payoffs at all time steps
    payoffs = calculate_payoffs(paths, K, option_type)  # Shape: (n_paths, n_steps+1)
    
    # Discount factor per time step
    dt = T / n_steps
    discount_factor = np.exp(-r * dt)
    
    # Initialize value function at maturity (last time step)
    # At maturity, exercise if in-the-money
    value_function = payoffs[:, -1].copy()  # Shape: (n_paths,)
    
    # Backward induction from n_steps-1 down to 0
    for t in range(n_steps - 1, -1, -1):
        # Current asset prices at time t
        current_prices = paths[:, t]  # Shape: (n_paths,)
        
        # Current payoffs at time t
        current_payoffs = payoffs[:, t]  # Shape: (n_paths,)
        
        # Discounted future values (from next time step)
        discounted_future_values = discount_factor * value_function  # Shape: (n_paths,)
        
        # Determine which paths are in-the-money at time t
        in_the_money_mask = (current_payoffs > 0)  # Shape: (n_paths,)
        
        # Step 4: Evaluate basis functions at current prices
        basis_matrix = evaluate_basis_functions(current_prices, n_basis, 'laguerre')
        
        # Step 5: Estimate continuation values via regression
        continuation_values = estimate_continuation_values(
            basis_matrix, discounted_future_values, time_cap_indices, t, in_the_money_mask
        )
        
        # Step 6: Determine exercise decision
        exercise_decision = determine_exercise_decision(
            current_payoffs, continuation_values, time_cap_indices, t
        )
        
        # Step 7: Update value function
        value_function = update_value_function(
            current_payoffs, continuation_values, exercise_decision
        )
    
    # Final option price: discounted average of value function at time 0
    # Since we've already discounted in the backward induction, we just average
    # Actually, we need to discount from time 0 to present (which is just the value at t=0)
    option_price = np.mean(value_function)
    
    return float(option_price)


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_price_time_capped_option(S0: float, mu: float, sigma: float, lambda_param: float,
                                     rho: float, T: float, n_steps: int, n_paths: int,
                                     seed: int, drawdown_threshold: float, initial_max: float,
                                     K: float, option_type: str, n_basis: int, r: float = 0.0) -> float:
    '''Reference implementation.'''
    # Step 1: Generate Monte Carlo paths
    paths = generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
    
    # Step 2: Determine time-cap occurrence
    time_cap_indices = determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
    
    # Step 3: Calculate payoffs at all time steps
    payoffs = calculate_payoffs(paths, K, option_type)  # Shape: (n_paths, n_steps+1)
    
    # Discount factor per time step
    dt = T / n_steps
    discount_factor = np.exp(-r * dt)
    
    # Initialize value function at maturity (last time step)
    # At maturity, exercise if in-the-money
    value_function = payoffs[:, -1].copy()  # Shape: (n_paths,)
    
    # Backward induction from n_steps-1 down to 0
    for t in range(n_steps - 1, -1, -1):
        # Current asset prices at time t
        current_prices = paths[:, t]  # Shape: (n_paths,)
        
        # Current payoffs at time t
        current_payoffs = payoffs[:, t]  # Shape: (n_paths,)
        
        # Discounted future values (from next time step)
        discounted_future_values = discount_factor * value_function  # Shape: (n_paths,)
        
        # Determine which paths are in-the-money at time t
        in_the_money_mask = (current_payoffs > 0)  # Shape: (n_paths,)
        
        # Step 4: Evaluate basis functions at current prices
        basis_matrix = evaluate_basis_functions(current_prices, n_basis, 'laguerre')
        
        # Step 5: Estimate continuation values via regression
        continuation_values = estimate_continuation_values(
            basis_matrix, discounted_future_values, time_cap_indices, t, in_the_money_mask
        )
        
        # Step 6: Determine exercise decision
        exercise_decision = determine_exercise_decision(
            current_payoffs, continuation_values, time_cap_indices, t
        )
        
        # Step 7: Update value function
        value_function = update_value_function(
            current_payoffs, continuation_values, exercise_decision
        )
    
    # Final option price: discounted average of value function at time 0
    # Since we've already discounted in the backward induction, we just average
    # Actually, we need to discount from time 0 to present (which is just the value at t=0)
    option_price = np.mean(value_function)
    
    return float(option_price)


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: small number of paths and steps ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 10
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'put'
n_basis = 5
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: call option ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 10
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'call'
n_basis = 5
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: different strike price ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 10
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 110.0
option_type = 'put'
n_basis = 5
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: more time steps ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 20
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'put'
n_basis = 5
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: more paths ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 10
n_paths = 500
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'put'
n_basis = 5
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: zero interest rate ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 10
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'put'
n_basis = 5
r = 0.0
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: more basis functions ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 1.0
n_steps = 10
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'put'
n_basis = 10
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
        # --- Case: different maturity ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.5
rho = 2.0
T = 0.5
n_steps = 10
n_paths = 100
seed = 42
drawdown_threshold = 0.2
initial_max = 100.0
K = 100.0
option_type = 'put'
n_basis = 5
r = 0.05
""",
            "call": "price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
            "gold_call": "_gold_price_time_capped_option(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed, drawdown_threshold, initial_max, K, option_type, n_basis, r)",
        },
    ]
