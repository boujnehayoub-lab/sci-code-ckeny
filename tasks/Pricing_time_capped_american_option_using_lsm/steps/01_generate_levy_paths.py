"""
Generate Monte Carlo paths for a geometric Lévy process.

Implement a function to simulate N independent trajectories of a geometric Lévy process
S_t = e^{X_t} where X_t is a spectrally negative Lévy process defined as:

    X_t = x + μt + σB_t - Σ_{k=1}^{N_t} U_k

where:
- x = log(S0) is the initial log-price
- μ is the drift parameter
- σ ≥ 0 is the volatility
- B_t is a standard Brownian motion
- N_t is a homogeneous Poisson process with intensity λ ≥ 0
- {U_k} is a sequence of independent identically distributed exponential random variables
  with mean ρ^{-1} (i.e., U_k ~ Exp(ρ))

The function should:
1. Discretize time into n_steps equal intervals from 0 to T
2. Generate n_paths independent trajectories
3. For each trajectory:
   - Generate Brownian motion increments: ΔB_i ~ N(0, dt) where dt = T/n_steps
   - Generate Poisson jump times and sizes
   - Combine to form X_t at each time step
   - Compute S_t = exp(X_t) at each time step

Constraints:
- S0 > 0, T > 0, n_steps >= 1, n_paths >= 1
- σ >= 0, λ >= 0, ρ > 0
- seed must be an integer for reproducibility

The function should return an array of shape (n_paths, n_steps+1) containing the asset
price paths, where paths[i, 0] = S0 and paths[i, j] = S_{t_j} for trajectory i.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def generate_levy_paths(S0: float, mu: float, sigma: float, lambda_param: float, 
                        rho: float, T: float, n_steps: int, n_paths: int, 
                        seed: int) -> np.ndarray:
    '''
    Generate Monte Carlo paths for a geometric Lévy process.
    
    Parameters
    ----------
    S0 : float
        Initial asset price (must be > 0).
    mu : float
        Drift parameter.
    sigma : float
        Volatility parameter (must be >= 0).
    lambda_param : float
        Jump intensity for Poisson process (must be >= 0).
    rho : float
        Parameter for exponential jump distribution (must be > 0). 
        Jumps U_k ~ Exp(rho), so mean jump size is 1/rho.
    T : float
        Time to maturity (must be > 0).
    n_steps : int
        Number of time steps for discretization (must be >= 1).
    n_paths : int
        Number of Monte Carlo paths to simulate (must be >= 1).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    paths : np.ndarray
        Array of shape (n_paths, n_steps+1) containing asset price trajectories.
        paths[i, 0] = S0 for all i, and paths[i, j] = S_{t_j} for trajectory i.
    '''
    return paths


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_generate_levy_paths(S0: float, mu: float, sigma: float, lambda_param: float,
                               rho: float, T: float, n_steps: int, n_paths: int,
                               seed: int) -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if lambda_param < 0:
        raise ValueError("lambda_param must be >= 0")
    if rho <= 0:
        raise ValueError("rho must be > 0")
    
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Time discretization
    dt = T / n_steps
    x0 = np.log(S0)
    
    # Initialize paths array
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate paths
    for i in range(n_paths):
        X = x0  # Current log-price
        
        for j in range(1, n_steps + 1):
            # Brownian motion increment
            dW = rng.normal(0, np.sqrt(dt))
            X += mu * dt + sigma * dW
            
            # Poisson jumps
            if lambda_param > 0:
                # Number of jumps in interval [t_{j-1}, t_j]
                num_jumps = rng.poisson(lambda_param * dt)
                
                # Generate jump sizes (exponential with mean 1/rho)
                if num_jumps > 0:
                    jumps = rng.exponential(1.0 / rho, size=num_jumps)
                    X -= np.sum(jumps)  # Subtract jumps (spectrally negative)
            
            # Convert to asset price
            paths[i, j] = np.exp(X)
    
    return paths


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: Geometric Brownian Motion (no jumps) ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.0  # No jumps
rho = 1.0
T = 1.0
n_steps = 10
n_paths = 5
seed = 42
""",
            "call": "generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
            "gold_call": "_gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
        },
        # --- Case with jumps ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.02
sigma = 0.3
lambda_param = 0.5
rho = 2.0
T = 0.5
n_steps = 20
n_paths = 10
seed = 123
""",
            "call": "generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
            "gold_call": "_gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
        },
        # --- Single path, single step ---
        {
            "setup": """
import numpy as np
S0 = 50.0
mu = 0.1
sigma = 0.15
lambda_param = 0.1
rho = 0.5
T = 1.0
n_steps = 1
n_paths = 1
seed = 0
""",
            "call": "generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
            "gold_call": "_gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
        },
        # --- Larger simulation ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.25
lambda_param = 0.2
rho = 1.0
T = 2.0
n_steps = 100
n_paths = 50
seed = 999
""",
            "call": "generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
            "gold_call": "_gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
        },
        # --- Edge case: zero volatility ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.0
lambda_param = 0.1
rho = 1.0
T = 1.0
n_steps = 10
n_paths = 5
seed = 42
""",
            "call": "generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
            "gold_call": "_gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)",
        },
        # --- Test input validation: S0 <= 0 ---
        {
            "setup": """
import numpy as np
S0 = 0.0
mu = 0.05
sigma = 0.2
lambda_param = 0.0
rho = 1.0
T = 1.0
n_steps = 10
n_paths = 5
seed = 42

def run_model():
    try:
        generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: T <= 0 ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.0
rho = 1.0
T = 0.0
n_steps = 10
n_paths = 5
seed = 42

def run_model():
    try:
        generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: rho <= 0 ---
        {
            "setup": """
import numpy as np
S0 = 100.0
mu = 0.05
sigma = 0.2
lambda_param = 0.0
rho = 0.0
T = 1.0
n_steps = 10
n_paths = 5
seed = 42

def run_model():
    try:
        generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_generate_levy_paths(S0, mu, sigma, lambda_param, rho, T, n_steps, n_paths, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
