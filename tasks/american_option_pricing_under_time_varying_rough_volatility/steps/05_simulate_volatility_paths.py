r"""
Complete the function `simulate_volatility_paths` which generates variance paths using 
the selected volatility engine (rough Bergomi or Heston).

This function implements two different volatility models:

1. **Rough Bergomi**: Uses a fractional kernel with time-varying Hurst parameter.
   The variance follows: v_t = ξ₀ * exp(η * fractional_integral - 0.5 * η² * t^{2H})
   where the fractional integral uses the time-varying Hurst path.

2. **Heston**: Uses mean-reverting square-root diffusion.
   The variance follows: dv_t = κ(θ - v_t)dt + η*√(v_t)*dW_t
   where κ is mean reversion speed, θ is long-run variance.

For SciCode, we use simplified but correct implementations:
- Rough Bergomi: Simplified fractional Brownian motion with time-varying Hurst
- Heston: Euler-Maruyama discretization of the square-root diffusion

The function should:
- Accept engine as a string ('rough_bergomi' or 'heston')
- Accept params as a dictionary with calibrated parameters
- Accept num_paths and num_steps for the simulation size
- Accept dt as the time step size
- Accept hurst_path as optional (required for rough Bergomi)
- Accept random_state for reproducibility
- Return a numpy array of shape (num_paths, num_steps) with all values > 0

Edge cases to handle:
- If engine is 'rough_bergomi' and hurst_path is None, raise ValueError
- If engine is 'heston', hurst_path is ignored
- Ensure all variance values are positive (variance must be > 0)
- Use random_state for all random number generation to ensure reproducibility
- Handle edge case where Heston variance could go negative (use reflection or truncation)
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def simulate_volatility_paths(engine: str, params: dict, num_paths: int,
                              num_steps: int, dt: float, 
                              hurst_path: np.ndarray = None,
                              random_state: int = None) -> np.ndarray:
    '''
    Simulate variance paths using selected volatility engine.
    
    Parameters
    ----------
    engine : str
        'rough_bergomi' or 'heston'.
    params : dict
        Calibrated parameters from step 4.
    num_paths : int
        Number of paths (M, typically 100-500 for SciCode).
    num_steps : int
        Number of time steps (N, typically 10-20 for SciCode).
    dt : float
        Time step size.
    hurst_path : np.ndarray, shape (num_steps,), optional
        Time-varying Hurst path (required for rough Bergomi).
    random_state : int, optional
        Random seed for reproducibility. Must be used for all random number generation.
    
    Returns
    -------
    variance_paths : np.ndarray, shape (num_paths, num_steps)
        Simulated variance paths v_t. All values must be > 0.
    '''
    return variance_paths


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_simulate_volatility_paths(engine: str, params: dict, num_paths: int,
                                    num_steps: int, dt: float, 
                                    hurst_path: np.ndarray = None,
                                    random_state: int = None) -> np.ndarray:
    '''Reference implementation.'''
    # Validate engine
    if engine not in ['rough_bergomi', 'heston']:
        raise ValueError(f"engine must be 'rough_bergomi' or 'heston', got '{engine}'")
    
    # Validate parameters
    required_params = ['rho', 'eta', 'xi0']
    for p in required_params:
        if p not in params:
            raise ValueError(f"Missing required parameter: {p}")
    
    # Validate inputs
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    
    # Set up random number generator
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState(42)  # Default seed
    
    # Extract parameters
    rho = params['rho']
    eta = params['eta']
    xi0 = params['xi0']
    
    # Initialize output array
    variance_paths = np.zeros((num_paths, num_steps))
    
    if engine == 'rough_bergomi':
        # Rough Bergomi: simplified fractional Brownian motion with time-varying Hurst
        if hurst_path is None:
            raise ValueError("hurst_path is required for rough_bergomi engine")
        
        hurst_path = np.asarray(hurst_path, dtype=float)
        if len(hurst_path) != num_steps:
            raise ValueError(f"hurst_path length ({len(hurst_path)}) must equal num_steps ({num_steps})")
        
        # Clip Hurst to valid range [0, 1]
        hurst_path = np.clip(hurst_path, 0.0, 1.0)
        
        # Simplified rough Bergomi: use fractional kernel approximation
        # Pre-generate all random numbers for better determinism
        total_randoms = num_paths * (num_steps - 1)
        all_randoms = rng.randn(total_randoms) * np.sqrt(dt)
        rand_idx = 0
        
        for path_idx in range(num_paths):
            v = xi0  # Start with initial variance
            variance_paths[path_idx, 0] = v
            
            fractional_integral = 0.0
            for t in range(1, num_steps):
                H_t = hurst_path[t]
                # Simplified fractional Brownian motion increment
                # Use simplified kernel approximation: (t*dt)^{H-0.5}
                t_scaled = t * dt
                if t_scaled > 0 and H_t > 0:
                    # Avoid numerical issues with very small H_t
                    H_safe = max(H_t, 0.01)
                    kernel = (t_scaled) ** (H_safe - 0.5)
                else:
                    kernel = 0.0
                
                dW = all_randoms[rand_idx]
                rand_idx += 1
                fractional_integral += kernel * dW
                
                # Rough Bergomi variance evolution
                # v_t = ξ₀ * exp(η * fractional_integral - 0.5 * η² * t^{2H})
                drift = -0.5 * eta**2 * (t_scaled ** (2 * H_t))
                v = xi0 * np.exp(eta * fractional_integral + drift)
                
                # Ensure positive
                v = max(v, 1e-6)
                variance_paths[path_idx, t] = v
    
    elif engine == 'heston':
        # Heston: mean-reverting square-root diffusion
        # dv_t = κ(θ - v_t)dt + η*√(v_t)*dW_t
        if 'kappa' not in params or 'theta' not in params:
            raise ValueError("Heston engine requires 'kappa' and 'theta' parameters")
        
        kappa = params['kappa']
        theta = params['theta']
        
        # Euler-Maruyama discretization
        # Pre-generate all random numbers for better determinism
        total_randoms = num_paths * (num_steps - 1)
        all_randoms = rng.randn(total_randoms) * np.sqrt(dt)
        rand_idx = 0
        
        for path_idx in range(num_paths):
            v = xi0  # Start with initial variance
            variance_paths[path_idx, 0] = v
            
            for t in range(1, num_steps):
                # Use pre-generated random increment
                dW = all_randoms[rand_idx]
                rand_idx += 1
                
                # Heston SDE: dv = κ(θ - v)dt + η*√(v)*dW
                drift = kappa * (theta - v) * dt
                diffusion = eta * np.sqrt(max(v, 1e-6)) * dW
                v = v + drift + diffusion
                
                # Ensure positive (reflection method)
                v = max(v, 1e-6)
                variance_paths[path_idx, t] = v
    
    return variance_paths


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
engine = 'rough_bergomi'
params = {'rho': -0.7, 'eta': 1.9, 'xi0': 0.04}
num_paths = 10
num_steps = 5
dt = 1/252
hurst_path = np.array([0.2, 0.25, 0.3, 0.28, 0.32])
random_state = 42
""",
            "call": "simulate_volatility_paths(engine, params, num_paths, num_steps, dt, hurst_path, random_state)",
            "gold_call": "_gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt, hurst_path, random_state)",
        },
        {
            "setup": """import numpy as np
engine = 'heston'
params = {'rho': -0.7, 'eta': 0.5, 'xi0': 0.04, 'kappa': 2.0, 'theta': 0.04}
num_paths = 10
num_steps = 5
dt = 1/252
random_state = 42
""",
            "call": "simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)",
            "gold_call": "_gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
# Test with different parameters
engine = 'rough_bergomi'
params = {'rho': 0.5, 'eta': 2.5, 'xi0': 0.09}
num_paths = 5
num_steps = 4
dt = 1/252
hurst_path = np.array([0.25, 0.3, 0.35, 0.32])
random_state = 999
""",
            "call": "simulate_volatility_paths(engine, params, num_paths, num_steps, dt, hurst_path, random_state)",
            "gold_call": "_gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt, hurst_path, random_state)",
        },
        {
            "setup": """import numpy as np
# Test Heston mean reversion (longer path)
engine = 'heston'
params = {'rho': -0.7, 'eta': 0.3, 'xi0': 0.02, 'kappa': 5.0, 'theta': 0.04}
num_paths = 5
num_steps = 10
dt = 1/252
random_state = 42
""",
            "call": "simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)",
            "gold_call": "_gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)",
        },
        # --- Error cases ---
        {
            "setup": """import numpy as np
engine = 'rough_bergomi'
params = {'rho': -0.7, 'eta': 1.9, 'xi0': 0.04}
num_paths = 10
num_steps = 5
dt = 1/252
# Missing hurst_path

def run_model():
    try:
        simulate_volatility_paths(engine, params, num_paths, num_steps, dt)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
engine = 'invalid_engine'
params = {'rho': -0.7, 'eta': 1.9, 'xi0': 0.04}
num_paths = 10
num_steps = 5
dt = 1/252
random_state = 42

def run_model():
    try:
        simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
engine = 'heston'
params = {'rho': -0.7, 'eta': 0.5, 'xi0': 0.04}  # Missing kappa and theta
num_paths = 10
num_steps = 5
dt = 1/252
random_state = 42

def run_model():
    try:
        simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_simulate_volatility_paths(engine, params, num_paths, num_steps, dt, random_state=random_state)
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
