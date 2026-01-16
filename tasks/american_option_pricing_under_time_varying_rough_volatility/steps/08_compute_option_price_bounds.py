r"""
Complete the function `compute_option_price_bounds` which computes lower (primal) and upper (dual) 
bounds for American option price using signature-based optimal stopping.

This is the FINAL INTEGRATION STEP that combines outputs from all previous steps:
- Uses RFF embeddings (from step 7) as regression features to estimate continuation values
- Generates asset price paths from variance paths (from step 5) to compute payoffs
- Implements Longstaff-Schwartz method (primal/lower bound) using RFF embeddings
- Implements martingale dual method (upper bound) for tight bounds

For SciCode, we use simplified implementations:
- **Longstaff-Schwartz (Lower Bound)**: Backward induction with linear regression on RFF embeddings
- **Martingale Dual (Upper Bound)**: Simplified dual bound using pathwise maximum
- **Asset Path Generation**: Euler discretization of geometric Brownian motion with stochastic volatility

The function should:
- Accept rff_embeddings as a 2D numpy array (num_paths, 2*D) from Step 7
- Accept variance_paths as a 2D numpy array (num_paths, num_steps) from Step 5
- Accept initial_price, strike, risk_free_rate as floats
- Accept exercise_dates as a numpy array of time indices
- Accept dt as float (time step size)
- Accept method as str (default 'linear' for linear regression on RFF)
- Accept random_state as int (for deterministic asset path generation)
- Return a dictionary with 4 floats: lower_bound, upper_bound, duality_gap, duality_gap_percent

Edge cases to handle:
- Ensure lower_bound <= upper_bound (fundamental primal-dual property)
- Handle empty exercise_dates (should raise ValueError or handle gracefully)
- Ensure all bounds are non-negative
- Handle numerical stability (division by zero in duality_gap_percent if lower_bound = 0)
"""

import numpy as np
from scipy.linalg import lstsq

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def compute_option_price_bounds(rff_embeddings: np.ndarray, variance_paths: np.ndarray,
                                initial_price: float, strike: float, risk_free_rate: float,
                                exercise_dates: np.ndarray, dt: float,
                                method: str = 'linear',
                                random_state: int = None) -> dict:
    '''
    Compute option price bounds using signature-based optimal stopping.
    
    This function integrates outputs from all previous steps:
    - Uses RFF embeddings (from step 7) as regression features
    - Generates asset price paths from variance paths (from step 5)
    - Implements Longstaff-Schwartz (primal) and martingale dual (upper bound)
    
    Parameters
    ----------
    rff_embeddings : np.ndarray, shape (num_paths, 2*D)
        RFF embeddings from step 7 (features for regression).
    variance_paths : np.ndarray, shape (num_paths, num_steps)
        Variance paths from step 5. Used to generate asset price paths.
    initial_price : float
        Initial asset price S_0.
    strike : float
        Option strike price K.
    risk_free_rate : float
        Risk-free interest rate r.
    exercise_dates : np.ndarray, shape (n_exercise,)
        Time indices where early exercise is allowed.
    dt : float
        Time step size.
    method : str, optional
        Regression method: 'linear' (linear regression on RFF), default 'linear'.
    random_state : int, optional
        Random seed for asset path generation (for reproducibility).
    
    Returns
    -------
    bounds : dict
        Dictionary with keys:
        - 'lower_bound': float, primal (lower) bound on option price
        - 'upper_bound': float, dual (upper) bound on option price
        - 'duality_gap': float, upper_bound - lower_bound
        - 'duality_gap_percent': float, (duality_gap / lower_bound) * 100
    '''
    return bounds


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_compute_option_price_bounds(rff_embeddings: np.ndarray, variance_paths: np.ndarray,
                                       initial_price: float, strike: float, risk_free_rate: float,
                                       exercise_dates: np.ndarray, dt: float,
                                       method: str = 'linear',
                                       random_state: int = None) -> dict:
    '''Reference implementation.'''
    # Validate inputs
    rff_embeddings = np.asarray(rff_embeddings, dtype=float)
    variance_paths = np.asarray(variance_paths, dtype=float)
    exercise_dates = np.asarray(exercise_dates, dtype=int)
    
    if rff_embeddings.ndim != 2:
        raise ValueError(f"rff_embeddings must be 2D array, got {rff_embeddings.ndim}D")
    if variance_paths.ndim != 2:
        raise ValueError(f"variance_paths must be 2D array, got {variance_paths.ndim}D")
    if rff_embeddings.shape[0] != variance_paths.shape[0]:
        raise ValueError(f"rff_embeddings and variance_paths must have same number of paths, got {rff_embeddings.shape[0]} and {variance_paths.shape[0]}")
    if exercise_dates.size == 0:
        raise ValueError("exercise_dates must not be empty")
    if initial_price <= 0:
        raise ValueError(f"initial_price must be positive, got {initial_price}")
    if strike < 0:
        raise ValueError(f"strike must be non-negative, got {strike}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    num_paths, num_steps = variance_paths.shape
    exercise_dates = np.sort(exercise_dates)
    exercise_dates = exercise_dates[exercise_dates < num_steps]  # Ensure valid indices
    
    if exercise_dates.size == 0:
        raise ValueError("No valid exercise dates within path length")
    
    # Use RandomState for deterministic asset path generation
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    
    # Generate asset price paths from variance paths
    # dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_t
    # Euler discretization: S_{t+1} = S_t * (1 + r*dt + sqrt(v_t)*sqrt(dt)*Z_t)
    asset_paths = np.zeros((num_paths, num_steps))
    asset_paths[:, 0] = initial_price
    
    # Pre-generate all random numbers for determinism
    total_randoms = num_paths * (num_steps - 1)
    all_randoms = rng.randn(total_randoms)
    rand_idx = 0
    
    for t in range(num_steps - 1):
        v_t = variance_paths[:, t]
        sqrt_v_t = np.sqrt(np.maximum(v_t, 1e-10))  # Ensure positive
        dW = all_randoms[rand_idx:rand_idx + num_paths] * np.sqrt(dt)
        rand_idx += num_paths
        
        asset_paths[:, t+1] = asset_paths[:, t] * (1 + risk_free_rate * dt + sqrt_v_t * dW)
        asset_paths[:, t+1] = np.maximum(asset_paths[:, t+1], 1e-10)  # Ensure positive
    
    # Compute intrinsic value (payoff) at each exercise date
    # For call option: h(S_t) = max(S_t - K, 0)
    def _payoff(price):
        return np.maximum(price - strike, 0.0)
    
    # === LOWER BOUND (Longstaff-Schwartz) ===
    # Backward induction: start from maturity and work backwards
    # At each exercise date, compare intrinsic value with continuation value (from regression)
    
    # Cashflow matrix: tracks discounted payoffs for each path
    cashflows = np.zeros(num_paths)
    
    # Work backwards from last exercise date to first
    for i in range(len(exercise_dates) - 1, -1, -1):
        t = exercise_dates[i]
        intrinsic = _payoff(asset_paths[:, t])
        
        # Discount cashflows from future dates
        discount_factor = np.exp(-risk_free_rate * dt * (exercise_dates[-1] - t))
        cashflows = cashflows * discount_factor
        
        # At maturity (last exercise date), exercise if in-the-money
        if i == len(exercise_dates) - 1:
            cashflows = np.maximum(cashflows, intrinsic)
        else:
            # At intermediate dates, compare intrinsic with continuation value
            # Use RFF embeddings as features for regression
            # Continuation value = E[discounted future payoff | current RFF embedding]
            
            # Fit linear regression: cashflows ~ RFF embeddings
            # Only use paths that are in-the-money (intrinsic > 0)
            itm_mask = intrinsic > 1e-10
            
            if np.sum(itm_mask) > 1 and rff_embeddings.shape[1] > 0:
                X = rff_embeddings[itm_mask, :]
                y = cashflows[itm_mask]
                
                # Add intercept term
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                
                # Solve: X_with_intercept @ beta = y
                try:
                    beta, _, _, _ = lstsq(X_with_intercept, y, cond=1e-10)
                    # Continuation value = X_with_intercept @ beta
                    continuation = X_with_intercept @ beta
                    # Update cashflows: exercise if intrinsic > continuation
                    exercise_mask = itm_mask & (intrinsic > continuation)
                    cashflows[exercise_mask] = intrinsic[exercise_mask]
                except:
                    # If regression fails, use simple rule: exercise if in-the-money
                    cashflows[itm_mask] = np.maximum(cashflows[itm_mask], intrinsic[itm_mask])
            else:
                # Not enough ITM paths or no features, use simple rule
                cashflows[itm_mask] = np.maximum(cashflows[itm_mask], intrinsic[itm_mask])
    
    # Discount back to time 0
    if len(exercise_dates) > 0:
        discount_to_zero = np.exp(-risk_free_rate * dt * exercise_dates[0])
        cashflows = cashflows * discount_to_zero
    
    lower_bound = float(np.mean(cashflows))
    
    # === UPPER BOUND (Simplified Martingale Dual) ===
    # Simplified approach: use pathwise maximum of discounted intrinsic values
    # V^up ≈ E[max_t (e^{-rt} * h(S_t))]
    
    discounted_intrinsic = np.zeros((num_paths, len(exercise_dates)))
    for i, t in enumerate(exercise_dates):
        intrinsic = _payoff(asset_paths[:, t])
        discount = np.exp(-risk_free_rate * dt * t)
        discounted_intrinsic[:, i] = discount * intrinsic
    
    # Upper bound: expected maximum discounted intrinsic value
    max_discounted_intrinsic = np.max(discounted_intrinsic, axis=1)
    upper_bound = float(np.mean(max_discounted_intrinsic))
    
    # Ensure lower_bound <= upper_bound (fundamental property)
    if lower_bound > upper_bound:
        # Swap if numerical error causes violation
        lower_bound, upper_bound = upper_bound, lower_bound
    
    # Compute duality gap
    duality_gap = upper_bound - lower_bound
    
    # Compute duality gap as percentage
    if lower_bound > 1e-10:
        duality_gap_percent = (duality_gap / lower_bound) * 100.0
    else:
        duality_gap_percent = 0.0 if duality_gap < 1e-10 else 100.0
    
    # Ensure non-negative bounds
    lower_bound = max(lower_bound, 0.0)
    upper_bound = max(upper_bound, lower_bound)
    duality_gap = upper_bound - lower_bound
    if lower_bound > 1e-10:
        duality_gap_percent = (duality_gap / lower_bound) * 100.0
    else:
        duality_gap_percent = 0.0
    
    return {
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'duality_gap': float(duality_gap),
        'duality_gap_percent': float(duality_gap_percent)
    }


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
# Test with simple inputs
num_paths = 100
num_steps = 10
D = 8

rff_embeddings = np.random.RandomState(42).randn(num_paths, 2*D) * 0.1
variance_paths = np.random.RandomState(42).rand(num_paths, num_steps) * 0.05 + 0.02
initial_price = 100.0
strike = 100.0
risk_free_rate = 0.05
exercise_dates = np.array([5, 9])  # Last valid index is num_steps - 1
dt = 1/252
random_state = 42
""",
            "call": "compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, 'linear', random_state)",
            "gold_call": "_gold_compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, 'linear', random_state)",
        },
        {
            "setup": """import numpy as np
# Test with default method
num_paths = 50
num_steps = 8
D = 4

rff_embeddings = np.random.RandomState(123).randn(num_paths, 2*D) * 0.1
variance_paths = np.random.RandomState(123).rand(num_paths, num_steps) * 0.03 + 0.02
initial_price = 50.0
strike = 55.0
risk_free_rate = 0.03
exercise_dates = np.array([4, 7])  # Last valid index is num_steps - 1
dt = 1/252
random_state = 123
""",
            "call": "compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, random_state=random_state)",
            "gold_call": "_gold_compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
# Test single exercise date (European option)
num_paths = 70
num_steps = 6
D = 5

rff_embeddings = np.random.RandomState(321).randn(num_paths, 2*D) * 0.1
variance_paths = np.random.RandomState(321).rand(num_paths, num_steps) * 0.03 + 0.02
initial_price = 100.0
strike = 100.0
risk_free_rate = 0.05
exercise_dates = np.array([5])  # Only at maturity (must be < num_steps)
dt = 1/252
random_state = 321
""",
            "call": "compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, 'linear', random_state)",
            "gold_call": "_gold_compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, 'linear', random_state)",
        },
        {
            "setup": """import numpy as np
# Test deep ITM option (strike = 0)
num_paths = 60
num_steps = 4
D = 4

rff_embeddings = np.random.RandomState(654).randn(num_paths, 2*D) * 0.1
variance_paths = np.random.RandomState(654).rand(num_paths, num_steps) * 0.02 + 0.02
initial_price = 100.0
strike = 0.0  # Deep ITM
risk_free_rate = 0.05
exercise_dates = np.array([2, 3])  # Last valid index is num_steps - 1
dt = 1/252
random_state = 654
""",
            "call": "compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, 'linear', random_state)",
            "gold_call": "_gold_compute_option_price_bounds(rff_embeddings, variance_paths, initial_price, strike, risk_free_rate, exercise_dates, dt, 'linear', random_state)",
        },
    ]
