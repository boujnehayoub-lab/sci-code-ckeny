"""
Estimate continuation values via regression with indicator function.

Implement a function to estimate continuation values using least squares regression
with the indicator function I_{ti<θ} to account for the time cap constraint.

In the Least Squares Monte Carlo method, continuation values are estimated by
regressing discounted future values against basis functions evaluated at current
asset prices. For time-capped American options, we only include paths where the
time cap has not yet occurred (ti < θ).

The regression problem is:
    min ||Y - X*β||^2

where:
    - Y is the vector of discounted future values (for paths that are in-the-money
      and haven't hit the time cap)
    - X is the matrix of basis functions (rows = paths, columns = basis functions)
    - β is the vector of regression coefficients

The continuation value for each path is then: C(S_t) = X * β

The function should:
1. Filter paths that are in-the-money at the current time step
2. Apply the indicator I_{ti<θ} to only include paths where time cap hasn't occurred
3. Extract discounted future values for these paths
4. Extract basis functions for these paths
5. Perform least squares regression
6. Compute continuation values for all paths (using the fitted coefficients)

Constraints:
- basis_matrix must be of shape (n_paths, n_basis)
- discounted_future_values must be of shape (n_paths,)
- time_cap_indices must be of shape (n_paths,) with values in [0, n_steps]
- current_time_step must be in [0, n_steps]
- in_the_money_mask must be of shape (n_paths,) with boolean values

The function should return an array of shape (n_paths,) containing continuation
values for all paths (even those not used in regression).
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def estimate_continuation_values(basis_matrix: np.ndarray,
                                  discounted_future_values: np.ndarray,
                                  time_cap_indices: np.ndarray,
                                  current_time_step: int,
                                  in_the_money_mask: np.ndarray) -> np.ndarray:
    '''
    Estimate continuation values using least squares regression with indicator.
    
    Parameters
    ----------
    basis_matrix : np.ndarray
        Basis function evaluations of shape (n_paths, n_basis).
    discounted_future_values : np.ndarray
        Discounted future values of shape (n_paths,).
    time_cap_indices : np.ndarray
        Time cap occurrence indices of shape (n_paths,).
        Values are integers in [0, n_steps]. If θ never occurs, value is n_steps.
    current_time_step : int
        Current time step index (0-indexed, in [0, n_steps]).
    in_the_money_mask : np.ndarray
        Boolean array of shape (n_paths,) indicating which paths are in-the-money
        at the current time step.
    
    Returns
    -------
    continuation_values : np.ndarray
        Array of shape (n_paths,) containing estimated continuation values.
        For paths not used in regression, the value may be set to 0 or NaN.
    '''
    # Input validation
    basis_matrix = np.asarray(basis_matrix)
    discounted_future_values = np.asarray(discounted_future_values)
    time_cap_indices = np.asarray(time_cap_indices, dtype=int)
    in_the_money_mask = np.asarray(in_the_money_mask, dtype=bool)
    
    n_paths, n_basis = basis_matrix.shape
    
    if discounted_future_values.shape != (n_paths,):
        raise ValueError("discounted_future_values must have shape (n_paths,)")
    if time_cap_indices.shape != (n_paths,):
        raise ValueError("time_cap_indices must have shape (n_paths,)")
    if in_the_money_mask.shape != (n_paths,):
        raise ValueError("in_the_money_mask must have shape (n_paths,)")
    if not (0 <= current_time_step):
        raise ValueError("current_time_step must be >= 0")
    
    # Indicator function: I_{ti<θ} = 1 if current_time_step < time_cap_indices, else 0
    indicator = (current_time_step < time_cap_indices).astype(float)
    
    # Filter: paths that are in-the-money AND haven't hit time cap
    regression_mask = in_the_money_mask & (indicator > 0.5)
    
    # If no paths qualify for regression, return zeros
    if not np.any(regression_mask):
        return np.zeros(n_paths)
    
    # Extract data for regression
    X_reg = basis_matrix[regression_mask]  # (n_regression_paths, n_basis)
    Y_reg = discounted_future_values[regression_mask]  # (n_regression_paths,)
    
    # Perform least squares regression: Y = X * β
    # Using numpy's lstsq (or we can use np.linalg.solve if X'X is invertible)
    # For numerical stability, we use lstsq with rcond=None (or a small value)
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X_reg, Y_reg, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback: if regression fails, return zeros
        return np.zeros(n_paths)
    
    # Compute continuation values for ALL paths using the fitted coefficients
    continuation_values = basis_matrix @ beta
    
    return continuation_values


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_estimate_continuation_values(basis_matrix: np.ndarray,
                                        discounted_future_values: np.ndarray,
                                        time_cap_indices: np.ndarray,
                                        current_time_step: int,
                                        in_the_money_mask: np.ndarray) -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    basis_matrix = np.asarray(basis_matrix)
    discounted_future_values = np.asarray(discounted_future_values)
    time_cap_indices = np.asarray(time_cap_indices, dtype=int)
    in_the_money_mask = np.asarray(in_the_money_mask, dtype=bool)
    
    n_paths, n_basis = basis_matrix.shape
    
    if discounted_future_values.shape != (n_paths,):
        raise ValueError("discounted_future_values must have shape (n_paths,)")
    if time_cap_indices.shape != (n_paths,):
        raise ValueError("time_cap_indices must have shape (n_paths,)")
    if in_the_money_mask.shape != (n_paths,):
        raise ValueError("in_the_money_mask must have shape (n_paths,)")
    if not (0 <= current_time_step):
        raise ValueError("current_time_step must be >= 0")
    
    # Indicator function: I_{ti<θ} = 1 if current_time_step < time_cap_indices, else 0
    indicator = (current_time_step < time_cap_indices).astype(float)
    
    # Filter: paths that are in-the-money AND haven't hit time cap
    regression_mask = in_the_money_mask & (indicator > 0.5)
    
    # If no paths qualify for regression, return zeros
    if not np.any(regression_mask):
        return np.zeros(n_paths)
    
    # Extract data for regression
    X_reg = basis_matrix[regression_mask]  # (n_regression_paths, n_basis)
    Y_reg = discounted_future_values[regression_mask]  # (n_regression_paths,)
    
    # Perform least squares regression: Y = X * β
    # Using numpy's lstsq (or we can use np.linalg.solve if X'X is invertible)
    # For numerical stability, we use lstsq with rcond=None (or a small value)
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X_reg, Y_reg, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback: if regression fails, return zeros
        return np.zeros(n_paths)
    
    # Compute continuation values for ALL paths using the fitted coefficients
    continuation_values = basis_matrix @ beta
    
    return continuation_values


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: simple regression ---
        {
            "setup": """
import numpy as np
n_paths = 5
n_basis = 3
basis_matrix = np.array([[1.0, 0.5, 0.25],
                         [1.0, 0.6, 0.36],
                         [1.0, 0.4, 0.16],
                         [1.0, 0.7, 0.49],
                         [1.0, 0.3, 0.09]])
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0])
time_cap_indices = np.array([10, 10, 10, 10, 10])  # No time cap hit
current_time_step = 5
in_the_money_mask = np.array([True, True, True, True, True])
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Case: some paths hit time cap ---
        {
            "setup": """
import numpy as np
n_paths = 5
n_basis = 3
basis_matrix = np.array([[1.0, 0.5, 0.25],
                         [1.0, 0.6, 0.36],
                         [1.0, 0.4, 0.16],
                         [1.0, 0.7, 0.49],
                         [1.0, 0.3, 0.09]])
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0])
time_cap_indices = np.array([10, 3, 10, 4, 10])  # Paths 1 and 3 hit time cap before step 5
current_time_step = 5
in_the_money_mask = np.array([True, True, True, True, True])
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Case: some paths not in-the-money ---
        {
            "setup": """
import numpy as np
n_paths = 5
n_basis = 3
basis_matrix = np.array([[1.0, 0.5, 0.25],
                         [1.0, 0.6, 0.36],
                         [1.0, 0.4, 0.16],
                         [1.0, 0.7, 0.49],
                         [1.0, 0.3, 0.09]])
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0])
time_cap_indices = np.array([10, 10, 10, 10, 10])
current_time_step = 5
in_the_money_mask = np.array([True, True, False, True, False])  # Paths 2 and 4 not ITM
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Case: combination of time cap and ITM filters ---
        {
            "setup": """
import numpy as np
n_paths = 6
n_basis = 4
rng = np.random.default_rng(42)
basis_matrix = rng.uniform(0.1, 1.0, size=(6, 4))
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0, 11.0])
time_cap_indices = np.array([10, 3, 10, 4, 10, 2])  # Some paths hit time cap
current_time_step = 5
in_the_money_mask = np.array([True, True, False, True, False, True])
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Case: larger number of paths ---
        {
            "setup": """
import numpy as np
n_paths = 20
n_basis = 5
rng = np.random.default_rng(42)
basis_matrix = rng.uniform(0.1, 1.0, size=(n_paths, n_basis))
discounted_future_values = rng.uniform(5.0, 15.0, size=n_paths)
time_cap_indices = rng.integers(5, 15, size=n_paths)
current_time_step = 8
in_the_money_mask = rng.choice([True, False], size=n_paths, p=[0.6, 0.4])
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Edge case: no paths qualify for regression (all hit time cap) ---
        {
            "setup": """
import numpy as np
n_paths = 5
n_basis = 3
basis_matrix = np.array([[1.0, 0.5, 0.25],
                         [1.0, 0.6, 0.36],
                         [1.0, 0.4, 0.16],
                         [1.0, 0.7, 0.49],
                         [1.0, 0.3, 0.09]])
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0])
time_cap_indices = np.array([3, 2, 4, 1, 2])  # All hit time cap before step 5
current_time_step = 5
in_the_money_mask = np.array([True, True, True, True, True])
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Edge case: no paths in-the-money ---
        {
            "setup": """
import numpy as np
n_paths = 5
n_basis = 3
basis_matrix = np.array([[1.0, 0.5, 0.25],
                         [1.0, 0.6, 0.36],
                         [1.0, 0.4, 0.16],
                         [1.0, 0.7, 0.49],
                         [1.0, 0.3, 0.09]])
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0])
time_cap_indices = np.array([10, 10, 10, 10, 10])
current_time_step = 5
in_the_money_mask = np.array([False, False, False, False, False])  # None ITM
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Edge case: current_time_step = 0 ---
        {
            "setup": """
import numpy as np
n_paths = 5
n_basis = 3
basis_matrix = np.array([[1.0, 0.5, 0.25],
                         [1.0, 0.6, 0.36],
                         [1.0, 0.4, 0.16],
                         [1.0, 0.7, 0.49],
                         [1.0, 0.3, 0.09]])
discounted_future_values = np.array([10.0, 12.0, 8.0, 14.0, 6.0])
time_cap_indices = np.array([10, 10, 10, 10, 10])
current_time_step = 0
in_the_money_mask = np.array([True, True, True, True, True])
""",
            "call": "estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
            "gold_call": "_gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)",
        },
        # --- Test input validation: shape mismatch ---
        {
            "setup": """
import numpy as np
basis_matrix = np.array([[1.0, 0.5], [1.0, 0.6]])
discounted_future_values = np.array([10.0, 12.0, 8.0])  # Wrong shape
time_cap_indices = np.array([10, 10])
current_time_step = 5
in_the_money_mask = np.array([True, True])

def run_model():
    try:
        estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: negative current_time_step ---
        {
            "setup": """
import numpy as np
basis_matrix = np.array([[1.0, 0.5], [1.0, 0.6]])
discounted_future_values = np.array([10.0, 12.0])
time_cap_indices = np.array([10, 10])
current_time_step = -1
in_the_money_mask = np.array([True, True])

def run_model():
    try:
        estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_continuation_values(basis_matrix, discounted_future_values, time_cap_indices, current_time_step, in_the_money_mask)
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
