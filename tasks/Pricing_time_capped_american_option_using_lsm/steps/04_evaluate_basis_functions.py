"""
Evaluate basis functions (weighted Laguerre polynomials) for regression.

Implement a function to compute weighted Laguerre polynomials φ_k(S) for each asset price
value. The weighted Laguerre polynomials are commonly used in the Least Squares Monte Carlo
method for American option pricing.

The weighted Laguerre polynomials of order k are defined as:
    L_k(x) = e^{-x/2} * L_k^standard(x)

where L_k^standard(x) are the standard Laguerre polynomials, which can be computed using
the recurrence relation:
    L_0(x) = 1
    L_1(x) = 1 - x
    L_k(x) = ((2k-1-x)*L_{k-1}(x) - (k-1)*L_{k-2}(x)) / k  for k >= 2

For the weighted version used in LSMC:
    φ_0(S) = e^{-S/2}
    φ_1(S) = e^{-S/2} * (1 - S)
    φ_k(S) = e^{-S/2} * L_k^standard(S)  for k >= 2

Alternatively, you can use the explicit formulas or recurrence relations directly
for the weighted polynomials.

The function should:
1. Normalize asset prices (typically by dividing by strike K, but here we use raw prices)
2. Compute the first n_basis weighted Laguerre polynomials for each price
3. Return a matrix where each row corresponds to a path and each column to a basis function

Constraints:
- asset_prices must be a 1D array of shape (n_paths,) with all values > 0
- n_basis must be >= 1
- basis_type should be 'laguerre' (other types like 'hermite' may be added later)

The function should return an array of shape (n_paths, n_basis) containing the basis
function evaluations.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def evaluate_basis_functions(asset_prices: np.ndarray, n_basis: int, 
                             basis_type: str = 'laguerre') -> np.ndarray:
    '''
    Evaluate basis functions (weighted Laguerre polynomials) for each asset price.
    
    Parameters
    ----------
    asset_prices : np.ndarray
        1D array of shape (n_paths,) containing asset prices at a given time.
        All values must be > 0.
    n_basis : int
        Number of basis functions to compute (must be >= 1).
    basis_type : str
        Type of basis functions. Currently only 'laguerre' is supported.
    
    Returns
    -------
    basis_matrix : np.ndarray
        Array of shape (n_paths, n_basis) containing basis function evaluations.
        basis_matrix[i, k] = φ_k(S_i) where S_i is the i-th asset price.
    '''
    # Input validation
    asset_prices = np.asarray(asset_prices)
    if asset_prices.ndim != 1:
        raise ValueError("asset_prices must be 1-dimensional")
    if np.any(asset_prices <= 0):
        raise ValueError("All asset prices must be > 0")
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    if basis_type.lower() != 'laguerre':
        raise ValueError("basis_type must be 'laguerre' (other types not yet supported)")
    
    n_paths = asset_prices.shape[0]
    S = asset_prices
    basis_matrix = np.zeros((n_paths, n_basis))
    
    # Weighted Laguerre polynomials: φ_k(S) = e^{-S/2} * L_k(S)
    # where L_k are standard Laguerre polynomials
    
    # Precompute the exponential weight
    exp_weight = np.exp(-S / 2.0)
    
    # φ_0(S) = e^{-S/2}
    if n_basis >= 1:
        basis_matrix[:, 0] = exp_weight
    
    # φ_1(S) = e^{-S/2} * (1 - S)
    if n_basis >= 2:
        basis_matrix[:, 1] = exp_weight * (1.0 - S)
    
    # φ_k(S) for k >= 2 using recurrence relation
    # L_k(x) = ((2k-1-x)*L_{k-1}(x) - (k-1)*L_{k-2}(x)) / k
    # For weighted: φ_k(S) = e^{-S/2} * L_k(S)
    
    # We need to compute standard Laguerre polynomials first, then multiply by weight
    if n_basis >= 3:
        # Initialize L_0 and L_1 (standard Laguerre)
        L_prev2 = np.ones(n_paths)  # L_0(x) = 1
        L_prev1 = 1.0 - S  # L_1(x) = 1 - x
        
        for k in range(2, n_basis):
            # Recurrence: L_k = ((2k-1-x)*L_{k-1} - (k-1)*L_{k-2}) / k
            L_k = ((2.0 * k - 1.0 - S) * L_prev1 - (k - 1.0) * L_prev2) / k
            # Weighted version
            basis_matrix[:, k] = exp_weight * L_k
            # Update for next iteration
            L_prev2 = L_prev1
            L_prev1 = L_k
    
    return basis_matrix


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_evaluate_basis_functions(asset_prices: np.ndarray, n_basis: int,
                                    basis_type: str = 'laguerre') -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    asset_prices = np.asarray(asset_prices)
    if asset_prices.ndim != 1:
        raise ValueError("asset_prices must be 1-dimensional")
    if np.any(asset_prices <= 0):
        raise ValueError("All asset prices must be > 0")
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    if basis_type.lower() != 'laguerre':
        raise ValueError("basis_type must be 'laguerre' (other types not yet supported)")
    
    n_paths = asset_prices.shape[0]
    S = asset_prices
    basis_matrix = np.zeros((n_paths, n_basis))
    
    # Weighted Laguerre polynomials: φ_k(S) = e^{-S/2} * L_k(S)
    # where L_k are standard Laguerre polynomials
    
    # Precompute the exponential weight
    exp_weight = np.exp(-S / 2.0)
    
    # φ_0(S) = e^{-S/2}
    if n_basis >= 1:
        basis_matrix[:, 0] = exp_weight
    
    # φ_1(S) = e^{-S/2} * (1 - S)
    if n_basis >= 2:
        basis_matrix[:, 1] = exp_weight * (1.0 - S)
    
    # φ_k(S) for k >= 2 using recurrence relation
    # L_k(x) = ((2k-1-x)*L_{k-1}(x) - (k-1)*L_{k-2}(x)) / k
    # For weighted: φ_k(S) = e^{-S/2} * L_k(S)
    
    # We need to compute standard Laguerre polynomials first, then multiply by weight
    if n_basis >= 3:
        # Initialize L_0 and L_1 (standard Laguerre)
        L_prev2 = np.ones(n_paths)  # L_0(x) = 1
        L_prev1 = 1.0 - S  # L_1(x) = 1 - x
        
        for k in range(2, n_basis):
            # Recurrence: L_k = ((2k-1-x)*L_{k-1} - (k-1)*L_{k-2}) / k
            L_k = ((2.0 * k - 1.0 - S) * L_prev1 - (k - 1.0) * L_prev2) / k
            # Weighted version
            basis_matrix[:, k] = exp_weight * L_k
            # Update for next iteration
            L_prev2 = L_prev1
            L_prev1 = L_k
    
    return basis_matrix


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: single basis function ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, 105.0, 95.0, 110.0])
n_basis = 1
basis_type = 'laguerre'
""",
            "call": "evaluate_basis_functions(asset_prices, n_basis, basis_type)",
            "gold_call": "_gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)",
        },
        # --- Case: two basis functions ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, 105.0, 95.0])
n_basis = 2
basis_type = 'laguerre'
""",
            "call": "evaluate_basis_functions(asset_prices, n_basis, basis_type)",
            "gold_call": "_gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)",
        },
        # --- Case: five basis functions (typical for LSMC) ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, 105.0, 95.0, 110.0, 90.0])
n_basis = 5
basis_type = 'laguerre'
""",
            "call": "evaluate_basis_functions(asset_prices, n_basis, basis_type)",
            "gold_call": "_gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)",
        },
        # --- Case: larger number of paths ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(42)
asset_prices = rng.uniform(80.0, 120.0, size=20)
n_basis = 3
basis_type = 'laguerre'
""",
            "call": "evaluate_basis_functions(asset_prices, n_basis, basis_type)",
            "gold_call": "_gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)",
        },
        # --- Case: higher order basis functions ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, 110.0, 90.0, 105.0, 95.0])
n_basis = 10
basis_type = 'laguerre'
""",
            "call": "evaluate_basis_functions(asset_prices, n_basis, basis_type)",
            "gold_call": "_gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)",
        },
        # --- Edge case: single path ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0])
n_basis = 5
basis_type = 'laguerre'
""",
            "call": "evaluate_basis_functions(asset_prices, n_basis, basis_type)",
            "gold_call": "_gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)",
        },
        # --- Test input validation: n_basis < 1 ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, 105.0])
n_basis = 0
basis_type = 'laguerre'

def run_model():
    try:
        evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: non-positive prices ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, -5.0, 105.0])
n_basis = 3
basis_type = 'laguerre'

def run_model():
    try:
        evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: invalid basis_type ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([100.0, 105.0])
n_basis = 3
basis_type = 'hermite'

def run_model():
    try:
        evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: 2D array ---
        {
            "setup": """
import numpy as np
asset_prices = np.array([[100.0, 105.0], [95.0, 110.0]])
n_basis = 3
basis_type = 'laguerre'

def run_model():
    try:
        evaluate_basis_functions(asset_prices, n_basis, basis_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_evaluate_basis_functions(asset_prices, n_basis, basis_type)
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
