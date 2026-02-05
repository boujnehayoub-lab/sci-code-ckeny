"""
Calculate immediate exercise payoffs for each path at each time step.

Implement a function to compute the immediate exercise value (payoff) G(S_t) at each
time step for each Monte Carlo path. The payoff function depends on the option type:

For a put option: G(S) = (K - S)^+ = max(K - S, 0)
For a call option: G(S) = (S - K)^+ = max(S - K, 0)

where:
- K is the strike price
- S is the current asset price
- (·)^+ denotes the positive part (max with zero)

The function should:
1. Check the option type ('put' or 'call')
2. For each path and each time step, compute the appropriate payoff
3. Return an array of payoffs with the same shape as the input paths

Constraints:
- paths must have shape (n_paths, n_steps+1) with all values > 0
- K must be > 0
- option_type must be either 'put' or 'call' (case-insensitive)

The function should return an array of shape (n_paths, n_steps+1) containing the
immediate exercise payoffs at each time step for each path.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def calculate_payoffs(paths: np.ndarray, K: float, option_type: str) -> np.ndarray:
    '''
    Calculate immediate exercise payoffs for each path at each time step.
    
    Parameters
    ----------
    paths : np.ndarray
        Asset price paths of shape (n_paths, n_steps+1). All values must be > 0.
    K : float
        Strike price (must be > 0).
    option_type : str
        Type of option: 'put' or 'call' (case-insensitive).
        For 'put': G(S) = max(K - S, 0)
        For 'call': G(S) = max(S - K, 0)
    
    Returns
    -------
    payoffs : np.ndarray
        Array of shape (n_paths, n_steps+1) containing immediate exercise payoffs
        G(S_t) at each time step for each path.
    '''
    return payoffs


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_calculate_payoffs(paths: np.ndarray, K: float, option_type: str) -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    if paths.ndim != 2:
        raise ValueError("paths must be 2-dimensional")
    if np.any(paths <= 0):
        raise ValueError("All path values must be > 0")
    if K <= 0:
        raise ValueError("K must be > 0")
    
    option_type_lower = option_type.lower()
    if option_type_lower not in ['put', 'call']:
        raise ValueError("option_type must be 'put' or 'call'")
    
    # Calculate payoffs based on option type
    if option_type_lower == 'put':
        # Put option: G(S) = max(K - S, 0)
        payoffs = np.maximum(K - paths, 0.0)
    else:  # call
        # Call option: G(S) = max(S - K, 0)
        payoffs = np.maximum(paths - K, 0.0)
    
    return payoffs


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: Put option ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 95.0, 90.0, 85.0, 80.0],
    [100.0, 105.0, 110.0, 115.0, 120.0],
    [100.0, 100.0, 100.0, 100.0, 100.0],
])
K = 100.0
option_type = 'put'
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Basic case: Call option ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 105.0, 110.0, 115.0, 120.0],
    [100.0, 95.0, 90.0, 85.0, 80.0],
    [100.0, 100.0, 100.0, 100.0, 100.0],
])
K = 100.0
option_type = 'call'
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Case: Out-of-the-money paths ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 110.0, 120.0, 130.0],  # All above strike for put
    [100.0, 90.0, 80.0, 70.0],  # All below strike for call
])
K = 100.0
option_type = 'put'
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Case: Different strike price ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 95.0, 90.0, 85.0],
    [100.0, 105.0, 110.0, 115.0],
])
K = 110.0
option_type = 'put'
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Case: Case-insensitive option type ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 95.0, 90.0, 85.0],
])
K = 100.0
option_type = 'PUT'  # Uppercase
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Case: Mixed case option type ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 105.0, 110.0, 115.0],
])
K = 100.0
option_type = 'Call'  # Mixed case
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Edge case: At-the-money (S = K) ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 100.0, 100.0, 100.0],
])
K = 100.0
option_type = 'put'
""",
            "call": "calculate_payoffs(paths, K, option_type)",
            "gold_call": "_gold_calculate_payoffs(paths, K, option_type)",
        },
        # --- Test input validation: K <= 0 ---
        {
            "setup": """
import numpy as np
paths = np.array([[100.0, 105.0, 110.0]])
K = 0.0
option_type = 'put'

def run_model():
    try:
        calculate_payoffs(paths, K, option_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_calculate_payoffs(paths, K, option_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: invalid option_type ---
        {
            "setup": """
import numpy as np
paths = np.array([[100.0, 105.0, 110.0]])
K = 100.0
option_type = 'invalid'

def run_model():
    try:
        calculate_payoffs(paths, K, option_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_calculate_payoffs(paths, K, option_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: non-positive paths ---
        {
            "setup": """
import numpy as np
paths = np.array([[100.0, -5.0, 110.0]])
K = 100.0
option_type = 'put'

def run_model():
    try:
        calculate_payoffs(paths, K, option_type)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_calculate_payoffs(paths, K, option_type)
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
