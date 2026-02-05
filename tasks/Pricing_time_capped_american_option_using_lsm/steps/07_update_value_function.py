"""
Update value function backward (one iteration).

Implement a function to update the value function for one time step in the backward
induction process of the Least Squares Monte Carlo algorithm.

The value function update rule is:
    V_t = max(payoff_t, continuation_value_t) if exercise decision is True
    V_t = continuation_value_t if exercise decision is False

However, in practice, when we exercise, we get the immediate payoff. When we continue,
we get the discounted future value (which is what the continuation value estimates).

The function should:
1. For paths where exercise_decision is True: set value = payoff (immediate exercise)
2. For paths where exercise_decision is False: set value = continuation_value (hold option)
3. Return the updated value function for the current time step

This is one iteration of the backward induction, updating values from time t+1 to time t.

Constraints:
- payoffs must be of shape (n_paths,)
- continuation_values must be of shape (n_paths,)
- exercise_decision must be of shape (n_paths,) with boolean values

The function should return an array of shape (n_paths,) containing the updated
value function at the current time step.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def update_value_function(payoffs: np.ndarray,
                          continuation_values: np.ndarray,
                          exercise_decision: np.ndarray) -> np.ndarray:
    '''
    Update value function for one time step in backward induction.
    
    Parameters
    ----------
    payoffs : np.ndarray
        Immediate exercise payoffs of shape (n_paths,).
    continuation_values : np.ndarray
        Estimated continuation values of shape (n_paths,).
    exercise_decision : np.ndarray
        Boolean array of shape (n_paths,) indicating exercise decisions.
        True indicates exercise, False indicates continuation.
    
    Returns
    -------
    value_function : np.ndarray
        Array of shape (n_paths,) containing the updated value function
        at the current time step.
    '''
    # Input validation
    payoffs = np.asarray(payoffs)
    continuation_values = np.asarray(continuation_values)
    exercise_decision = np.asarray(exercise_decision, dtype=bool)
    
    n_paths = payoffs.shape[0]
    
    if continuation_values.shape != (n_paths,):
        raise ValueError("continuation_values must have shape (n_paths,)")
    if exercise_decision.shape != (n_paths,):
        raise ValueError("exercise_decision must have shape (n_paths,)")
    
    # Update value function:
    # - If exercise: value = payoff (immediate exercise value)
    # - If continue: value = continuation_value (estimated continuation value)
    value_function = np.where(exercise_decision, payoffs, continuation_values)
    
    return value_function


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_update_value_function(payoffs: np.ndarray,
                                 continuation_values: np.ndarray,
                                 exercise_decision: np.ndarray) -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    payoffs = np.asarray(payoffs)
    continuation_values = np.asarray(continuation_values)
    exercise_decision = np.asarray(exercise_decision, dtype=bool)
    
    n_paths = payoffs.shape[0]
    
    if continuation_values.shape != (n_paths,):
        raise ValueError("continuation_values must have shape (n_paths,)")
    if exercise_decision.shape != (n_paths,):
        raise ValueError("exercise_decision must have shape (n_paths,)")
    
    # Update value function:
    # - If exercise: value = payoff (immediate exercise value)
    # - If continue: value = continuation_value (estimated continuation value)
    value_function = np.where(exercise_decision, payoffs, continuation_values)
    
    return value_function


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: mix of exercise and continuation ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 8.5, 11.0, 6.0, 14.0])
exercise_decision = np.array([True, False, True, False, True])
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: all exercise ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 7.5, 11.0, 4.0, 14.0])
exercise_decision = np.array([True, True, True, True, True])
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: all continue ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([11.0, 9.0, 13.0, 6.0, 16.0])
exercise_decision = np.array([False, False, False, False, False])
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: payoff equals continuation (boundary) ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([10.0, 8.0, 12.0, 5.0, 15.0])  # Equal to payoffs
exercise_decision = np.array([True, False, True, False, True])
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: larger number of paths ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(42)
n_paths = 20
payoffs = rng.uniform(5.0, 15.0, size=n_paths)
continuation_values = rng.uniform(5.0, 15.0, size=n_paths)
exercise_decision = rng.choice([True, False], size=n_paths)
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: single path ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0])
continuation_values = np.array([9.0])
exercise_decision = np.array([True])
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: zero payoffs and continuation values ---
        {
            "setup": """
import numpy as np
payoffs = np.array([0.0, 0.0, 0.0])
continuation_values = np.array([0.0, 0.0, 0.0])
exercise_decision = np.array([True, False, True])
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Case: negative continuation values (out-of-the-money) ---
        {
            "setup": """
import numpy as np
payoffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
continuation_values = np.array([-1.0, -0.5, -2.0, -0.1, -1.5])  # Negative (OTM)
exercise_decision = np.array([False, False, False, False, False])  # All continue
""",
            "call": "update_value_function(payoffs, continuation_values, exercise_decision)",
            "gold_call": "_gold_update_value_function(payoffs, continuation_values, exercise_decision)",
        },
        # --- Test input validation: shape mismatch ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0])
continuation_values = np.array([9.0, 8.5])  # Wrong shape
exercise_decision = np.array([True, False, True])

def run_model():
    try:
        update_value_function(payoffs, continuation_values, exercise_decision)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_update_value_function(payoffs, continuation_values, exercise_decision)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: exercise_decision shape mismatch ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0])
continuation_values = np.array([9.0, 8.5, 11.0])
exercise_decision = np.array([True, False])  # Wrong shape

def run_model():
    try:
        update_value_function(payoffs, continuation_values, exercise_decision)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_update_value_function(payoffs, continuation_values, exercise_decision)
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
