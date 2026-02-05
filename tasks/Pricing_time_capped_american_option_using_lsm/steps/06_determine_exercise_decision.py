"""
Determine exercise decision at a time step.

Implement a function to determine the optimal exercise decision at a given time step
by comparing immediate exercise payoffs with estimated continuation values, while
accounting for the time cap constraint.

The exercise decision rule is:
    Exercise if:
        1. Immediate payoff >= Continuation value (standard American option logic)
        2. Current time step < Time cap index (time cap constraint: I_{ti<θ} = 1)
    
    Otherwise, continue (hold the option).

The function should:
1. Compare immediate payoffs with continuation values
2. Check if the time cap has been reached (current_time_step >= time_cap_index)
3. Determine exercise decision for each path
4. Return a boolean array indicating which paths should be exercised

Constraints:
- payoffs must be of shape (n_paths,)
- continuation_values must be of shape (n_paths,)
- time_cap_indices must be of shape (n_paths,) with values in [0, n_steps]
- current_time_step must be in [0, n_steps]

The function should return a boolean array of shape (n_paths,) where True indicates
exercise and False indicates continuation.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def determine_exercise_decision(payoffs: np.ndarray,
                                 continuation_values: np.ndarray,
                                 time_cap_indices: np.ndarray,
                                 current_time_step: int) -> np.ndarray:
    '''
    Determine exercise decision at a given time step.
    
    Parameters
    ----------
    payoffs : np.ndarray
        Immediate exercise payoffs of shape (n_paths,).
    continuation_values : np.ndarray
        Estimated continuation values of shape (n_paths,).
    time_cap_indices : np.ndarray
        Time cap occurrence indices of shape (n_paths,).
        Values are integers in [0, n_steps]. If θ never occurs, value is n_steps.
    current_time_step : int
        Current time step index (0-indexed, in [0, n_steps]).
    
    Returns
    -------
    exercise_decision : np.ndarray
        Boolean array of shape (n_paths,) where True indicates exercise and
        False indicates continuation.
    '''
    # Input validation
    payoffs = np.asarray(payoffs)
    continuation_values = np.asarray(continuation_values)
    time_cap_indices = np.asarray(time_cap_indices, dtype=int)
    
    n_paths = payoffs.shape[0]
    
    if continuation_values.shape != (n_paths,):
        raise ValueError("continuation_values must have shape (n_paths,)")
    if time_cap_indices.shape != (n_paths,):
        raise ValueError("time_cap_indices must have shape (n_paths,)")
    if not (0 <= current_time_step):
        raise ValueError("current_time_step must be >= 0")
    
    # Indicator function: I_{ti<θ} = 1 if current_time_step < time_cap_indices, else 0
    indicator = (current_time_step < time_cap_indices).astype(float)
    
    # Exercise decision: exercise if (payoff >= continuation) AND (time cap not reached)
    # That is: (payoff >= continuation) AND (indicator == 1)
    exercise_decision = (payoffs >= continuation_values) & (indicator > 0.5)
    
    return exercise_decision


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_determine_exercise_decision(payoffs: np.ndarray,
                                       continuation_values: np.ndarray,
                                       time_cap_indices: np.ndarray,
                                       current_time_step: int) -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    payoffs = np.asarray(payoffs)
    continuation_values = np.asarray(continuation_values)
    time_cap_indices = np.asarray(time_cap_indices, dtype=int)
    
    n_paths = payoffs.shape[0]
    
    if continuation_values.shape != (n_paths,):
        raise ValueError("continuation_values must have shape (n_paths,)")
    if time_cap_indices.shape != (n_paths,):
        raise ValueError("time_cap_indices must have shape (n_paths,)")
    if not (0 <= current_time_step):
        raise ValueError("current_time_step must be >= 0")
    
    # Indicator function: I_{ti<θ} = 1 if current_time_step < time_cap_indices, else 0
    indicator = (current_time_step < time_cap_indices).astype(float)
    
    # Exercise decision: exercise if (payoff >= continuation) AND (time cap not reached)
    # That is: (payoff >= continuation) AND (indicator == 1)
    exercise_decision = (payoffs >= continuation_values) & (indicator > 0.5)
    
    return exercise_decision


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: exercise when payoff > continuation and time cap not reached ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 8.5, 11.0, 6.0, 14.0])
time_cap_indices = np.array([10, 10, 10, 10, 10])  # No time cap hit
current_time_step = 5
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: some paths hit time cap ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 8.5, 11.0, 6.0, 14.0])
time_cap_indices = np.array([10, 3, 10, 4, 10])  # Paths 1 and 3 hit time cap before step 5
current_time_step = 5
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: payoff equals continuation (boundary) ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([10.0, 8.0, 12.0, 5.0, 15.0])  # Equal to payoffs
time_cap_indices = np.array([10, 10, 10, 10, 10])
current_time_step = 5
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: all paths hit time cap ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 8.5, 11.0, 6.0, 14.0])
time_cap_indices = np.array([3, 2, 4, 1, 2])  # All hit time cap before step 5
current_time_step = 5
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: payoff < continuation (should not exercise) ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([11.0, 9.0, 13.0, 6.0, 16.0])  # All > payoffs
time_cap_indices = np.array([10, 10, 10, 10, 10])
current_time_step = 5
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: current_time_step = 0 ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 8.5, 11.0, 6.0, 14.0])
time_cap_indices = np.array([10, 10, 10, 10, 10])
current_time_step = 0
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: current_time_step equals time_cap_index (boundary) ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0, 5.0, 15.0])
continuation_values = np.array([9.0, 8.5, 11.0, 6.0, 14.0])
time_cap_indices = np.array([5, 5, 5, 5, 5])  # Time cap at step 5
current_time_step = 5
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Case: larger number of paths ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(42)
n_paths = 20
payoffs = rng.uniform(5.0, 15.0, size=n_paths)
continuation_values = rng.uniform(5.0, 15.0, size=n_paths)
time_cap_indices = rng.integers(5, 15, size=n_paths)
current_time_step = 8
""",
            "call": "determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
            "gold_call": "_gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)",
        },
        # --- Test input validation: shape mismatch ---
        {
            "setup": """
import numpy as np
payoffs = np.array([10.0, 8.0, 12.0])
continuation_values = np.array([9.0, 8.5])  # Wrong shape
time_cap_indices = np.array([10, 10, 10])
current_time_step = 5

def run_model():
    try:
        determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)
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
payoffs = np.array([10.0, 8.0])
continuation_values = np.array([9.0, 8.5])
time_cap_indices = np.array([10, 10])
current_time_step = -1

def run_model():
    try:
        determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_determine_exercise_decision(payoffs, continuation_values, time_cap_indices, current_time_step)
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
