"""
Determine the time-cap occurrence for each Monte Carlo path based on drawdown threshold.

Implement a function to find the first time when a drawdown event occurs for each path.
The time cap θ is defined as the first time when the asset price falls below its historical
maximum by a fixed percentage threshold:

    θ = inf{t ≥ 0 : 1 - S_t/S̄_t ≥ C}

where:
- S_t is the asset price at time t
- S̄_t = s ∨ max_{0≤s≤t} S_s is the running maximum of the asset price
- s is the initial maximum (historical maximum until the option issue date)
- C is the drawdown threshold (0 < C ≤ 1)

The function should:
1. For each path, track the running maximum S̄_t at each time step
2. At each time step, compute the drawdown: 1 - S_t/S̄_t
3. Find the first time index where drawdown ≥ C
4. Return the time index for each path (or n_steps if drawdown never exceeds C)

Constraints:
- paths must have shape (n_paths, n_steps+1) with all values > 0
- drawdown_threshold must satisfy 0 < C ≤ 1
- initial_max must be > 0
- n_steps must match paths.shape[1] - 1

The function should return an array of shape (n_paths,) containing the time step index
where θ occurs for each path. If θ never occurs (drawdown never exceeds C), return n_steps.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def determine_time_cap(paths: np.ndarray, drawdown_threshold: float, 
                       initial_max: float, n_steps: int) -> np.ndarray:
    '''
    Determine the time-cap occurrence index for each path based on drawdown threshold.
    
    Parameters
    ----------
    paths : np.ndarray
        Asset price paths of shape (n_paths, n_steps+1). All values must be > 0.
    drawdown_threshold : float
        Drawdown threshold C where 0 < C ≤ 1. The time cap occurs when
        1 - S_t/S̄_t ≥ C.
    initial_max : float
        Initial maximum s (historical maximum until option issue date). Must be > 0.
    n_steps : int
        Number of time steps (should equal paths.shape[1] - 1).
    
    Returns
    -------
    time_cap_indices : np.ndarray
        Array of shape (n_paths,) containing the time step index where θ occurs.
        Values are integers in [0, n_steps]. If θ never occurs, returns n_steps.
    '''
    return time_cap_indices


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_determine_time_cap(paths: np.ndarray, drawdown_threshold: float,
                             initial_max: float, n_steps: int) -> np.ndarray:
    '''Reference implementation.'''
    # Input validation
    if paths.ndim != 2:
        raise ValueError("paths must be 2-dimensional")
    if paths.shape[1] != n_steps + 1:
        raise ValueError(f"paths.shape[1] must equal n_steps + 1, got {paths.shape[1]} and {n_steps + 1}")
    if np.any(paths <= 0):
        raise ValueError("All path values must be > 0")
    if not (0 < drawdown_threshold <= 1):
        raise ValueError("drawdown_threshold must satisfy 0 < C ≤ 1")
    if initial_max <= 0:
        raise ValueError("initial_max must be > 0")
    
    n_paths = paths.shape[0]
    time_cap_indices = np.full(n_paths, n_steps, dtype=int)
    
    for i in range(n_paths):
        # Track running maximum: S̄_t = s ∨ max_{0≤s≤t} S_s
        running_max = initial_max
        
        for j in range(n_steps + 1):
            # Update running maximum
            running_max = max(running_max, paths[i, j])
            
            # Compute drawdown: 1 - S_t/S̄_t
            if running_max > 0:
                drawdown = 1.0 - paths[i, j] / running_max
                
                # Check if drawdown threshold is exceeded
                if drawdown >= drawdown_threshold:
                    time_cap_indices[i] = j
                    break
    
    return time_cap_indices


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Basic case: drawdown occurs early ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 105.0, 85.0, 95.0, 100.0],  # Path 1: drawdown at index 2 (85/105 = 0.810, 1-0.810=0.190 >= 0.15)
    [100.0, 110.0, 120.0, 115.0, 110.0],  # Path 2: no drawdown (always increasing or small drops)
    [100.0, 95.0, 80.0, 85.0, 90.0],  # Path 3: drawdown at index 2 (80/100 = 0.8, 1-0.8=0.2 >= 0.15)
])
drawdown_threshold = 0.15
initial_max = 100.0
n_steps = 4
""",
            "call": "determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
            "gold_call": "_gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
        },
        # --- Case: no drawdown occurs (all paths stay above threshold) ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 105.0, 102.0, 108.0, 110.0],
    [100.0, 98.0, 99.0, 101.0, 103.0],
])
drawdown_threshold = 0.3  # Large threshold
initial_max = 100.0
n_steps = 4
""",
            "call": "determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
            "gold_call": "_gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
        },
        # --- Case: drawdown at first step ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 80.0, 85.0, 90.0],  # Immediate 20% drop
    [100.0, 70.0, 75.0, 80.0],  # Immediate 30% drop
])
drawdown_threshold = 0.15
initial_max = 100.0
n_steps = 3
""",
            "call": "determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
            "gold_call": "_gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
        },
        # --- Case: initial_max affects running maximum ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 95.0, 90.0, 85.0],  # Decreasing path
])
drawdown_threshold = 0.1
initial_max = 110.0  # Higher initial max
n_steps = 3
""",
            "call": "determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
            "gold_call": "_gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
        },
        # --- Case: multiple paths with different behaviors ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 120.0, 100.0, 110.0, 105.0],  # Drawdown at index 2: 100/120 = 0.833, 1-0.833=0.167 >= 0.15
    [100.0, 105.0, 110.0, 115.0, 120.0],  # No drawdown (always increasing)
    [100.0, 90.0, 85.0, 80.0, 75.0],  # Drawdown at index 1: 90/100 = 0.9, 1-0.9=0.1 < 0.15, at index 2: 85/100=0.85, 1-0.85=0.15 >= 0.15
    [100.0, 110.0, 90.0, 100.0, 105.0],  # Drawdown at index 2: 90/110 = 0.818, 1-0.818=0.182 >= 0.15
])
drawdown_threshold = 0.15
initial_max = 100.0
n_steps = 4
""",
            "call": "determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
            "gold_call": "_gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
        },
        # --- Edge case: threshold = 1.0 (very strict) ---
        {
            "setup": """
import numpy as np
paths = np.array([
    [100.0, 0.1, 50.0, 100.0],  # Massive drop at index 1
])
drawdown_threshold = 1.0
initial_max = 100.0
n_steps = 3
""",
            "call": "determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
            "gold_call": "_gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)",
        },
        # --- Test input validation: invalid drawdown_threshold ---
        {
            "setup": """
import numpy as np
paths = np.array([[100.0, 105.0, 110.0]])
drawdown_threshold = 0.0
initial_max = 100.0
n_steps = 2

def run_model():
    try:
        determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: initial_max <= 0 ---
        {
            "setup": """
import numpy as np
paths = np.array([[100.0, 105.0, 110.0]])
drawdown_threshold = 0.2
initial_max = 0.0
n_steps = 2

def run_model():
    try:
        determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Test input validation: paths with non-positive values ---
        {
            "setup": """
import numpy as np
paths = np.array([[100.0, -5.0, 110.0]])
drawdown_threshold = 0.2
initial_max = 100.0
n_steps = 2

def run_model():
    try:
        determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_determine_time_cap(paths, drawdown_threshold, initial_max, n_steps)
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
