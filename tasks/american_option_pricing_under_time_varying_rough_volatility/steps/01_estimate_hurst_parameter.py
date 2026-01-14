r"""
Complete the function `estimate_hurst_parameter` which computes the Hurst parameter H_t from 
demeaned log returns using the rescaled-range (R/S) statistic. 

The Hurst parameter measures volatility roughness:
- Values below 0.5 indicate rough (anti-persistent) volatility
- Values above 0.5 indicate smooth (persistent) volatility
- Value of 0.5 indicates a random walk

The R/S statistic is calculated as follows:
1. Compute the cumulative sum of demeaned returns: Y_k = Σᵢ₌₁ᵏ (rᵢ - r̄) for k = 1, ..., n
2. Calculate the range: R = max(Y_k) - min(Y_k)
3. Calculate the standard deviation: S = std(returns)
4. Estimate Hurst parameter: H = log₂(R/S) / log₂(n)

You should require that the input array has length n >= 8 (raise ValueError if n < 8), 
and the function should return a float value H in the range [0, 1].

Edge cases to handle:
- If S = 0 (constant returns), the formula R/S is undefined. In this case, return H = 0.5 
  (random walk behavior for constant returns).
- If R = 0 (all cumulative sums are equal), return H = 0.5.
- Ensure the result is clipped to [0, 1] if numerical issues cause it to go outside this range.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def estimate_hurst_parameter(returns: np.ndarray) -> float:
    '''
    Estimate Hurst parameter using rescaled-range statistic.
    
    Parameters
    ----------
    returns : np.ndarray, shape (n,)
        Demeaned log returns (n >= 8 required for meaningful estimation).
        Must be deterministic synthetic data (no external sources).
    
    Returns
    -------
    hurst : float
        Estimated Hurst parameter H in [0, 1].
        Formula: H = log₂(R/S) / log₂(n), where:
        - R = max(cumulative_demeaned) - min(cumulative_demeaned) is the range
        - S = std(returns) is the standard deviation
        - n = len(returns)
    '''
    return hurst


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_estimate_hurst_parameter(returns: np.ndarray) -> float:
    '''Reference implementation.'''
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    
    # Check minimum length requirement
    if n < 8:
        raise ValueError("returns must have length >= 8 for meaningful Hurst estimation")
    
    # Demean the returns (they should already be demeaned, but ensure it)
    mean_return = np.mean(returns)
    demeaned = returns - mean_return
    
    # Calculate standard deviation S
    S = np.std(demeaned, ddof=0)  # Population std (ddof=0)
    
    # Handle edge case: constant returns (S = 0)
    if S == 0.0:
        return 0.5  # Random walk for constant returns
    
    # Compute cumulative sum of demeaned returns: Y_k = Σᵢ₌₁ᵏ (rᵢ - r̄)
    cumulative = np.cumsum(demeaned)
    
    # Calculate range R = max(Y_k) - min(Y_k)
    R = np.max(cumulative) - np.min(cumulative)
    
    # Handle edge case: R = 0 (all cumulative sums equal)
    if R == 0.0:
        return 0.5
    
    # Calculate R/S ratio
    RS_ratio = R / S
    
    # Handle edge case: RS_ratio <= 0 (shouldn't happen, but protect against numerical issues)
    if RS_ratio <= 0:
        return 0.5
    
    # Estimate Hurst parameter: H = log₂(R/S) / log₂(n)
    # Using log2 for base-2 logarithm
    log2_RS = np.log2(RS_ratio)
    log2_n = np.log2(n)
    
    hurst = log2_RS / log2_n
    
    # Clip to [0, 1] range to handle any numerical issues
    hurst = np.clip(hurst, 0.0, 1.0)
    
    return float(hurst)


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
returns = np.zeros(20)
""",
            "call": "estimate_hurst_parameter(returns)",
            "gold_call": "_gold_estimate_hurst_parameter(returns)",
        },
        {
            "setup": """import numpy as np
# Create synthetic data with known properties (alternating pattern - anti-persistent)
returns = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
""",
            "call": "estimate_hurst_parameter(returns)",
            "gold_call": "_gold_estimate_hurst_parameter(returns)",
        },
        {
            "setup": """import numpy as np
# Create synthetic data with persistent trend
returns = np.linspace(0.01, 0.20, 20)
""",
            "call": "estimate_hurst_parameter(returns)",
            "gold_call": "_gold_estimate_hurst_parameter(returns)",
        },
        {
            "setup": """import numpy as np
# Minimum valid length (n = 8)
returns = np.array([0.1, -0.05, 0.02, 0.03, -0.01, 0.04, -0.02, 0.01])
""",
            "call": "estimate_hurst_parameter(returns)",
            "gold_call": "_gold_estimate_hurst_parameter(returns)",
        },
        {
            "setup": """import numpy as np
# Random walk-like data (should give H ≈ 0.5)
np.random.seed(42)
returns = np.random.randn(50) * 0.01
""",
            "call": "estimate_hurst_parameter(returns)",
            "gold_call": "_gold_estimate_hurst_parameter(returns)",
        },
        # --- Error cases ---
        {
            "setup": """import numpy as np
returns = np.array([0.1, -0.05, 0.02])  # Length < 8

def run_model():
    try:
        estimate_hurst_parameter(returns)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_hurst_parameter(returns)
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
returns = np.array([])  # Empty array

def run_model():
    try:
        estimate_hurst_parameter(returns)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_hurst_parameter(returns)
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
returns = np.array([0.1])  # Single element

def run_model():
    try:
        estimate_hurst_parameter(returns)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_estimate_hurst_parameter(returns)
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
