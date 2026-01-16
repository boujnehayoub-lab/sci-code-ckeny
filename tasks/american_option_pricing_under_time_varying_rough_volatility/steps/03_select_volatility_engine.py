r"""
Complete the function `select_volatility_engine` which chooses between rough Bergomi and 
Heston volatility models based on the average forecasted Hurst parameter. This implements 
the paper's second innovation: regime-switching between volatility models.

The selection logic is:
- If the average forecasted Hurst parameter is below the threshold (default 0.5), 
  select 'rough_bergomi' (for rough volatility regimes)
- If the average forecasted Hurst parameter is at or above the threshold, 
  select 'heston' (for smooth volatility regimes)

The function should:
- Accept forecasted_hurst as a numpy array of shape (horizon,) containing forecasted Hurst values
- Accept threshold as a float (default 0.5) for the selection criterion
- Compute the mean of forecasted_hurst
- Return 'rough_bergomi' if mean < threshold, otherwise return 'heston'

Edge cases to handle:
- Empty array: should raise ValueError
- Single value: compute mean of that single value
- Boundary case: when mean exactly equals threshold, return 'heston' (>= threshold)
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def select_volatility_engine(forecasted_hurst: np.ndarray, 
                             threshold: float = 0.5) -> str:
    '''
    Select volatility engine based on average forecasted Hurst parameter.
    
    This implements the paper's second innovation: regime-switching between
    rough Bergomi (for rough volatility, H < 0.5) and Heston (for smooth
    volatility, H >= 0.5) based on forecasted Hurst parameter.
    
    Parameters
    ----------
    forecasted_hurst : np.ndarray, shape (horizon,)
        Forecasted Hurst parameter path.
    threshold : float, optional
        Threshold for selection (default 0.5).
        If average < threshold → 'rough_bergomi'
        If average >= threshold → 'heston'
    
    Returns
    -------
    engine : str
        Either 'rough_bergomi' or 'heston'.
    '''
    return engine


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_select_volatility_engine(forecasted_hurst: np.ndarray, 
                                    threshold: float = 0.5) -> str:
    '''Reference implementation.'''
    forecasted_hurst = np.asarray(forecasted_hurst, dtype=float)
    
    # Validate input
    if forecasted_hurst.size == 0:
        raise ValueError("forecasted_hurst must not be empty")
    
    # Compute average forecasted Hurst parameter
    hurst_avg = np.mean(forecasted_hurst)
    
    # Select engine based on threshold
    if hurst_avg < threshold:
        return 'rough_bergomi'
    else:  # hurst_avg >= threshold
        return 'heston'


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
forecasted_hurst = np.array([0.3, 0.32, 0.31, 0.33, 0.30])
threshold = 0.5
""",
            "call": "select_volatility_engine(forecasted_hurst, threshold)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst, threshold)",
        },
        {
            "setup": """import numpy as np
forecasted_hurst = np.array([0.6, 0.62, 0.61, 0.63, 0.60])
threshold = 0.5
""",
            "call": "select_volatility_engine(forecasted_hurst, threshold)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst, threshold)",
        },
        {
            "setup": """import numpy as np
# Boundary case: exactly at threshold
forecasted_hurst = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
threshold = 0.5
""",
            "call": "select_volatility_engine(forecasted_hurst, threshold)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst, threshold)",
        },
        {
            "setup": """import numpy as np
# Mixed values averaging to threshold
forecasted_hurst = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
threshold = 0.5
""",
            "call": "select_volatility_engine(forecasted_hurst, threshold)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst, threshold)",
        },
        {
            "setup": """import numpy as np
# Custom threshold
forecasted_hurst = np.array([0.4, 0.45, 0.5, 0.55, 0.6])
threshold = 0.6
""",
            "call": "select_volatility_engine(forecasted_hurst, threshold)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst, threshold)",
        },
        {
            "setup": """import numpy as np
# Single value
forecasted_hurst = np.array([0.3])
threshold = 0.5
""",
            "call": "select_volatility_engine(forecasted_hurst, threshold)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst, threshold)",
        },
        {
            "setup": """import numpy as np
# Default threshold (not specified)
forecasted_hurst = np.array([0.3, 0.32, 0.31, 0.33, 0.30])
""",
            "call": "select_volatility_engine(forecasted_hurst)",
            "gold_call": "_gold_select_volatility_engine(forecasted_hurst)",
        },
        # --- Error cases ---
        {
            "setup": """import numpy as np
# Empty array
forecasted_hurst = np.array([])

def run_model():
    try:
        select_volatility_engine(forecasted_hurst)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_select_volatility_engine(forecasted_hurst)
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
