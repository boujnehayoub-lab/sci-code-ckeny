r"""
Complete the function `calibrate_parameters` which estimates volatility model parameters 
from synthetic price and volatility time series using deterministic statistical formulas.

This function calibrates parameters for two volatility models:
1. **Rough Bergomi**: Requires ρ (correlation), η (vol-of-vol), ξ₀ (initial variance)
2. **Heston**: Requires the above plus κ (mean reversion speed) and θ (long-run mean variance)

The calibration uses simple statistical formulas:
- **Correlation (ρ)**: Pearson correlation between price returns and volatility changes
- **Initial variance (ξ₀)**: Mean of squared volatility values
- **Vol-of-vol (η)**: For rough Bergomi, scaled by Hurst parameter: std(vol_changes) / (dt^H)
- **Mean reversion (κ)**: For Heston, simplified estimate: 2 / (mean(vol²) * dt)
- **Long-run variance (θ)**: For Heston, mean of squared volatility values

The function should:
- Accept engine as a string ('rough_bergomi' or 'heston')
- Accept price_data as a numpy array (log prices or log returns)
- Accept vol_data as a numpy array (implied volatility series)
- Accept historical_hurst as a float (most recent Hurst parameter)
- Accept dt as a float (time step, default 1/252 for daily)
- Return a dictionary with calibrated parameters

Edge cases to handle:
- If price_data is log prices, compute returns using np.diff
- If price_data is already returns, use directly
- Handle constant volatility (division by zero in eta calculation)
- Ensure ρ is clipped to [-1, 1]
- Ensure all positive parameters (eta, xi0, kappa, theta) are > 0
- Raise ValueError if engine is not 'rough_bergomi' or 'heston'
"""

import numpy as np
from scipy.stats import pearsonr

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def calibrate_parameters(engine: str, price_data: np.ndarray, 
                         vol_data: np.ndarray, historical_hurst: float,
                         dt: float = 1/252) -> dict:
    '''
    Calibrate volatility model parameters from synthetic price and volatility arrays.
    
    Uses deterministic statistical formulas (correlation, standard deviation) on the
    provided arrays. No external data sources or optimization routines.
    
    Parameters
    ----------
    engine : str
        'rough_bergomi' or 'heston'.
    price_data : np.ndarray, shape (n,)
        Synthetic log price series (or log returns).
    vol_data : np.ndarray, shape (n,)
        Synthetic implied volatility series (ATM).
    historical_hurst : float
        Most recent historical Hurst parameter.
    dt : float, optional
        Time step (default 1/252 for daily).
    
    Returns
    -------
    params : dict
        Dictionary with keys:
        - 'rho': float, correlation parameter in [-1, 1]
        - 'eta': float, volatility-of-volatility > 0
        - 'xi0': float, initial variance > 0
        - 'kappa': float (Heston only), mean reversion speed > 0
        - 'theta': float (Heston only), long-run mean variance > 0
    '''
    return params


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_calibrate_parameters(engine: str, price_data: np.ndarray, 
                                vol_data: np.ndarray, historical_hurst: float,
                                dt: float = 1/252) -> dict:
    '''Reference implementation.'''
    # Validate engine
    if engine not in ['rough_bergomi', 'heston']:
        raise ValueError(f"engine must be 'rough_bergomi' or 'heston', got '{engine}'")
    
    # Convert to numpy arrays
    price_data = np.asarray(price_data, dtype=float)
    vol_data = np.asarray(vol_data, dtype=float)
    
    # Validate inputs
    if len(price_data) != len(vol_data):
        raise ValueError(f"price_data and vol_data must have the same length, got {len(price_data)} and {len(vol_data)}")
    
    if len(price_data) < 2:
        raise ValueError("price_data and vol_data must have length >= 2 for calibration")
    
    # Compute price returns (handle both log prices and returns)
    # If price_data looks like returns (mean near 0, std reasonable), use directly
    # Otherwise, assume it's log prices and compute diff
    price_mean = np.mean(np.abs(price_data))
    if price_mean > 1.0:  # Likely log prices (e.g., log(100) = 4.6)
        price_returns = np.diff(price_data)
    else:  # Likely already returns
        price_returns = price_data[1:]  # Use all but first to match vol_changes length
    
    # Compute volatility changes
    vol_changes = np.diff(vol_data)
    
    # Ensure same length for correlation
    min_len = min(len(price_returns), len(vol_changes))
    price_returns = price_returns[:min_len]
    vol_changes = vol_changes[:min_len]
    
    # Calculate correlation (rho)
    if len(price_returns) > 1 and np.std(price_returns) > 0 and np.std(vol_changes) > 0:
        rho, _ = pearsonr(price_returns, vol_changes)
        rho = float(rho)
    else:
        rho = 0.0  # Default if correlation cannot be computed
    
    # Clip rho to [-1, 1]
    rho = np.clip(rho, -1.0, 1.0)
    
    # Calculate initial variance (xi0) - mean of squared volatility
    xi0 = float(np.mean(vol_data ** 2))
    xi0 = max(xi0, 1e-6)  # Ensure positive
    
    # Calculate vol-of-vol (eta)
    vol_changes_std = np.std(vol_changes)
    if vol_changes_std > 0 and dt > 0:
        # For rough Bergomi: eta = std(vol_changes) / (dt^H)
        if historical_hurst > 0:
            eta = vol_changes_std / (dt ** historical_hurst)
        else:
            eta = vol_changes_std / dt  # Fallback if H <= 0
        eta = max(eta, 1e-6)  # Ensure positive
    else:
        eta = 0.1  # Default small value if cannot compute
    
    # Build base parameters
    params = {
        'rho': rho,
        'eta': eta,
        'xi0': xi0
    }
    
    # Add Heston-specific parameters
    if engine == 'heston':
        # Mean reversion speed (kappa) - simplified estimate
        if xi0 > 0 and dt > 0:
            kappa = 2.0 / (xi0 * dt)
        else:
            kappa = 1.0  # Default
        kappa = max(kappa, 1e-6)  # Ensure positive
        
        # Long-run variance (theta) - same as xi0 for simplicity
        theta = xi0
        
        params['kappa'] = kappa
        params['theta'] = theta
    
    return params


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
price_data = np.array([100.0, 101.0, 99.5, 100.5, 101.5])
vol_data = np.array([0.15, 0.16, 0.14, 0.17, 0.18])
historical_hurst = 0.3
dt = 1/252
""",
            "call": "calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
            "gold_call": "_gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
        },
        {
            "setup": """import numpy as np
engine = 'heston'
price_data = np.array([100.0, 101.0, 99.5, 100.5, 101.5])
vol_data = np.array([0.15, 0.16, 0.14, 0.17, 0.18])
historical_hurst = 0.6
dt = 1/252
""",
            "call": "calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
            "gold_call": "_gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
        },
        {
            "setup": """import numpy as np
# Test with log returns (already returns, not log prices)
engine = 'rough_bergomi'
price_data = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
vol_data = np.array([0.15, 0.16, 0.14, 0.17, 0.18])
historical_hurst = 0.3
dt = 1/252
""",
            "call": "calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
            "gold_call": "_gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
        },
        {
            "setup": """import numpy as np
# Test with perfect correlation
engine = 'rough_bergomi'
price_data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
vol_data = np.array([0.15, 0.16, 0.17, 0.18, 0.19])  # Perfectly correlated
historical_hurst = 0.3
dt = 1/252
""",
            "call": "calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
            "gold_call": "_gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
        },
        {
            "setup": """import numpy as np
# Test with constant volatility (edge case)
engine = 'rough_bergomi'
price_data = np.array([100.0, 101.0, 99.5, 100.5, 101.5])
vol_data = np.array([0.15, 0.15, 0.15, 0.15, 0.15])  # Constant
historical_hurst = 0.3
dt = 1/252
""",
            "call": "calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
            "gold_call": "_gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)",
        },
        {
            "setup": """import numpy as np
# Test Heston with default dt
engine = 'heston'
price_data = np.array([100.0, 101.0, 99.5, 100.5, 101.5, 100.0, 101.0, 99.5])
vol_data = np.array([0.15, 0.16, 0.14, 0.17, 0.18, 0.16, 0.15, 0.17])
historical_hurst = 0.6
""",
            "call": "calibrate_parameters(engine, price_data, vol_data, historical_hurst)",
            "gold_call": "_gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst)",
        },
        # --- Error cases ---
        {
            "setup": """import numpy as np
engine = 'invalid_engine'
price_data = np.array([100.0, 101.0, 99.5])
vol_data = np.array([0.15, 0.16, 0.14])
historical_hurst = 0.3
dt = 1/252

def run_model():
    try:
        calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)
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
engine = 'rough_bergomi'
price_data = np.array([100.0])  # Too short
vol_data = np.array([0.15])
historical_hurst = 0.3
dt = 1/252

def run_model():
    try:
        calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_calibrate_parameters(engine, price_data, vol_data, historical_hurst, dt)
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
engine = 'rough_bergomi'
price_data = np.array([100.0, 101.0, 99.5])
vol_data = np.array([0.15, 0.16])  # Different lengths

def run_model():
    try:
        calibrate_parameters(engine, price_data, vol_data, 0.3, 1/252)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_calibrate_parameters(engine, price_data, vol_data, 0.3, 1/252)
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
