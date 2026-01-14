r"""
Complete the function `forecast_hurst_path` which forecasts future Hurst parameter values 
from historical values using an XGBoost gradient-boosted ensemble. This implements the 
paper's first innovation: using a gradient-boosted ensemble to forecast the time-varying 
Hurst parameter.

The forecasting method works as follows:
1. Use time indices as features: X = [[0], [1], [2], [3], [4]] for the 5 historical Hurst values
2. Train an XGBoost regressor on (X, historical_hurst) where historical_hurst = [H_{t-4}, ..., H_t]
3. Forecast future values by extrapolating time indices: [5], [6], ..., [4+horizon]
4. Clip all forecasted values to the range [0, 1] (Hurst parameter must be in this range)

You should use `xgb.XGBRegressor` with the following parameters:
- n_estimators: number of boosting rounds (default 100)
- learning_rate: learning rate (default 0.1)
- max_depth: maximum tree depth (default 3)
- random_state: random seed for reproducibility (must be used to ensure deterministic behavior)

The function should:
- Accept historical_hurst as a numpy array of shape (5,) containing 5 most recent Hurst values
- Accept horizon as an integer specifying how many future steps to forecast
- Return a numpy array of shape (horizon,) containing the forecasted Hurst values
- Ensure all outputs are clipped to [0, 1] range
- Be deterministic: same inputs + same random_state → identical output

Edge cases to handle:
- If all historical values are the same, the forecast should remain approximately constant
- If forecasted values go outside [0, 1], clip them to this range
- Ensure random_state is used for all random operations to guarantee reproducibility
"""

import numpy as np
import xgboost as xgb

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def forecast_hurst_path(historical_hurst: np.ndarray, horizon: int, 
                        n_estimators: int = 100, learning_rate: float = 0.1,
                        max_depth: int = 3, random_state: int = None) -> np.ndarray:
    '''
    Forecast Hurst parameter path using XGBoost gradient-boosted ensemble.
    
    This implements the paper's first innovation: using a gradient-boosted
    ensemble (XGBoost) to forecast the time-varying Hurst parameter.
    
    Parameters
    ----------
    historical_hurst : np.ndarray, shape (5,)
        Five most recent Hurst values [H_{t-4}, ..., H_t].
        Must be deterministic synthetic data.
    horizon : int
        Forecast horizon (number of steps, typically 5-20).
    n_estimators : int, optional
        Number of boosting rounds (default 100, reduced from paper's full scale).
    learning_rate : float, optional
        Learning rate (default 0.1).
    max_depth : int, optional
        Maximum tree depth (default 3).
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    forecasted_hurst : np.ndarray, shape (horizon,)
        Forecasted Hurst values [Ĥ_{t+1}, ..., Ĥ_{t+horizon}].
        Values are clipped to [0, 1] if needed.
    '''
    return forecasted_hurst


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_forecast_hurst_path(historical_hurst: np.ndarray, horizon: int, 
                              n_estimators: int = 100, learning_rate: float = 0.1,
                              max_depth: int = 3, random_state: int = None) -> np.ndarray:
    '''Reference implementation.'''
    historical_hurst = np.asarray(historical_hurst, dtype=float)
    
    # Validate input shape
    if historical_hurst.shape != (5,):
        raise ValueError("historical_hurst must have shape (5,)")
    
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    
    # Clip historical values to [0, 1] if needed
    historical_hurst = np.clip(historical_hurst, 0.0, 1.0)
    
    # Prepare training data: time indices as features
    # X = [[0], [1], [2], [3], [4]] for 5 historical values
    X_train = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y_train = historical_hurst
    
    # Train XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    # Prepare forecast features: extrapolate time indices
    # [5], [6], ..., [4+horizon]
    X_forecast = np.array([[i] for i in range(5, 5 + horizon)], dtype=float)
    
    # Forecast future values
    forecasted = model.predict(X_forecast)
    
    # Clip to [0, 1] range
    forecasted = np.clip(forecasted, 0.0, 1.0)
    
    return forecasted.astype(float)


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
historical_hurst = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
horizon = 5
random_state = 42
""",
            "call": "forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
            "gold_call": "_gold_forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
historical_hurst = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
horizon = 10
random_state = 123
""",
            "call": "forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
            "gold_call": "_gold_forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
historical_hurst = np.array([0.6, 0.55, 0.5, 0.45, 0.4])
horizon = 8
random_state = 42
""",
            "call": "forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
            "gold_call": "_gold_forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
historical_hurst = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
horizon = 5
n_estimators = 50
learning_rate = 0.05
max_depth = 2
random_state = 42
""",
            "call": "forecast_hurst_path(historical_hurst, horizon, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)",
            "gold_call": "_gold_forecast_hurst_path(historical_hurst, horizon, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
# Test with specific random state for deterministic output
historical_hurst = np.array([0.3, 0.32, 0.31, 0.33, 0.30])
horizon = 5
random_state = 999
""",
            "call": "forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
            "gold_call": "_gold_forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
        },
        {
            "setup": """import numpy as np
# Test clipping: values that would go outside [0, 1] should be clipped
historical_hurst = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
horizon = 3
random_state = 42
""",
            "call": "forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
            "gold_call": "_gold_forecast_hurst_path(historical_hurst, horizon, random_state=random_state)",
        },
        # --- Error cases ---
        {
            "setup": """import numpy as np
historical_hurst = np.array([0.3, 0.32, 0.31])  # Wrong shape: should be (5,)
horizon = 5

def run_model():
    try:
        forecast_hurst_path(historical_hurst, horizon)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_forecast_hurst_path(historical_hurst, horizon)
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
historical_hurst = np.array([0.3, 0.32, 0.31, 0.33, 0.30])
horizon = 0  # Invalid: horizon must be >= 1

def run_model():
    try:
        forecast_hurst_path(historical_hurst, horizon)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _gold_forecast_hurst_path(historical_hurst, horizon)
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
