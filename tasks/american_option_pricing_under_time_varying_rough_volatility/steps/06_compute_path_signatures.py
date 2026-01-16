r"""
Complete the function `compute_path_signatures` which extracts truncated path signatures 
from variance paths.

Path signatures capture essential features of paths algebraically. They are:
- **Deterministic**: Same input path always produces the same signature
- **Feature-rich**: Different paths produce different signatures

For SciCode, we use:
- **Degree 2**: Truncate signature at degree 2 (reduced from paper's degree 3)
- **1D paths**: Each variance path is treated as a 1-dimensional path
- **Simplified implementation**: Direct computation of signature components for degree 2

For a 1D path with degree 2, the signature has 3 components:
- Component 0: Path increment (final - initial value)
- Component 1: First iterated integral: ∫₀ᵀ (path(t) - path(0)) dt
- Component 2: Second iterated integral: ∫₀ᵀ ∫₀ˢ (path(s) - path(0)) ds dt

The signature dimension for degree d is: d + 1
For degree=2, dimension = 3.

The function should:
- Accept variance_paths as a 2D numpy array (num_paths, num_steps)
- Accept degree as an integer (default 2)
- Accept level as an integer (default 2, reserved for future use)
- Return a 2D numpy array (num_paths, signature_dim) where signature_dim = degree + 1
- Compute signature for each path independently using numerical integration

Edge cases to handle:
- Empty paths array (raise ValueError)
- Paths with only one point (signature components are zeros except component 0)
- Ensure deterministic output (no randomness in signature computation)
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def compute_path_signatures(variance_paths: np.ndarray, degree: int = 2,
                            level: int = 2) -> np.ndarray:
    '''
    Compute truncated path signatures from variance paths.
    
    Computes path signatures directly using numerical integration. The signature dimension
    is determined by the degree parameter: dimension = degree + 1.
    
    Parameters
    ----------
    variance_paths : np.ndarray, shape (num_paths, num_steps)
        Variance paths from step 5.
    degree : int, optional
        Signature truncation degree (default 2 for SciCode).
    level : int, optional
        Signature truncation level (default 2 for SciCode, reserved for future use).
    
    Returns
    -------
    signatures : np.ndarray, shape (num_paths, m)
        Signature vectors where m = degree + 1.
        For degree=2, m = 3.
    '''
    return signatures


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_compute_path_signatures(variance_paths: np.ndarray, degree: int = 2,
                                   level: int = 2) -> np.ndarray:
    '''Reference implementation.'''
    # Validate inputs
    variance_paths = np.asarray(variance_paths, dtype=float)
    
    if variance_paths.ndim != 2:
        raise ValueError(f"variance_paths must be 2D array, got {variance_paths.ndim}D")
    
    if variance_paths.shape[0] == 0:
        raise ValueError("variance_paths must have at least one path")
    
    if variance_paths.shape[1] == 0:
        raise ValueError("variance_paths must have at least one time step")
    
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    
    # Signature dimension for 1D paths: degree + 1
    sig_dim = degree + 1
    
    num_paths = variance_paths.shape[0]
    num_steps = variance_paths.shape[1]
    signatures = np.zeros((num_paths, sig_dim))
    
    # Compute signature for each path
    for path_idx in range(num_paths):
        path = variance_paths[path_idx, :]
        
        # Component 0: Path increment (final - initial)
        signatures[path_idx, 0] = path[-1] - path[0]
        
        if degree >= 1 and num_steps > 1:
            # Component 1: First iterated integral
            # ∫₀ᵀ (path(t) - path(0)) dt
            # Approximate using trapezoidal rule
            path_centered = path - path[0]
            dt = 1.0 / (num_steps - 1)  # Normalized time step
            # Use trapezoidal rule: h * [f[0]/2 + f[1] + f[2] + ... + f[n-1] + f[n]/2]
            # Or: h * [sum - (first + last)/2]
            integral_1 = dt * (np.sum(path_centered) - 0.5 * (path_centered[0] + path_centered[-1]))
            signatures[path_idx, 1] = integral_1
        
        if degree >= 2 and num_steps > 1:
            # Component 2: Second iterated integral
            # ∫₀ᵀ ∫₀ˢ (path(s) - path(0)) ds dt
            # Approximate using double integration
            path_centered = path - path[0]
            dt = 1.0 / (num_steps - 1)
            
            # First integral: ∫₀ˢ (path(s) - path(0)) ds for each s
            first_integral = np.zeros(num_steps)
            for s in range(1, num_steps):
                # Trapezoidal rule for segment [0, s] with s+1 points
                segment = path_centered[:s+1]
                if len(segment) > 1:
                    # For segment with s+1 points, spacing is dt, total width is s*dt
                    # But we use uniform spacing dt, so: integral = dt * (sum - (first+last)/2)
                    first_integral[s] = dt * (np.sum(segment) - 0.5 * (segment[0] + segment[-1]))
                else:
                    first_integral[s] = 0.0
            
            # Second integral: ∫₀ᵀ first_integral(t) dt
            integral_2 = dt * (np.sum(first_integral) - 0.5 * (first_integral[0] + first_integral[-1]))
            signatures[path_idx, 2] = integral_2
    
    return signatures


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
# Test with simple variance paths
variance_paths = np.array([[0.04, 0.05, 0.06, 0.07, 0.08],
                           [0.04, 0.03, 0.02, 0.03, 0.04]])
degree = 2
level = 2
""",
            "call": "compute_path_signatures(variance_paths, degree, level)",
            "gold_call": "_gold_compute_path_signatures(variance_paths, degree, level)",
        },
        {
            "setup": """import numpy as np
# Test with linear path (known signature properties)
linear_path = np.linspace(1.0, 2.0, 10).reshape(1, -1)
degree = 2
level = 2
""",
            "call": "compute_path_signatures(linear_path, degree, level)",
            "gold_call": "_gold_compute_path_signatures(linear_path, degree, level)",
        },
        {
            "setup": """import numpy as np
# Test with default parameters
variance_paths = np.array([[0.04, 0.05, 0.06, 0.07, 0.08],
                           [0.04, 0.03, 0.02, 0.03, 0.04]])
""",
            "call": "compute_path_signatures(variance_paths)",
            "gold_call": "_gold_compute_path_signatures(variance_paths)",
        },
        {
            "setup": """import numpy as np
# Test with single path
variance_paths = np.array([[0.04, 0.05, 0.06, 0.07, 0.08]])
degree = 2
level = 2
""",
            "call": "compute_path_signatures(variance_paths, degree, level)",
            "gold_call": "_gold_compute_path_signatures(variance_paths, degree, level)",
        },
    ]
