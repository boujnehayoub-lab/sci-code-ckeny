r"""
Complete the function `apply_random_fourier_features` which embeds signature vectors 
into lower-dimensional space using Random Fourier Features (RFF) to approximate RBF kernel efficiently.

Random Fourier Features (RFF) is a technique to approximate RBF (Radial Basis Function) kernels
efficiently, reducing computational cost from O(m) to O(D) per kernel evaluation. This implements
the paper's third innovation: efficient kernel approximation via Random Fourier Features.

For SciCode, we use:
- **D = 32**: RFF dimension (reduced from paper's D=128 for computational efficiency)
- **gamma = 1.0**: Kernel bandwidth parameter (default)
- **Deterministic behavior**: Using random_state for reproducibility

The RFF embedding formula:
1. Generate projection matrix W: W_{ij} ~ N(0, 1) with shape (m, D)
2. For each signature vector x: z = sqrt(2/D) * [cos(W^T x / gamma), sin(W^T x / gamma)]
3. Result shape: (num_paths, 2*D) where 2*D comes from concatenating cosine and sine components

The function should:
- Accept signatures as a 2D numpy array (num_paths, m)
- Accept D as an integer (default 32)
- Accept gamma as a float (default 1.0)
- Accept random_state as an integer (optional, for reproducibility)
- Return a 2D numpy array (num_paths, 2*D)
- Use np.random.RandomState(random_state) for all random number generation
- Ensure deterministic behavior: same inputs + same random_state → identical output

Edge cases to handle:
- Empty signatures array (raise ValueError)
- Invalid D (D <= 0, raise ValueError)
- Invalid gamma (gamma <= 0, raise ValueError)
- random_state=None should still work (use a default seed or allow randomness)
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

def apply_random_fourier_features(signatures: np.ndarray, D: int = 32,
                                  gamma: float = 1.0, 
                                  random_state: int = None) -> np.ndarray:
    '''
    Apply Random Fourier Features to signature vectors.
    
    Embeds signature vectors into lower-dimensional space using RFF to approximate
    RBF kernel efficiently. This reduces computational cost while preserving
    kernel approximation quality.
    
    Parameters
    ----------
    signatures : np.ndarray, shape (num_paths, m)
        Signature vectors from step 6.
    D : int, optional
        RFF dimension (default 32 for SciCode).
    gamma : float, optional
        Kernel bandwidth parameter (default 1.0).
    random_state : int, optional
        Random seed for projection matrix. Must be used for all random number generation.
        If None, uses a default seed for reproducibility.
    
    Returns
    -------
    rff_embeddings : np.ndarray, shape (num_paths, 2*D)
        RFF embeddings (sine and cosine concatenated).
    '''
    return rff_embeddings


# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _gold_apply_random_fourier_features(signatures: np.ndarray, D: int = 32,
                                        gamma: float = 1.0, 
                                        random_state: int = None) -> np.ndarray:
    '''Reference implementation.'''
    # Validate inputs
    signatures = np.asarray(signatures, dtype=float)
    
    if signatures.ndim != 2:
        raise ValueError(f"signatures must be 2D array, got {signatures.ndim}D")
    
    if signatures.shape[0] == 0:
        raise ValueError("signatures must have at least one path")
    
    if signatures.shape[1] == 0:
        raise ValueError("signatures must have at least one feature")
    
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    
    num_paths, m = signatures.shape
    
    # Use RandomState for deterministic behavior
    # If random_state is None, use 0 as default for reproducibility
    rng = np.random.RandomState(random_state if random_state is not None else 0)
    
    # Generate projection matrix W: shape (m, D)
    # W_{ij} ~ N(0, 1)
    W = rng.normal(0, 1, size=(m, D))
    
    # Compute W^T x for all paths: shape (num_paths, D)
    # This is: signatures @ W, which gives (num_paths, m) @ (m, D) = (num_paths, D)
    WTx = signatures @ W
    
    # Scale by gamma: W^T x / gamma
    WTx_scaled = WTx / gamma
    
    # Compute cosine and sine components
    cos_components = np.cos(WTx_scaled)  # shape (num_paths, D)
    sin_components = np.sin(WTx_scaled)  # shape (num_paths, D)
    
    # Concatenate: [cos, sin] -> shape (num_paths, 2*D)
    rff_embeddings = np.concatenate([cos_components, sin_components], axis=1)
    
    # Scale by sqrt(2/D)
    scale = np.sqrt(2.0 / D)
    rff_embeddings = scale * rff_embeddings
    
    return rff_embeddings


# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid cases ---
        {
            "setup": """import numpy as np
# Test with simple signatures
signatures = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])
D = 10
gamma = 1.0
random_state = 42
""",
            "call": "apply_random_fourier_features(signatures, D, gamma, random_state)",
            "gold_call": "_gold_apply_random_fourier_features(signatures, D, gamma, random_state)",
        },
        {
            "setup": """import numpy as np
# Test with default parameters
signatures = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8]])
""",
            "call": "apply_random_fourier_features(signatures)",
            "gold_call": "_gold_apply_random_fourier_features(signatures)",
        },
        {
            "setup": """import numpy as np
# Test with single path
signatures = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
D = 4
gamma = 1.0
random_state = 42
""",
            "call": "apply_random_fourier_features(signatures, D, gamma, random_state)",
            "gold_call": "_gold_apply_random_fourier_features(signatures, D, gamma, random_state)",
        },
    ]
