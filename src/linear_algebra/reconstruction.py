# src/linear_algebra/reconstruction.py

import numpy as np

def reconstruction_error(X, X_reconstructed):
    """
    Computes Frobenius norm reconstruction error.
    """
    return np.linalg.norm(X - X_reconstructed, ord="fro")
