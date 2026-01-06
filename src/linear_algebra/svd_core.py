# src/linear_algebra/svd_core.py

import numpy as np
from sklearn.decomposition import TruncatedSVD

def compute_truncated_svd(X, n_components=50, random_state=42):
    """
    Computes truncated SVD on high-dimensional data.

    Parameters:
    X : ndarray or DataFrame
        Normalized gene expression matrix
    n_components : int
        Number of singular components
    random_state : int

    Returns:
    U_reduced : ndarray
        Low-dimensional sample representation
    singular_values : ndarray
        Singular values
    svd_model : TruncatedSVD object
    """

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state
    )

    U_reduced = svd.fit_transform(X)
    singular_values = svd.singular_values_

    return U_reduced, singular_values, svd
