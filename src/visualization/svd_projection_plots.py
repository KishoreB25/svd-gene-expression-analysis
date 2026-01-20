# src/visualization/svd_projection_plots.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_svd_2d(U_reduced, labels, save_path=None):
    """
    2D scatter plot of first two SVD components.
    """

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(
        U_reduced[:, 0],
        U_reduced[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D Projection using SVD")
    plt.colorbar(scatter, label="Class Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_svd_3d(U_reduced, labels, save_path=None):
    """
    3D scatter plot of first three SVD components.
    """

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        U_reduced[:, 0],
        U_reduced[:, 1],
        U_reduced[:, 2],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )

    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    ax.set_zlabel("SVD Component 3")
    ax.set_title("3D Projection using SVD")

    fig.colorbar(scatter, label="Class Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
