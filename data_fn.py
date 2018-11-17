"""Some simple example functions for generating data for testing. (It's expected to feed this into a BatchData as the
data_fn argument.)
"""

import numpy as np


def batch_single(data_fn):
    """Converts a function that returns (feature, label) pairs as Python objects into a function that returns numpy
    arrays instead.
    """
    def wrapper():
        X, y = data_fn()
        return np.array([X]), np.array([y])
    return wrapper


def identity(X):
    """Returns a number chosen from Uniform[0,1] as both feature and label."""
    if X is None:
        X = np.random.rand(1)
    return X, X


def difficult(X=None):
    """An awkward function."""
    if X is None:
        X = np.random.rand(1)
    y = X * np.sin(X ** 2) + np.sin(10 * X) + np.sin((1 + X) ** 3)
    return X, y
