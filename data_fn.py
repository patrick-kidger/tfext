"""Some simple example functions for generating data for testing. (It's expected to feed this into a BatchData as the
data_fn argument.)
"""

import numpy as np


def batch_single(data_fn):
    def wrapper():
        X, y = data_fn()
        return np.array([X]), np.array([y])
    return wrapper


def identity():
    X = np.random.rand(1)
    return X, X
