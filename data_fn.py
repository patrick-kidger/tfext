"""Some simple example functions for generating data for testing. (It's expected to feed this into a BatchData as the
data_fn argument.)
"""

import numpy as np


def identity():
    X = np.random.rand(1)
    return X, X
