"""Provides for testing and evaluating regressors."""

import numpy as np
import tensorflow as tf
import tools
tflog = tf.logging

from . import batch_data as bd


def _eval_regressor(regressor, X, y):
    """Evaluates a regressor on some test data :X:, :y:."""

    if hasattr(regressor, 'no_tf'):
        input_fn = lambda: (X, y)
    else:
        input_fn = bd.BatchData.to_dataset((X, y))

    predictor = regressor.predict(input_fn=input_fn,
                                  yield_single_examples=False)
    prediction = next(predictor)
        
    diff = prediction - y
    squared_error = np.square(diff)
    total_loss = np.sum(squared_error)
    result = tools.Object(prediction=prediction,
                          X=X,
                          y=y,
                          diff=diff,
                          max_deviation=np.max(np.abs(diff)),
                          average_loss=np.mean(squared_error),
                          loss=total_loss / len(X),
                          total_loss=total_loss)
    return result


def _eval_regressors(regressors, X, y, names=None):
    """Evaluates an iterable of regressors on some test data :X:, :y:."""

    results = []
    if names is None:
        names = [None] * len(regressors)
    for regressor, name in zip(regressors, names):
        if name is not None:
            tflog.info("Evaluating {}".format(name))
        result = _eval_regressor(regressor, X, y)
        results.append(result)
    return results


def eval_regressor(regressor, data_fn, batch_size=1):
    """Evaluates a regressor on some test data of size :batch_size: generated from :data_fn:."""
    X, y = bd.BatchData.batch(data_fn, batch_size)
    return _eval_regressor(regressor, X, y)


def eval_regressors(regressors, data_fn, batch_size=1, names=None):
    """Evaluates an iterable of regressors on some test data of size :batch_size: generated from :data_fn:."""
    X, y = bd.BatchData.batch(data_fn, batch_size)
    return _eval_regressors(regressors, X, y, names=names)


def regressor_as_func(regressor):
    """Converts a regressor to a Python function that can be called in the normal manner."""
    def as_func(*args):
        def data_fn():
            return args, None
        X, y = bd.BatchData.batch(data_fn=data_fn, batch_size=1)
        input_fn = bd.BatchData.to_dataset((X, y))
        predictor = regressor.predict(input_fn=input_fn, yield_single_examples=False)
        prediction = next(predictor)
        return prediction[0]  # unpack it from its batch size of 1
    return as_func


def regressor_as_func_multi(regressor):
    """Converts a regressor to a Python function that can be called multiples times simultaneously, by batching the
    desired inputs together.

    Each argument passed to the resulting function should be a tuple of the arguments for one function call.
    Example:
    >>> func = regressor_as_func_multi(regressor)
    >>> func((1, 2), (3, 4), (5, 6))
    calls regressor on the input (1, 2), then on the input (3, 4), the on the input (5, 6).
    """
    def as_func(*args):
        index = -1
        def data_fn():
            nonlocal index
            index += 1
            return args[index], None
        X, y = bd.BatchData.batch(data_fn=data_fn, batch_size=len(args))
        input_fn = bd.BatchData.to_dataset((X, y))
        predictor = regressor.predict(input_fn=input_fn, yield_single_examples=False)
        prediction = next(predictor)
        return prediction
    return as_func
