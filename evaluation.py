"""Provides for testing and evaluating regressors."""

import numpy as np
import tensorflow as tf
import tools
tflog = tf.logging

from . import batch_data as dg


def _eval_regressor(regressor, X, y):
    """Evaluates a regressor on some test data :X:, :y:.
    """

    if hasattr(regressor, 'no_tf'):
        data_func = lambda: (X, y)
    else:
        data_func = dg.BatchData.to_dataset((X, y))

    predictor = regressor.predict(input_fn=data_func,
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
    """Evaluates an iterable of regressors on some test data
    :X:, :y:."""

    results = []
    if names is None:
        names = [None] * len(regressors)
    for regressor, name in zip(regressors, names):
        if name is not None:
            tflog.info("Evaluating {}".format(name))
        result = _eval_regressor(regressor, X, y)
        results.append(result)
    return results


def eval_regressor(regressor, gen_one_data, batch_size=1):
    """Evaluates a regressor on some test data of size :batch_size:
    generated from :gen_one_data:.
    """
    X, y = dg.BatchData.batch(gen_one_data, batch_size)
    return _eval_regressor(regressor, X, y)


def eval_regressors(regressors, gen_one_data, batch_size=1, names=None):
    """Evaluates an iterable of regressors on some test data of size
    :batch_size: generated from :gen_one_data:.
    """
    X, y = dg.BatchData.batch(gen_one_data, batch_size)
    return _eval_regressors(regressors, X, y, names=names)
