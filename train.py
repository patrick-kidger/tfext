import os
import tensorflow as tf
tfce = tf.contrib.estimator
tfe = tf.estimator
tft = tf.train


def train_adaptively(model, input_fn, max_steps=1000,
                     learning_rate=0.01, gradient_clip=1.0,
                     learning_divisor=10, gradient_divisor=10,
                     steps_without_decrease=1000,
                     **kwargs):
    learning_rate = learning_rate
    gradient_clip = gradient_clip
    while True:
        dnn = model.compile(gradient_clip=gradient_clip, learning_rate=learning_rate, **kwargs)
        os.makedirs(dnn.eval_dir(), exist_ok=True)
        hook = tfce.stop_if_no_decrease_hook(dnn, 'loss', steps_without_decrease, run_every_secs=None,
                                             run_every_steps=2 * steps_without_decrease)
        train_spec = tfe.TrainSpec(input_fn=input_fn, max_steps=max_steps, hooks=[hook])
        eval_spec = tfe.EvalSpec(input_fn=input_fn, steps=steps_without_decrease, start_delay_secs=60)
        tfe.train_and_evaluate(dnn, train_spec, eval_spec)
        if learning_divisor is not None:
            learning_rate = learning_rate / learning_divisor
        if gradient_divisor is not None:
            gradient_clip = gradient_clip / gradient_divisor
        print(tft.get_global_step())
        break


import numpy as np
from . import batch_data
from . import dnn_from_seq as ds
def test():
    def f():
        X = np.array([np.random.uniform(-3, 3)])
        return X, np.square(X)
    i = batch_data.BatchData(f, num_processes=min(os.cpu_count(), 8))
    model = ds.Network.define_dnn((8, 8, 8), 1)
    return model, i
