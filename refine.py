import os
import tensorflow as tf
tfi = tf.initializers

from . import batch_data as bd
from . import data_fn as df
from . import dnn_from_seq as ds

num_processes = min(os.cpu_count(), 8)


class RememberedTensor:
    def __call__(self, inputs, debugging=False):
        var = tf.Variable(inputs, name='Remembered', validate_shape=False, use_resource=True)
        # assignment = var.assign(inputs)
        with tf.control_dependencies([var]):
            inputs = tf.identity(inputs)
        return inputs
        # shape = inputs.get_shape()
        # inputs = tf.Variable(inputs, name='Remembered', validate_shape=False)  # Todo: better naming
        # inputs.set_shape(shape)
        # return inputs


def simple_dnn(hidden_units=(3, 3, 3), logits=1, activation=tf.nn.relu):
    ki = tfi.truncated_normal(mean=0.0, stddev=0.05)
    model = ds.Network()
    for layer_size in hidden_units:
        layer_elements = []
        for _ in range(layer_size):
            subnetwork = ds.Subnetwork()
            subnetwork.add(ds.dense(units=1, activation=activation, kernel_initializer=ki))
            subnetwork.add(RememberedTensor(), debug=True)
            layer_elements.append(subnetwork)
        layer = ds.concat(*layer_elements)
        model.add(layer, mode=True, params=True)
    model.add(ds.dense(units=logits))
    return model


def train(model, data_fn=df.identity):
    dnn = model.compile()
    with bd.BatchData.context(data_fn=data_fn, batch_size=64, num_processes=num_processes) as input_fn:
        dnn.train(input_fn=input_fn, max_steps=1000)
    return dnn


def eval_subnetworks(dnn):
    pass


# TODO: test the variable names got by training such a DNN (hopefully it's not doing clever linalg speedups so we can
# TODO: extract the variables easily. (Not that we should really need to do do this, and don't want to refine just
# TODO: single neurons anyway
