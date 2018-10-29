import os
import tensorflow as tf
tfi = tf.initializers

from . import batch_data as bd
from . import data_fn as df
from . import network as net

num_processes = min(os.cpu_count(), 8)


def simple_dnn(hidden_units=(3, 3, 3), logits=1, activation=tf.nn.relu):
    var_init = tfi.truncated_normal(mean=0.0, stddev=0.05)
    model = net.Network()
    for layer_size in hidden_units:
        layer_elements = []
        for _ in range(layer_size):
            subnetwork = net.Subnetwork()
            subnetwork.add(net.dense(units=1, activation=activation, kernel_initializer=var_init,
                                     bias_initializer=var_init))
            subnetwork.add(net.RememberTensor(model), debug=True)
            layer_elements.append(subnetwork)
        layer = net.concat(*layer_elements)
        model.add(layer, mode=True, params=True)
    model.add(net.dense(units=logits, kernel_initializer=var_init, bias_initializer=var_init))
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
