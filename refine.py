import tensorflow as tf

from . import dnn_from_seq as ds


def simple_dnn(hidden_units=(3, 3, 3), logits=1, activation=tf.nn.relu):
    model = ds.Network()
    for layer_size in hidden_units:
        neurons = [ds.Subnetwork.define_dnn(hidden_units=(1,), logits=0, activation=activation)
                   for _ in range(layer_size)]
        layer = ds.concat(*neurons)
        model.add(layer)
    model.add(ds.dense(units=logits))
    return model


# TODO: test the variable names got by training such a DNN (hopefully it's not doing clever linalg speedups so we can
# TODO: extract the variables easily. (Not that we should really need to do do this, and don't want to refine just
# TODO: single neurons anyway
