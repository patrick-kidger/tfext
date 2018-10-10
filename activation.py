import tensorflow as tf
import tools


def concat_activations(activation_funcs, funcname=None):
    @tools.rename(funcname)
    def concat_activation(features, name=None, axis=-1):
        with tf.name_scope(name, "ConcatActivation", [features]) as name:
            features = tf.convert_to_tensor(features, name="features")
            activations = [activation_func(features) for activation_func in activation_funcs]
            return tf.concat(activations, axis, name=name)
    return concat_activation


def minus_activation(activation_func):
    @tools.rename(f'minus_{activation_func.__name__}')
    def minus(x, *args, **kwargs):
        return activation_func(-x, *args, **kwargs)
    return minus
