import tensorflow as tf


_get_logger = tf.logging.info.__globals__['_get_logger']  # The usual awful hack to get at the things TF doesn't expose

def get_logger(name=None):
    if name is None:
        return _get_logger()
    else:
        return _get_logger().getChild(name)
