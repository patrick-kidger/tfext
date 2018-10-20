"""Provides for creating DNNs (specifically FFNN) ab initio."""

import functools as ft
import itertools as it
import tensorflow as tf
tfe = tf.estimator
tfi = tf.initializers
tfla = tf.layers
tflo = tf.losses
tft = tf.train
import tools

from . import processor as pc


def dense(*args, **kwargs):
    def dense(*args_, **kwargs_):
        return tfla.Dense(*args, **kwargs)(*args_, **kwargs_)
    return dense


def dropout(*args, **kwargs):
    def dropout(*args_, **kwargs_):
        return tfla.Dropout(*args, **kwargs)(*args_, **kwargs_)
    return dropout


# Keras-inspired nice interface, just without the slow speed and lack of 
# multicore functionality of Keras...
# (Plus it allows us to integrate our preprocessing)
class Network:
    """Defines a neural network. Expected usage is roughly:
    >>> model = Network()
    >>> model.add(dense(units=100, activation=tf.nn.relu))
    >>> model.add_train(dropout(rate=0.4))
    >>> model.add(dense(units=50, activation=tf.nn.relu))
    >>> model.add_train(dropout(rate=0.4))
    >>> model.add(dense(units=1, activation=tf.nn.relu))
    
    to define the neural network in the abstract (note that the last dense layer are treated as the logits), followed
    by:
    >>> dnn = model.compile()
    to actually create it in TensorFlow. Here, 'dnn' is a tf.Estimator, so may
    be used like:
    >>> dnn.train(...)
    >>> dnn.predict(...)
    >>> dnn.evaluate(...)
    """
    
    def __init__(self):
        """Creates a Network. See Network.__doc__ for more info."""
        self._layer_funcs = []
        self._layer_train = []
        self.processor = lambda: pc.IdentityProcessor()
        self.model_dir = None
        # self.compile_kwargs = tools.Object()
        
    def add(self, layer):
        """Add a layer to the network."""
        self._layer_funcs.append(layer)
        self._layer_train.append(False)
        
    def add_train(self, layer):
        """Add a layer to the network which needs to know if the network is in training or not."""
        self.add(layer)
        self._layer_train[-1] = True

    def register_processor(self, processor):
        """Used to set a :processor: to apply preprocessing to the input data."""
        if processor is not None:
            self.processor = processor

    def register_model_dir(self, model_dir):
        """Used to set a :model_dir:, to load any existing data for the model, and to save training results in."""
        if model_dir is not None:
            self.model_dir = model_dir

    # def add_compile_kwargs(self, **kwargs):
    #     self.update_compile_kwargs(kwargs)
    #
    # def update_compile_kwargs(self, dct):
    #     self.compile_kwargs.update(dct)
    #
    # def reset_compile_kwargs(self):
    #     self.compile_kwargs = tools.Object()
        
    def compile(self, optimizer=None, loss_fn=tflo.mean_squared_error, gradient_clip=None, learning_rate=None,
                **kwargs):
        """Takes its abstract neural network definition and compiles it into a tf.estimator.Estimator.
        
        May be given an :optimizer:, defaulting to tf.train.AdamOptimizer().
        May be given a :loss_fn:, defaulting to tf.losses.mean_squared_error.
        May be given a :gradient_clip:, defaulting to no clipping.
        May be given a :learning_rate:, which will set the learning rate of the default optimizer. Will be ignored if an
            actual optimizer is passed as well.
        
        Any additional kwargs are passed into the creation of the tf.estimator.Estimator.
        """
            
        def model_fn(features, labels, mode):
            # Create processor variables
            processor = self.processor()
            processor.init(model_dir=self.model_dir, training=mode == tfe.ModeKeys.TRAIN)
            
            # Apply any preprocessing to the features
            processed_features = processor.transform(features)
            
            # First layer is the feature inputs.
            layers = [processed_features]
            
            for prev_layer, layer_func, train in zip(layers, self._layer_funcs, 
                                                     self._layer_train):
                if train:
                    layer = layer_func(inputs=prev_layer, 
                                       training=mode == tfe.ModeKeys.TRAIN)
                else:
                    layer = layer_func(inputs=prev_layer)
                    
                # Deliberately using the generator nature of zip to add elements
                # to the layers list as we're iterating through it.
                # https://media.giphy.com/media/3oz8xtBx06mcZWoNJm/giphy.gif
                layers.append(layer)
                
            logits = layers[-1]
            predicted_labels = processor.inverse_transform(features, logits)
            
            if mode == tfe.ModeKeys.PREDICT:
                return tfe.EstimatorSpec(mode=mode, predictions=predicted_labels)
            
            loss = loss_fn(labels, predicted_labels)

            if mode == tfe.ModeKeys.TRAIN:
                nonlocal optimizer
                if optimizer is None:
                    if learning_rate is None:
                        optimizer = tft.AdamOptimizer()
                    else:
                        optimizer = tft.AdamOptimizer(learning_rate=learning_rate)

                g_step = tft.get_global_step()
                if gradient_clip is None:
                    train_op = optimizer.minimize(loss=loss, global_step=g_step)
                else:
                    # Perform Gradient clipping
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        gradients, variables = zip(*optimizer.compute_gradients(loss))
#                         gradients0 = tf.Print(gradients[0], [tf.global_norm(gradients)], 'Global norm: ')
#                         gradients = tuple([gradients0, *gradients[1:]])
                        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
                        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=g_step)
                if self.model_dir is None:
                    training_hooks = []
                else:
                    training_hooks = [pc.ProcessorSavingHook(self.processor, self.model_dir)]
                return tfe.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                         training_hooks=training_hooks)
            
            if mode == tfe.ModeKeys.EVAL:
                return tfe.EstimatorSpec(mode=mode, loss=loss)
            
            raise ValueError("mode '{}' not understood".format(mode))
                
        return tfe.Estimator(model_fn=model_fn, model_dir=self.model_dir, **kwargs)

    @classmethod
    def define_dnn(cls, hidden_units, logits, activation=tf.nn.relu, drop_rate=0.0, processor=None, model_dir=None):
        """Simple shortcut for defining a DNN via the Network interface, without compiling it.

        (Mostly useful as a way of integrating preprocessing, which the usual DNNRegressor does not allow.)

        Arguments:
        :[int] hidden_units: A list of integers describing the number of neurons in each hidden layer.
        :int logits: The number of output logits.
        :activation: The activation function for the hidden units. Defaults to tf.nn.relu.
        :float drop_rate: A number in the interval [0, 1) for the drop rate. Defaults to 0.
        :processor: A processor for pre- and post-postprocessing the data. Should be a function of no arguments
            returning a subclass of ProcessorBase.
        :str model_dir: A string describing the directory to save details about the trained model in.
        """

        self = cls()
        self.register_processor(processor)
        self.register_model_dir(model_dir)
        kernel_initializer = tfi.truncated_normal(mean=0.0, stddev=0.05)
        for units in hidden_units:
            self.add(dense(units=units, activation=activation, kernel_initializer=kernel_initializer))
            if drop_rate != 0:
                self.add_train(dropout(rate=drop_rate))
        self.add(dense(units=logits, kernel_initializer=kernel_initializer))

        return self


def create_dnn(hidden_units, logits, activation=tf.nn.relu, drop_rate=0.0, processor=None, model_dir=None,
               log_steps=100, **kwargs):
    """Simple shortcut for creating a DNN via the Network interface.

    (Mostly useful as a way of integrating preprocessing, which the usual DNNRegressor does not allow.)

    Arguments:
    :[int] hidden_units: A list of integers describing the number of neurons in each hidden layer.
    :int logits: The number of output logits.
    :activation: The activation function for the hidden units. Defaults to tf.nn.relu.
    :float drop_rate: A number in the interval [0, 1) for the drop rate. Defaults to 0.
    :int log_steps: How frequently to log results to stdout during training. Defaults to 1000.

    Any further kwargs are passed on to the call to compile the Network, for example to specify a processor or model
    directory.
    """

    model = Network.define_dnn(hidden_units, logits, activation, drop_rate, processor, model_dir)
    return model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps), **kwargs)
    
    
def model_dir_str(model_dir, hidden_units, logits, processor=lambda: pc.IdentityProcessor(),
                  activation=tf.nn.relu, uuid=None):
    """Returns a string for the model directory describing the network.
    
    Note that it only stores the information that describes the layout of the network - in particular it does not
    describe any training hyperparameters (in particular dropout rate).
    """
    
    layer_counter = [(k, sum(1 for _ in g)) for k, g in it.groupby(hidden_units)]
    for layer_size, layer_repeat in layer_counter:
        if layer_repeat == 1:
            model_dir += '{}_'.format(layer_size)
        else:
            model_dir += '{}x{}_'.format(layer_size, layer_repeat)
    model_dir += '{}__'.format(logits)
    model_dir += processor().__class__.__name__
    
    if isinstance(activation, ft.partial):
        activation_fn = activation.func
        alpha = str(activation.keywords['alpha']).replace('.', '')
    else:
        activation_fn = activation
        alpha = '02'
        
    model_dir += '_' + activation_fn.__name__.replace('_', '')
    if activation_fn is tf.nn.leaky_relu:
        model_dir += alpha

    if uuid not in (None, ''):
        model_dir += '_' + str(uuid)
    return model_dir
