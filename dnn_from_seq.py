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


DEBUG = 'debug'


def dense(*args, **kwargs):
    def dense(*args_, **kwargs_):
        return tfla.Dense(*args, **kwargs)(*args_, **kwargs_)
    return dense


def dropout(*args, **kwargs):
    def dropout(*args_, **kwargs_):
        return tfla.Dropout(*args, **kwargs)(*args_, **kwargs_)
    return dropout


class Subnetwork:
    def __init__(self, name=None, **kwargs):
        self._layer_funcs = []
        self._layer_train = []
        self._layer_debug = []
        self._layer_mode = []
        self._layer_params = []
        self._name = name
        super(Subnetwork, self).__init__(**kwargs)

    def add(self, layer, train=False, debug=False, mode=False, params=False):
        """Add a layer to the network. Pass :train: as True if the layer needs to know if the network is in training or
        not. Pass :debug: as True is the layer needs to know if the network is in debug mode or not.
        """
        self._layer_funcs.append(layer)
        self._layer_train.append(train)
        self._layer_debug.append(debug)
        self._layer_mode.append(mode)
        self._layer_params.append(params)

    def __call__(self, inputs, mode, params=None):
        try:
            debug = params['dynamic'].debug
        except (KeyError, TypeError):
            debug = False
        with tf.variable_scope(self._name, self.__class__.__name__):
            prev_layer = inputs

            for layer_func, train_, debug_, mode_, params_ in zip(self._layer_funcs, self._layer_train,
                                                                  self._layer_debug, self._layer_mode,
                                                                  self._layer_params):
                extra_kwargs = {}
                if train_:
                    extra_kwargs['training'] = mode == tfe.ModeKeys.TRAIN
                if debug_:
                    extra_kwargs['debugging'] = debug
                if mode_:
                    extra_kwargs['mode'] = mode
                if params_:
                    extra_kwargs['params'] = params
                prev_layer = layer_func(inputs=prev_layer, **extra_kwargs)

        return prev_layer

    @classmethod
    def define_dnn(cls, hidden_units, logits, activation=tf.nn.relu, drop_rate=0.0):
        """Simple shortcut for defining a DNN via the Subnetwork interface.

        Arguments:
        :[int] hidden_units: A list of integers describing the number of neurons in each hidden layer.
        :int logits: The number of output logits. Set to anything Falsey to not use any logits (and expose the
            last hidden layer instead)
        :activation: The activation function for the hidden units. Defaults to tf.nn.relu.
        :float drop_rate: A number in the interval [0, 1) for the drop rate. Defaults to 0.
        """

        self = cls()
        kernel_initializer = tfi.truncated_normal(mean=0.0, stddev=0.05)
        for units in hidden_units:
            self.add(dense(units=units, activation=activation, kernel_initializer=kernel_initializer))
            if drop_rate != 0:
                self.add(dropout(rate=drop_rate), train=True)
        if logits:
            self.add(dense(units=logits, kernel_initializer=kernel_initializer, name='logits'))

        return self


# Todo: allow mixing of training/debugging/mode inputs to different subnetworks, i.e. when one of them isn't actually a
# subnetwork.
def concat(*subnetworks):
    def concat_wrapper(inputs, mode, params):
        logits = [subnetwork(inputs=inputs, mode=mode, params=params) for subnetwork in subnetworks]
        return tf.concat(logits, 1)  # axis 0 is the batch size
    return concat_wrapper


class DebugEstimator(tfe.Estimator):
    def __init__(self, *args, **kwargs):
        super(DebugEstimator, self).__init__(*args, **kwargs)
        if 'dynamic' in self._params:
            raise ValueError(f"{self.__class__.__name__} reserves 'dynamic' as a key in params; pleased use a different"
                             f" key.")
        else:
            self._params['dynamic'] = tools.Object(debug=False)

    def debug(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True):
        # Not a pretty hack. Doesn't seem to be a nice way to add extra modes though, sadly.
        self._params['dynamic'].debug = True
        for val in self.predict(input_fn, predict_keys, hooks, checkpoint_path, yield_single_examples):
            self._params['dynamic'].debug = False
            yield val
            self._params['dynamic'].debug = True
        self._params['dynamic'].debug = False


# Keras-inspired nice interface, just without the slow speed and lack of 
# multicore functionality of Keras...
# (Plus it allows us to integrate our preprocessing)
class Network(Subnetwork):
    """Defines a neural network. Expected usage is roughly:
    >>> model = Network()
    >>> model.add(dense(units=100, activation=tf.nn.relu))
    >>> model.add(dropout(rate=0.4), train=True)
    >>> model.add(dense(units=50, activation=tf.nn.relu))
    >>> model.add(dropout(rate=0.4), train=True)
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
    
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.processor = lambda: pc.IdentityProcessor()
        self.model_dir = None
        self.compile_kwargs = tools.Object()

    def register_processor(self, processor):
        """Used to set a :processor: to apply preprocessing to the input data."""
        if processor is not None:
            self.processor = processor

    def register_model_dir(self, model_dir):
        """Used to set a :model_dir:, to load any existing data for the model, and to save training results in."""
        if model_dir is not None:
            self.model_dir = model_dir

    def add_compile_kwargs(self, **kwargs):
        """Alternative default keyword arguments may be set for Sequential.compile for this instance of Sequential.
        Pass them as :**kwargs: here.
        """
        self.compile_kwargs.update(kwargs)

    def reset_compile_kwargs(self):
        """Reset the default keyword arguments for Sequential.compile."""
        self.compile_kwargs = tools.Object()
        
    def compile(self, optimizer=None, loss_fn=None, gradient_clip=None, learning_rate=None, **kwargs):
        """Takes its abstract neural network definition and compiles it into a tf.estimator.Estimator.

        May be given an :optimizer:, defaulting to tf.train.AdamOptimizer().
        May be given a :loss_fn:, defaulting to tf.losses.mean_squared_error.
        May be given a :gradient_clip:, defaulting to no clipping.
        May be given a :learning_rate:, which will set the learning rate of the default optimizer. Will be ignored if an
            actual optimizer is passed as well.

        Any additional kwargs are passed into the creation of the tf.estimator.Estimator.
        """

        if optimizer is None and 'optimizer' in self.compile_kwargs:
            optimizer = self.compile_kwargs.optimizer
        if gradient_clip is None and 'gradient_clip' in self.compile_kwargs:
            gradient_clip = self.compile_kwargs.gradient_clip
        if learning_rate is None and 'learning_rate' in self.compile_kwargs:
            learning_rate = self.compile_kwargs.learning_rate
        if loss_fn is None:
            if 'loss_fn' in self.compile_kwargs:
                loss_fn = self.compile_kwargs.loss_fn
            else:
                loss_fn = tflo.mean_squared_error
            
        def model_fn(features, labels, mode, params=None):
            # Create processor variables
            processor = self.processor()
            processor.init(model_dir=self.model_dir, mode=mode, params=params)
            
            # Apply any preprocessing to the features
            processed_features = processor.transform(features)

            # Apply the network to the input
            logits = self(inputs=processed_features, mode=mode, params=params)

            # Apply postprocessing to the results
            predicted_labels = processor.inverse_transform(features, logits)
            
            if mode == tfe.ModeKeys.PREDICT:
                return tfe.EstimatorSpec(mode=mode, predictions=predicted_labels)
            
            loss = loss_fn(labels, predicted_labels)

            if mode == tfe.ModeKeys.TRAIN:
                if optimizer is None:
                    if learning_rate is None:
                        optimizer_ = tft.AdamOptimizer()
                    else:
                        optimizer_ = tft.AdamOptimizer(learning_rate=learning_rate)
                else:
                    optimizer_ = optimizer

                g_step = tft.get_global_step()
                if gradient_clip is None:
                    train_op = optimizer_.minimize(loss=loss, global_step=g_step)
                else:
                    # Perform Gradient clipping
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        gradients, variables = zip(*optimizer_.compute_gradients(loss))
                        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
                        train_op = optimizer_.apply_gradients(zip(gradients, variables), global_step=g_step)
                if self.model_dir is None:
                    training_hooks = []
                else:
                    training_hooks = [pc.ProcessorSavingHook(processor, self.model_dir)]
                return tfe.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                         training_hooks=training_hooks)
            
            if mode == tfe.ModeKeys.EVAL:
                return tfe.EstimatorSpec(mode=mode, loss=loss)
            
            raise ValueError("mode '{}' not understood".format(mode))
                
        return DebugEstimator(model_fn=model_fn, model_dir=self.model_dir, params={}, **kwargs)

    @classmethod
    def define_dnn(cls, hidden_units, logits, activation=tf.nn.relu, drop_rate=0.0, processor=None, model_dir=None):
        """Simple shortcut for defining a DNN via the Network interface, without compiling it.

        (Mostly useful as a way of integrating preprocessing, which the usual DNNRegressor does not allow.)

        Arguments:
        :processor: A processor for pre- and post-postprocessing the data. Should be a function of no arguments
            returning a subclass of ProcessorBase.
        :str model_dir: A string describing the directory to save details about the trained model in.

        Other arguments as for Subnetwork.define_dnn.__doc__.
        """

        self = super(Network, cls).define_dnn(hidden_units, logits, activation, drop_rate)
        self.register_processor(processor)
        self.register_model_dir(model_dir)
        return self


# TODO? Remove? It's not a terribly useful function, and the docstring is out of date.
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
