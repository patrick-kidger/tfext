# tfext (TensorFlow Extension)
Some basic stuff for using with TensorFlow.

* `BatchData`, with takes an arbitrary Python function that generates (feature, label) pairs, sets it going in multiple threads, and feeds the results into a tf.data.Dataset.
* `Network`, a Keras-inspired interface for assembling layers of neurons.
* `RegressorAverager`, for averaging an ensemble.
* `ProcessorBase`, for building in pre- and post-processing either side of the DNN.
* `concat_activations` and friends, for modifying existing activation functions to combine them into new more complex activation functions.

Plus a bit more.

Depends upon the [tools](https://github.com/patrick-kidger/tools) library.