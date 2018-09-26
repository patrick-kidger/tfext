# tfext (TensorFlow Extension)
Some basic stuff for using with TensorFlow.

* BatchData, with takes a function that generates (feature, label) pairs, sets it going in multiple threads, and feeds the results into a tf.data.Dataset.
* Sequential, a Keras-inspired interface for assembling layers of neurons.
* RegressorAverager, for averaging an ensemble.
* ProcessorBase, for building in pre- and post-processing either side of the DNN.

Plus a bit more.
