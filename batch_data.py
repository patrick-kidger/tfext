"""Provides for generating and batching data, and converting it into the
tf.data.Dataset format expected by TensorFlow.
"""

import multiprocessing as mp
import numpy as np
import os

import tensorflow as tf
tfd = tf.data

from . import exceptions as ex


### Multithreaded generating and batching of data

class BatchData:
    """Multithreading wrapper around tf.data.Dataset. Note that its
    terminate() method should be called when the instance is finished
    with.
    """
    
    def __init__(self, gen_one_data, batch_size=1, queue_size=50, 
                 X_dtype=None, y_dtype=None, X_shape=None, y_shape=None,
                 num_processes=None):
        """Initialising this class will create a queue of length :queue_size:
        and start populating it with the return values from :gen_one_data:.
        The argument :num_processes: determines how many processes will be
        used to call :gen_one_data:. (Note then that calls of :gen_one_data:
        might return their results out of order with the order that they were
        called in.) It defaults to the result of os.cpu_count().
        
        
        The argument :batch_size: is used to determine the size of the
        batches that it later produces. The arguments :X_dtype:, :y_dtype:,
        :X_shape: and :y_shape: should be the dtypes and shapes of the
        features (X) and labels (y) produced by gen_one_data. If any of them
        are set to None (their default), then :gen_one_data: will be called
        once to determine them automatically.
        """
        
        self.batch_size = batch_size
        self.queue = mp.Queue(maxsize=queue_size)

        if any([i is None] for i in (X_dtype, y_dtype, X_shape, y_shape)):
            X, y = gen_one_data()
            X_dtype = X.dtype
            y_dtype = y.dtype
            X_shape = X.shape
            y_shape = y.shape
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.X_shape = X_shape
        self.y_shape = y_shape

        def _gen_one_data(thread, max_thread):
            def gen_one_data_wrapper():
                gen_one_data.thread_prepare(thread, max_thread)
                while True:
                    self.queue.put(gen_one_data())
            return gen_one_data_wrapper
                
        if num_processes is None:
            num_processes = os.cpu_count()
        self.processes = [mp.Process(target=_gen_one_data(i, num_processes))
                          for i in range(num_processes)]
        
        for process in self.processes:
            process.start()
            
        self.terminated = False
        
    def __call__(self):
        """Creates a tf.data.Dataset gives batches of the appropriate size.
        """
        
        if not self.terminated:
            def generator():
                while True:
                    yield self.queue.get()
            # As we want a Dataset that keeps producing (feature, label) pairs
            # forever, we have to use the from_generator constructor. (I don't
            # think any of the others allow for online data production like this.)
            ds = tfd.Dataset.from_generator(generator, (self.X_dtype, self.y_dtype),
                                            (self.X_shape, self.y_shape))
            return ds.batch(self.batch_size)
        else:
            raise ex.TerminatedBatchData
    
    def terminate(self):
        """Terminates the processes that this instance uses."""
        for process in self.processes:
            process.terminate()
        self.terminated = True
            
    @classmethod
    def context(cls, *args, **kwargs):
        """For use in with statements. Creates a BatchData and automatically
        terminates it afterwards.
        """
        
        class _BatchDataContext:
            def __enter__(self_context):
                self = cls(*args, **kwargs)
                self_context.instance = self
                return self
            
            def __exit__(self_context, exc_type, exc_val, exc_tb):
                self_context.instance.terminate()
                
        return _BatchDataContext()
    
    @classmethod
    def batch(cls, gen_one_data, batch_size=1):
        """Takes a function :gen_one_data: which returns a generator and a
        :batch_size:, which defaults to 1, and returns a batch of that size. 
        Its return value is not wrapped in a tf.data.Dataset.
        """
        
        with cls.context(gen_one_data=gen_one_data) as self:
            X_batch = []
            y_batch = []
            for _ in range(batch_size):
                X, y = self.queue.get()
                X_batch.append(X)
                y_batch.append(y)
            return np.array(X_batch), np.array(y_batch)
        
    @staticmethod
    def to_dataset(data):
        """Returns a tf.data.Dataset which endlessly repeats :data:."""
        # Lambda wrapper is because in order to be part of the same graph as
        # the DNN, so it has to be called later on.
        return lambda: tfd.Dataset.from_tensors(data).repeat()
