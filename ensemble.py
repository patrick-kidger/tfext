"""Provides a regressor averager to handle an ensemble of regressors."""

import numpy as np
import tensorflow as tf
import tools
tflog = tf.logging

from . import dnn_from_folder as df
from . import evaluation


class RegressorAverager:
    """Regressor that averages the results of other regressors to make its prediction."""
    
    def __init__(self, regressors, mask=None, **kwargs):
        """Should be passed an iterable of :regressors:. It will make predictions according to their average.
        
        May also pass a :mask: argument, which should be a tuple of bools the same length as the number of regressors,
        specifying whether or not a particular regressor should be used when making predictions.
        """
        self.regressors = tuple(regressors)
        self.mask = None
        self.reset_mask()
        if mask is not None:
            self.set_mask(mask)
        super(RegressorAverager, self).__init__(**kwargs)
        
    def set_mask(self, mask):
        """Sets a mask to only use some of the regressors."""
        assert len(mask) == len(self.regressors)
        if not mask:
            tflog.warn('Setting empty mask for {}.'.format(self.__class__.__name__))
        self.mask = tuple(mask)
        return self  # for chaining
        
    def reset_mask(self):
        """Resets the mask, so as to use all regressors."""
        self.mask = [True for _ in range(len(self.regressors))]
        return self  # for chaining
    
    def auto_mask(self, gen_one_data, batch_size=1000, *, thresh=None, top=None):
        """Automatically creates a mask to only use the regressors that are deemed to be 'good'.
        
        The function :gen_one_data: will be called :batch_size: times to generate the  (feature, label) pairs on which
        the regressors are tested.
        
        Precisely one of :thresh: or :top: should be passed. The best :top: number of regressors whose loss is smaller
        than :thresh: will be deemed to be 'good'. If :thresh: is None (which it defaults to), then simply the best
        :top: regressors will be used. If :top: is None (which it defaults to) then simply every regressor whose loss is
        at least :thresh: will be used.
        """
        if thresh is None and top is None:
            raise RuntimeError('At least one of thresh or top must be not '
                               'None.')
            
        if thresh is None:
            thresh = np.inf
        if top is None:
            top = len(self.regressors)
        top = top - 1
        
        results = evaluation.eval_regressors(self.regressors, gen_one_data, batch_size)
        loss_values = [result.loss for result in results]            
        thresh_loss = min(sorted(loss_values)[top], thresh)
        
        dnn_mask = []
        for loss in loss_values:
            dnn_mask.append(loss <= thresh_loss)
            
        self.set_mask(dnn_mask)
        
    def predict(self, input_fn, *args, **kwargs):
        X, y = input_fn()
        
        returnval = tools.AddBase()
        counter = 0
        for regressor, mask in zip(self.regressors, self.mask):
            if mask:
                counter += 1
                returnval += evaluation._eval_regressor(regressor, X, y).prediction
        returnval = returnval / counter
        
        while True:
            yield returnval

    @classmethod
    def from_dir(cls, dir_, compile_kwargs=None, gen_one_data=None, batch_size=1000, *, thresh=None, top=None):
        """Creates an average regressor from all regressors in a specified directory."""

        if compile_kwargs is None:
            compile_kwargs = {}
        models, names = df.models_from_dir(dir_)
        self = cls(regressors=[model.compile(**compile_kwargs) for model in models])
        if gen_one_data is not None:
            self.auto_mask(gen_one_data=gen_one_data, batch_size=batch_size, thresh=thresh, top=top)
            
        return self
