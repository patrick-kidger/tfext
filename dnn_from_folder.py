"""Provides for creating DNN (specifically FFNN) from saved models."""

import functools as ft
import os
import tensorflow as tf
import tools
tflog = tf.logging

from . import dnn_from_seq as ds
from . import processor as pc


def _dnn_hyperparameters_from_dir(dir_name):
    """Creates DNN hyperparameters from the name of the directory of the DNN."""
    
    details = tools.Object()

    units, rest = dir_name.split('__')
    units = units.split('_')
    rest = rest.split('_')
    
    all_units = []
    for unit in units:
        if 'x' in unit:
            unit_size, unit_repeat = unit.split('x')
            unit_size, unit_repeat = int(unit_size), int(unit_repeat)
            all_units.extend([unit_size for _ in range(unit_repeat)])
        else:
            all_units.append(int(unit))
    details.hidden_units = all_units[:-1]
    details.logits = all_units[-1]
    
    processor_name = rest[0]
    processor_class = pc.ProcessorBase.find_subclass(processor_name)
    details.processor = processor_class()
    
    activation_name = rest[1].lower()
    
    # Not a great way to do this inversion, admittedly
    if activation_name[:9] == 'leakyrelu':
        alpha = float(str(activation_name[9]) + '.' + str(activation_name[10:]))
        details.activation = ft.partial(tf.nn.leaky_relu, alpha=alpha)
    else:
        try:
            activation_fn = getattr(tf.nn, activation_name)
        except AttributeError:
            raise ValueError(f"Activation '{activation_name}' not understood.")
        else:
            details.activation = activation_fn
        
    remaining = rest[2:]
    if len(remaining) == 0:
        uuid = None
    elif len(remaining) == 1:
        uuid = remaining[0]
    else:
        raise ValueError(f"Bad dir_name string '{dir_name}'. Too many remaining arguments: {remaining}")
        
    return details, uuid


def model_from_model_dir(model_dir, **kwargs):
    """Creates a model for a DNN from the :model_dir: argument. Any additional keyword arguments provided override the
    details of the DNN found.
    """

    if not os.path.isdir(model_dir):
        raise RuntimeError(f'Model dir {model_dir} does not exist')

    if model_dir[-1] in ('/', '\\'):
        model_dir = model_dir[:-1]
    dir_name = tools.split(['/', '\\'], model_dir)[-1]
    # I suspect that we should be able to restore the DNN just from the
    # information saved in the model directory, without needing to know
    # its structure from the directory name...
    details, uuid = _dnn_hyperparameters_from_dir(dir_name)
    details.update(kwargs)
    model = ds.Sequential.define_dnn(hidden_units=details.hidden_units, logits=details.logits,
                                     activation=details.activation, drop_rate=0.0,
                                     processor=details.processor, model_dir=model_dir)
    return model


def models_from_dir(dir_, exclude_start=('.',), exclude_end=(), exclude_in=(), **kwargs):
    """Creates multiple DNNs and processors from a directory containing the directories for multiple DNNs and
    processors.
    
    Its arguments :exclude_start:, :exclude_end:, :exclude_in: are each tuples which allow for excluding particular
    models, if their model directories start, end, or include any of the strings specified in each tuple respectively.
    
    Essentially just a wrapper around model_from_model_dir, to run it multiple times. It will forward any additional
    keyword arguments onto each call of model_from_model_dir.
    """
    
    subdirectories = sorted(next(os.walk(dir_))[1])
    if dir_[-1] in ('/', '\\'):
        dir_ = dir_[:-1]
    models = []
    names = []
    
    for subdir in subdirectories:
        if any(subdir.startswith(ex) for ex in exclude_start):
            tflog.info("Excluding '{}' based on start.".format(subdir))
            continue
        if any(subdir.endswith(ex) for ex in exclude_end):
            tflog.info("Excluding '{}' based on end.".format(subdir))
            continue
        if any(ex in subdir for ex in exclude_in):
            tflog.info("Excluding '{}' based on containment.".format(subdir))
            continue
            
        model_dir = dir_ + '/' + subdir
        try:
            model = model_from_model_dir(model_dir, **kwargs)
        except (FileNotFoundError, ValueError) as e:
            tflog.info("Could not load DNN from '{}'. Error message: '{}'"
                       .format(subdir, e))
        else:
            models.append(model)
            names.append(subdir)
            
    return models, names
