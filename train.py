import os
import tensorflow as tf
tfce = tf.contrib.estimator
tfe = tf.estimator
tflog = tf.logging
tft = tf.train


# Yikes
context = tfe.Estimator.train.__globals__['context']
_load_global_step_from_checkpoint_dir = tfe.Estimator.train.__globals__['_load_global_step_from_checkpoint_dir']


# TODO? Allow for using 'steps' instead of 'max_steps', and allow for setting a particular optimizer.
def train_adaptively(model, input_fn, max_steps,
                     learning_rate=0.01, gradient_clip=1.0,
                     learning_divisor=10, gradient_divisor=10,
                     step_check=1000,
                     start_delay_secs=120,
                     throttle_secs=60,
                     **kwargs):
    """Trains a model with an adaptively decreasing learning rate or gradient clipping.

    The model will be trained for some number of steps, and the loss is periodically checked. If the loss ever goes up
    between checks then the training will adapt by reducing the learning rate and/or gradient clipping.

    Args:
        model: An instance of `Network` for training.
        input_fn: As in the tf.estimator.Estimator.train documentation.
        max_steps: As in the tf.estimator.Estimator.train documentation.
        learning_rate: `float`, the initial learning rate.
        gradient_clip: `float` or `None`, the initial gradient clip. May be set to None to turn off gradient clipping.
        learning_divisor: `float`, the amount to divide the learning rate by each time we adapt.
        gradient_divisor: `float`, the amount to divide the gradient clip by each time we adapt.
        step_check: `int`, how often (how many steps) the loss is checked.
        start_delay_secs: As in tf.estimator.EvalSpec documentation. May be used to avoid adapting early on during
            training.
        throttle_secs: As in the tf.estimator.EvalSpec documentation. May be used to avoid adapting too frequently.
        **kwargs: Are passed on to each compilation of the Network.

    Returns:
        model, for chaining.
    """
    learning_rate = learning_rate
    gradient_clip = gradient_clip
    if model.model_dir.startswith('/tmp'):
        tflog.warn(f"The model's model_dir is {model.model_dir}, which appears to be temporary: adaptive training "
                   f"requires a persistent directory to work from, so this is unlikely to work.")
    while True:
        dnn = model.compile(gradient_clip=gradient_clip, learning_rate=learning_rate, **kwargs)
        os.makedirs(dnn.eval_dir(), exist_ok=True)
        hook = tfce.stop_if_no_decrease_hook(estimator=dnn,
                                             metric_name='loss',
                                             max_steps_without_decrease=step_check,
                                             run_every_secs=None,
                                             run_every_steps=step_check)

        train_spec = tfe.TrainSpec(input_fn=input_fn, max_steps=max_steps, hooks=[hook])
        eval_spec = tfe.EvalSpec(input_fn=input_fn, steps=step_check, start_delay_secs=start_delay_secs,
                                 throttle_secs=throttle_secs)
        tfe.train_and_evaluate(dnn, train_spec, eval_spec)
        # Emulating how tf.estimator.Estimator.train handles getting the current step that a model has been trained to.
        with context.graph_mode():
            current_step = _load_global_step_from_checkpoint_dir(dnn._model_dir)
            if max_steps <= current_step:
                break
        learning_rate = learning_rate / learning_divisor
        tflog.info(f'Adaptively decreasing learning rate to {learning_rate}.')
        if gradient_clip is not None:
            gradient_clip = gradient_clip / gradient_divisor
            tflog.info(f'Adaptively decreasing gradient clip to {gradient_clip}.')
    return model
