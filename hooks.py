import tensorflow as tf

tft = tf.train


class ActionEverySecondOrStepHook(tft.SessionRunHook):
    """Takes an action every now and again according to the number of seconds or steps which have passed since the last
    time it took an action.
    """
    # Adapted from the source code for tf.train.CheckpointSaverHook

    def __init__(self, save_secs=600, save_steps=None, **kwargs):
        self._timer = tft.SecondOrStepTimer(every_secs=save_secs,
                                            every_steps=save_steps)
        self._global_step_tensor = None
        super(ActionEverySecondOrStepHook, self).__init__(**kwargs)

    def begin(self):
        self._global_step_tensor = tft.get_global_step()

    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        self._action(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):
        return tft.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self._action(run_context.session, global_step)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._action(session, last_step)

    def _action(self, session, step):
        raise NotImplementedError


class ProcessorSavingHook(ActionEverySecondOrStepHook):
    """Saves the processor data. Use as a SessionRunHook."""

    def __init__(self, processor, model_dir, **kwargs):
        self.processor = processor
        self.model_dir = model_dir
        super(ProcessorSavingHook, self).__init__(**kwargs)

    def _action(self, session, step):
        self.processor.save(session, step, self.model_dir)


class GlobalStepLogger(tft.SessionRunHook):
    def __init__(self, logger, subnetwork_name, **kwargs):
        self.logger = logger
        self.subnetwork_name = subnetwork_name
        super(GlobalStepLogger, self).__init__(**kwargs)

    # noinspection PyUnusedLocal
    def _action(self, session, step):
        self.logger.put_nowait(f'{self.subnetwork_name} trained to step {step}')
