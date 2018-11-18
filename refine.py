import multiprocessing as mp
import os
import queue as queue_lib
import tensorflow as tf
import tools

from . import batch_data as bd
from . import data_fn as df
from . import hooks as hooks_lib
from . import network as net

# noinspection PyCallingNonCallable
mp = mp.get_context('forkserver')  # The default 'fork' doesn't play nice with tensorflow

tfe = tf.estimator
tfi = tf.initializers
tflog = tf.logging
tft = tf.train
Def = tools.HasDefault  # Renamed so that our argument lists aren't massively long


# TODO: Find a really hard problem that we can't just train a FFNN to solve.
# TODO: Figure out why subtraining takes longer than it should to go through all the steps

# TODO: subnetwork logging not happening?
# TODO: subtraining taking ages to start?
# TODO: define_dnn not working?

defaults = tools.Record(hidden_units=(5, 5, 2), logits=1,
                        initial_sub_hidden_units=(2, 2), initial_sub_logits=1,
                        sub_hidden_units=(5, 5), sub_logits=1,
                        simple_train_steps=10000, sub_train_steps=2000, fractal_train_steps=5000,
                        activation=tf.nn.relu, data_fn=df.difficult, log_steps=1000,
                        subnetwork_logging=True,
                        num_processes=min(os.cpu_count(), 8))


@tools.with_defaults(defaults)
def create_network(subnetwork_fn, hidden_units=Def, logits=Def):
    var_init = tfi.truncated_normal(mean=0.0, stddev=0.05)
    model = net.Network()
    subnetwork_names = []
    for layer_index, layer_size in enumerate(hidden_units):
        layer_elements = []
        for neuron in range(layer_size):
            subnetwork_name = f'Subnetwork_{layer_index}_{neuron}'
            subnetwork = subnetwork_fn(model=model, subnetwork_name=subnetwork_name, var_init=var_init)
            layer_elements.append(subnetwork)
            subnetwork_names.append(subnetwork_name)
        layer = net.concat(*layer_elements)
        model.add(layer, mode=True, params=True)
    model.add(net.dense(units=logits, kernel_initializer=var_init, bias_initializer=var_init))
    return model, subnetwork_names


@tools.with_defaults(defaults)
def create_simple_dnn(hidden_units=Def, logits=Def, initial_sub_hidden_units=Def, initial_sub_logits=Def,
                      activation=Def):
    def subnetwork_fn(model, subnetwork_name, var_init):
        subnetwork = net.Subnetwork(name=subnetwork_name)
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_input'), debug=True)
        for unit in initial_sub_hidden_units:
            subnetwork.add(net.dense(units=unit, activation=activation, kernel_initializer=var_init,
                                     bias_initializer=var_init))
        if initial_sub_logits:
            subnetwork.add(net.dense(units=initial_sub_logits, kernel_initializer=var_init, bias_initializer=var_init,
                                     name='logits'))
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_output'), debug=True)
        return subnetwork
    return create_network(subnetwork_fn=subnetwork_fn, hidden_units=hidden_units, logits=logits)


@tools.with_defaults(defaults)
def create_fractal_dnn(hidden_units=Def, logits=Def, sub_hidden_units=Def, sub_logits=Def, activation=Def, uuid=None):
    def subnetwork_fn(model, subnetwork_name, var_init):


        # subnetwork = net.Subnetwork.define_dnn(hidden_units=sub_hidden_units, logits=sub_logits, activation=activation,
        #                                        name=subnetwork_name)


        subnetwork = net.Subnetwork(name=subnetwork_name)
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_input'), debug=True)
        for unit in sub_hidden_units:
            subnetwork.add(net.dense(units=unit, activation=activation, kernel_initializer=var_init,
                                     bias_initializer=var_init))
        if sub_logits:
            subnetwork.add(net.dense(units=sub_logits, kernel_initializer=var_init, bias_initializer=var_init,
                                     name='logits'))
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_output'), debug=True)


        return subnetwork
    model_, _ = create_network(subnetwork_fn=subnetwork_fn, hidden_units=hidden_units, logits=logits)
    if uuid is not None:
        model_.register_model_dir(f'/tmp/fractal_{uuid}')
    return model_


# TODO: switch this to use train_adaptively?
@tools.with_defaults(defaults)
def train(dnn, data_fn=Def, max_steps=1000, hooks=None, batch_size=64, use_processes=True, num_processes=Def):
    with bd.BatchData.context(data_fn=data_fn, batch_size=batch_size, num_processes=num_processes,
                              use_processes=use_processes) as input_fn:
        dnn.train(input_fn=input_fn, hooks=hooks, max_steps=max_steps)


@tools.with_defaults(defaults)
def eval_simple_dnn_for_subnetwork_training(queues, responders, loggers, simple_dnn, subnetwork_names, data_fn=Def,
                                            num_processes=Def):
    predict_keys = [f'{sub}_{type_}' for sub in subnetwork_names for type_ in ('input', 'output')]
    with bd.BatchData.context(data_fn=data_fn, batch_size=64, num_processes=num_processes) as input_fn:
        predictor = simple_dnn.debug(input_fn=input_fn, predict_keys=predict_keys)

        subnetworks_not_yet_trained = set(subnetwork_names)

        tflog.info(f'Starting training all subnetworks')
        while True:
            # Check if all of the subnetworks have finished training
            for sub in set(subnetworks_not_yet_trained):  # shallow copy
                responder = responders[sub]
                if not responder.empty():
                    subnetworks_not_yet_trained.remove(sub)

            if not subnetworks_not_yet_trained:
                tflog.info(f'Done training all subnetworks')
                break

            # Log out anything that the subnetworks want us to log
            for sub in subnetworks_not_yet_trained:
                if loggers[sub] is not None:
                    while True:
                        try:
                            sub_log = loggers[sub].get_nowait()
                        except queue_lib.Empty:
                            break
                        else:
                            tflog.info(sub_log)

            # Give the subnetworks data to train from
            prediction = next(predictor)
            for sub in subnetworks_not_yet_trained:
                X = prediction[f'{sub}_input']
                y = prediction[f'{sub}_output']
                try:
                    queues[sub].put_nowait((X, y))
                except queue_lib.Full:
                    pass


@tools.with_defaults(defaults)
def train_subnetwork(queue, responder, logger, subnetwork_name, sub_hidden_units=Def, sub_logits=Def,
                     sub_train_steps=Def, activation=Def, log_steps=Def):
    # Create the subnetworks
    submodel = net.Network.define_dnn(hidden_units=sub_hidden_units, logits=sub_logits, activation=activation,
                                      model_dir=f'/tmp/{subnetwork_name}_' + str(tools.uuid(6)), name=subnetwork_name)
    sub_dnn = submodel.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))

    def data_fn():
        return queue.get()
    if logger is not None:
        hooks = [hooks_lib.GlobalStepLogger(logger=logger, subnetwork_name=subnetwork_name, every_steps=log_steps,
                                            every_secs=None)]
    else:
        hooks = []

    # Train the subnetwork
    if logger is not None:
        logger.put_nowait(f'Started training {subnetwork_name}')
    # We're just pulling data off a queue so using processes is unnecessary here
    train(dnn=sub_dnn, data_fn=data_fn, hooks=hooks, max_steps=sub_train_steps, use_processes=False)
    if logger is not None:
        logger.put_nowait(f'Finished training {subnetwork_name}')

    # Return the subnetwork
    responder.put_nowait(sub_dnn.get_variable_values(filter_=lambda var: var.startswith(subnetwork_name),
                                                     map_name=lambda var: f'Network/{var}:0'))


def set_fractal_weights(fractal_dnn, sub_dnns_weights, simple_dnn):
    with tf.Session(graph=tf.Graph()) as sess:
        checkpoint = tft.get_checkpoint_state(fractal_dnn.model_dir)
        saver = tft.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        new_variable_values = simple_dnn.get_variable_values(filter_=lambda var: ('Subnetwork' not in var and
                                                                                  'Network' in var),
                                                             # Not using f-strings just to keep PyCharm happy
                                                             map_name=lambda var: '{var}:0'.format(var=var))
        for sub_dnn_weight in sub_dnns_weights.values():
            new_variable_values.update(sub_dnn_weight)
        variables_to_update = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                               if var.name in new_variable_values]
        sess.run(variables_to_update)
        variable_update_ops = [var.assign(new_variable_values[var.name]) for var in variables_to_update]
        sess.run(variable_update_ops)

        saver.save(sess, checkpoint.model_checkpoint_path)


@tools.with_defaults(defaults)
def overall(hidden_units=Def, logits=Def,
            initial_sub_hidden_units=Def, initial_sub_logits=Def,
            sub_hidden_units=Def, sub_logits=Def,
            simple_train_steps=Def, sub_train_steps=Def, fractal_train_steps=Def,
            activation=Def, data_fn=Def, log_steps=Def, subnetwork_logging=None,
            uuid=None):

    # Create and train the simple model
    simple_model, subnetwork_names = create_simple_dnn(hidden_units=hidden_units, logits=logits,
                                                       initial_sub_hidden_units=initial_sub_hidden_units,
                                                       initial_sub_logits=initial_sub_logits,
                                                       activation=activation)
    simple_dnn = simple_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    train(dnn=simple_dnn, data_fn=data_fn, max_steps=simple_train_steps)

    # Create and train the submodels
    queues = {}
    responders = {}
    loggers = {}
    processes = []
    for subnetwork_name in subnetwork_names:
        queue = mp.Queue(maxsize=100)
        responder = mp.Queue(maxsize=1)
        logger = mp.Queue(maxsize=5) if subnetwork_logging else None
        queues[subnetwork_name] = queue
        responders[subnetwork_name] = responder
        loggers[subnetwork_name] = logger
        p = mp.Process(target=train_subnetwork, args=(queue, responder, logger, subnetwork_name),
                       kwargs={'sub_train_steps': sub_train_steps})
        processes.append(p)
        p.start()
    try:
        eval_simple_dnn_for_subnetwork_training(queues=queues, responders=responders, loggers=loggers,
                                                simple_dnn=simple_dnn, subnetwork_names=subnetwork_names,
                                                data_fn=data_fn)
        sub_dnns_weights = {subnetwork_name: responders[subnetwork_name].get() for subnetwork_name in subnetwork_names}
    finally:
        for p in processes:
            p.terminate()
            p.join()

    # Create and train the fractal model
    fractal_model = create_fractal_dnn(hidden_units=hidden_units, logits=logits, sub_hidden_units=sub_hidden_units,
                                       sub_logits=sub_logits, activation=activation, uuid=uuid)
    fractal_dnn = fractal_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    # Not actually looking to train it now; this is just the easiest way to create a checkpoint to start from.
    train(dnn=fractal_dnn, data_fn=data_fn, max_steps=1, use_processes=False)
    set_fractal_weights(fractal_dnn=fractal_dnn, sub_dnns_weights=sub_dnns_weights, simple_dnn=simple_dnn)
    # Now we start training it
    if fractal_train_steps > 1:
        train(dnn=fractal_dnn, data_fn=data_fn, max_steps=fractal_train_steps)

    # An upper bound on the number of steps that we could possibly need to train our naive models for. If they still do
    # worse then our refinement system is definitely worth something.
    naive_steps = simple_train_steps + fractal_train_steps + sum(hidden_units) * sub_train_steps

    # Create and train the naive fractal model
    naive_fractal_model = create_fractal_dnn(hidden_units=hidden_units, logits=logits,
                                             sub_hidden_units=sub_hidden_units, sub_logits=sub_logits,
                                             activation=activation)
    naive_fractal_dnn = naive_fractal_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    train(dnn=naive_fractal_dnn, data_fn=data_fn, max_steps=naive_steps)

    # Create and train the naive fully connected model
    naive_hidden_units = []
    for x in hidden_units:
        for y in sub_hidden_units:
            naive_hidden_units.append(x * y)
    naive_simple_model = net.Network.define_dnn(hidden_units=naive_hidden_units, logits=logits, activation=activation)
    naive_simple_dnn = naive_simple_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    train(dnn=naive_simple_dnn, data_fn=data_fn, max_steps=naive_steps)

    return tools.Record(fractal_dnn=fractal_dnn, simple_dnn=simple_dnn, naive_fractal_dnn=naive_fractal_dnn,
                        naive_simple_dnn=naive_simple_dnn)
