import multiprocessing as mp
mp = mp.get_context('forkserver')
import os
import queue as queue_lib
import tools

Def = tools.HasDefault  # Renamed so that our argument lists aren't massively long


tf = None
bd = None
df = None
net = None
tfe = None
tfi = None
tflog = None
tft = None


def import_tf():
    # As you might have guessed, this is a bit hacky.
    # Basically, we're looking to spin up several processes and train a model in each.
    # Problem is, tensorflow doesn't like being copied over to each process
    # So we have to import it separately in each process, and in particular, not import it in the main process until
    # we've started the subprocesses.

    # However, fun fact: uncomment the following three lines...

    # import sys
    # if 'tensorflow' in sys.modules:
    #     raise RuntimeError('TensorFlow has already been imported!')

    # ...and the RuntimeError will be raised.
    # WTF?

    global tf; import tensorflow as tf

    global bd; from . import batch_data as bd
    global df; from . import data_fn as df
    global net; from . import network as net

    global tfe; tfe = tf.estimator
    global tfi; tfi = tf.initializers
    global tflog; tflog = tf.logging
    global tft; tft = tf.train


# TODO: Find a really hard problem that we can't just train a FFNN to solve.
# TODO: Figure out subtraining bottleneck
# TODO: Figure out what's going on with the TF stuff above. Something to do with tf.data.Dataset.from_generator, perhaps
# TODO: Memory maps! (+subprocess or os.exec(?) to start the extra processes.) Hopefully allowing for program repetition
# TODO:     and getting rid of these stupid import hacks

# Indirection so that we can specify these functions in defaults before they're actually available
def tf_nn_relu(*args, **kwargs):
    return tf.nn.relu(*args, **kwargs)


def df_difficult(*args, **kwargs):
    return df.difficult(*args, **kwargs)


defaults = tools.Record(hidden_units=(5, 5, 2), logits=1,
                        initial_sub_hidden_units=(2, 2), initial_sub_logits=1,
                        sub_hidden_units=(5, 5), sub_logits=1,
                        simple_train_steps=10000, sub_train_steps=2000, fractal_train_steps=5000,
                        activation=tf_nn_relu, data_fn=df_difficult, log_steps=1000,
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
    model, _ = create_network(subnetwork_fn=subnetwork_fn, hidden_units=hidden_units, logits=logits)
    if uuid is not None:
        model.register_model_dir(f'/tmp/fractal_{uuid}')
    return model


# TODO: switch this to use train_adaptively?
@tools.with_defaults(defaults)
def train(dnn, data_fn=Def, max_steps=1000, hooks=None, batch_size=64, use_processes=True, num_processes=Def):
    with bd.BatchData.context(data_fn=data_fn, batch_size=batch_size, num_processes=num_processes,
                              use_processes=use_processes) as input_fn:
        dnn.train(input_fn=input_fn, hooks=hooks, max_steps=max_steps)


@tools.with_defaults(defaults)
def eval_simple_dnn_for_subnetwork_training(queues, responders, simple_dnn, subnetwork_names, data_fn=Def,
                                            num_processes=Def):
    predict_keys = [f'{sub}_{type_}' for sub in subnetwork_names for type_ in ('input', 'output')]
    with bd.BatchData.context(data_fn=data_fn, batch_size=64, num_processes=num_processes) as input_fn:
        predictor = simple_dnn.debug(input_fn=input_fn, predict_keys=predict_keys)

        subnetworks_not_yet_trained = set(subnetwork_names)

        while True:
            # Check if all of the subnetworks have finished training
            for subnetwork in set(subnetworks_not_yet_trained):  # shallow copy
                responder = responders[subnetwork]
                if not responder.empty():
                    subnetworks_not_yet_trained.remove(subnetwork)

            if not subnetworks_not_yet_trained:
                tflog.info(f'Done training all subnetworks')
                break

            prediction = next(predictor)
            for sub in subnetworks_not_yet_trained:
                X = prediction[f'{sub}_input']
                y = prediction[f'{sub}_output']
                try:
                    queues[sub].put_nowait((X, y))
                except queue_lib.Full:
                    pass


@tools.with_defaults(defaults)
def train_subnetwork(queue, responder, subnetwork_name, sub_hidden_units=Def, sub_logits=Def, sub_train_steps=Def,
                     activation=Def, log_steps=Def):
    import_tf()

    # Create the subnetworks
    submodel = net.Network.define_dnn(hidden_units=sub_hidden_units, logits=sub_logits, activation=activation,
                                      model_dir=f'/tmp/{subnetwork_name}_' + str(tools.uuid(6)), name=subnetwork_name)
    sub_dnn = submodel.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))

    def data_fn():
        return queue.get()

    # Train the subnetwork
    # We're just pulling data off a queue so using processes is unnecessary here
    train(dnn=sub_dnn, data_fn=data_fn, max_steps=sub_train_steps, use_processes=False)

    # Return the subnetwork
    responder.put(sub_dnn.get_variable_values(filter=lambda var: var.startswith(subnetwork_name),
                                              map_name=lambda var: f'Network/{var}:0'))


def set_fractal_weights(fractal_dnn, sub_dnns_weights, simple_dnn):
    with tf.Session(graph=tf.Graph()) as sess:
        checkpoint = tft.get_checkpoint_state(fractal_dnn.model_dir)
        saver = tft.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        new_variable_values = simple_dnn.get_variable_values(filter=lambda var: ('Subnetwork' not in var and
                                                                                 'Network' in var),
                                                             map_name=lambda var: f'{var}:0')
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
            activation=Def, data_fn=Def, log_steps=Def, uuid=None):

    queues = {}
    responders = {}
    processes = []
    _subnetwork_names = []
    for layer_index, layer_size in enumerate(hidden_units):
        for neuron in range(layer_size):
            _subnetwork_names.append(f'Subnetwork_{layer_index}_{neuron}')
    for subnetwork_name in _subnetwork_names:
        queue = mp.Queue(maxsize=100)
        responder = mp.Queue(maxsize=1)
        queues[subnetwork_name] = queue
        responders[subnetwork_name] = responder
        p = mp.Process(target=train_subnetwork, args=(queue, responder, subnetwork_name),
                       kwargs={'sub_train_steps': sub_train_steps})
        processes.append(p)
        p.start()

    import_tf()

    # Create and train the simple model
    simple_model, subnetwork_names = create_simple_dnn(hidden_units=hidden_units, logits=logits,
                                                       initial_sub_hidden_units=initial_sub_hidden_units,
                                                       initial_sub_logits=initial_sub_logits,
                                                       activation=activation)
    simple_dnn = simple_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    train(dnn=simple_dnn, data_fn=data_fn, max_steps=simple_train_steps)

    # Create and train the submodels
    try:
        eval_simple_dnn_for_subnetwork_training(queues=queues, responders=responders, simple_dnn=simple_dnn,
                                                subnetwork_names=subnetwork_names, data_fn=data_fn)
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
    naive_steps = simple_train_steps + fractal_train_steps + sum(defaults.hidden_units) * sub_train_steps

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
