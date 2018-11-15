import collections as co
#import multiprocessing as mp
#import queue as queue_lib
import os
import tensorflow as tf
tfe = tf.estimator
tfi = tf.initializers
tft = tf.train
import tools
from tools import HasDefault as Def  # So that our argument lists aren't massively long

from . import batch_data as bd
from . import data_fn as df
from . import network as net

num_processes = min(os.cpu_count(), 8)


defaults = tools.Record(hidden_units=(15, 15, 10), logits=1, sub_hidden_units=(3, 3), sub_logits=1,
                        activation=tf.nn.relu, data_fn=df.difficult)
simple_train_steps = 10000
sub_train_steps = 2000
sub_train_increments = 100
fractal_train_steps = 5000

log_steps = 1000


@tools.with_defaults(defaults)
def _create_network(subnetwork_fn, hidden_units=Def, logits=Def):
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
def create_simple_dnn(hidden_units=Def, logits=Def, activation=Def):
    def subnetwork_fn(model, subnetwork_name, var_init):
        subnetwork = net.Subnetwork(name=subnetwork_name)
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_input'), debug=True)
        subnetwork.add(net.dense(units=1, activation=activation, kernel_initializer=var_init,
                                 bias_initializer=var_init))
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_output'), debug=True)
        return subnetwork
    return _create_network(subnetwork_fn=subnetwork_fn, hidden_units=hidden_units, logits=logits)


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
            subnetwork.add(net.dense(units=sub_logits, kernel_initializer=var_init, bias_initializer=var_init, name='logits'))
        subnetwork.add(net.RememberTensor(network=model, name=f'{subnetwork_name}_output'), debug=True)


        return subnetwork
    model, _ = _create_network(subnetwork_fn=subnetwork_fn, hidden_units=hidden_units, logits=logits)
    if uuid is not None:
        model.register_model_dir(f'/tmp/fractal_{uuid}')
    return model


# TODO: switch this to use train_adaptively?
@tools.with_defaults(defaults)
def train(dnn, data_fn=Def, max_steps=1000, batch_size=64, use_processes=True):
    with bd.BatchData.context(data_fn=data_fn, batch_size=batch_size, num_processes=num_processes,
                              use_processes=use_processes) as input_fn:
        dnn.train(input_fn=input_fn, max_steps=max_steps)


@tools.with_defaults(defaults)
def _eval_subnetworks(dnn, subnetwork_names, data_fn=Def):
    queues = {sub: co.deque() for sub in subnetwork_names}
    predict_keys = [f'{sub}_{type_}' for sub in subnetwork_names for type_ in  ('input', 'output')]
    with bd.BatchData.context(data_fn=data_fn, batch_size=64, num_processes=num_processes) as input_fn:
        predictor = dnn.debug(input_fn=input_fn, predict_keys=predict_keys)

        #i = 0
        def make_prediction(str_='default'):
            #nonlocal i
            #print(f'{str_}: before {i}')
            prediction = next(predictor)
            #print(f'{str_}: after {i}')
            #i += 1
            for sub in subnetwork_names:
                X = prediction[f'{sub}_input']
                y = prediction[f'{sub}_output']
                queues[sub].append((X, y))

        # Prepopulate the queues to a certain amount
        # for _ in range(64):
        #     make_prediction('initial')
        requested_sub = (yield)
        while True:
            queue = queues[requested_sub]
            if len(queue) == 0:
                make_prediction()
            requested_sub = (yield queue.popleft())
            # try:
            #     val = queue.get()
            # except queue_lib.Empty:
            #     make_prediction('pred')
            # else:
            #     requested_sub = (yield val)
            # for _ in range(len(subnetwork_names)):
            #     queue = queues[requested_sub]
            #     if len(queue) == 0:
            #         for __ in range(10):
            #             make_prediction('emergency')
            #     requested_sub = (yield queue.popleft())
            # make_prediction('running')


@tools.with_defaults(defaults)
def create_input_fns(dnn, subnetwork_names, data_fn=Def):
    inputs_fns = {}
    data_generator = _eval_subnetworks(dnn, subnetwork_names, data_fn)
    next(data_generator)  # Start the coroutine
    for sub in subnetwork_names:
        def data_fn(_sub=sub):
            return data_generator.send(_sub)
        inputs_fns[sub] = data_fn
    return inputs_fns


# TODO: make the number of logits dynamically determined by whatever the input_fns return
@tools.with_defaults(defaults)
def train_subnetworks(sub_dnns, input_fns):
    # Train them all simultaneously to make use of the fact that we generate data for all of them simultaneously
    for max_steps in range(sub_train_increments, sub_train_steps + 1, sub_train_increments):
        for sub, input_fn in input_fns.items():
            sub_dnn = sub_dnns[sub]
            # Threading mucks up the way we're pulling data through, and isn't necessary here because we're just getting
            # values that are generated by the original network; there's no difficult computation anywhere in the
            # Python.
            train(dnn=sub_dnn, data_fn=input_fn, max_steps=max_steps, use_processes=False)
    return sub_dnns


def set_fractal_weights(fractal_dnn, sub_dnns, simple_dnn):
    with tf.Session(graph=tf.Graph()) as sess:
        checkpoint = tft.get_checkpoint_state(fractal_dnn.model_dir)
        saver = tft.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        new_variable_values = simple_dnn.get_variable_values(filter=lambda var: 'Subnetwork' not in var and 'Network' in var,
                                                             map_name=lambda var: f'{var}:0')
        for name, sub_dnn in sub_dnns.items():
            new_variable_values.update(sub_dnn.get_variable_values(filter=lambda var: var.startswith(name),
                                                                   map_name=lambda var: f'Network/{var}:0'))
        variables_to_update = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                               if var.name in new_variable_values]
        sess.run(variables_to_update)
        variable_update_ops = [var.assign(new_variable_values[var.name]) for var in variables_to_update]
        sess.run(variable_update_ops)

        saver.save(sess, checkpoint.model_checkpoint_path)


@tools.with_defaults(defaults)
def overall(hidden_units=Def, logits=Def, sub_hidden_units=Def, sub_logits=Def, activation=Def, data_fn=Def, uuid=None):

    # Create and train the simple model
    simple_model, subnetwork_names = create_simple_dnn(hidden_units=hidden_units, logits=logits, activation=activation)
    simple_dnn = simple_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    train(dnn=simple_dnn, data_fn=data_fn, max_steps=simple_train_steps)

    # Create and train the submodels
    input_fns = create_input_fns(dnn=simple_dnn, subnetwork_names=subnetwork_names, data_fn=data_fn)
    sub_dnns = {}
    for sub in input_fns:
        submodel = net.Network.define_dnn(hidden_units=sub_hidden_units, logits=sub_logits, activation=activation,
                                          model_dir=f'/tmp/{sub}_' + str(tools.uuid(6)), name=sub)
        sub_dnns[sub] = submodel.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    train_subnetworks(sub_dnns=sub_dnns, input_fns=input_fns)

    # Create and train the fractal model
    fractal_model = create_fractal_dnn(hidden_units=hidden_units, logits=logits, sub_hidden_units=sub_hidden_units,
                                       sub_logits=sub_logits, activation=activation, uuid=uuid)
    fractal_dnn = fractal_model.compile(config=tfe.RunConfig(log_step_count_steps=log_steps))
    # Not actually looking to train it now; this is just the easiest way to create a checkpoint to start from.
    train(dnn=fractal_dnn, data_fn=data_fn, max_steps=1, use_processes=False)
    set_fractal_weights(fractal_dnn=fractal_dnn, sub_dnns=sub_dnns, simple_dnn=simple_dnn)
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

    return tools.Record(sub_dnns=sub_dnns, fractal_dnn=fractal_dnn, simple_dnn=simple_dnn,

                        naive_fractal_dnn=naive_fractal_dnn, naive_simple_dnn=naive_simple_dnn)
