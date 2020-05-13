import tensorflow as tf
import numpy as np
import matplotlib
import urllib
from sim_hLN import *
from init_hLN import *
from utils import *
from tqdm import tqdm
from scipy import signal
# from plot import *

matplotlib.rcParams["legend.frameon"] = False


class hLN_Model(object):

    #   will need fleshing out/editing according to form of sim_hLN/general Python class for hLN model
    # to define hLN model we just need its structure (Jc) and how the input neurons connect to its subunits
    def __init__(self, Jc, Wce, Wci, sig_on):
        # Initialize the parameters in some way
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        M = len(Jc)
        self.n_e = np.concatenate(Wce).ravel().shape[0]
        self.n_i = np.concatenate(Wci).ravel().shape[0]
        self.Jc, self.Wce, self.Wci, self.sig_on = Jc, Wce, Wci, sig_on
        self.Jw = tf.random.uniform(shape=[M], minval=1, maxval=1, dtype=tf.float32)
        self.logJw = tf.Variable(tf.math.log(self.Jw))
        self.Wwe = tf.Variable(tf.random.uniform(shape=[self.n_e], minval=0.05, maxval=0.15, dtype=tf.float32))
        self.Wwi = tf.Variable(tf.random.uniform(shape=[self.n_i], minval=-0.15, maxval=-0.05, dtype=tf.float32))
        self.Taue = tf.random.uniform(shape=[self.n_e], minval=10, maxval=20, dtype=tf.float32)
        self.logTaue = tf.Variable(tf.math.log(self.Taue))
        self.Taui = tf.random.uniform(shape=[self.n_i], minval=5, maxval=10, dtype=tf.float32)
        self.logTaui = tf.Variable(tf.math.log(self.Taui))
        # can initialise Th to anything as model is initially linear, and then Th will be initialised by another
        # routine when it becomes nonlinear
        self.Th = tf.Variable(tf.random.uniform(shape=[M], minval=-3, maxval=3, dtype=tf.float32))
        self.Delay = tf.random.uniform(shape=[M], minval=0.1, maxval=5, dtype=tf.float32)
        self.logDelay = tf.Variable(tf.math.log(self.Delay))
        self.v0 = tf.Variable(tf.random.uniform(shape=(), minval=-0.1, maxval=0.1, dtype=tf.float32))
        self.params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                       self.Th, self.logDelay)
        self.trainable_params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                                 self.Th, self.logDelay)


        if M == 1 and not sig_on[0]:
            # if single subunit linear model, take out parameters from trainable_params list
            self.logJw.assign([0])
            self.trainable_params = (self.v0, self.Wwe, self.Wwi, self.logTaue, self.logTaui, self.logDelay)

        # initialise model in untied state - synapses have different weights and time constants
        self.tied = False



    def __call__(self, x):
        return sim_hLN_tf2(X=x, dt=1, Jc=self.Jc, Wce=self.Wce, Wci=self.Wci, params=self.params, sig_on=self.sig_on)

    def randomise_parameters(self):
        M = len(self.Jc)
        self.Jw = tf.random.uniform(shape=[M], minval=1, maxval=1, dtype=tf.float32)
        self.logJw.assign(tf.math.log(self.Jw))
        self.Wwe.assign(tf.random.uniform(shape=[self.n_e], minval=0.05, maxval=0.15, dtype=tf.float32))
        self.Wwi.assign(tf.random.uniform(shape=[self.n_i], minval=-0.15, maxval=-0.05, dtype=tf.float32))
        self.Taue = tf.random.uniform(shape=[self.n_e], minval=10, maxval=20, dtype=tf.float32)
        self.logTaue.assign(tf.math.log(self.Taue))
        self.Taui = tf.random.uniform(shape=[self.n_i], minval=5, maxval=10, dtype=tf.float32)
        self.logTaui.assign(tf.math.log(self.Taui))
        # can initialise Th to anything as model is initially linear, and then Th will be initialised by another
        # routine when it becomes nonlinear
        self.Th.assign(tf.random.uniform(shape=[M], minval=-3, maxval=3, dtype=tf.float32))
        self.Delay = tf.random.uniform(shape=[M], minval=0.1, maxval=5, dtype=tf.float32)
        self.logDelay.assign(tf.math.log(self.Delay))
        self.v0.assign(tf.random.uniform(shape=(), minval=-0.1, maxval=0.1, dtype=tf.float32))
        self.params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                       self.Th, self.logDelay)
        self.trainable_params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                                 self.Th, self.logDelay)

        if M == 1 and not self.sig_on[0]:
            # if single subunit linear model, take out parameters from trainable_params list
            self.logJw.assign([0])
            self.trainable_params = (self.v0, self.Wwe, self.Wwi, self.logTaue, self.logTaui, self.logDelay)

        return


class hLN_TiedModel(object):

    # hLN model with tied parameters - easier than adjusting current model to account for it. Probably want 2 options:
    # tie all synapses to have the same parameters, or one set of parameters for each subunit
    def __init__(self, Jc, Wce, Wci, sig_on):
        # Initialize the parameters in some way
        M = len(Jc)
        self.n_e = np.concatenate(Wce).ravel().shape[0]
        self.n_i = np.concatenate(Wci).ravel().shape[0]
        self.Jc, self.Wce, self.Wci, self.sig_on = Jc, Wce, Wci, sig_on
        self.Jw = tf.random.uniform(shape=[M], minval=1, maxval=1, dtype=tf.float32)
        self.logJw = tf.Variable(tf.math.log(self.Jw))
        self.Wwe = tf.Variable(tf.random.uniform(shape=[M], minval=0.05, maxval=0.15, dtype=tf.float32))
        self.Wwi = tf.Variable(tf.random.uniform(shape=[M], minval=-0.15, maxval=-0.05, dtype=tf.float32))
        self.Taue = tf.random.uniform(shape=[M], minval=10, maxval=20, dtype=tf.float32)
        self.logTaue = tf.Variable(tf.math.log(self.Taue))
        self.Taui = tf.random.uniform(shape=[M], minval=5, maxval=10, dtype=tf.float32)
        self.logTaui = tf.Variable(tf.math.log(self.Taui))
        # can initialise Th to anything as model is initially linear, and then Th will be initialised by another
        # routine when it becomes nonlinear
        self.Th = tf.Variable(tf.random.uniform(shape=[M], minval=-3, maxval=3, dtype=tf.float32))
        self.Delay = tf.random.uniform(shape=[M], minval=0.1, maxval=5, dtype=tf.float32)
        self.logDelay = tf.Variable(tf.math.log(self.Delay))
        self.v0 = tf.Variable(tf.random.uniform(shape=(), minval=-0.1, maxval=0.1, dtype=tf.float32))
        self.trainable_params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                                 self.Th, self.logDelay)
        self.params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                       self.Th, self.logDelay)

        if M == 1 and not sig_on[0]:
            # if single subunit linear model, take out parameters from trainable_params list
            self.logJw.assign([0])
            self.trainable_params = (self.v0, self.Wwe, self.Wwi, self.logTaue, self.logTaui, self.logDelay)

    def __call__(self, x):

        # feedforward model function - before carrying out, create vectors from the single weight and time
        # constant values and assing to parameters

        # will concatenate vectors together to form parameter vectors from the tied synaptic weights and time
        # constants, so intialise empty first
        logTaues = []
        logTauis = []
        Wwes = []
        Wwis = []
        for m in range(len(self.Jc)):
            logTaues = tf.concat((logTaues, tf.fill(dims=[len(self.Wce[m])], value=self.logTaue[m])), axis=0)
            logTauis = tf.concat((logTauis, tf.fill(dims=[len(self.Wci[m])], value=self.logTaui[m])), axis=0)
            Wwes = tf.concat((Wwes, tf.fill(dims=[len(self.Wce[m])], value=self.Wwe[m])), axis=0)
            Wwis = tf.concat((Wwis, tf.fill(dims=[len(self.Wci[m])], value=self.Wwi[m])), axis=0)

        self.params = (self.v0, self.logJw, Wwes, Wwis, logTaues, logTauis,
                       self.Th, self.logDelay)

        # now option with different parameters on each subunit

        return sim_hLN_tf2(X=x, dt=1, Jc=self.Jc, Wce=self.Wce, Wci=self.Wci, params=self.params, sig_on=self.sig_on)

    def randomise_parameters(self):
        M = len(self.Jc)
        self.Jw = tf.random.uniform(shape=[M], minval=1, maxval=1, dtype=tf.float32)
        self.logJw.assign(tf.math.log(self.Jw))
        self.Wwe.assign(tf.random.uniform(shape=[M], minval=0.05, maxval=0.15, dtype=tf.float32))
        self.Wwi.assign(tf.random.uniform(shape=[M], minval=-0.15, maxval=-0.05, dtype=tf.float32))
        self.Taue = tf.random.uniform(shape=[M], minval=10, maxval=20, dtype=tf.float32)
        self.logTaue.assign(tf.math.log(self.Taue))
        self.Taui = tf.random.uniform(shape=[M], minval=5, maxval=10, dtype=tf.float32)
        self.logTaui.assign(tf.math.log(self.Taui))
        # can initialise Th to anything as model is initially linear, and then Th will be initialised by another
        # routine when it becomes nonlinear
        self.Th.assign(tf.random.uniform(shape=[M], minval=-3, maxval=3, dtype=tf.float32))
        self.Delay = tf.random.uniform(shape=[M], minval=0.1, maxval=5, dtype=tf.float32)
        self.logDelay.assign(tf.math.log(self.Delay))
        self.v0.assign(tf.random.uniform(shape=(), minval=-0.1, maxval=0.1, dtype=tf.float32))
        self.trainable_params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                                 self.Th, self.logDelay)
        self.params = (self.v0, self.logJw, self.Wwe, self.Wwi, self.logTaue, self.logTaui,
                       self.Th, self.logDelay)

        if M == 1 and not self.sig_on[0]:
            # if single subunit linear model, take out parameters from trainable_params list
            self.logJw.assign([0])
            self.trainable_params = (self.v0, self.Wwe, self.Wwi, self.logTaue, self.logTaui, self.logDelay)

        return


# define loss and gradient function
@tf.function
def loss(predicted_v, target_v):
    """loss function is mean squared error between hLN model output and target membrane potential"""
    return tf.reduce_mean(tf.square(predicted_v - target_v))

# use gradient tape to calculate gradients used to optimise model
@tf.function
def grad(model, inputs, targets):
    """find value of loss function and its gradient with respect to the trainable parameters of the model"""
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_params)
        loss_value = loss(model(inputs), targets)
        grads = tape.gradient(loss_value, sources=model.trainable_params,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return loss_value, grads

@tf.function
def grad_subset(model, inputs, targets):
    """find the value of loss function and its gradient with respect to the trainable parameters of the model. Update
    for batch training, to only evaluate the loss on a subset of the batch data so as to reduce impact of inputs
    outside the batch window"""

    # define how much of the batch window we won't include in loss calculation - 100ms should be fine as target time
    # constants are maximum 20ms to begin with
    subset_delay = 100

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_params)
        loss_value = loss(model(inputs)[subset_delay:], targets[subset_delay:])
        grads = tape.gradient(loss_value, sources=model.trainable_params,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)

    return loss_value, grads


# define training function
def train(model, num_epochs, optimizer, inputs, target):
    """perform gradient descent training on a given hLN model for a specified number of epochs, while recording
    loss and accuracy information."""
    loss_values = []
    accuracies = []
    for epoch in tqdm(range(num_epochs)):
        loss_value, grads = grad(model=model, inputs=inputs, targets=target)
        accuracy = 100 * (1 - (loss_value/np.var(target)))
        loss_values.append(loss_value.numpy())
        accuracies.append(max(accuracy.numpy(), 0))
        optimizer.apply_gradients(zip(grads, model.trainable_params))

    return loss_values, accuracies


def train_sgd(model, num_epochs, optimizer, inputs, target):
    """perform gradient descent training on a given hLN model for a specified number of epochs, while recording
    loss and accuracy information. Adjusted to perform SGD to prevent overfitting."""
    loss_values = []
    accuracies = []
    n_points = 1000
    n_train = int(len(target.numpy()))
    for epoch in tqdm(range(num_epochs)):
        t_start = int(np.random.uniform(0, n_train - n_points))
        loss_value, grads = grad_subset(model=model, inputs=inputs[:, t_start: t_start + n_points],
                                        targets=target[t_start:t_start + n_points])
        accuracy = 100 * (1 - (loss_value/np.var(target[t_start:t_start + n_points])))
        loss_values.append(loss_value.numpy())
        accuracies.append(max(accuracy.numpy(), 0))
        optimizer.apply_gradients(zip(grads, model.trainable_params))

    return loss_values, accuracies


def train_sgd_decay(model, num_epochs, initial_rate, inputs, target, decay):
    """perform gradient descent training on a given hLN model for a specified number of epochs, while recording
    loss and accuracy information. Adjusted to perform SGD to prevent overfitting."""
    # create decaying learning rates
    iterations = np.arange(0, num_epochs, 1)
    l_rates = initial_rate * (1 / (1 + decay * iterations))

    loss_values = []
    accuracies = []
    n_points = 1000
    n_train = int(len(target.numpy()))
    for epoch in tqdm(range(num_epochs)):
        t_start = int(np.random.uniform(0, n_train - n_points))
        loss_value, grads = grad_subset(model=model, inputs=inputs[:, t_start: t_start + n_points],
                                        targets=target[t_start:t_start + n_points])
        accuracy = 100 * (1 - (loss_value / np.var(target[t_start:t_start + n_points])))
        loss_values.append(loss_value.numpy())
        accuracies.append(max(accuracy.numpy(), 0))
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=l_rates[epoch])
        optimizer.apply_gradients(zip(grads, model.trainable_params))

    return loss_values, accuracies


def train_until(model, train_inputs, train_target, val_inputs, val_target):
    """Function to perform SGD with Adam optimizer as previously, but this time train until condition instead of
    for specified number of epochs. Every n_check epochs, check the performance on validation data. If validation
    loss starts to increase, stop training"""

    train_losses = []
    val_losses = []
    n_points = 1000
    n_train = int(len(train_target.numpy()))
    last_val_loss1 = 999  # initialise big so training does not stop straight away
    last_val_loss2 = 1000

    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                              epsilon=1e-07, amsgrad=True)

    # set maximum training epochs at 10000 - stop before if condition satisfied
    max_epochs = 50000
    for epoch in tqdm(range(max_epochs)):

        t_start = int(np.random.uniform(0, n_train - n_points))
        loss_value, grads = grad_subset(model=model, inputs=train_inputs[:, t_start: t_start + n_points],
                                        targets=train_target[t_start:t_start + n_points])
        # accuracy = 100 * (1 - (loss_value / np.var(train_target[t_start:t_start + n_points])))
        # loss_values.append(loss_value.numpy())
        # accuracies.append(max(accuracy.numpy(), 0))
        # train_loss = loss(model(train_inputs), train_target)
        # val_loss = loss(model(val_inputs), val_target)
        # train_losses.append(train_loss)
        # val_losses.append(val_loss)

         # check validation loss every 100 training epochs and store it :
        if epoch % 100 == 0:
            val_loss = loss(model(val_inputs), val_target)
            val_losses.append(val_loss)
            # minimum 3000 epochs before we try to determine val loss gradient
            if epoch >= 4000:
                # then smooth validation losses by using a moving average across 5 stored datapoints
                N = 5
                smoothed_val_losses = np.convolve(val_losses, np.ones((N,))/N, mode='valid')

                # then use the last p smoothed values to decide if the validation loss is increasing, fit a 1d polynomial
                # by using p=16 and N=5, we take into account val loss over (N+p-1)*100 = 2000 epochs to determine if
                # the val loss in increasing
                p = 16
                lastp = smoothed_val_losses[-p:]
                m, c = np.polyfit(x=range(p), y=lastp, deg=1)
                if m > 0:
                    print(f"Epochs trained:{epoch}")
                    break



        optimizer_adam.apply_gradients(zip(grads, model.trainable_params))

    print(f"Epochs trained:{max_epochs}")

    return

