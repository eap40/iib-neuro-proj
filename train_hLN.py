import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
import urllib
from sim_hLN import *
from init_hLN import *
from utils import *
from tqdm import tqdm

matplotlib.rcParams["legend.frameon"] = False

# SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 24, 24, 24
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title


class hLN_Model(object):

    #   will need fleshing out/editing according to form of sim_hLN/general Python class for hLN model
    # to define hLN model we just need its structure (Jc) and how the input neurons connect to its subunits
    def __init__(self, Jc, Wce, Wci, sig_on):
        # Initialize the parameters in some way
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        M = len(Jc)
        self.Jc, self.Wce, self.Wci, self.sig_on = Jc, Wce, Wci, sig_on
        self.log_Jw = tf.Variable(np.full(M, 0), dtype=tf.float32) #coupling weights 1 for all branches intially
        self.Wwe = tf.Variable([np.full(X_e.shape[0], 1.0)], dtype=tf.float32)
        self.Wwi = tf.Variable([np.full(X_i.shape[0], -1.0)], dtype=tf.float32)
        self.log_Tau_e = tf.Variable([np.full(X_e.shape[0], 0)], dtype=tf.float32)
        self.log_Tau_i = tf.Variable([np.full(X_i.shape[0], 0)], dtype=tf.float32)
        self.Th = tf.Variable([1.0], dtype=tf.float32)
        self.log_Delay = tf.Variable(np.zeros([M, 1]), dtype=tf.float32)
        self.v0 = tf.Variable(0, dtype=tf.float32)
        self.params = (self.v0, self.log_Jw, self.Wwe, self.Wwi, self.log_Tau_e, self.log_Tau_i,
                       self.Th, self.log_Delay)
        self.trainable_params = (self.v0, self.log_Jw, self.Wwe, self.Wwi, self.log_Tau_e, self.log_Tau_i,
                                 self.Th, self.log_Delay)


    def __call__(self, x):
        return sim_hLN_tf(X=x, dt=1, Jc=self.Jc, Wce=self.Wce, Wci=self.Wci, params=self.params, sig_on=self.sig_on)

    def randomise_parameters(self):
        # self.Wwe = tf.Variable([np.random.uniform(0, 2, X_e.shape[0])], dtype=tf.float32)
        # self.Wwi = tf.Variable([np.random.uniform(-2, 0, X_i.shape[0])], dtype=tf.float32)
        # self.Tau_e = tf.Variable([np.random.uniform(0, 2, X_e.shape[0])], dtype=tf.float32)
        # self.Tau_i = tf.Variable([np.random.uniform(0, 2, X_i.shape[0])], dtype=tf.float32)
        # self.v0 = tf.Variable(np.mean(target), dtype=tf.float32)
        self.log_Jw.assign(np.random.uniform(0, 1, M))
        self.Wwe.assign([np.random.uniform(0, 2, X_e.shape[0])])
        self.Wwi.assign([np.random.uniform(-2, 0, X_i.shape[0])])
        self.log_Tau_e.assign([np.random.uniform(0, 1, X_e.shape[0])])
        self.log_Tau_i.assign([np.random.uniform(0, 1, X_i.shape[0])])
        self.Th.assign([np.random.uniform(-0.5, 0.5)])
        self.v0.assign(np.random.uniform(0, 0.2))
        self.params = (self.v0, self.log_Jw, self.Wwe, self.Wwi, self.log_Tau_e, self.log_Tau_i,
                       self.Th, self.log_Delay)
        self.trainable_params = (self.v0, self.log_Jw, self.Wwe, self.Wwi, self.log_Tau_e, self.log_Tau_i,
                                 self.Th, self.log_Delay)
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
        grads = tape.gradient(loss_value, sources=model.trainable_params, unconnected_gradients=tf.UnconnectedGradients.NONE)
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
        optimizer.apply_gradients(zip(grads, model.params))

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
        loss_value, grads = grad(model=model, inputs=inputs[:, t_start: t_start + n_points],
                                 targets=target[t_start:t_start + n_points])
        accuracy = 100 * (1 - (loss_value/np.var(target[t_start:t_start + n_points])))
        loss_values.append(loss_value.numpy())
        accuracies.append(max(accuracy.numpy(), 0))
        optimizer.apply_gradients(zip(grads, model.params))

    return loss_values, accuracies



# print(tf.autograph.to_code(sim_hLN_tf.python_function))

#### now apply same format to hLN model instead of regression model ####

# first we need our inputs and target output
# E_spikes, I_spikes = gen_realistic_inputs(Tmax=3000)
# X_e = spikes_to_input(E_spikes, Tmax=48000)
# X_i = spikes_to_input(I_spikes, Tmax=48000)
# X_tot = np.vstack((X_e, X_i)) #this is our final input
X_tot = tf.convert_to_tensor(np.load('real_inputs.npy'), dtype=tf.float32)  # real inputs made earlier
X_e = X_tot[:629]
X_i = X_tot[629:]

# true parameters to produce output:
N_soma = 420
Jc_sing = np.array([0])
M = len(Jc_sing)
Jw_sing = np.full([M,1], 1) #coupling weights 1 for all branches intially
M = len(Jc_sing) #number of subunits
Wce_sing = [np.arange(0, X_e.shape[0], 1)] #all input excitatory neurons connected to root subunit
Wwe_sing = [np.ones(X_e.shape[0])] #weighting matrix - all excitatory neurons connected with weight 1
Wci_sing = [np.arange(X_e.shape[0], X_e.shape[0] + X_i.shape[0] - 1, 1)] #all input inhibitory neurons connected to root subunit
Wwi_sing = [np.full(X_i.shape[0], -1)] #weighting matrix - all inhibitory neurons connected with weight -1
Tau_e = [np.full(X_e.shape[0], 1)] #all excitatory time constants 1
Tau_i = [np.full(X_i.shape[0], 1)] #all inhibitory time constants 1
Th = [1] #no offset in all sigmoids
v0 = 0 #no offset in membrane potential

params_sing = [v0, Jw_sing, Wwe_sing, Wwi_sing, Tau_e, Tau_i, Th]


# target = sim_hLN_tf(X=X_tot, dt=1, Jc=Jc_sing, Wce=Wce_sing, Wci=Wci_sing, params=hLN_model.params)


#
# # first_output = hLN_model(X_tot[:, start:start + n_timepoints])
#
# # define optimizer
# optimizer_slow = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.2)
# optimizer_fast = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.4)
#
#
# epochs = range(1)
# loss_values = []
# accuracies = []
# for epoch in tqdm(epochs):
#     logdir = 'log'
#     writer = tf.summary.create_file_writer(logdir)
#     tf.summary.trace_on(graph=True, profiler=True)
#     loss_value, grads = grad(model=hLN_model, inputs=X_tot[:, start:start + n_timepoints], targets=target)
#     with writer.as_default():
#         tf.summary.trace_export(name="test", step=0, profiler_outdir=logdir)
#     accuracy = 100 * (1 - (loss_value/np.var(target)))
#     loss_values.append(loss_value.numpy())
#     accuracies.append(accuracy)
#     optimizer_slow.apply_gradients(zip(grads, hLN_model.trainable_params))
#
# accuracies = np.clip(accuracies, a_min=0, a_max=None)

# plt.figure(1)
# plt.title('Loss and accuracy during primary training')
#
# plt.subplot(1, 2, 1)
# plt.plot(loss_values)
# plt.title('Loss value')
# plt.xlabel('Epoch number')
#
# plt.subplot(1, 2, 2)
# plt.plot(accuracies)
# plt.title('Prediction accuracy (%)')
# plt.xlabel('Epoch number')
#
#
# plt.tight_layout()
#
# output = hLN_model(X_tot[:, start:start + n_timepoints])
#
# plt.figure(2)
# plt.plot(target.numpy(), label='Target signal')
# plt.plot(output.numpy(), label='Model after primary training')
# # plt.plot(first_output.numpy(), label='Model before training')
# plt.xlabel('Time (s)')
# # plt.ylabel('Membrane potential (arbitrary units)')
# plt.title('Membrane potential (in arbitrary \n units) over time')
# plt.legend()
# plt.show()

### TRAINING PROCEDURE FOR GENERIC HLN MODEL

# # 1. Generate inputs, then generate output signal using some hLN model
# n_timepoints = 100
# start = 24000
#
# X_tot = tf.convert_to_tensor(np.load('real_inputs.npy'), dtype=tf.float32)  # real inputs made earlier
# X_e = X_tot[:629]
# X_i = X_tot[629:]
#
# target = np.load('target.npy')[start:start + n_timepoints]
# target = tf.convert_to_tensor(target, dtype=tf.float32)
# print(target)
#
# # 2. Initialise a linear, single subunit hLN model, and optimise to fit data
#
#
# # initialise model
# hLN_lin = hLN_Model(Jc=Jc_sing, Wci=Wci_sing, Wce=Wce_sing, sig_on=tf.constant([False]))
#
# # train model - should be close to a linear regression problem
#
# # first randomise parameters to be different from those which created the target
# hLN_lin.randomise_parameters()
#
# # then define optimizer - simple gradient descent
# #tensor here for initially slow learning, followed by fast
# learning_rate_lin = 0.01
# optimizer_lin = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate_lin)
#
# epochs = range(10)
# loss_values = []
# accuracies = []
# for epoch in tqdm(epochs):
#     if epoch < 10:
#         learning_rate_lin = 0.01
#     elif 10 <= epoch < 100:
#         learning_rate_lin = 0.6
#
#     # logdir = 'log'
#     # writer = tf.summary.create_file_writer(logdir)
#     # tf.summary.trace_on(graph=True, profiler=True)
#     loss_value, grads = grad(model=hLN_lin, inputs=X_tot[:, start:start + n_timepoints], targets=target)
#     # with writer.as_default():
#     #     tf.summary.trace_export(name="test", step=0, profiler_outdir=logdir)
#     accuracy = 100 * (1 - (loss_value/np.var(target)))
#     loss_values.append(loss_value.numpy())
#     accuracies.append(max(accuracy, 0))
#     optimizer_lin = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate_lin)
#     optimizer_lin.apply_gradients(zip(grads, hLN_lin.trainable_params))
#
#
# plt.figure(1)
#
#
# plt.subplot(1, 2, 1)
# plt.plot(loss_values)
# plt.title('Loss value')
# plt.xlabel('Epoch number')
#
# plt.subplot(1, 2, 2)
# plt.plot(accuracies)
# plt.title('Prediction accuracy (%)')
# plt.xlabel('Epoch number')
#
# plt.title('Loss and accuracy during linear training')
# plt.tight_layout()
#
# output = hLN_lin(X_tot[:, start:start + n_timepoints])
#
# plt.figure(2)
# plt.plot(target.numpy(), label='Target signal')
# plt.plot(output.numpy(), label='Linear model after training')
# # plt.plot(first_output.numpy(), label='Model before training')
# plt.xlabel('Time (s)')
# # plt.ylabel('Membrane potential (arbitrary units)')
# plt.title('Membrane potential (in arbitrary \n units) over time')
# plt.legend()
# plt.show()
#
# # 3. Introduce non-linearity by adjusting parameters to approximate linear integration, and optimise further. Accuracy
# #  of linear model should be a lower bound on non-linear model.
#
# # initialise nonlinear model, with sig_on = true
# hLN_nonlin = hLN_Model(Jc=Jc_sing, Wci=Wci_sing, Wce=Wce_sing, sig_on=tf.constant([True]))
#
# # set parameters to those of the linear model
# hLN_nonlin.params, hLN_nonlin.trainable_params = hLN_lin.params, hLN_lin.trainable_params
#
# # now adjust parameters so nonlinear approximates linear
# nSD = 100
# init_nonlin(X=X_tot[:, start:start + n_timepoints], model=hLN_nonlin, nSD=nSD)
#
# output_nonlin = hLN_nonlin(X_tot[:, start:start + n_timepoints])
#
# # plt.figure(3)
# # plt.plot(output.numpy(), label='Linear model after training')
# # plt.plot(output_nonlin.numpy(), label='Nonlinear model after initialisation')
# # plt.plot(output_nonlin.numpy() - output.numpy(), label='Difference')
# # plt.legend()
# # plt.title(f'Effect of adding non-linearity to single subunit model, with nSD={nSD}')
# # plt.show()
#
# # now train new nonlinear model
# learning_rate_nonlin = 0.01
# optimizer_nonlin = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate_nonlin)
#
# epochs = range(500)
# loss_values = []
# accuracies = []
# for epoch in tqdm(epochs):
#     # logdir = 'log'
#     # writer = tf.summary.create_file_writer(logdir)
#     # tf.summary.trace_on(graph=True, profiler=True)
#     loss_value, grads = grad(model=hLN_nonlin, inputs=X_tot[:, start:start + n_timepoints], targets=target)
#     # with writer.as_default():
#     #     tf.summary.trace_export(name="test", step=0, profiler_outdir=logdir)
#     accuracy = 100 * (1 - (loss_value/np.var(target)))
#     loss_values.append(loss_value.numpy())
#     accuracies.append(max(accuracy, 0))
#     optimizer_nonlin.apply_gradients(zip(grads, hLN_nonlin.trainable_params))
#
# output_nonlin2 = hLN_nonlin(X_tot[:, start:start + n_timepoints])
#
#
# plt.figure(4)
#
# plt.subplot(1, 2, 1)
# plt.plot(loss_values)
# plt.title('Loss value')
# plt.xlabel('Epoch number')
#
# plt.subplot(1, 2, 2)
# plt.plot(accuracies)
# plt.title('Prediction accuracy (%)')
# plt.xlabel('Epoch number')
#
# plt.title('Loss and accuracy during nonlinear training')
# plt.tight_layout()
#
# plt.figure(5)
# plt.plot(output_nonlin2.numpy(), label='Non-linear model after training')
# plt.plot(target.numpy(), label='Target signal')
# plt.legend()
# plt.title(f'Change in nonlinear output signal from training')
# plt.show()

# 4. Expand to more subunits...

# first we need to specify the new architecture - should involve adding subunits, all of which
# are initially linear

# Adding new linear subunits requires simply re assigning inputs to the new leaves

# Once new linear subunits added (model still functionally the same) add nonlinearities to all new subunits


