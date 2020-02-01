import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
import urllib
from sim_hLN import *
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
    def __init__(self, Jc, Wci, Wce, sig_on):
        # Initialize the parameters in some way
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        M = len(Jc)
        self.Jc, self.Wci, self.Wce, self.sig_on = Jc, Wci, Wce, sig_on
        self.Jw = tf.Variable(np.full(M, 1.0), dtype=tf.float32) #coupling weights 1 for all branches intially
        self.Wwe = tf.Variable([np.full(X_e.shape[0], 1.0)], dtype=tf.float32)
        self.Wwi = tf.Variable([np.full(X_i.shape[0], -1.0)], dtype=tf.float32)
        self.Tau_e = tf.Variable([np.full(X_e.shape[0], 1.0)], dtype=tf.float32)
        self.Tau_i = tf.Variable([np.full(X_i.shape[0], 1.0)], dtype=tf.float32)
        self.Th = tf.Variable([1.0], dtype=tf.float32)
        self.Delay = tf.Variable(np.zeros([M, 1]), dtype=tf.float32)
        self.v0 = tf.Variable(0, dtype=tf.float32)
        self.params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th, self.Delay)
        self.trainable_params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th)


    def __call__(self, x):
        return sim_hLN_tf(X=x, dt=1, Jc=self.Jc, Wce=self.Wce, Wci=self.Wci, params=self.params, sig_on=self.sig_on)

    def randomise_parameters(self):
        # self.Wwe = tf.Variable([np.random.uniform(0, 2, X_e.shape[0])], dtype=tf.float32)
        # self.Wwi = tf.Variable([np.random.uniform(-2, 0, X_i.shape[0])], dtype=tf.float32)
        # self.Tau_e = tf.Variable([np.random.uniform(0, 2, X_e.shape[0])], dtype=tf.float32)
        # self.Tau_i = tf.Variable([np.random.uniform(0, 2, X_i.shape[0])], dtype=tf.float32)
        # self.v0 = tf.Variable(np.mean(target), dtype=tf.float32)
        self.Jw.assign(np.random.uniform(0, 2, M))
        self.Wwe.assign([np.random.uniform(0, 2, X_e.shape[0])])
        self.Wwi.assign([np.random.uniform(-2, 0, X_i.shape[0])])
        self.Tau_e.assign([np.random.uniform(0, 2, X_e.shape[0])])
        self.Tau_i.assign([np.random.uniform(0, 2, X_i.shape[0])])
        self.Th.assign([np.random.uniform(-0.5, 0.5)])
        self.v0.assign(np.random.uniform(0, 0.2))
        self.params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th, self.Delay)
        self.trainable_params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th)
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
Wci_sing = [np.arange(N_soma, N_soma + X_i.shape[0] - 1, 1)] #all input inhibitory neurons connected to root subunit
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

# 1. Generate inputs, then generate output signal using some hLN model
n_timepoints = 100
start = 24000

X_tot = tf.convert_to_tensor(np.load('real_inputs.npy'), dtype=tf.float32)  # real inputs made earlier
X_e = X_tot[:629]
X_i = X_tot[629:]

# target = np.load('target.npy')[start:start + n_timepoints]
# target = tf.convert_to_tensor(target, dtype=tf.float32)

# 2. Initialise a linear, single subunit hLN model, and optimise to fit data


# initialise model
hLN_model_lin = hLN_Model(Jc=Jc_sing, Wci=Wci_sing, Wce=Wce_sing, sig_on=tf.constant([False]))

hLN_model_nonlin = hLN_Model(Jc=Jc_sing, Wci=Wci_sing, Wce=Wce_sing, sig_on=tf.constant([True]))

target_lin = hLN_model_lin(X_tot[:, start:start + n_timepoints])
target_nonlin = hLN_model_nonlin(X_tot[:, start:start + n_timepoints])

plt.plot(target_lin.numpy(), label='Linear subunit')
plt.plot(target_nonlin.numpy(), label='Nonlinear subunit')
plt.plot(sigm(target_lin.numpy(), tau=1), label='Linear followed by sigmoid')
plt.plot(target_nonlin.numpy()-sigm(target_lin.numpy(), tau=1), label='Difference')
plt.legend()

plt.show()

# train model - should be close to a linear regression problem


# 3. Introduce non-linearity by adjusting parameters to approximate linear integration, and optimise further. Accuracy
#  of linear model should be a lower bound on non-linear model.

# 4. Expand to more subunits...


