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



@tf.function
def sim_hLN_tf(X, dt, Jc, Wce, Wci, params, alpha=True, double=False, mult_inputs=False):
    """
    # function to simulate subthreshold response of a hGLM model, with TensorFlow friendly functions
    #
    # args:
    # X: NxT binary input matrix of presynaptic spikes; N: # of input neurons; T: # of timesteps
    # dt: the time resolution of X in miliseconds
    # pars: list with the following elements:
        # v0: baseline of somatic voltage
        # Jc: M length connectivity vector for dendritic branches; M: # of dendritic branches
            # for each branch its parent is given. 1 is the root which has a value 0.
            # e.g., a 3 layer deep binary tree has 7 subunits is given by: [0, 1, 1, 2, 2, 3, 3]
        # Jw: M length vector for the coupling weights associated with the dendritic branches
        # Wc.e, Wc.i: a list of M components. The component m is a vector indicating the neurons connected to branch m
        # Ww.e, (Ww.e2), Ww.i: LISTS of M components indicating the synaptic weights associated to the neurons connected to branch m
        # Th: M length vector of the thresholds for the sigmoidal nonlinearities in the branches
        # Tau.e, Tau.i: LISTS of M components of the (log) synaptic time constants in miliseconds
        # dTau.e: optional LISTS of M components of the (log) synaptic time constants for the dExp synapses in miliseconds
        # delay.t: optional vector of the synaptic delays in ms (default: 1/10 ms)
    # vv: response, if provided only the error is calculated
    # logpars: parameters Tau, Jw and Ww are strictly positive, so they can be defined by their log
    # regpars: NULL (no regularisation) or list with the prior mean and inverse variance for the log of Ww and Tau
    #     Ww.e, (Ww.e2), Ww.i, Tau.e, Tau.i, alpha.Ww, alpha.Tau
    # double: logical; simple or double alpha kernels
    # scale.tau2: if double kernels are used, the scaling factor between their time constants
    # verbose: regularisation details are provided
    # calc.minis: also returns the calculated value of the individual synaptic amplitudes
    # X.ahp: output spike train convolved with the basis functions for simulating the after-spike currents
    # mult_inputs: if true, allow an input neuron to be connected to multiple subunits
    #
    # returns: either
    # - v_soma, if vv is not provided; or
    # - the error between vv and its predicion, + regularisation terms
    """

    # if multiple inputs are not allowed, then check Wce argument for any multiple input cases. If some are found, print
    # error message and exit function
    # if mult_inputs == False:
    #     neuron_cons = np.sum(Wce, 1)
    #     assert np.max(neuron_cons) <= 1, 'One or more neurons are connected to multiple subunits. Please revise your Wce matrix.'

    # WILL NEED MORE PARAMETER CHECKS, ERROR MESSAGES ETC HERE BEFORE FUNCTION STARTS PROPER
    v0, Jw, Wwe, Wwi, Tau_e, Tau_i, Th, Delay = params

    N = X.shape[0]  # number of input neurons
    Ne = len(Wce)  # number of excitatory neurons
    Ni = len(Wci)  # number of inhibitory neurons
    L = X.shape[1]  # number of timesteps

    M = len(Jc)  # number of subunits
    # delay = np.zeros([M, 1])  # default delay to 0 for all subunits

    # gain = Jw

    # second calculate synaptic input to each dendritic branch
    # Y = tf.zeros([M, L], dtype=tf.float32)  # this will be the matrix of inputs for each subunit at each given timestep
    Ym_list = []
    for m in range(M):

        # create empty vector for each subunit - stack all of these at end to get full Y matrix
        Y_m = tf.zeros(L, dtype=tf.float32)
        # # subunit gain = Jw[m] * f'(0) - evaluated in the absence of inputs
        # gain[m] = Jw[m]
        #
        # # Connectivity is defined from the root - soma - towards the leaves.
        # # So the parent has already been processed when processing the leaves!
        # total_gain[m] = gain[m]
        # parent = Jc[m-1]
        # if parent !=0:
        #     total_gain[m] *= total_gain[parent]

        # calculate synaptic input to each dendritic branch

        # Wc.e, Wc.i: a list of M components. The component m is a vector indicating the neurons connected to branch m
        # Ww.e, (Ww.e2), Ww.i: LISTS of M components indicating the synaptic weights associated to the neurons connected to branch m
        if len(Wce[m]) > 0:  # if subunit has any excitatory neurons connected to it
            # # need to swap out list(enumerate(Wce[m])) for something TF friendly
            # synapses = tf.scan(lambda a, x: a + 1, elems=Wce[m], initializer=-1)
            # synapses = tf.dtypes.cast(synapses, tf.int64)
            # list_enum = tf.stack([synapses, Wce[m]], axis=1)
            # print(list_enum)
            synapse = 0
            for neuron in Wce[m]:
                if alpha:
                    # add convolved input to Y matrix if alpha set true
                    increment = int_spikes(X=X, dt=dt, Wc=Wce[m][synapse], Ww=Wwe[m][synapse], Tau=Tau_e[m][synapse],
                                           delay=Delay[m])
                    Y_m += increment

                else:
                    # add simple binary input if not
                    Y_m += Wwe[m][synapse] * X[neuron]

                synapse += 1  # changed to this format as tf didn't like list(enumerate) or aliases

            # should be one weight linking each input neuron to each subunit
        #
        if len(Wci[m]) > 0:
            # if inhibitory input exists for subunit, calculate response
            synapse = 0
            for neuron in Wci[m]:
                Y_m += int_spikes(X=X, dt=dt, Wc=Wci[m][synapse], Ww=Wwi[m][synapse], Tau=Tau_i[m][synapse],
                                  delay=Delay[m])
                synapse += 1

        # append Y_m to list of Y_ms, then stack them all at the end of the for loop in ms
        Ym_list.append(Y_m)

    # now stack all the Y_ms we stored during the loop
    Y = tf.stack(Ym_list)
    # then start from leaves and apply nonlinearities as well as inputs from the children


    R = tf.zeros([M, L], dtype=tf.float32)  # a matrix with the activation of the subunits
    # 0 vector subunits_done - when subunit processed, set one zero value to the subunit number
    # i.e. subunit 3 processed in 3 unit structure: subunits done -> [0, 0, 3]
    subunits_done = tf.zeros((len(Jc), 1), dtype=tf.int64)
    Jc_orig = tf.identity(Jc)  # make a copy as Jc will be altered

    while tf.math.count_nonzero(subunits_done) < M:  # repeat until all subunits processed
        # leaves are subunits which don't receive input from other subunits i.e. indices <= M which do not appear in Jc
        leaves = tf.sets.difference(tf.range(1, M + 1, 1, dtype=tf.int64)[None, :], Jc[None, :])
        leaves = tf.sparse.to_dense(leaves)
        remain = tf.sets.difference(tf.range(1, M + 1, 1, dtype=tf.int64)[None, :],
                                    tf.reshape(subunits_done, [-1])[None, :])
        remain = tf.sparse.to_dense(remain)  # all unprocessed subunits
        # then find any subunits in both leaves and remain - if none then stop as no remaining leaves
        current_leaves = tf.sets.intersection(leaves, remain)
        # convert from sparse to dense tensor
        current_leaves = tf.sparse.to_dense(current_leaves)
        # reshape to 1D tensor so we can iterate over it, and use elements as indices
        current_leaves = tf.reshape(current_leaves, [-1])
        if len(current_leaves) == 0:
            # no further leaves found, end loop
            break

        # if any found then condition
        for leaf in current_leaves:
            # apply the sigmoidal nonlinearity to the dendritic inputs for every leaf
            # add option here for linear processing (if Th=none, just add Y row to R row)
            # define paddings - this way we can add non linearity output to whole R Tensor instead of assigning 1 row
            paddings = ([[leaf - 1, M - leaf], [0, 0]])
            # sigmoid threshold defined per subunit
            increment = tf.pad(tf.reshape(sigm(Y[leaf - 1, :], tau=Th[leaf - 1]), (1, L)), paddings, "CONSTANT")
            R += increment
            R = tf.reshape(R, (M, L))

            # add the input from the child to the parent
            # when we process parent its input Y will be dendritic input + input from its children
            # can't assign values in a tensor so we add something of same shape as y - use paddings
            parent = Jc[leaf - 1]
            if parent > 0:
                increment = tf.reshape(Jw[leaf - 1] * R[leaf - 1, :], (1, L))
                paddings = ([[parent - 1, M - parent], [0, 0]])
                Y += tf.pad(increment, paddings, "CONSTANT")
                Y = tf.reshape(Y, (M, L))

            # Jc[leaf - 1] = 0  # subunit already processed, does not give further input
            # set Jc[leaf-1] value to zero
            Jc -= Jc[leaf - 1] * tf.one_hot(indices=leaf - 1, depth=len(Jc), dtype=tf.int64)
            subunits_done = tf.concat([subunits_done[:leaf - 1], tf.reshape(leaf, (1, 1)), subunits_done[leaf:]],
                                      axis=0)
            subunits_done = tf.reshape(subunits_done, (len(Jc), 1))
            # tf.print(subunits_done)

    Jc = Jc_orig  # return to original state

    # should now be able to calculate response
    v_soma = Jw[0] * R[0, :] + v0

    return v_soma


###hLN model training, following same structure as simple regression model in tensorflow examples###

class hLN_Model(object):
    #   will need fleshing out/editing according to form of sim_hLN/general Python class for hLN model
    # to define hLN model we just need its structure (Jc) and how the input neurons connect to its subunits
    def __init__(self, Jc, Wci, Wce):
        # Initialize the parameters in some way
        # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
        M = len(Jc)
        self.Jc, self.Wci, self.Wce = Jc, Wci, Wce
        self.Jw = tf.Variable(np.full(M, 1.0), dtype=tf.float32) #coupling weights 1 for all branches intially
        self.Wwe = tf.Variable([np.full(X_e.shape[0], 1.0)], dtype=tf.float32)
        self.Wwi = tf.Variable([np.full(X_i.shape[0], -1.0)], dtype=tf.float32)
        self.Tau_e = tf.Variable([np.full(X_e.shape[0], 1.0)])
        self.Tau_i = tf.Variable([np.full(X_i.shape[0], 1.0)])
        self.Th = tf.Variable([1.0], dtype=tf.float32)
        self.Delay = tf.Variable(np.zeros([M, 1]), dtype=tf.float32)
        self.v0 = tf.Variable(0, dtype=tf.float32)
        self.params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th, self.Delay)
        self.trainable_params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Delay)

    def __call__(self, x):
        return sim_hLN_tf(X=x, dt=1, Jc=self.Jc, Wce=self.Wce, Wci=self.Wci, params=self.params)

    def randomise_parameters(self):
        # self.Wwe = tf.Variable([np.full(X_e.shape[0], np.random.uniform(0, 2))], dtype=tf.float32)
        self.Wwe = tf.Variable([np.random.uniform(0, 2, X_e.shape[0])], dtype=tf.float32)
        # self.Wwi = tf.Variable([np.full(X_i.shape[0], -np.random.uniform(0, 2))], dtype=tf.float32)
        self.Wwi = tf.Variable([np.random.uniform(-2, 0, X_i.shape[0])], dtype=tf.float32)
        # self.Tau_e = tf.Variable([np.full(X_e.shape[0], np.random.uniform(0, 2))])
        self.Tau_e = tf.Variable([np.random.uniform(0, 2, X_e.shape[0])], dtype=tf.float32)
        # self.Tau_i = tf.Variable([np.full(X_i.shape[0], np.random.uniform(0, 2))])
        self.Tau_i = tf.Variable([np.random.uniform(0, 2, X_i.shape[0])], dtype=tf.float32)
        self.v0 = tf.Variable(np.random.uniform(-1, 1), dtype=tf.float32)
        self.params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Th, self.Delay)
        self.trainable_params = (self.v0, self.Jw, self.Wwe, self.Wwi, self.Tau_e, self.Tau_i, self.Delay)
        return



# define loss and gradient function

def loss(predicted_v, target_v):
    """loss function is mean squared error between hLN model output and target membrane potential"""
    return tf.reduce_mean(tf.square(predicted_v - target_v))

# use gradient tape to calculate gradients used to optimise model
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

# initialise model
hLN_model = hLN_Model(Jc=Jc_sing, Wci=Wci_sing, Wce=Wce_sing)

# target = sim_hLN_tf(X=X_tot, dt=1, Jc=Jc_sing, Wce=Wce_sing, Wci=Wci_sing, params=hLN_model.params)
n_timepoints = 500
start = 24000

# target = hLN_model(X_tot[:n_timepoints])
# print(Y, R, target)
# Y = Y.numpy()
# np.save('target.npy', target.numpy())
target = np.load('target.npy')[start:start + n_timepoints]
target = tf.convert_to_tensor(target, dtype=tf.float32)
# print(Y.shape)
# # plt.plot(target, label='target')
# plt.plot(Y.T)
# plt.show()

# print(target)

# randomise parameters of model before starting training
hLN_model.randomise_parameters()

# first_output = hLN_model(X_tot[:, start:start + n_timepoints])

# define optimizer
optimizer_slow = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.3)
optimizer_fast = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=3)


epochs = range(5)
loss_values = []
accuracies = []
for epoch in tqdm(epochs):
    loss_value, grads = grad(model=hLN_model, inputs=X_tot[:, start:start + n_timepoints], targets=target)
    accuracy = 100 * (1 - (loss_value/np.var(target)))
    loss_values.append(loss_value.numpy())
    accuracies.append(accuracy)
    optimizer_slow.apply_gradients(zip(grads, hLN_model.trainable_params))

accuracies = np.clip(accuracies, a_min=0, a_max=None)

plt.figure(1)

plt.subplot(1, 2, 1)
plt.plot(loss_values)
plt.title('Loss value')
plt.xlabel('Epoch number')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title('Prediction accuracy (%)')
plt.xlabel('Epoch number')


plt.tight_layout()

output = hLN_model(X_tot[:, start:start + n_timepoints])

plt.figure(2)
plt.plot(target.numpy(), label='Target signal')
plt.plot(output.numpy(), label='Model after primary training')
# plt.plot(first_output.numpy(), label='Model before training')
plt.xlabel('Time (s)')
# plt.ylabel('Membrane potential (arbitrary units)')
plt.title('Membrane potential (in arbitrary \n units) over time')
plt.legend()
plt.show()


# epochs = range(5)
# loss_values = []
# accuracies = []
# for epoch in tqdm(epochs):
#     loss_value, grads = grad(model=hLN_model, inputs=X_tot[:, start:start + n_timepoints], targets=target)
#     accuracy = 100 * (1 - (loss_value/np.var(target)))
#     loss_values.append(loss_value.numpy())
#     accuracies.append(accuracy)
#     optimizer_fast.apply_gradients(zip(grads, hLN_model.trainable_params))
#
# accuracies = np.clip(accuracies, a_min=0, a_max=None)
#
# plt.figure(3)
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
# plt.figure(4)
# plt.plot(target.numpy(), label='Target signal')
# plt.plot(output.numpy(), label='Model after secondary training')
# # plt.plot(first_output.numpy(), label='Model before training')
# plt.xlabel('Time (s)')
# # plt.ylabel('Membrane potential (arbitrary units)')
# plt.title('Membrane potential (in arbitrary \n units) over time')
# plt.legend()
# plt.show()
