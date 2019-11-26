# a script to simulate a hierarchical linear non-linear model and its response
# first draft should be simplest skeleton of Ujfalussy full R script that can take inputs and produce and output

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy import sparse as sps
import seaborn as sns
from gen_inputs import *

sns.set()

# to do:
# define all parameters required by function - simplify as much as possible to begin with
# be careful with Python vs R indexing, could be a problem in subunit inputs CHECK THIS PROPERLY



# DEFINE:
# Jc, Jw, Tau matrix, Wc, gain matrix

# RANDOMISE:
# Ww (weights), X?
# could be helpful to define X to check if soma output is plausible.



# pars = [v0, Jc, Jw, Wce, Wwe, Th, Tau, dTau, delay]


# to define:
# subunit parameters:


Jc_sing = np.array([0])
Jc_five = np.array([0, 1, 1, 1, 1])


def sim_hLN(X, dt, Jc, Jw, Wce, Wci, Wwe, Wwi,  Tau_e, Tau_i, Th, alpha=True, double=False, v0=0, mult_inputs=False):
    """
    # function to simulate subthreshold response of a hGLM model
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

    N = X.shape[0] # number of input neurons
    Ne = len(Wce) # number of excitatory neurons
    Ni = len(Wci) # number of inhibitory neurons
    L = X.shape[1]  # number of timesteps

    M = len(Jc)  # number of subunits
    Delay = np.zeros([M, 1])  # default delay to 0 for all subunits


    gain = Jw

    # # first calculate baseline for all subunits, starting from the leaves - probably leave this out for first go
    # subunits_done=np.zeros([0,0])#indices of processed subunits will be stored here
    #
    # Jc_orig = Jc #make copy as Jc will be altered
    #
    # while len(subunits_done) < M: #repeat until all subunits processed
    #   # leaves are subunits which don't receive input from other subunits i.e. indices <= M which do not appear in Jc
    #   leaves = np.setdiff1d(np.arange(0, M, 1), Jc)
    #   remain = np.setdiff1d(np.arange(0, M, 1), subunits_done)#all unprocessed subunits
    #   #then find any subunits in both leaves and remain - if none then stop as no remaining leaves
    #   current_leaves = np.intersect1d(leaves, remain)
    #   if len(current_leaves)==0:
    #     # no further leaves found, end loop
    #     break
    #
    #
    #   #if any found then condition
    #   for leaf in current_leaves:
    #        #find children of leaf
    #        children =np.where(Jc==leaf)[0]
    #        if len(children) > 0:
    #            for child in children:
    #                # update baseline response - child is an integer representing the index of a subunit
    #                baseline[leaf] += Jw[child] * sigm(baseline[child], c=Th[child])
    #                Jc[leaf]=None #subunit already processed, does not give further input
    #                subunits_done = np.append(subunits_done, leaf)
    #
    # Jc = Jc_orig #reset Jc


    # second calculate synaptic input to each dendritic branch
    Y = np.zeros([M, L]) #this will be the matrix of inputs for each subunit at each given timestep

    for m in range(M):
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
        if len(Wce[m]) > 0:# if subunit has any excitatory neurons connected to it

            for synapse, neuron in list(enumerate(Wce[m])):

                if alpha:
                    # add convolved input to Y matrix if alpha set true
                    Y[m, :] += int_spikes(X=X, dt=dt, Wc=Wce[m][synapse], Ww=Wwe[m][synapse], Tau=Tau_e[m][synapse], delay=Delay[m])

                else:
                    # add simple binary input if not
                    Y[m, :] += Wwe[m][synapse] * X[neuron]

            #should be one weight linking each input neuron to each subunit
        #
        if len(Wci[m])>0:
            #if inhibitory input exists for subunit, calculate response
            for synapse, neuron in list(enumerate(Wci[m])):

                Y[m, :] += int_spikes(X=X, dt=dt, Wc=Wci[m][synapse], Ww=Wwi[m][synapse], Tau=Tau_i[m][synapse], delay=Delay[m])




    # then start from leaves and apply nonlinearities as well as inputs from the children

    R = np.zeros([M, L]) #a matrix with the activation of the subunits
    subunits_done=np.zeros([0, 0])#indices of processed subunits will be stored here
    Jc_orig = Jc.copy() #make a copy as Jc will be altered

    while len(subunits_done) < M: #repeat until all subunits processed
        # leaves are subunits which don't receive input from other subunits i.e. indices <= M which do not appear in Jc
        leaves = np.setdiff1d(np.arange(1, M+1, 1), Jc)
        remain = np.setdiff1d(np.arange(1, M+1, 1), subunits_done)  # all unprocessed subunits
        # then find any subunits in both leaves and remain - if none then stop as no remaining leaves
        current_leaves = np.intersect1d(leaves, remain)
        if len(current_leaves) == 0:
            # no further leaves found, end loop
            break

        # if any found then condition
        for leaf in current_leaves:
            # apply the sigmoidal nonlinearity to the dendritic inputs for every leaf
            # add option here for linear processing (if Th=none, just add Y row to R row)

            R[leaf-1, :] = sigm(Y[leaf-1, :], tau=Th[leaf-1]) #sigmoid threshold defined per subunit

            #add the input from the child to the parent
            # when we process parent its input Y will be dendritic input + input from its children
            parent = Jc[leaf-1]
            if parent > 0:
                Y[parent-1, :] += Jw[leaf-1] * R[leaf-1, :]

            Jc[leaf-1] = 0  # subunit already processed, does not give further input
            subunits_done = np.append(subunits_done, leaf)
            print(subunits_done)

    Jc = Jc_orig #return to original state

    # should now be able to calculate response
    v_soma = Jw[0] * R[0, :] + v0


    return v_soma


#initial parameters for:
L = 100 #number of timesteps
# N = 2 #number of input neurons
# X_det = binary_input(N, L, kind='delta', delay=0)
# X_rand = binary_input(N, L, kind='rand', delay=0) #random input
#
# t = np.linspace(0, 100, L)
#
# spikes = np.where(X_rand == 1, 1, np.nan)
# conv = convolve(s=X_rand[0], dt=1, tau=1, delay=0)
#
# plt.plot(t, spikes[0, :], 'o', label='spikes')
# plt.plot(t, conv.T)
# plt.show()




# # try inputting realistic input to model, plot response
# E_spikes, I_spikes = gen_realistic_inputs(Tmax=3000)
# X_e = spikes_to_input(E_spikes, Tmax=48001)
# X_i = spikes_to_input(I_spikes, Tmax=48001)
#
# # these parameters for 1 subunit
# Jc_sing = np.array([0])
# M = len(Jc_sing) #number of subunits
# Wce_sing = [np.arange(0, X_e.shape[0], 1)] #all input neurons connected to root subunit
# Wwe_sing = [np.ones(X_e.shape[0])] #weighting matrix - all neurons connected with weight 1 initially
# Tau_e = [np.full(X_e.shape[0], 1)] #all excitatory time constants 1
# Th = [[0]] #no offset in all sigmoids
#
#
# resp_sing = sim_hLN(X=X_e, dt=1, Jc=Jc_sing, Wce=Wce_sing, Wwe=Wwe_sing, Wci=[[0]], Wwi=[[0]], Tau_e=Tau_e, Tau_i=[], Th=Th, mult_inputs=True)
#
# plt.plot(resp_sing)
# # plt.plot(t, 0.5*spikes[0, :], 'bo', label='Neuron 1 spikes', color='green')
# # plt.plot(t, 0.5*spikes[1, :], 'bo', label='Neuron 2 spikes', color='magenta')
#
# plt.title("Single subunit response to two random inputs, 1 excitatory and 1 inhibitory")
# plt.xlabel("Time step")
# plt.ylabel("Root subunit response")
# plt.legend()
# plt.show()

# these parameters for 2 subunits
# Jc_double = np.array([0, 1])
# M = len(Jc_double) #number of subunits
# Wce_double = np.array([[1, 1], [0, 0]]) #both input neurons connected to both subunits
# Wwe_double = np.array([[1, -1], [1, -1]]) #weighting matrix - basically 1 excitatory and 1 inhibitory again
#
# resp_doub = sim_hLN(X=X_rand, dt=1, Jc=Jc_double, Wce=Wce_double, Wwe=Wwe_double, mult_inputs=True)
# spikes=np.where(X_rand==1, 1, np.nan)
#
# plt.plot(t, resp_doub)
# plt.plot(t, 0.5*spikes[0, :], 'bo', label='Neuron 1 spikes', color='green')
# plt.plot(t, 0.5*spikes[1, :], 'bo', label='Neuron 2 spikes', color='magenta')
# plt.title("Two subunit soma response to random inputs at end subunit, 1 excitatory and 1 inhibitory")
# plt.xlabel("Time step")
# plt.ylabel("Root subunit response")
# plt.legend()
# plt.show()












