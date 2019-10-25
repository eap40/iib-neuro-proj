# a script to simulate a hierarchical linear non-linear model and its response
# first draft should be simplest skeleton of Ujfalussy full R script that can take inputs and produce and output

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sps
import seaborn as sns
import tensorflow as tf

sns.set()

# to do:
# define all parameters required by function - simplify as much as possible to begin with
# be careful with Python vs R indexing, could be a problem in subunit inputs CHECK THIS PROPERLY
# get to work with single subunit, single input to begin with

# simplest case parameters:
# Jc = [0] i.e. single subunit, Jw should then be irrelevant (or 1 maybe?)
# 1 excitatory input neuron, 1 inhibitory - READ PAPER EQUATIONS FOR IDEAS ON WHAT EXACTLY TO CODE
# 10 time bins
# initially define global sigmoidal function with no subunit dependent parameters
# WE WANT TO GIVE MODEL SOME INPUT AND GET SOME OUTPUT - ONLY KEY RESULT FOR THIS WEEK!!!


# R matrix creation should be fine (non-linearity and adding of inputs from children)
# Y matrix creation - going from spikes trains to 'signals' will need a lot of work - maybe randomise input to begin with?
# for now should probably create Y matrix directly, then use to see if R process works
# one time point to start with

# DEFINE:
# Jc, Jw, Tau matrix, Wc, gain matrix

# RANDOMISE:
# Ww (weights), X?
# could be helpful to define X to check if soma output is plausible.



# pars = [v0, Jc, Jw, Wce, Wwe, Th, Tau, dTau, delay]

def binary_input(M, L, kind='rand'):
    """function to generate a binary input matrix of dimensions M x L, where M is the number of input neurons
    and L the number of time bins. Kind argument controls the type of input e.g. randomly spaced inputs"""
    if kind=='rand':
        # create sparse binary array with randomly distributed values
        sparse_array = sps.random(M, L, density=0.1, dtype='bool')
        # convert sparse array to numpy array for later use
        array_out = sps.csr_matrix.todense(sparse_array)

    return array_out





def sigm(x, tau=0):
    """sigmoid function for initial global non-linearity"""
    return 1/(1 + np.exp(-(x-tau)))


def sim_hLN(X, Jc, Wce, Wwe, Tau=None, v0=0):

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
    #
    # returns: either
    # - v_soma, if vv is not provided; or
    # - the error between vv and its predicion, + regularisation terms


    N = X.shape[0] #number of input neurons
    L = X.shape[1] #number of timesteps

    M = len(Jc) #number of subunits

    Jw = np.ones([M,1]) #coupling weights 1 for all branches intially
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
    Y = np.zeros([M,L]) #this will be the matrix of inputs for each subunit at each given timestep


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


        #calculate synaptic input to each dendritic branch
        # Wc is MxN matrix, indicating neurons connected to subunit (1 if connected, 0 if not connnected
        # Ww is MxN matrix of synaptic weights
        # just have excitory inputs to begin with
        if np.sum(Wce[m, :])>0: #mif subunit has any neurons connected to it
            Y[m,:] = np.matmul((Wce[m] * Wwe[m]), X) #i.e. at each timepoint, the dot product of weights vector with input vector
            #should be one weight linking each input neuron to each subunit
        #
        # if len(Wci[m])>0:
        #     #if inhibitory input exists for subunit, calculate response
        #     #see above  for Y update




    # then start from leaves and apply nonlinearities as well as inputs from the children

    R = np.zeros([M,L]) #a matrix with the activation of the subunits
    subunits_done=np.zeros([0,0])#indices of processed subunits will be stored here
    Jc_orig = Jc.copy() #make a copy as Jc will be altered

    while len(subunits_done) < M: #repeat until all subunits processed
        # leaves are subunits which don't receive input from other subunits i.e. indices <= M which do not appear in Jc
        leaves = np.setdiff1d(np.arange(1, M+1, 1), Jc)
        print(leaves)
        remain = np.setdiff1d(np.arange(1, M+1, 1), subunits_done)  # all unprocessed subunits
        # then find any subunits in both leaves and remain - if none then stop as no remaining leaves
        current_leaves = np.intersect1d(leaves, remain)
        if len(current_leaves) == 0:
            # no further leaves found, end loop
            break

        # if any found then condition
        for leaf in current_leaves:
            # apply the sigmoidal nonlinearity to the dendritic inputs for every leaf
            R[leaf-1, :] = sigm(Y[leaf-1, :], tau=0) #need to define sigmoid function, $tau=Tau[leaf] eventually

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
N = 2 #number of input neurons
X_const = np.ones([N, L]) #constant input
X_rand = np.random.rand(N, L) #random input

# these parameters for 1 subunit
# Jc_single = np.array([0])
# M = len(Jc_single) #number of subunits
# Wce_single = np.array([[1, 1]]) #both input neurons connected to root subunit
# Wwe_single = np.array([[1, -1]]) #weighting matrix - basically 1 excitatory and 1 inhibitory
#
#
# resp_sing = sim_hLN(X=X_rand, Jc=Jc_single, Wce=Wce_single, Wwe=Wwe_single)
#
# plt.plot(resp_sing)
# plt.title("Single subunit response to two random inputs")
# plt.xlabel("Time step")
# plt.ylabel("Root subunit response")
# plt.show()

# these parameters for 2 subunits
# Jc_double = np.array([0, 1])
# M = len(Jc_double) #number of subunits
# Wce_double = np.ones([2, 2]) #both input neurons connected to both subunits
# Wwe_double = np.array([[1, -1], [1, -1]]) #weighting matrix - basically 1 excitatory and 1 inhibitory again
#
# resp_doub = sim_hLN(X=X_rand, Jc=Jc_double, Wce=Wce_double, Wwe=Wwe_double)
#
#
# plt.plot(resp_doub)
# plt.title("Two subunit soma response to random inputs")
# plt.xlabel("Time step")
# plt.ylabel("Root subunit response")
# plt.show()



# print(np.matmul((Wce_test * Wwe_test), X_const).shape)


