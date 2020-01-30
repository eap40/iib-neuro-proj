#############################################
# A function to randomlly intialise the parameters Ww.e, Jw and/or Th to generate data or to initialise learning
# We initialise these parameters randomly, though we try to be close to linear in general
# The output is standardised - its mean is 0 and its variance is 1
# initialization for the parameters (including W.ahp) is formally correct, but this does not mean that it is biologically meaningful


import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow import keras
import urllib
from sim_hLN import *
from utils import *
from train_hLN import *
from tqdm import tqdm


def init_hLN(X, dt, Jc, Wce, Wci, params, nSD):
    """function to initialise sensible values for Ww, Jw and Th before beginning learning. Some randomness, but
    generally the sigmoid parameters are set such that the computation is approximately linear
    nSD: setting of standard deviation of input signals to sigmoid functions - equivalent to 1/alpha in calcs
    All other parameters: as in sim_hLN function"""

    # params should be parameters of a previously created hLN model
    v0, Jw, Wwe, Wwi, Tau_e, Tau_i, Th, Delay = params

    N = X.shape[0]
    L = X.shape[1]
    Tmax = L / dt
    M = len(Jc)

    # reset parameter values to sensible ones/ ones that produce ~linear behaviour
    Jw = tf.fill(Jw.shape, 4)  # to compensate for gain of sigmoid ~0.25
    Wwe = tf.fill(Wwe.shape, np.random.uniform(low=0.5, high=2))
    Wwi = tf.fill(Wwi.shape, np.random.uniform(low=0.1, high=1))
    Th = tf.fill(Th.shape, 1)
    Tau_e = tf.fill(Tau_e.shape, np.random.uniform(low=2, high=8))
    Tau_i = tf.fill(Tau_i.shape, np.random.uniform(low=2, high=30))
    Delay = tf.fill(Delay.shape, 0.1)

    # now calculate synaptic input to each dendritic branch
    Ym_list = []
    for m in range(M):

        # create empty vector for each subunit - stack all of these at end to get full Y matrix
        Y_m = tf.zeros(L, dtype=tf.float32)

        # Wc.e, Wc.i: a list of M components. The component m is a vector indicating the neurons connected to branch m
        # Ww.e, (Ww.e2), Ww.i: LISTS of M components indicating the synaptic weights associated to the neurons connected to branch m
        if len(Wce[m]) > 0:  # if subunit has any excitatory neurons connected to it
            # # need to swap out list(enumerate(Wce[m])) for something TF friendly
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

    # next, we start from the leaves and apply the nonlinearities as well as add the inputs from the children
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

        # if any found then apply sigmoidal non-linearity
        for leaf in current_leaves:
            if leaf != 1: # if not the soma
                # we want to rescale inputs to linear range first
                range_Y = tf.math.reduce_std(Y[leaf-1, :])
                rescale = 1 / (nSD * range_Y)
                Wwe = rescale * Wwe  # need to change this for single subunit case, but will have trouble assigning values in tensor
                Wwi = rescale * Wwi

                # rescale inputs from other subunits similarly
                children = np.argwhere(Jc_orig == leaf)
                Jw[children - 1] *= rescale
                Y[leaf-1, :] = rescale * Y[leaf-1, :]

                # compensate in output
                Jw[leaf - 1] /= rescale

                Th[leaf - 1] = np.mean(Y[leaf-1, :])

                # now apply sigmoid
                R = sigm(Y[leaf-1, :], tau=Th[leaf - 1])

            else:  # soma
                children = np.argwhere(Jc_orig == leaf)
                # for root subunit offset, we need the sum of all Y row means excluding the 0th row
                means = np.mean(Y, axis=1)
                means[0] = 0  # we don't want to include input to the root subunit in its offset
                Th[leaf-1] += (np.sum(Jw[children])/2 - np.sum(means))
                R = sigm(Y[leaf - 1, :], tau=Th[leaf - 1])



            # add the input from the child to the parent
            parent = Jc[leaf-1]
            if parent > 0: # not the soma
                Y[parent, :] += Jw[leaf-1] * R

            else: # the soma...
                v_soma = Jw[leaf-1] * R + v0


        # Jc[leaf - 1] = 0  # subunit already processed, does not give further input
        # set Jc[leaf-1] value to zero
        Jc -= Jc[leaf - 1] * tf.one_hot(indices=leaf - 1, depth=len(Jc), dtype=tf.int64)
        subunits_done = tf.concat([subunits_done[:leaf - 1], tf.reshape(leaf, (1, 1)), subunits_done[leaf:]],
                                  axis=0)
        subunits_done = tf.reshape(subunits_done, (len(Jc), 1))
        # tf.print(subunits_done)

    v = v_soma

    params = v0, Jw, Wwe, Wwi, Tau_e, Tau_i, Th, Delay

    return