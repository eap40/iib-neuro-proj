#############################################
# File containing functions for initialising hLN models. init_nonlin initialises nonlinearities for architectures
# previously containing linear subunits. <update_architecture> will add new linear subunits to an existing
# architecture, which should essentially involve just redistributing inputs.

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sim_hLN import *
from utils import *
import copy

# @tf.function
def init_nonlin(X, model, lin_model, nSD, dt=1):
    """function to initialise the nonlinearities in subunits which were previously nonlinear. The parameters
     of the linear model should already have been optimised, and as such its accuracy should be a lower bound on the
    accuracy of the new non-linear model
    X: binary matrix of inputs
    model: nonlinear model to set parameters for
    lin_model: linear model with same architecture, parameters of which have been optimised
    """

    # first set parameters of new nonlinear model to those of optimised linear model
    model.logJw.assign(lin_model.logJw)
    model.Wwe.assign(lin_model.Wwe)
    model.Wwi.assign(lin_model.Wwi)
    model.logTaue.assign(lin_model.logTaue)
    model.logTaui.assign(lin_model.logTaui)
    model.Th.assign(lin_model.Th)
    model.logDelay.assign(lin_model.logDelay)
    model.v0.assign(lin_model.v0)
    # model.params = (model.v0, model.log_Jw, model.Wwe, model.Wwi, model.log_Tau_e, model.log_Tau_i,
    #                model.Th, model.log_Delay)
    # model.trainable_params = (model.v0, model.log_Jw, model.Wwe, model.Wwi, model.log_Tau_e, model.log_Tau_i,
    #                          model.Th, model.log_Delay)



    # to extend to more subunits, make everything numpy so we can assign it. Then convert back to tensors at the end
    Jc, Wce, Wci = model.Jc, model.Wce, model.Wci
    # params should be parameters of a previously created hLN model - convert to numpy so we can assign
    v0, log_Jw, Wwe, Wwi, log_Tau_e, log_Tau_i, Th, log_Delay = [param.numpy() for param in model.params]



    # these parameters defined by their logs to ensure positivity - convert here
    Jw, Tau_e, Tau_i, Delay = np.exp(log_Jw), np.exp(log_Tau_e), np.exp(log_Tau_i), np.exp(log_Delay)

    N = X.shape[0]
    L = X.shape[1]
    Tmax = L / dt
    M = len(Jc)
    if M == 1: #single subunit model - adjust parameters such that Jw is 1 initially
        Wwe *= Jw
        Wwi *= Jw
        Jw /= Jw

    # first find which subunits we want to initialise nonlinearities for - should just be the leaves:
    leaves = np.setdiff1d(np.arange(1, M + 1, 1), Jc)

    # now calculate synaptic input to each dendritic branch - will use this to scale
    # need to assign rescaled parameters first
    model.Wwe.assign(Wwe)
    model.Wwi.assign(Wwi)
    model.logJw.assign(np.log(Jw))
    Y = sim_inputs(X=X, dt=1, Jc=model.Jc, Wce=model.Wce, Wci=model.Wci, params=model.params, sig_on=model.sig_on)
    Y = Y.numpy()

    #  we only need to initialise nonlinearities in the leaves:
    for leaf in leaves:

        range_Y = np.std(Y[leaf - 1, :])
        alpha = (nSD * range_Y)
        print(alpha)
        if len(Wce[leaf-1] > 0):  # if leaf has any e neurons connected to it
            Wwe[(Wce[leaf-1])] /= alpha
        if len(Wci[leaf-1] > 0):  # if leaf has any i neurons connected to it
            Wwi[(Wci[leaf-1] - model.n_e)] /= alpha

        Th[leaf-1] = np.mean(Y[leaf - 1, :]) / alpha

        Jw[leaf-1] *= (4 * alpha)

        if leaf != 1:  #if not the soma
            parent = Jc[leaf-1]
            Th[parent-1] -= (np.mean(Y[leaf - 1, :]) - 2 * alpha)

        else:  # soma: add offset into v0
            v0 += (np.mean(Y[leaf - 1, :]) - 2 * alpha)


    # while tf.math.count_nonzero(subunits_done) < M:  # repeat until all subunits processed
    #     # leaves are subunits which don't receive input from other subunits i.e. indices <= M which do not appear in Jc
    #     leaves = tf.sets.difference(tf.range(1, M + 1, 1, dtype=tf.int64)[None, :], Jc[None, :])
    #     leaves = tf.sparse.to_dense(leaves)
    #     remain = tf.sets.difference(tf.range(1, M + 1, 1, dtype=tf.int64)[None, :],
    #                                 tf.reshape(subunits_done, [-1])[None, :])
    #     remain = tf.sparse.to_dense(remain)  # all unprocessed subunits
    #     # then find any subunits in both leaves and remain - if none then stop as no remaining leaves
    #     current_leaves = tf.sets.intersection(leaves, remain)
    #     # convert from sparse to dense tensor
    #     current_leaves = tf.sparse.to_dense(current_leaves)
    #     # reshape to 1D tensor so we can iterate over it, and use elements as indices
    #     current_leaves = tf.reshape(current_leaves, [-1])
    #     if len(current_leaves) == 0:
    #         # no further leaves found, end loop
    #         break
    #
    #     # if any found then apply sigmoidal non-linearity
    #     for leaf in current_leaves:
    #         if leaf != 1: # if not the soma
    #             # we want to rescale inputs to linear range first
    #             range_Y = tf.math.reduce_std(Y[leaf-1, :])
    #             alpha = nSD * range_Y
    #
    #             if len(Wce[m] > 0):  # if leaf has any e neurons connected to it
    #                 Wwe[[Wce[m]]] /= alpha
    #             if len(Wci[m] > 0):  # if leaf has any i neurons connected to it
    #                 Wwi[[Wci[m]]] /= alpha
    #             #
    #             # # rescale inputs from other subunits similarly
    #             # children = np.argwhere(Jc_orig == leaf)
    #             # Jw[children - 1] *= rescale
    #             # Y[leaf-1, :] = rescale * Y[leaf-1, :]
    #             #
    #             # # compensate in output
    #             # Jw[leaf - 1] /= rescale
    #             #
    #             # Th[leaf - 1] = np.mean(Y[leaf-1, :])
    #             #
    #             # # now apply sigmoid
    #             # R = sigm(Y[leaf-1, :], tau=Th[leaf - 1])
    #
    #         else:  # soma
    #             # children = np.argwhere(Jc_orig == leaf)
    #             range_Y = tf.math.reduce_std(Y[leaf - 1, :])
    #             alpha = (nSD * range_Y)
    #             if len(Wce[m] > 0):  # if leaf has any e neurons connected to it
    #                 Wwe[[Wce[m]]] /= alpha
    #             if len(Wci[m] > 0):  # if leaf has any i neurons connected to it
    #                 Wwi[[Wci[m]]] /= alpha
    #             # for root subunit offset, we need the sum of all Y row means excluding the 0th row
    #             # means = tf.math.reduce_mean(Y, axis=1)
    #             # means[0] = 0  # we don't want to include input to the root subunit in its offset
    #             Th[leaf-1] = (tf.reduce_mean(Y[leaf - 1, :]) / alpha)
    #             v0.assign_add(tf.reduce_mean(Y[leaf - 1, :]) - 2 * alpha)
    #             Jw = Jw * 4.0 * alpha
    #
    #
    #         # add the input from the child to the parent
    #         # when we process parent its input Y will be dendritic input + input from its children
    #         # can't assign values in a tensor so we add something of same shape as y - use paddings
    #         parent = Jc[leaf - 1]
    #         if parent > 0:
    #             increment = tf.reshape(Jw[leaf - 1] * R[leaf - 1, :], (1, L))
    #             paddings = ([[parent - 1, M - parent], [0, 0]])
    #             Y += tf.pad(increment, paddings, "CONSTANT")
    #             Y = tf.reshape(Y, (M, L))
    #
    #         else: # the soma...
    #             v_soma = Jw[leaf-1] * R + v0
    #
    #
    #         # Jc[leaf - 1] = 0  # subunit already processed, does not give further input
    #         # set Jc[leaf-1] value to zero
    #         Jc -= Jc[leaf - 1] * tf.one_hot(indices=leaf - 1, depth=len(Jc), dtype=tf.int64)
    #         subunits_done = tf.concat([subunits_done[:leaf - 1], tf.reshape(leaf, (1, 1)), subunits_done[leaf:]],
    #                                   axis=0)
    #         subunits_done = tf.reshape(subunits_done, (len(Jc), 1))
    #         # tf.print(subunits_done)

    # v = v_soma
    # #
    logJw, logTaue, logTaui, logDelay = np.log(Jw), np.log(Tau_e), np.log(Tau_i), np.log(Delay)

    model.logJw.assign(logJw)
    model.Wwe.assign(Wwe)
    model.Wwi.assign(Wwi)
    model.logTaue.assign(logTaue)
    model.logTaui.assign(logTaui)
    model.Th.assign(Th)
    model.logDelay.assign(logDelay)
    model.v0.assign(v0)

    return


def update_arch(prev_model, next_model):
    """Function to assign the correct parameters to a new architecture which has added new linear leaf subunits
    from the previous architecture. The Jc, Wce and Wci of the new architecture are known, so the synaptic
    parameters just need to be redistributed accordingly"""

    # first change hLN attributes into numpy - allows assignment and should be easier to manipulate
    logJw = next_model.logJw.numpy()
    logDelay = next_model.logDelay.numpy()


    # work out which subunits we have just added - for these cases should just be the leaves
    M = len(next_model.Jc)
    leaves = np.setdiff1d(np.arange(1, M+1, 1), next_model.Jc)
    for leaf in leaves:
        logJw[leaf-1] = 0  # set subunit gain to 1 for all new leaves
        # then set delay to the delay of the subunit parent from the previous model
        parent = next_model.Jc[leaf-1]
        logDelay[leaf-1] = prev_model.logDelay.numpy()[parent-1]



    # assign the newly calculated parameters to the new model
    next_model.Wwe.assign(prev_model.Wwe)
    next_model.Wwi.assign(prev_model.Wwi)
    next_model.logTaue.assign(prev_model.logTaue)
    next_model.logTaui.assign(prev_model.logTaui)
    next_model.logJw.assign(logJw)
    next_model.logDelay.assign(logDelay)

    return
