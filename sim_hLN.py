# a script to simulate a hierarchical linear non-linear model and its response
from utils import *


@tf.function
def sim_inputs(X, dt, Jc, Wce, Wci, params, sig_on, alpha=True, double=False, mult_inputs=False):
    """Function to simulate synaptic input to all subunits. Essentially a cut off version of sim_hLN_tf,
    that returns Y matrix before inputs from children are added and nonlinearities applied"""

    v0, logJw, Wwe, Wwi, logTaue, logTaui, Th, logDelay = params

    # some parameters defined by their logs to ensure positivity - convert here
    Jw, Taue, Taui, Delay = tf.exp(logJw), tf.exp(logTaue), tf.exp(logTaui), tf.exp(logDelay)

    # get number of subunits, synapses, timesteps etc.
    M = len(Jc)
    N = X.shape[0]
    L = X.shape[1]

    # first we calculate the synaptic input to each subunit: will need to loop over subunits, but don't loop over
    # synapses within each subunit

    Ym_list = []  # empty list to store synaptic input vector for each subunit
    # then concatenate excitatory and inhibitory parameters for easier convolution
    Taus = tf.concat((Taue, Taui), axis=0)
    Wws = tf.concat((Wwe, Wwi), axis=0)

    for m in range(M):

        # create empty vector for each subunit - stack all of these at end to get full Y matrix
        Ym = tf.zeros(shape=L, dtype=tf.float32)

        # concatenate Wce[m] and Wci[m] and then check if full: then we can keep the existing Wce syntax to work
        # for both inhibitory and excitatory synapses in 1 convolution

        Wc = tf.concat((Wce[m], Wci[m]), axis=0)

        if len(Wc) > 0:  # if subunit has any neurons connected to it: multiple conditions
            # so we only have to do one convolution
            n_syn = len(Wc)

            # create alpha kernel matrix, one row for each excitatory synapse. Then convolve kernel matrix with
            # the input rows corresponding to the subunit excitatory input neurons

            # find Taus for all subunit synapses, then find maximum of these
            Taus_m = tf.gather(Taus, Wc)
            Wws_m = tf.gather(Wws, Wc)
            Tau_max = tf.math.reduce_max(Taus_m)
            # kernel decays quickly, so only consider times up to 10 * max(tau) after spikes - this will be the
            # length of the filter
            filt_length = tf.cast(10 * Tau_max, tf.int32)
            # shape of filter matrix
            shape = [n_syn, 1]
            # create filter matrix
            filt_times = tf.tile(tf.reshape(tf.range(0, filt_length, dt), (1, filt_length)), (n_syn, 1))
            filt_times = tf.cast(filt_times, tf.float32)
            filt = ((filt_times - Delay[m]) / tf.reshape(Taus_m, (n_syn, 1))) * tf.math.exp(
                -(filt_times - Delay[m]) / tf.reshape(Taus_m, (n_syn, 1)))
            filt = tf.clip_by_value(filt, clip_value_min=0, clip_value_max=10)
            filt = filt * tf.reshape(Wws_m, (n_syn, 1))

            # now select input rows we will convolve with, according to values in Wce[m]
            X_m = tf.gather_nd(X, tf.reshape(Wc, (n_syn, 1)))

            # we have the filter and the input, now extend the inputs and convolve
            X_extend = tf.concat((tf.zeros((n_syn, int(filt_length / 2)), dtype=tf.float32), X_m), axis=1)

            # reshape and convolve
            filt = tf.reshape(tf.transpose(filt), shape=[1, filt_length, n_syn, 1])
            X_extend = tf.reshape(tf.transpose(X_extend), shape=[1, 1, L + int(filt_length / 2), n_syn])

            strides = [1, 1, 1, 1]
            filtered_m = tf.nn.depthwise_conv2d(X_extend, filt[:, ::-1], strides, padding='SAME', data_format="NHWC")
            #             filtered_m = tf.reshape(filtered_m, (L+int(filt_length/2), n_syn))

            Ym = tf.reduce_sum(filtered_m, axis=[0, 1, 3])

            # now discard extra points:
            Ym = Ym[:L]

        Ym_list.append(Ym)

    # now stack all the Yms we stored during the loop
    Y = tf.stack(Ym_list)

    return Y


@tf.function
def sim_hLN_tf2(X, dt, Jc, Wce, Wci, params, sig_on):
    """new function to simulate subthreshold response of an hLN model. Should be simpler than old version (i.e. less
    for loops) and hopefully help achieve 100% accuracy on training data. Note Jw, Tau and Delay are defined by their
    logs to ensure positivity.

    X: N x T binarry input matrix of presynaptic spikes; N: # of inputs; T: # of timesteps
    dt: time resolution of X in milliseconds, default 1
    params: list with following element describing hLN model
        v0: baseline somatic voltage
        Jc:M length connectivity vector for dendritic branches; M: # of dendritic branches
             for each branch its parent is given. 1 is the root which has a value 0.
             e.g., a 3 layer deep binary tree has 7 subunits is given by: [0, 1, 1, 2, 2, 3, 3]
        Jw: M length vector for coupling weights associated with dendritic branches
        Wce, Wci: list of M components. Component m is a vector indicating neurons connected to subunit m
        Wwe, Wwi: vectors of synaptic weights for each input neuron (excitatory/inhibitory)
        Taue, Taui: time constants for each input neuron
        Delay: M length vector with delay for each subunit
    sig_on: M length boolean vector determining whether each subunit has non-linear step or not."""

    v0, logJw, Wwe, Wwi, logTaue, logTaui, Th, logDelay = params

    # some parameters defined by their logs to ensure positivity - convert here
    Jw, Taue, Taui, Delay = tf.exp(logJw), tf.exp(logTaue), tf.exp(logTaui), tf.exp(logDelay)

    # get number of subunits, synapses, timesteps etc.
    M = len(Jc)
    N = X.shape[0]
    L = X.shape[1]

    # first we calculate the synaptic input to each subunit: will need to loop over subunits, but don't loop over
    # synapses within each subunit

    Ym_list = []  # empty list to store synaptic input vector for each subunit
    # then concatenate excitatory and inhibitory parameters for easier convolution
    Taus = tf.concat((Taue, Taui), axis=0)
    Wws = tf.concat((Wwe, Wwi), axis=0)

    for m in range(M):

        # create empty vector for each subunit - stack all of these at end to get full Y matrix
        Ym = tf.zeros(shape=L, dtype=tf.float32)

        # concatenate Wce[m] and Wci[m] and then check if full: then we can keep the existing Wce syntax to work
        # for both inhibitory and excitatory synapses in 1 convolution

        Wc = tf.concat((Wce[m], Wci[m]), axis=0)

        if len(Wc) > 0:  # if subunit has any neurons connected to it: multiple conditions
            # so we only have to do one convolution
            n_syn = len(Wc)

            # create alpha kernel matrix, one row for each excitatory synapse. Then convolve kernel matrix with
            # the input rows corresponding to the subunit excitatory input neurons

            # find Taus for all subunit synapses, then find maximum of these
            Taus_m = tf.gather(Taus, Wc)
            Wws_m = tf.gather(Wws, Wc)
            Tau_max = tf.math.reduce_max(Taus_m)
            # kernel decays quickly, so only consider times up to 10 * max(tau) after spikes - this will be the
            # length of the filter
            filt_length = tf.cast(10 * Tau_max, tf.int32)
            # shape of filter matrix
            shape = [n_syn, 1]
            # create filter matrix
            filt_times = tf.tile(tf.reshape(tf.range(0, filt_length, dt), (1, filt_length)), (n_syn, 1))
            filt_times = tf.cast(filt_times, tf.float32)
            filt = ((filt_times - Delay[m]) / tf.reshape(Taus_m, (n_syn, 1))) * tf.math.exp(
                -(filt_times - Delay[m]) / tf.reshape(Taus_m, (n_syn, 1)))
            filt = tf.clip_by_value(filt, clip_value_min=0, clip_value_max=10)
            filt = filt * tf.reshape(Wws_m, (n_syn, 1))

            # now select input rows we will convolve with, according to values in Wce[m]
            X_m = tf.gather_nd(X, tf.reshape(Wc, (n_syn, 1)))

            # we have the filter and the input, now extend the inputs and convolve
            X_extend = tf.concat((tf.zeros((n_syn, int(filt_length / 2)), dtype=tf.float32), X_m), axis=1)

            # reshape and convolve
            filt = tf.reshape(tf.transpose(filt), shape=[1, filt_length, n_syn, 1])
            X_extend = tf.reshape(tf.transpose(X_extend), shape=[1, 1, L + int(filt_length / 2), n_syn])

            strides = [1, 1, 1, 1]
            filtered_m = tf.nn.depthwise_conv2d(X_extend, filt[:, ::-1], strides, padding='SAME', data_format="NHWC")
            #             filtered_m = tf.reshape(filtered_m, (L+int(filt_length/2), n_syn))

            Ym = tf.reduce_sum(filtered_m, axis=[0, 1, 3])

            # now discard extra points:
            Ym = Ym[:L]

        Ym_list.append(Ym)

    # now stack all the Yms we stored during the loop
    Y = tf.stack(Ym_list)

    # now we have synaptic inputs for each subunit, we start from the leaves and apply nonlinearites, and then
    # add leaf output to its parent - use same code as first sim_hLN_tf, given we need to loop over leaves anyway

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
            # define paddings - this way we can add non linearity output to whole R Tensor instead of assigning 1 row
            paddings = ([[leaf - 1, M - leaf], [0, 0]])

            # sigmoid threshold defined per subunit - apply only if subunit defined as non-linear
            if sig_on[leaf - 1]:  # if subunit is nonlinear
                increment = tf.pad(tf.reshape(sigm(Y[leaf - 1, :], tau=Th[leaf - 1]), (1, L)), paddings, "CONSTANT")
            else:  # subunit is just linear
                increment = tf.pad(tf.reshape(Y[leaf - 1, :], (1, L)), paddings, "CONSTANT")

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




