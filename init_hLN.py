#############################################
# File containing functions for initialising hLN models. init_nonlin initialises nonlinearities for architectures
# previously containing linear subunits. <update_architecture> will add new linear subunits to an existing
# architecture, which should essentially involve just redistributing inputs.

from sim_hLN import *
from utils import *


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
        # print(alpha)
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
    Th = next_model.Th.numpy()

    # then initialise 'substructure' of new model to be identical to previous model (i.e. subsection of new
    # architecture that made up the previous model)
    M_old = len(prev_model.Jc)
    logJw[:M_old] = prev_model.logJw.numpy()
    logDelay[:M_old] = prev_model.logDelay.numpy()
    Th[:M_old] = prev_model.Th.numpy()

    # work out which subunits we have just added - for these cases should just be the leaves
    M = len(next_model.Jc)
    leaves = np.setdiff1d(np.arange(1, M + 1, 1), next_model.Jc)
    for leaf in leaves:
        logJw[leaf - 1] = 0  # set subunit gain to 1 for all new leaves
        # then set delay to the delay of the subunit parent from the previous model
        parent = next_model.Jc[leaf - 1]
        logDelay[leaf - 1] = prev_model.logDelay.numpy()[parent - 1]


    # assign the newly calculated parameters to the new model
    next_model.Wwe.assign(prev_model.Wwe)
    next_model.Wwi.assign(prev_model.Wwi)
    next_model.logTaue.assign(prev_model.logTaue)
    next_model.logTaui.assign(prev_model.logTaui)
    next_model.logJw.assign(logJw)
    next_model.logDelay.assign(logDelay)
    next_model.Th.assign(Th)
    next_model.v0.assign(prev_model.v0)

    return


def init_nonlin_tied(X, model, lin_model, nSD, dt=1):
    """function to initialise the nonlinearities in subunits which were previously nonlinear. The parameters
     of the linear model should already have been optimised, and as such its accuracy should be a lower bound on the
    accuracy of the new non-linear model. This function is for models of type hLN_TiedModel only, as parameter
    definitions differ slightly from the untied version
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
    v0, log_Jw, Wwe, Wwi, log_Tau_e, log_Tau_i, Th, log_Delay = [param.numpy() for param in model.trainable_params]

    # these parameters defined by their logs to ensure positivity - convert here
    Jw, Tau_e, Tau_i, Delay = np.exp(log_Jw), np.exp(log_Tau_e), np.exp(log_Tau_i), np.exp(log_Delay)

    N = X.shape[0]
    L = X.shape[1]
    Tmax = L / dt
    M = len(Jc)
    if M == 1:  # single subunit model - adjust parameters such that Jw is 1 initially
        Wwe *= Jw[0]
        Wwi *= Jw[0]
        Jw /= Jw

    # first find which subunits we want to initialise nonlinearities for - should just be the leaves:
    leaves = np.setdiff1d(np.arange(1, M + 1, 1), Jc)

    # now calculate synaptic input to each dendritic branch - will use this to scale
    # need to assign rescaled parameters first
    model.Wwe.assign(Wwe)
    model.Wwi.assign(Wwi)
    model.logJw.assign(np.log(Jw))

    #     logTaues = tf.fill(dims=[model.n_e], value=model.logTaue)
    #     logTauis = tf.fill(dims=[model.n_i], value=model.logTaui)
    #     Wwes = tf.fill(dims=[model.n_e], value=model.Wwe)
    #     Wwis = tf.fill(dims=[model.n_i], value=model.Wwi)

    logTaues = []
    logTauis = []
    Wwes = []
    Wwis = []
    for m in range(len(model.Jc)):
        logTaues = tf.concat((logTaues, tf.fill(dims=[len(model.Wce[m])], value=model.logTaue[m])), axis=0)
        logTauis = tf.concat((logTauis, tf.fill(dims=[len(model.Wci[m])], value=model.logTaui[m])), axis=0)
        Wwes = tf.concat((Wwes, tf.fill(dims=[len(model.Wce[m])], value=model.Wwe[m])), axis=0)
        Wwis = tf.concat((Wwis, tf.fill(dims=[len(model.Wci[m])], value=model.Wwi[m])), axis=0)

    model.params = (model.v0, model.logJw, Wwes, Wwis, logTaues, logTauis,
                    model.Th, model.logDelay)
    Y = sim_inputs(X=X, dt=1, Jc=model.Jc, Wce=model.Wce, Wci=model.Wci, params=model.params, sig_on=model.sig_on)
    Y = Y.numpy()

    # now we have synaptic input, convert Wwes and Wwis back into numpy for easier processing
    Wwes, Wwis = Wwes.numpy(), Wwis.numpy()

    #  we only need to initialise nonlinearities in the leaves:
    for leaf in leaves:

        range_Y = np.std(Y[leaf - 1, :])
        alpha = (nSD * range_Y)
        # print(alpha)
        if len(Wce[leaf - 1] > 0):  # if leaf has any e neurons connected to it
            Wwes[(Wce[leaf - 1])] /= alpha
            Wwe[leaf - 1] /= alpha

        if len(Wci[leaf - 1] > 0):  # if leaf has any i neurons connected to it
            Wwis[(Wci[leaf - 1] - model.n_e)] /= alpha
            Wwi[leaf - 1] /= alpha

        Th[leaf - 1] = np.mean(Y[leaf - 1, :]) / alpha

        Jw[leaf - 1] *= (4 * alpha)

        if leaf != 1:  # if not the soma
            parent = Jc[leaf - 1]
            Th[parent - 1] -= (np.mean(Y[leaf - 1, :]) - 2 * alpha)

        else:  # soma: add offset into v0
            v0 += (np.mean(Y[leaf - 1, :]) - 2 * alpha)

    logJw, logTaue, logTaui, logDelay = np.log(Jw), np.log(Tau_e), np.log(Tau_i), np.log(Delay)

    model.logJw.assign(logJw)
    model.Wwe.assign(Wwe)
    model.Wwi.assign(Wwi)
    model.Th.assign(Th)
    model.logDelay.assign(logDelay)
    model.v0.assign(v0)

    return



def update_arch_tied(prev_model, next_model):
    """Function to assign the correct parameters to a new architecture which has added new linear leaf subunits
    from the previous architecture. The Jc, Wce and Wci of the new architecture are known, so the synaptic
    parameters just need to be redistributed accordingly. This new function is required for the tied models,
    as the parameters are defined differently and so need to be redistributed differently also."""

    # first change hLN attributes into numpy - allows assignment and should be easier to manipulate
    logJw = next_model.logJw.numpy()
    logDelay = next_model.logDelay.numpy()
    Th = next_model.Th.numpy()
    Wwe, Wwi = next_model.Wwe.numpy(), next_model.Wwi.numpy()
    logTaue, logTaui = next_model.logTaue.numpy(), next_model.logTaui.numpy()

    # then initialise 'substructure' of new model to be identical to previous model (i.e. subsection of new
    # architecture that made up the previous model)
    M_old = len(prev_model.Jc)
    logJw[:M_old] = prev_model.logJw.numpy()
    logDelay[:M_old] = prev_model.logDelay.numpy()
    Th[:M_old] = prev_model.Th.numpy()
    Wwe[:M_old], Wwi[:M_old] = prev_model.Wwe.numpy(), prev_model.Wwi.numpy()
    logTaue[:M_old], logTaui[:M_old] = prev_model.logTaue.numpy(), prev_model.logTaui.numpy()

    # work out which subunits we have just added - for these cases should just be the leaves
    M = len(next_model.Jc)
    leaves = np.setdiff1d(np.arange(1, M + 1, 1), next_model.Jc)
    for leaf in leaves:
        logJw[leaf - 1] = 0  # set subunit gain to 1 for all new leaves
        # then set delay to the delay of the subunit parent from the previous model
        parent = next_model.Jc[leaf - 1]
        logDelay[leaf - 1] = prev_model.logDelay.numpy()[parent - 1]
        # also set weights and time constants to those of subunit parent in previous model
        Wwe[leaf - 1] = prev_model.Wwe.numpy()[parent - 1]
        Wwi[leaf - 1] = prev_model.Wwi.numpy()[parent - 1]
        logTaue[leaf - 1] = prev_model.logTaue.numpy()[parent - 1]
        logTaui[leaf - 1] = prev_model.logTaui.numpy()[parent - 1]

    # assign the newly calculated parameters to the new model
    next_model.Wwe.assign(Wwe)
    next_model.Wwi.assign(Wwi)
    next_model.logTaue.assign(logTaue)
    next_model.logTaui.assign(logTaui)
    next_model.logJw.assign(logJw)
    next_model.logDelay.assign(logDelay)
    next_model.Th.assign(Th)
    next_model.v0.assign(prev_model.v0)

    return