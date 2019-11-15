import numpy as np
from scipy import sparse as sps
import matplotlib.pyplot as plt
import seaborn as sns



sns.set()




def binary_input(M, L, kind='rand', delay=0):
    """function to generate a binary input matrix of dimensions M x L, where M is the number of input neurons
    and L the number of time bins. Kind argument controls the type of input e.g. randomly spaced inputs"""
    if kind == 'rand':
        # create sparse binary array with randomly distributed values
        sparse_array = sps.random(M, L, density=0.05, dtype='bool')
        # convert sparse array to numpy array for later use
        array_out = sps.csr_matrix.todense(sparse_array)

    if kind == 'delta':
        array_out = np.zeros([M,L])
        array_out[:, delay] = np.ravel(np.ones([M, 1]))
        array_out = np.asmatrix(array_out)

    array_out = np.asarray(array_out)

    return array_out



def alpha_syn(t, tau):
    """standard alpha function for synaptic kernels. t is time, tau is time constant defined for each synapse."""
    kernel = np.heaviside(t, 0) * (t / tau) * np.exp(-(t / tau))
    return kernel

def alpha_filt(tau, spikes, delay=0, dt=1):
    """function to convolve an input spike train with a standard alpha kernel"""

    # lots of checks on inputs eventually - just want to get some output for now

    # check if spikes is vector or matrix
    if len(spikes.shape) < 2:
        # spikes is a vector, convert it to a 1 X len(spikes) array:
        spikes = np.reshape(spikes, (1, len(spikes)))

    N = spikes.shape[0]
    L_max = spikes.shape[1]
    Tmax = dt * L_max
    t = np.arange(0, Tmax, dt)
    # kernel decays quickly, so only consider times up to 10 * tau after spikes
    tf = np.arange(0, 10 * tau, dt)
    L = len(tf)
    filt = (tf - delay) / tau * np.exp(-(tf - delay)/tau)
    filt = np.where(filt < 0, 0, filt)

    # now we need to work out whether we loop over the spikes, or apply filter to whole array. This will depend on
    # how many spikes are in the train

    # for now assume for loop is quicker:
    f0_spikes = np.zeros((N, L + L_max))
    for n in range(N):
        ispn = np.nonzero(np.ravel(spikes[n, :]))[0]
        # print('ispn:', ispn)
        if len(ispn) > 0:
            for isp in ispn:
                f0_spikes[n, isp:(isp + L)] += spikes[n, isp] * filt

    f_spikes = f0_spikes[:, :L_max]

    return f_spikes



def sigm(x, tau=0):
    """sigmoid function for initial global non-linearity"""
    return 1/(1 + np.exp(-(x-tau)))

def convolve(s, dt, tau, delay=0):
    """function to efficiently convolve a binary input (s) with a synaptic kernel. Given the input is binary
    and sparse, and the kernel decays quickly in time, this is quicker than using a standard convolution function.
    dt is time resolution of time bins in spikes, and delay is synaptic delay at each synapse. Returns the
    convolved signal for the time period of s (result should be same shape as s)"""

    # print(s.shape, np.nonzero(s)[0])
    times = dt * np.arange(0, len(s), 1)
    # initialise response array - will add to it for each spike
    resp = np.zeros(s.shape)

    # find indexes of spikes in s
    spikes = np.nonzero(s)[0]
    # print(spikes)

    for spike in spikes:
        effect = alpha_syn(times-(times[spike] + delay), tau=tau)
        resp += effect

    return resp


def numpy_convolve(s, dt, tau, delay=0):
    """function to perform built in numpy convolution, for comparison with custom function"""
    times = dt * np.arange(0, s.shape[1], 1)
    kernel = alpha_syn(times-delay, tau=tau)

    resp = np.convolve(kernel, np.ravel(s), mode='full')

    return resp[:len(times)]

def int_spikes(X, dt, Wc, Ww, Tau, delay):
    """function to integrate synaptic input arriving to a given subunit - should perform better that initial attempt
    'convolve'
    INPUT:
    X: NxL binary input matrix of presynaptic spikes; N: # of input neurons; L: # of timesteps
    dt: the time resolution of X in milliseconds
    Wc: a vector with the indices of the presynaptic neurons.
    Ww: either a scalar or a vector with the synaptic weights.
    Tau: time constant of the synapses - positive a scalar
    delay.t: propagation delay of the synapses - a positive scalar
    grad.Tau: specifies whether input is filtered with the synaptic kernel or its derivative wrt. Tau. if T, alpha must be T
    alpha.ampl: filter is parametrised by keeping the amplitude (instead of the integral) independent of Tau

    OUTPUT:
    out: a vector of length L  - the total synaptic input to a given subunit"""

    # start with parameter checks - fill in later, just want some output for now

    L = X.shape[1]
    Tmax = L/dt
    N = X.shape[0]
    y = np.zeros(L)

    # the weights of the cells
    w = np.zeros(N)

    # 2 different cases required: Wc is either a single value (so len is invalid) or list, for which we use len
    # replace try with if laterd
    try:
        n = len(Wc)
        # more checks on Wc and Ww input parameters here
        for i in range(n):
            w[Wc[i]] = Ww[i]

    except TypeError:
        # single element in Wc (so len command produces TypeError
        w[Wc] = Ww


    # might want something here to limit in the case of very large positive or negative values
    # add up inputs going into subunit
    x = np.matmul(w, X)
    if len(x.shape) < 2:
        # if x is 1d, reshape to 2d array (1 row, column for each timestep
        x = np.reshape(x, (1, len(x)))

    # print("x shape:", x.shape)

    # MIGHT NEED TO BULK OUT ALPHA SYN FUNCTION TO WORK WITH THIS ONE
    # change alpha function to filter a spike train with the kernel, not just generate the kernel.
    out = alpha_filt(tau=Tau, spikes=x, delay=delay, dt=dt)

    return np.ravel(out)


