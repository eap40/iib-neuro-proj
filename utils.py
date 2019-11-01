import numpy as np
from scipy import sparse as sps
import matplotlib.pyplot as plt
import seaborn as sns



sns.set()




def binary_input(M, L, kind='rand', delay=0):
    """function to generate a binary input matrix of dimensions M x L, where M is the number of input neurons
    and L the number of time bins. Kind argument controls the type of input e.g. randomly spaced inputs"""
    if kind=='rand':
        # create sparse binary array with randomly distributed values
        sparse_array = sps.random(M, L, density=0.03, dtype='bool')
        # convert sparse array to numpy array for later use
        array_out = sps.csr_matrix.todense(sparse_array)


    if kind=='delta':
        array_out = np.zeros([M,L])
        array_out[:, delay] = np.ravel(np.ones([M, 1]))
        array_out= np.asmatrix(array_out)

    return array_out



def alpha_syn(t, tau):
    """standard alpha function for synaptic kernels. t is time, tau is time constant defined for each synapse."""
    kernel = np.heaviside(t, 0) * (t / tau) * np.exp(-(t / tau))
    return kernel



def sigm(x, tau=0):
    """sigmoid function for initial global non-linearity"""
    return 1/(1 + np.exp(-(x-tau)))

def convolve(s, dt, tau, delay=0):
    """function to efficiently convolve a binary input (s) with a synaptic kernel. Given the input is binary
    and sparse, and the kernel decays quickly in time, this is quicker than using a standard convolution function.
    dt is time resolution of time bins in spikes, and delay is synaptic delay at each synapse. Returns the
    convolved signal for the time period of s (result should be same shape as s)"""


    times = dt * np.arange(0, s.shape[1], 1)

    # initialise response array - will add to it for each spike
    resp = np.zeros(s.shape)

    for n in range(s.shape[0]):
        print(n)
        # find indexes of spikes in row n
        spikes = np.nonzero(s[n])[1]
        print(spikes)

        for spike in spikes:
            effect = alpha_syn(times-(times[spike] + delay), tau=tau)
            resp[n, :] += effect

    return resp


def numpy_convolve(s, dt, tau, delay=0):
    """function to perform built in numpy convolution and compare with custom function"""
    times = dt * np.arange(0, s.shape[1], 1)
    kernel = alpha_syn(times-delay, tau=tau)

    resp = np.convolve(kernel, np.ravel(s), mode='full')

    return resp[:len(times)]