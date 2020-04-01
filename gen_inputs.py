from functs_inputs import *
import numpy as np
# np.set_printoptions(threshold=1000)


def gen_realistic_inputs(Tmax):
    """generate synaptic inputs with different mixing orientations"""

    # define list of alphas
    alphas = [10.54, 4.96, 1.07, 0.5, 1.82, 6.33, 11.84, 14.01, 10.54, 4.96, 1.07, 0.5, 1.82, 6.33, 11.84, 14.01]
    beta = 20

    # generate preferred orientations of 13 'dendrites' from normal distribution
    # multiple cells pooled into one ensemble
    ori_dends = np.floor(np.random.normal(loc=0, scale=1.5, size=16) % 16)
    print(ori_dends)

    # define how many excitatory/inhibitory cells in each of 13 ensembles
    Ensyn = [48, 58, 52, 34, 45, 39, 44, 68, 50, 62, 30, 60, 39]
    Insyn = [11, 11, 9, 6, 8, 5, 8, 12, 11, 13, 6, 11, 8]
    N_inh = np.sum(Insyn)

    Erate = [5, 20] #firing rate of excitatory inputs for low and high states
    Esd = [2.5, 10] #standard deviation of firing rate (for OU process)
    Irate = [20, 30] #firing rate of inhibitory neurons in each state
    N_soma = 420 #number of inhibitory neurons connected directly to soma - change to 420 for real thing

    # generate transition times between upstate and down state
    for ori in range(16):
        # present 16 stimulus orientations
        rate_inh = np.zeros(Tmax)
        for den in range(13):
            # generate 13 independent ensembles
            st = gen_events_sin(Tmax=Tmax, alpha=alphas[int((ori-ori_dends[den]) % 16)], beta=beta, maxL=150)
            rate_inh += Insyn[den] * (st * (Irate[1] - Irate[0]) + Irate[0])
            spt_Eden = gen_spikes_states(Tmax=Tmax, N=Ensyn[den], mu=Erate, tau=500, x=st, sd=Esd)
            if den > 0:
                spt_Eden[:, 0] += np.sum(Ensyn[:den])  # i.e. number dendrites from different ensembles independently
                spt_E = np.vstack((spt_E, spt_Eden))
            else:
                spt_E = spt_Eden

        rate_inh /= N_inh
        # generate inhibitory spikes from rate_inh
        sp_i_d = gen_poisson_train_rescale(rate_inh, N_inh)
        sp_i_d[:, 0] += 1 #soma is now 'first' inhibitory neuron, so increment the number of the others
        sp_i_soma = gen_poisson_spikes(rate_inh * N_soma)

        # join somatic and dendritic inhibitory neurons into one inhibitory input, then sort by spiking times
        sp_i = np.vstack((sp_i_soma, sp_i_d))
        spt_I = sp_i[sp_i[:, 1].argsort()]

        # sort the spiking events by time rather than the neuron number that produced them
        spt_E = spt_E[spt_E[:, 1].argsort()]

        if ori == 0:
            E_spikes = spt_E
            I_spikes = spt_I

        else:
            spt_E[:, 1] += ori * Tmax
            spt_I[:, 1] += ori * Tmax
            E_spikes = np.vstack((E_spikes, spt_E))
            I_spikes = np.vstack((I_spikes, spt_I))

        print("orientation", ori, "finished")

    return E_spikes, I_spikes


# E_spikes, I_spikes = gen_realistic_inputs(Tmax=1000)
# print(E_spikes.shape, I_spikes.shape)
#
# Ensyn = [48, 58, 52, 34, 45, 39, 44, 68, 50, 62, 30, 60, 39]
# Ensyn_cum = np.cumsum(Ensyn)
# Ensyn_cum = Ensyn_cum.reshape(13, 1)
# print(Ensyn_cum)
#
# ens_num = np.argmax(Ensyn_cum > E_spikes[:, 0] - 1, axis=0)
#
# plt.scatter(E_spikes[:, 1], E_spikes[:, 0], s=1, c=ens_num, cmap="prism", marker='.', label="Excitatory")
# plt.scatter(I_spikes[:, 1], np.sum(Ensyn) + I_spikes[:, 0], s=1, marker='.', color='black', label='Inhibitory')
# plt.xlabel("Time (ms)")
# plt.ylabel("Neuron number")
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.title("Input spike trains")
# plt.show()



def spikes_to_input(spikes, Tmax):
    """function to convert output from gen_realistic_inputs into a form that can be fed to the hLN model
    spikes = output from gen_realistic_inputs, a S X 2 array where S is the total number of spiking events.
     The 2nd column of S contains the time of the spiking event, and the 1st contains the number of the neuron
     that produced the spike."""

    neurons = spikes[:, 0]
    times = spikes[:, 1]
    N = int(np.amax(neurons)) #number of neurons
    T = Tmax  # number of timesteps
    # empty array which will be binary, 1 indicating a spiking event at time t from neuron n
    bin_array = np.zeros((N + 1, T))
    times = np.floor(times) # times must be integer valued for indexing, multiple spikes in same step counted as 1
    neurons, times = neurons.astype(int), times.astype(int)
    # where spiking event, fill binary array with 1s
    bin_array[neurons, times] = 1
    return bin_array


#
# E_spikes, I_spikes = gen_realistic_inputs(Tmax=3000)
# #
# X_e = spikes_to_input(E_spikes, Tmax=48000)
# X_i = spikes_to_input(I_spikes, Tmax=48000)
# X_tot = np.vstack((X_e, X_i))
#
# np.save("Data/inputs_equalrate.npy", X_tot)
#
#
# print(spikes_to_input(E_spikes, Tmax=16000))
