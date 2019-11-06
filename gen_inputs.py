import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# script to generate realistic inputs for the hLN model, based on Ujfalussy R code

# functions required:
# 1. Homogeneous poisson process
# 2. Inhomogeneous poisson process
# 3. Function to generate events according to sin process as in paper
# 4. Function to generate inhomogeneous poisson process with time rescaling methods
# 5. Main script to tie all together (probably in different file)

# Remember: first we need to generate transitions between low and high, once we have these we can then generate spikes
# Can probably write smaller functions first, maybe will help to understand main function in R code


def gen_poisson_events(Tmax, rate):
    """function to generate events from a homogeneous poisson process
    Tmax = time over which to generate in ms
    rate = event frequency in 1/ms
    returns list of event times"""

    t0 = np.random.exponential(scale=1/rate, size=None)
    if t0 > Tmax:
        sp = None

    else:
        sp = [t0]
        tmax = t0
        while tmax < Tmax:
            t_next = tmax + np.random.exponential(scale=1/rate, size=None)
            if t_next < Tmax:
                sp.append(t_next)
            tmax = t_next

    return sp


def gen_poisson_train(rates, N):
    """function to generate events from an inhomogeneous poisson process, with rates defined at each ms"""

    return

def gen_poisson_train_rescale(rates, N):
    """function to generate events from an inhomogenous poisson process using the tinm-rescaling theorem - should be
    quick than gen_poisson_train.
    rates = Poisson rate defined per ms
    N = number of cells producing train
    returns array of event times"""
    Tmax = len(rates) #Tmax in ms
    mean_rate = np.mean(rates)
    t_spikes = np.array([[0, 0]])
    # print(t_spikes.shape)

    for cell in range(N):
        t_sp = gen_poisson_events(Tmax=Tmax, rate=mean_rate/1000) #convert from 1/s to 1/ms
        try:
            if len(t_sp) > 0:
                T_sp = mean_rate * np.array(t_sp)
                Cr = np.cumsum(rates)
                t = np.arange(Tmax)
                t_rescaled = np.interp(x=T_sp, xp=Cr, fp=t)
                # later: add lines to output both event time and the neuron that produced it
                cells = np.full(t_rescaled.shape, cell)
                t_sp_cell = np.vstack((cells, t_rescaled)).T
                t_spikes = np.vstack((t_spikes, t_sp_cell))
        except TypeError:
            pass

    t_spikes = t_spikes[t_spikes[:, 1].argsort()]

    return t_spikes



# const_rates = [100] * 100
#
# print(gen_poisson_train_rescale(rates=const_rates, N=10))

# plt.plot([10.54,  4.96,  1.07,  0.5,  1.82,  6.33, 11.84, 14.01, 10.54,  4.96,  1.07,  0.5,  1.82,  6.33, 11.84, 14.01])
# plt.show()

def gen_events_sin(Tmax, alpha, beta, maxL=None):
    """function to generate transition times between up and down states based on rules set out in Lengyel paper.
    Tmax = time to generate transitions over
    alpha = max of the down to up transition rate in Hz
    beta = up to down transition rate in Hz
    maxL = maximum duration of an up state, default no maximum
    """

    # convert rates from Hz to 1/ms
    alpha /= 1000
    beta /= 1000

    max_rate = max(alpha, beta)
    events = gen_poisson_events(Tmax=Tmax, rate=max_rate)
    state = 0 #start from down state, up state is 1
    st = np.zeros(Tmax)
    events_kept = np.array([[0, 0]])

    if len(events) > 0:
        for tt in events:
            # we keep the transition with some probability - this is thinning
            if state == 0:
                rr = (np.sin(tt/500 * 2 * np.pi + 150) + 1) * alpha/2
            else:
                rr = beta #constant rate for up to down
            p_keep = rr / max_rate
            if np.random.uniform() < p_keep:
                # print("Transition kept")
                state = 1 - state
                st[round(tt):] = state
                if maxL is not None:
                    if state == 0:
                        # if we have just transitioned to down state, check that the time in up state did not exceed the max
                        t_last = events_kept[-1][1]
                        Li = tt - t_last
                        if Li > maxL:
                            t1 = np.random.uniform(low=t_last + Li/5, high=t_last + 2*Li/5)
                            t2 = np.random.uniform(low=t_last + 3*Li/5, high=t_last + 4*Li/5)
                            events_kept = np.append(events_kept, [[0, t1]], axis=0)
                            events_kept = np.append(events_kept, [[1, t2]], axis=0)
                            st[round(t1):round(t2)] = 0

                events_kept = np.append(events_kept, [[state, tt]], axis=0)

            else:
                # print("Transition rejected")
                pass

    return st






def gen_spikes_states(Tmax, N, mu, tau, x, sd=0):
    """Given a sequence of states and transition times, generate a spike train from the resulting inhomogeneous
    poisson process
    Tmax = time over which to generate
    N = number of neurons
    mu = vector of firing rates (Hz)
    sd = standard deviation of firing rates (Hz)
    tau = time constant (ms)
    x = sequence of states
    """

    dt_rates = 1
    L = Tmax / dt_rates
    # create rates object using transitions described by x
    rates = np.where(x, mu[1], mu[0])
    # print(rates)

    # add random elements to rates later (OU process)

    t_sp = gen_poisson_train_rescale(rates=rates, N=N)

    return t_sp


# st = gen_events_sin(Tmax=100, alpha=200, beta=20, maxL=150)
# print(st)
#
# spikes = gen_spikes_states(Tmax=100, N=10, mu=Mu, tau=0, x=st)
#
# print(spikes)


def gen_realistic_inputs(Tmax):
    """generate synaptic inputs with different mixing orientations"""

    # define list of alphas
    alphas = [10.54, 4.96, 1.07, 0.5, 1.82, 6.33, 11.84, 14.01, 10.54, 4.96, 1.07, 0.5, 1.82, 6.33, 11.84, 14.01]
    beta = 20

    # generate preferred orientations of 13 'dendrites' from normal distribution
    # multiple cells pooled into one ensemble
    ori_dends = np.floor(np.random.normal(loc=0, scale=1.5, size=16) % 16)
    print(ori_dends)

    # define how many cells in each of 13 ensembles
    Ensyn = [48, 58, 52, 34, 45, 39, 44, 68, 50, 62, 30, 60, 39]

    Erate = [5, 20] #firing rate of excitatory inputs for low and high states

    # generate transition times between upstate and down state
    for ori in range(16):
        # present 16 stimulus orientations
        for den in range(13):
            # generate 13 independent ensembles
            st = gen_events_sin(Tmax=Tmax, alpha=alphas[int((ori-ori_dends[den]) % 16)], beta=beta, maxL=150)
            spt_Eden = gen_spikes_states(Tmax=Tmax, N=Ensyn[den], mu=Erate, tau=0, x=st, sd=0)
            if den > 0:
                spt_Eden[:, 0] += np.sum(Ensyn[:den]) #i.e. number dendrites from different ensembles independently
                spt_E = np.vstack((spt_E, spt_Eden))
            else:
                spt_E = spt_Eden

        spt_E = spt_E[spt_E[:, 1].argsort()]

        if ori == 0:
            E_spikes = spt_E

        else:
            spt_E[:, 1] += ori * Tmax
            E_spikes = np.vstack((E_spikes, spt_E))

        print("orientation", ori, "finished")

    return E_spikes


spikes = gen_realistic_inputs(Tmax=1000)
print(spikes.shape)

plt.plot(spikes[:, 1], spikes[:, 0], marker='.', markersize=1, linestyle='')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron number")
plt.ylim(bottom=0)
plt.title("Excitatory input spike trains")
plt.show()
