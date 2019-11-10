import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# file containing smaller core functions required for larger input generation function

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
                cells = np.full(t_rescaled.shape, cell)
                t_sp_cell = np.vstack((cells, t_rescaled)).T
                t_spikes = np.vstack((t_spikes, t_sp_cell))
        except TypeError:
            pass

    t_spikes = t_spikes[t_spikes[:, 1].argsort()]

    return t_spikes

def gen_poisson_spikes(r):
    """r is a vector of rates (Hz) over time, generates poisson spike counts with same resolution + some jitter
    function works by generating the number of spikes in each time interval from a poisson process with lambda =
    rate(t), then draws the times of the spikes from a uniform distribution across the time interval """
    sp = np.array([[0, 0]])
    L = len(r)
    for i in range(L):
        n_sp = np.random.poisson(lam=r[i]/1000, size=None)
        if n_sp > 0:
            new_entry = np.vstack((np.zeros(n_sp), i + np.sort(np.random.uniform(size=n_sp)))).T
            sp = np.vstack((sp, new_entry))

    return sp


const_rates = [100] * 100
#
print(gen_poisson_spikes(r=const_rates))

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
    try:
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
    except TypeError:
        pass

    return st






def gen_spikes_states(Tmax, N, mu, tau, x, sd):
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
    L = int(L)
    # create rates object using transitions described by x
    rates = np.where(x, mu[1], mu[0])
    # print(rates)

    # add random elements to rates later (OU process)
    # simulate an OU process with sd = 1 and add it to the rates
    z = np.zeros(L)
    z0 = 0
    Q = np.sqrt(2 / tau)

    for i in range(1, L):
        z[i] = z[i - 1] + (z0 - z[i-1]) * dt_rates / tau + Q * np.random.normal() * np.sqrt(dt_rates)
        r0 = mu[int(x[i])]
        rates[i] = r0 + sd[int(x[i])] * z[i]

    # random element could produce negative rates, which are not allowed so clip rates to 0 minimum
    rates = np.clip(rates, a_min=0, a_max=None)

    t_sp = gen_poisson_train_rescale(rates=rates, N=N)

    return t_sp


def input_demo(Tmax, N):
    """create plot to demonstrate input function working. Generate state intervals, then using states defined
    by these intervals generate poisson events. Plot spiking events and actual rate on same plot (so we can
    see more events when rate is higher)"""
    Erate = [5, 20]
    Esd = [2.5, 10]
    st = gen_events_sin(Tmax=Tmax, alpha=20, beta=20, maxL=150)
    spt = gen_spikes_states(Tmax=Tmax, N=N, mu=Erate, tau=500, x=st, sd=Esd)
    rates = np.where(st, Erate[1], Erate[0])
    fig, ax = plt.subplots()
    ax.grid(False)
    plt.scatter(spt[:, 1], spt[:, 0], s=1, marker='.')
    rates_2d = np.tile(rates, (N, 1))
    print(rates_2d.shape)
    im = ax.imshow(rates_2d, aspect='auto', cmap='Reds', origin='lower')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Neuron spiking events as firing rate moves \n between the background and elevated state")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron number")
    plt.show()

    