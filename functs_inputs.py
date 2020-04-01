import numpy as np
from scipy.stats import truncexpon

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
    t_spikes = t_spikes[1:]  #get rid of 'first' entry used to initalise t_spikes array

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


# const_rates = [100] * 100
# #
# print(gen_poisson_spikes(r=const_rates))

# plt.plot([10.54,  4.96,  1.07,  0.5,  1.82,  6.33, 11.84, 14.01, 10.54,  4.96,  1.07,  0.5,  1.82,  6.33, 11.84, 14.01])
# plt.show()

def gen_events_sin(Tmax, alpha, beta, maxL=None):
    """function to generate transition times between up and down states based on rules set out in Ujfalussy paper.
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


def gen_events_sin2(Tmax, alpha, beta, maxL=None):
    """function to generate transition times between up and down states based on rules set out in Ujfalussy paper.
    Uses thinning to generate transitions from down to up, and then draws up state durations from a truncated
    exponential distribution.
    Tmax = time to generate transitions over
    alpha = max of the down to up transition rate in Hz
    beta = up to down transition rate in Hz
    maxL = maximum duration of an up state, default no maximum
    """

    # convert rates from Hz to 1/ms
    alpha /= 1000
    beta /= 1000
    max_rate = max(alpha, beta)

    # generate events from a homogeneous poisson process with rate max_rate
    events = gen_poisson_events(Tmax=Tmax, rate=max_rate)
    events_kept = np.array([])
    # use thinning to generate event times according to down to up transition rate
    for event_time in events:
        # rate rr is sine function of stimulus orientation and time
        rr = (np.sin(event_time / 500 * 2 * np.pi + 150) + 1) * alpha / 2
        p_keep = rr / max_rate # probability of keeping Poisson event at time tt
        if np.random.uniform() < p_keep:
            # keep event
            events_kept = np.append(events_kept, event_time)

    # events kept should now contain events generated according to inhomogeneous Poisson process, with rate defined by
    # down to up sine function

    # define truncated exponential distribution to draw up state duration times from
    scale = 1/beta
    trunc_expon = truncexpon(b=maxL/scale, scale=scale)

    # start from down state, up state is 1
    st = np.zeros(Tmax)  # array of states, will be binary

    # first transition is down to up, with event time the first event in events_kept
    tt = events_kept[0]
    last_down_tt = events_kept[-1]  # last time at which there is a down to up transition
    state = 1  # in up state after this first transition
    durations = []

    while tt < Tmax:

        if state == 0:
            # next down to up transition is first event in events_kept after the last transition time
            if tt < last_down_tt: # if there is a down to up transition after time tt
                next_tt = events_kept[np.argmax(events_kept > tt)]
            else:
                next_tt = Tmax # no more down to up transitions, so should be in down state until Tmax

        elif state == 1:
            # draw up state duration from truncated exponential distribution
            duration = trunc_expon.rvs(1)[0]
            durations.append(duration)
            # print("duration:", duration)
            next_tt = tt + duration

        # print('tt:', tt, 'next_tt:,', next_tt)

        # print("index:", round(tt), type(round(tt)))

        st[int(round(tt)):] = state

        state = 1 - state
        tt = next_tt

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
    rates = np.where(x, mu[1], mu[0]).astype(float)
    # print(rates)

    # add random elements to rates later (OU process)
    # simulate an OU process with sd = 1 and add it to the rates
    z = np.zeros(L)
    z0 = 0  # set baseline to 0 - OU process will tend to this
    z[0] = np.random.normal()  # sample initial condition from stationary distribution of OU process
    Q = np.sqrt(2 / tau)

    for i in range(1, L):
        z[i] = z[i - 1] + ((z0 - z[i-1]) * dt_rates / tau) + Q * np.random.normal() * np.sqrt(dt_rates)

    sds = np.where(x, sd[1], sd[0])
    rates += z * sds

    # plt.plot(z)
    # plt.show()

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
    plt.plot(rates)
    plt.show()
    fig, ax = plt.subplots()
    ax.grid(False)
    plt.scatter(spt[:, 1], spt[:, 0], s=1, marker='.')
    rates_2d = np.tile(rates, (N, 1))
    im = ax.imshow(rates_2d, aspect='auto', cmap='Reds', origin='lower')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Neuron spiking events as firing rate moves \n between the background and elevated state")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron number")
    plt.show()

