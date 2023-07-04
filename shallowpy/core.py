import matplotlib.pyplot as plt
import numpy as np

# ### main run function


def run_model(model, W0, tmax, dx, plot_fig=True, dN_fig=200, dt_save=None, x=None, Z=None):
    if dt_save is None:
        dt_save = tmax/100
    #
    # Initialization
    W = np.copy(W0)
    t = 0  # time tracking
    Nt = 0  # time steps
    #
    U_save = [W[:-1, :]]
    t_save = [0]
    #
    if plot_fig:
        fig, axarr, lines = init_plot(W0, x)
    #
    # Running simulation
    while t <= tmax:
        # update time step
        W, dt = model.temporalstep(W, dx)
        t += dt
        Nt += 1
        # update saved time steps
        if (t - t_save[-1]) >= dt_save:
            t_save.append(t)
            U_save.append(W[:-1, :])
        # update interactive figure
        if plot_fig & (Nt % dN_fig == 0):
            plt.suptitle('{:.1e} s, {:0d}'.format(t, Nt))
            lines = update_plot(axarr, W, x, lines)
    return np.array(U_save), np.array(t_save)


# %% plot functions

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_allvar(axarr, W, x):
    lines = [ax.plot(x, w, color=color)[0]
             for ax, w, color in zip(axarr.flatten(), W, color_cycle)]
    return lines


def clearlines(lines):
    for l in lines:
        l.remove()


def update_plot(axarr, W, x, lines):
    clearlines(lines)
    lines = plot_allvar(axarr, W, x)
    plt.pause(0.005)
    return lines


def init_plot(W0, x):
    fig, axarr = plt.subplots(
        W0.shape[0], 1, constrained_layout=True, sharex=True)
    #
    axarr.flatten()[-1].set_xlabel('Horizontal coordinate [m]')
    for i, ax in enumerate(axarr.flatten()):
        ax.set_ylabel('W[{:0d}]'.format(i))
    #
    lines = plot_allvar(axarr, W0, x)
    return fig, axarr, lines


# #### default parameters

default_pars = {
    'g': 9.81,
    'r': 0.95,
    'theta': 1,
    'hmin': 1e-10,
    'epsilon': 1.e-15,
    'dt_fact': 0.5,
}
