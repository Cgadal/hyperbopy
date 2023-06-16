import matplotlib.pyplot as plt
import numpy as np

from shallowpy.models.twolayer_layerwise import temporalStep as temporalStep_2L_layerwise
from shallowpy.models.twolayer_local import temporalStep as temporalStep_2L_local
from shallowpy.models.onelayer_global import temporalStep as temporalStep_1L_global
from shallowpy.models.onelayer_local import temporalStep as temporalStep_1L_local


dic_models = {'2L_layerwise': temporalStep_2L_layerwise,
              '2L_local': temporalStep_2L_local,
              '1L_global': temporalStep_1L_global,
              '1L_local': temporalStep_1L_local}


def finder(model):
    if model in dic_models.keys():
        return dic_models[model]
    else:
        raise ValueError(
            "'{}' not recognised, check implemented models".format(model)) from None


def update_step(temporalStep, W, g, r, dx, theta, dt_fact=0.5, dt=None):
    RHS, dtmax = temporalStep(W, g, r, dx, theta)
    if dt is None:
        dt = dtmax*dt_fact
    #
    W_next = np.copy(W)
    W_next[:-1, 1:-1] = W[:-1, 1:-1] + dt*RHS
    # boundary conditions (0 for q, reflective fo h)
    W_up = np.hstack([np.array([W_next[2*i, 1], 0])
                     for i in range((W_next.shape[0] - 1)//2)])
    W_down = np.hstack([np.array([W_next[2*i, -2], 0])
                       for i in range((W_next.shape[0] - 1)//2)])
    W_next[:-1, 0], W_next[:-1, -1] = W_up, W_down
    return W_next, dt


def run_model(model, W0, tmax, dx, g=9.81, r=0.95, theta=1, plot_fig=True, dN_fig=200,
              dt_save=None, x=None, Z=None, dt_fact=0.5):
    if dt_save is None:
        dt_save = tmax/100
    temporalStep = finder(model)
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
    while t <= tmax:
        # # Computing RightHandSide
        # # # Euler in time
        # W = update_step(W, g, r, dx, theta, dt_fact)
        #
        # # (3, 3) eSSPRK(3,3)
        w1, dt = update_step(temporalStep, W, g, r, dx, theta, dt_fact)
        w2 = (3/4)*W + (1/4)*update_step(temporalStep,
                                         w1, g, r, dx, theta, dt=dt)[0]
        W = (1/3)*W + (2/3)*update_step(temporalStep,
                                        w2, g, r, dx, theta, dt=dt)[0]
        #
        t += dt
        Nt += 1
        #
        if (t - t_save[-1]) >= dt_save:
            t_save.append(t)
            U_save.append(W[:-1, :])
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
