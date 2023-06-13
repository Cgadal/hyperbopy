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


def update_step(temporalStep, W, g, r, dx, theta, dt_fact):
    RHS, dtmax = temporalStep(W, g, r, dx, theta)
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
        fig, axarr = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        axarr[0].plot(x, Z, 'k')
        #
        lines = [axarr[0].plot([], [])[0] for i in range(W.shape[0] - 1)]
        #
        axarr[1].set_xlabel('Horizontal coordinate [m]')
        axarr[0].set_ylabel('h [m]')
        axarr[1].set_ylabel('u [m/s]')
    #
    while t <= tmax:
        # # Computing RightHandSide
        # # # Euler in time
        # W = update_step(W, g, r, dx, theta, dt_fact)
        #
        # # (3, 3) eSSPRK(3,3)
        w1, dt = update_step(temporalStep, W, g, r, dx, theta, dt_fact)
        w2 = (3/4)*W + (1/4)*update_step(temporalStep,
                                         w1, g, r, dx, theta, dt_fact)[0]
        W = (1/3)*W + (2/3)*update_step(temporalStep,
                                        w2, g, r, dx, theta, dt_fact)[0]
        #
        t += dt
        Nt += 1
        #
        if (t - t_save[-1]) >= dt_save:
            t_save.append(t)
            U_save.append(W[:-1, :])
        if plot_fig & (Nt % dN_fig == 0):
            plt.suptitle('{:.1e} s, {:0d}'.format(t, Nt))
            for l in lines:
                l.remove()
            lines = []
            axarr[0].set_prop_cycle(None)
            axarr[1].set_prop_cycle(None)
            for i in range((W.shape[0] - 1)//2):
                # q plots
                h, = axarr[0].plot(x, np.sum(W[::-2, :][:2*(i+1), :], axis=0))
                q, = axarr[1].plot(x, W[2*i + 1, :]/W[2*i, :])
                #
                lines.append(h)
                lines.append(q)
            plt.pause(0.005)
    return np.array(U_save), np.array(t_save)
