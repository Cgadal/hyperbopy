import matplotlib.pyplot as plt
import numpy as np

from shallowpy.models.twolayer_layerwise import temporalStep as temporalStep_2L_layerwise

def finder(model):
    if model == '2L_layerwise':
        return temporalStep_2L_layerwise
    else:
        raise ValueError("'{}' not recognised, check implemented models".format(model)) from None 


def update_step(temporalStep, W, g, r, dx, theta, dt_fact):
    RHS, dtmax = temporalStep(W, g, r, dx, theta)
    dt = dtmax*dt_fact
    #
    W_next = np.copy(W)
    W_next[:-1, 1:-1] = W[:-1, 1:-1] + dt*RHS
    # boundary conditions
    W_up = np.array([W_next[0, 1], 0, W_next[2, 1], 0])
    W_down = np.array([W_next[0, -2], 0, W_next[2, -2], 0])
    W_next[:-1, 0], W_next[:-1, -1] = W_up, W_down
    return W_next, dt


def run_model(model, W0, tmax, dx, g=9.81, r=1.2, theta=1, plot_fig=True, dN_fig=200,
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
        h2, = axarr[0].plot([], [])
        h1, = axarr[0].plot([], [])
        u1, = axarr[1].plot([], [])
        u2, = axarr[1].plot([], [])
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
        w2 = (3/4)*W + (1/4)*update_step(temporalStep, w1, g, r, dx, theta, dt_fact)[0]
        W = (1/3)*W + (2/3)*update_step(temporalStep, w2, g, r, dx, theta, dt_fact)[0]
        #
        t += dt
        Nt += 1
        #
        if (t - t_save[-1]) >= dt_save:
            t_save.append(t)
            U_save.append(W[:-1, :])
        if plot_fig & (Nt % dN_fig == 0):
            plt.suptitle('{:.1e} s, {:0d}'.format(t, Nt))
            h1.remove()
            h2.remove()
            u1.remove()
            u2.remove()
            #
            h2, = axarr[0].plot(x, W[2, :] + W[-1, :], color='tab:orange')
            h1, = axarr[0].plot(x, W[0, :] + W[2, :] +
                                W[-1, :], color='tab:blue')
            u1, = axarr[1].plot(x, W[1, :]/W[0, :], color='tab:blue')
            u2, = axarr[1].plot(x, W[3, :]/W[0, :], color='tab:orange')
            plt.pause(0.005)
    return np.array(U_save), np.array(t_save)
