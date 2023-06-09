import numpy as np
from twolayerSW.core import temporalStep
import matplotlib.pyplot as plt


def update_step(W, g, r, dx, theta, dt_fact):
    RHS, dtmax = temporalStep(W, g, r, dx, theta)
    dt = dtmax*dt_fact
    #
    W_next = np.copy(W)
    W_next[:-1, 1:-1] = W[:-1, 1:-1] + dt*RHS
    # boundary conditions
    W_up = np.array([W_next[0, 1], 0, W_next[2, 1], 0])
    W_down = np.array([W_next[0, -2], 0, W_next[2, -2], 0])
    W_next[:-1, 0], W_next[:-1, -1] = W_up, W_down
    return W_next


def run_model(W0, Nt, dx, g=9.81, r=1.2, theta=1, plot_fig=True, dN_fig=200, x=None, Z=None, dt_fact=0.5):
    W = np.copy(W0)
    #
    if plot_fig:
        fig, axarr = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        axarr[0].plot(x, Z, 'k')
        #
        w, = axarr[0].plot([], [])
        h1, = axarr[0].plot([], [])
        u1, = axarr[1].plot([], [])
        u2, = axarr[1].plot([], [])
        #
        axarr[1].set_xlabel('Horizontal coordinate [m]')
        axarr[0].set_ylabel('h [m]')
        axarr[1].set_ylabel('u [m/s]')
    #
    for n in range(Nt):
        # # Computing RightHandSide
        # # # Euler in time
        # W = update_step(W, g, r, dx, theta, dt_fact)
        #
        # # (3, 3) eSSPRK(3,3)
        w1 = update_step(W, g, r, dx, theta, dt_fact)
        w2 = (3/4)*W + (1/4)*update_step(w1, g, r, dx, theta, dt_fact)
        W = (1/3)*W + (2/3)*update_step(w2, g, r, dx, theta, dt_fact)
        #
        if plot_fig & (n % dN_fig == 0):
            plt.suptitle('{:0d}'.format(n))
            h1.remove()
            w.remove()
            u1.remove()
            u2.remove()
            #
            w, = axarr[0].plot(x, W[2, :], color='tab:orange')
            h1, = axarr[0].plot(x, W[0, :] + W[2, :], color='tab:blue')
            u1, = axarr[1].plot(x, W[1, :], color='tab:blue')
            u2, = axarr[1].plot(x, W[3, :], color='tab:orange')
            # u1, = axarr[1].plot(x, W[1, :]/W[0, :], color='tab:blue')
            # u2, = axarr[1].plot(
            #     x, W[3, :]/(W[2, :] - W[-1, :]), color='tab:orange')
            plt.pause(0.005)


if __name__ == '__main__':
    # ## Domain size
    L = 3     # domain length [m]
    H = 0.5   # water height [m]

    # ## Grid parameters
    Nt = 5  # time steps number
    Nx = 10000  # spatial grid points number (evenly spaced)
    x = np.linspace(0, L, Nx)
    dx = L/(Nx - 1)

    # ## Numerical parameters
    theta = 1

    # ## Reservoir parameters
    h0 = 0.1  # lock height [m]
    l0 = 0.5  # lock length [m]
    x0 = L/2  # lock center [m]

    # ## Initial condition
    # Bottom topography
    Z = np.zeros_like(x)

    # layer 2 (lower)
    h2 = np.ones_like(x)*0.2
    h2[(x >= x0 - l0/2) & ((x <= x0 + l0/2))
       ] = h2[(x >= x0 - l0/2) & ((x <= x0 + l0/2))] + h0
    q2 = np.zeros_like(x)

    # layer 1 (upper)
    h1 = H - h2
    q1 = np.zeros_like(x)

    # h1[h1 < hmin] = hmin
    # h2[h2 < hmin] = hmin

    w = h2 + Z
    W0 = np.array([h1, q1, w, q2, Z])

    # %% Run model
    run_model(W0, Nt, dx, g=9.81, r=1.2, plot_fig=True,
              dN_fig=1, x=x, Z=Z, theta=theta)
