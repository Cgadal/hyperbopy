import numpy as np
from twolayerSW.core import temporalStep
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def run_model(W0, Nt, dx, g=9.81, r=1.2, theta=1, plot_fig=True, dN_fig=200, x=None, Z=None):
    W = np.copy(W0)
    #
    if plot_fig:
        fig, axarr = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        axarr[0].plot(x, Z, 'k')
        #
        w, = axarr[0].plot([], [])
        h1, = axarr[0].plot([], [])
        q1, = axarr[1].plot([], [])
        q2, = axarr[1].plot([], [])
        #
        axarr[1].set_xlabel('Horizontal coordinate [m]')
        axarr[0].set_ylabel('h [m]')
        axarr[1].set_ylabel('q [m/s]')
    #
    for n in range(Nt):
        # # Computing RightHandSide
        RHS, dtmax = temporalStep(W, g, r, dx, theta)
        # # Euler in time
        dt = dtmax/100
        W[:-1, 1:-1] = W[:-1, 1:-1] + dt*RHS
        # # Applying boundary conditions
        W[0, 0], W[0, -1] = W[0, 1], W[0, -2]  # h1
        W[2, 0], W[2, -1] = W[2, 1], W[2, -2]  # w
        W[1, 0], W[1, -1] = 0, 0  # q1
        W[3, 0], W[3, -1] = 0, 0  # q2
        #
        if plot_fig & (n % dN_fig == 0):
            plt.suptitle('{:0d}'.format(n))
            h1.remove()
            w.remove()
            q1.remove()
            q2.remove()
            #
            w, = axarr[0].plot(x, W[2, :], color='tab:orange')
            h1, = axarr[0].plot(x, W[0, :] + W[2, :], color='tab:blue')
            q1, = axarr[1].plot(x, W[1, :], color='tab:blue')
            q2, = axarr[1].plot(x, W[3, :], color='tab:orange')
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
    x0 = L/2 # lock center [m]

    # ## Initial condition
    # Bottom topography
    Z = np.zeros_like(x)

    # layer 2 (lower)
    h2 = np.ones_like(x)*0.2
    h2[(x >= x0 - l0/2) & ((x <= x0 + l0/2))] = h2[(x >= x0 - l0/2) & ((x <= x0 + l0/2))] + h0
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
