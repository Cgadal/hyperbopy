"""
Here we solve the following system of equations:
    - d[h]/dt + d[hu]/dx = 0
    - d[u]/dt + d[g*(h + Z)]/dx = -u*d[u]/dx

variables:
    - U  = [h, u]:
    - W = [h, u, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (3, Nx)

dim(W_int) = (3, 2, Nx):
    - 2: [h, u, Z]
    - 2: [pos, min]

REFERENCE: 
    - Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.
    - Fyhn, E. H., Lervåg, K. Y., Ervik, Å., & Wilhelmsen, Ø. (2019). A consistent reduction of the two-layer shallow-water equations to an accurate one-layer spreading model. Physics of Fluids, 31(12), 122103.

"""

import numpy as np

from .general import H, RHSS_func, Variables_int

# #### model specific functions

def F(W_int, g):
    return np.swapaxes(
        np.array([W_int[0, ...]*W_int[1, ...],
                  g*(W_int[0, ...] + W_int[-1, ...]),
                  ]),
        0, 1)


def B_func(W, W_int):
    l1 = -(1/2)*W[1, 1:-1]*(W_int[1, 1, 1:] - W_int[1, 0, :-1])
    return np.array([np.zeros_like(l1), l1])


def Bpsi_int_func(W_int):
    l1 = -(1/2)*(W_int[1, 0, :] + W_int[1, 1, :]) * \
        (W_int[1, 0, :] - W_int[1, 1, :])
    return np.array([np.zeros_like(l1), l1])


def Ainv_int_func(W_int, g):
    zero = np.zeros_like(W_int[0, 0, :])
    one = np.ones_like(W_int[0, 0, :])
    #
    l1 = np.array([zero, one/g])
    l2 = np.array([2/(W_int[0, 0, :] + W_int[0, 1, :]), zero])
    return np.array([l1, l2])


def LocalSpeeds(h_int, u_int, g, dx):
    ap_int = np.row_stack((u_int + np.sqrt(g*h_int),
                          np.zeros_like(u_int[0, :]))).max(axis=0)
    am_int = np.row_stack((u_int - np.sqrt(g*h_int),
                          np.zeros_like(u_int[0, :]))).min(axis=0)
    return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))


# #### Spatial discretization step

def temporalStep(W, g, r, dx, theta, epsilon=1.e-15):
    # Compute intercell variables
    W_int = Variables_int(W, dx, theta)
    # Compute Local speeds
    a_int, dtmax = LocalSpeeds(W_int[0, ...], W_int[1, ...], g*(1-r), dx)
    # Compute intermediate matrices
    Ainv_int = Ainv_int_func(W_int, g*(1-r))
    B, Bpsi_int = B_func(W, W_int), Bpsi_int_func(W_int)
    S, Spsi_int = np.zeros_like(B), np.zeros_like(Bpsi_int)
    # Compute Fluxes
    Fluxes = F(W_int, g*(1-r))
    H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
    # Compute sources
    RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, a_int)
    # #### Computing right hand side
    return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax