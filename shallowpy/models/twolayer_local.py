"""
Here we solve the following system of equations:
    - d[h1]/dt + d[q1]/dx = 0
    - d[u1]/dt + d[u1**2/2 + g*(h1 + h2 + Z)]/dx = 0
    - d[h2]/dt + d[q2]/dt = 0
    - d[u2]/dt + d[u2**2/2 + g*(r*h1 + h2 + Z)]/dx = 0

with:
    - r = rho1/rho2 

variables:
    - U  = [h1, q1, h2, q2]:
    - W = [h1, u1, h2, u2, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (5, Nx)

dim(W_int) = (5, 2, Nx):
    - 2: [h1, u1, h2, u2, Z]
    - 2: [pos, min]

REFERENCE: Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

"""

import numpy as np

from .general import H, RHSS_func, Variables_int

# #### model specific functions


def F(W_int, g, r):
    return np.swapaxes(
        np.array([W_int[0, ...]*W_int[1, ...],
                  W_int[1, ...]**2/2 + g *
                      (W_int[0, ...] + W_int[2, ...] + W_int[-1, ...]),
                  W_int[2, ...]*W_int[3, ...],
                  W_int[3, ...]**2/2 + g *
                      (r*W_int[0, ...] + W_int[2, ...] + W_int[-1, ...]),
                  ]),
        0, 1)


def Ainv_int_func(W_int, g, r):
    zero = np.zeros_like(W_int[0, 0, :])
    one = np.ones_like(W_int[0, 0, :])
    #
    l1 = np.array(
        [zero, one/(g*(1-r)), 2/((1-r)*(W_int[2, 0, :] + W_int[2, 1, :])), one/(g*(1-r))])
    l2 = np.array([2/(W_int[0, 0, :] + W_int[0, 1, :]), zero, zero, zero])
    l3 = -l1
    l4 = np.array([zero, zero, 2/(W_int[2, 0, :] + W_int[2, 1, :]), zero])
    return np.array([l1, l2, l3, l4])


def LocalSpeeds(W_int, g, dx):
    h2_int = W_int[2, ...] - W_int[4, ...]
    um = (W_int[1, ...] + W_int[3, ...])/(W_int[0, ...] + h2_int)
    #
    ap_int = np.row_stack(
        (um + np.sqrt(g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).max(axis=0)
    am_int = np.row_stack(
        (um - np.sqrt(g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).min(axis=0)
    return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))

# #### Spatial discretization step


def temporalStep(W, g, r, dx, theta):
    # Compute intercell variables
    W_int = Variables_int(W, dx, theta)
    # Compute Local speeds
    a_int, dtmax = LocalSpeeds(W_int, g, dx)
    # Compute intermediate matrices
    Ainv_int = Ainv_int_func(W_int, g, r)
    B, S = np.zeros_like(W[:-1, 1:-1]), np.zeros_like(W[:-1, 1:-1])
    Bpsi_int, Spsi_int = np.zeros_like(
        W_int[:-1, 0, :]), np.zeros_like(W_int[:-1, 0, :])
    # Compute Fluxes
    Fluxes = F(W_int, g, r)
    H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
    # Compute sources
    # no sources
    # Computing right hand side
    return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]), dtmax
