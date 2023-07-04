"""
Here we solve the following system of equations:
    - d[h1]/dt + d[q1]/dx = 0
    - d[q1]/dt + d[q1**2/h1 + (g/2)*h1]/dx = -g*h1*(d[h2 + Z]/dx)
    - d[h2]/dt + d[q2]/dt = 0
    - d[q2]/dt + d[q2**2/h2 + (g/2)*h2**2]/dx = -g*h2*(d[r*h1 + Z]/dx)

with:
    - r = rho1/rho2 

variables:
    - U  = [h1, q1, h2, q2]:
    - W = [h1, q1, h2, q2, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (5, Nx)

dim(W_int) = (5, 2, Nx):
    - 2: [h1, q1, h2, q2, Z]
    - 2: [pos, min]

REFERENCE: Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

"""

import numpy as np

from ..general import model

# #### model specific functions


def F(W_int, g, r):
    return np.swapaxes(
        np.array([W_int[1, ...],
                  W_int[1, ...]**2/W_int[0, ...] + (g/2)*W_int[0, ...]**2,
                  W_int[3, ...],
                  W_int[3, ...]**2/W_int[2, ...] + (g/2)*W_int[2, ...]**2,
                  ]),
        0, 1)


def B(W, W_int, g, r):
    l1 = -g*W[0, 1:-1]*(W_int[2, 1, 1:] - W_int[2, 0, :-1])
    l2 = -g*r*W[2, 1:-1]*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def S(W, W_int, g):
    l1 = -g*W[0, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
    l2 = -g*W[2, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def Bpsi_int(W_int, g, r):
    l1 = -(g/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
        (W_int[2, 0, :] - W_int[2, 1, :])
    l2 = -(g*r/2)*(W_int[2, 0, :] + W_int[2, 1, :]) * \
        (W_int[0, 0, :] - W_int[0, 1, :])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def Spsi_int(W_int, g, **kwargs):
    l1 = -(g/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
        (W_int[-1, 0, :] - W_int[-1, 1, :])
    l2 = -(g/2)*(W_int[2, 0, :] + W_int[2, 1, :]) * \
        (W_int[-1, 0, :] - W_int[-1, 1, :])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def Ainv_int(W_int, g, r):
    zero = np.zeros_like(W_int[0, 0, :])
    one = np.ones_like(W_int[0, 0, :])
    #
    l1 = np.array([zero, 2/(g*(1-r)*(W_int[0, 0, :] + W_int[0, 1, :])),
                  zero, 2/(g*(r-1)*(W_int[2, 0, :] + W_int[2, 1, :]))])
    l2 = np.array([one, zero, zero, zero])
    l3 = -l1
    l4 = np.array([zero, zero, one, zero])
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


# #### model class object
phypars_default = {'g': 9.81, 'r': 0.95}

Model = model('2L_layerwise', phypars_default,
              F, Ainv_int, S, B, Spsi_int, Bpsi_int, LocalSpeeds)


# #### Spatial discretization step


# def temporalStep(W, g, r, dx, theta):
#     # Compute intercell variables
#     W_int = Variables_int(W, dx, theta)
#     # Compute Local speeds
#     a_int, dtmax = LocalSpeeds(W_int, g, dx)
#     # Compute intermediate matrices
#     Ainv_int = Ainv_int_func(W_int, g, r)
#     Bpsi_int, Spsi_int = Bpsi_int_func(W_int, g, r), Spsi_int_func(W_int, g, r)
#     B, S = B_func(W, W_int, g, r), S_func(W, W_int, g)
#     # Compute Fluxes
#     Fluxes = F(W_int, g, r)
#     H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
#     # Compute sources
#     RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, a_int)
#     # Computing right hand side
#     return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
