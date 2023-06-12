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

"""

import numpy as np

# #### model specific functions


def F(W_int, g, r):
    return np.swapaxes(
        np.array([W_int[1, ...],
                  W_int[1, ...]**2/W_int[0, ...] + (g/2)*W_int[0, ...]**2,
                  W_int[3, ...],
                  W_int[3, ...]**2/W_int[2, ...] + (g/2)*W_int[2, ...]**2,
                  ]),
        0, 1)


def B_func(W, W_int, g, r):
    l1 = -g*W[0, 1:-1]*(W_int[2, 1, 1:] - W_int[2, 0, :-1])
    l2 = -g*r*W[2, 1:-1]*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def S_func(W, W_int, g):
    l1 = -g*W[0, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
    l2 = -g*W[2, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def Bpsi_int_func(W_int, g, r):
    l1 = -(g/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
        (W_int[2, 0, :] - W_int[2, 1, :])
    l2 = -(g*r/2)*(W_int[2, 0, :] + W_int[2, 1, :]) * \
        (W_int[0, 0, :] - W_int[0, 1, :])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def Spsi_int_func(W_int, g, r):
    l1 = -(g/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
        (W_int[-1, 0, :] - W_int[-1, 1, :])
    l2 = -(g/2)*(W_int[2, 0, :] + W_int[2, 1, :]) * \
        (W_int[-1, 0, :] - W_int[-1, 1, :])
    return np.array([np.zeros_like(l1), l1, np.zeros_like(l1), l2])


def Ainv_int_func(W_int, g, r):
    zero = np.zeros_like(W_int[0, 0, :])
    one = np.ones_like(W_int[0, 0, :])
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

# #### General functions


def H(Fluxes, a_int, W_int, Ainv_int, Spsi_int):
    #
    return (a_int[0, :]*Fluxes[1, ...]
            - a_int[1, :]*Fluxes[0, ...]
            + a_int[0, :]*a_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :]
                                       - np.einsum('ikj,kj -> ij', Ainv_int, Spsi_int))) / (a_int[0, :] - a_int[1, :])


# def minmod(alpha, beta):
#     return (np.sign(alpha) + np.sign(beta))/2 * np.min(np.array([np.abs(alpha), np.abs(beta)]), axis=0)


# def Variables_int(var, dx):
#     alpha = (var[:, 2:] - var[:, 1:-1])/dx
#     beta = (var[:, 1:-1] - var[:, :-2])/dx
#     var_x = minmod(alpha, beta)
#     var_x = np.concatenate([(var[:, 1:2] - var[:, 0:1])/dx,
#                             var_x,
#                             (var[:, -1:] - var[:, -2:-1])/dx], axis=1)
#     #
#     var_m_int = var[:, 0:-1] + var_x[:, 0:-1]*dx/2
#     var_p_int = var[:, 1:] - var_x[:, 1:]*dx/2
#     return np.swapaxes(np.array([var_p_int, var_m_int]), 0, 1)


def Variables_int(var, dx, theta):
    # ##### Minmod limiter for interpolation
    zk = np.array([theta*(var[:, 1:-1] - var[:, :-2]),
                   (var[:, 2:] - var[:, :-2])/2,
                   theta*(var[:, 2:] - var[:, 1:-1])])
    A = np.array([np.min(zk, axis=0), np.max(zk, axis=0)])
    var_x = np.concatenate([(var[:, 1:2] - var[:, 0:1]),
                            A[0, ...]*(A[0, ...] > 0) +
                            A[1, ...]*(A[1, ...] < 0),
                            (var[:, -1:] - var[:, -2:-1])], axis=1)/dx
    #
    var_m_int = var[:, :-1] + var_x[:, :-1]*dx/2
    var_p_int = var[:, 1:] - var_x[:, 1:]*dx/2
    #
    return np.swapaxes(np.array([var_p_int, var_m_int]), 0, 1)


def RHSS_func(B, S, Bpsi_int, Spsi_int, a_int):
    jump_part = (a_int[1, 1:]*(Bpsi_int[:, 1:] + Spsi_int[:, 1:])) / (a_int[0, 1:] - a_int[1, 1:]) - \
        (a_int[0, :-1]*(Bpsi_int[:, :-1] + Spsi_int[:, :-1])) / \
        (a_int[0, :-1] - a_int[1, :-1])
    #
    centered_part = - B - S
    return centered_part + jump_part


def temporalStep(W, g, r, dx, theta):
    # Compute intercell variables
    W_int = Variables_int(W, dx, theta)
    # Compute Local speeds
    a_int, dtmax = LocalSpeeds(W_int, g, dx)
    # Compute intermediate matrices
    Ainv_int = Ainv_int_func(W_int, g, r)
    Bpsi_int, Spsi_int = Bpsi_int_func(W_int, g, r), Spsi_int_func(W_int, g, r)
    B, S = B_func(W, W_int, g, r), S_func(W, W_int, g)
    # Compute Fluxes
    Fluxes = F(W_int, g, r)
    H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
    # Compute sources
    RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, a_int)
    # breakpoint()
    # #### Computing right hand side
    # return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
    return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
