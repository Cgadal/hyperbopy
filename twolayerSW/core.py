"""
Here we solve the following system of equations:
    - d[h1]/dt + d[q1]/dx = 0
    - d[q1]/dt + d[q1**2/h1 + g*e*h1]/dx = -g*e*(d[h1]/dx)
    - d[w]/dt + d[q2]/dt = 0
    - d[q2]/dt + d[q2**2/(w - Z) + g*(w**2 - r*h1**2)/2 - g*e_c*Z]/dx = -g*r*e*(d[h1]/dx) - g*e_c*d[Z]/dx

with:
    - w = h2 + Z
    - e = h1 + w
    - e_c = r*h1 + w
    - r = rho1/rho2 

variables:
    - U  = [h1, q1, w, q2]:
    - W = [h1, q1, w, q2, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (5, Nx)

dim(W_int) = (5, 2, Nx):
    - 2: [h1, q1, w, q2, Z
    - 2: [pos, min]

"""

import numpy as np

# #### model specific functions


def F(W_int, g, r):
    return np.swapaxes(
        np.array([W_int[1, ...],
                  W_int[1, ...]**2/W_int[0, ...] + g *
                      (W_int[0, ...] + W_int[2, ...])*W_int[0, ...],
                  W_int[3, ...],
                  W_int[3, ...]**2/(W_int[2, ...] - W_int[-1, ...]) +
                  (g/2)*(W_int[2, ...]**2 - r*W_int[0, ...]**2)
                  - g*(r*W_int[0, ...] + W_int[2, ...])*W_int[4, ...]
                  ]),
        0, 1)


def Bpsi_int_func(W_int, g, r):
    l = (g/2)*(W_int[0, 0, :] + W_int[2, 0, :]
               + W_int[0, 1, :] + W_int[2, 1, :])*(W_int[1, 0, :] - W_int[1, 1, :])
    return np.array([np.zeros_like(l), l, np.zeros_like(l), -l*r/2])


def Spsi_int_func(W_int, g, r):
    l = (g/2)*(r*W_int[0, 0, :] + (W_int[2, 0, :])
               + r*W_int[0, 1, :] + W_int[2, 1, :])*(W_int[-1, 0, :] - W_int[-1, 1, :])
    return np.array([np.zeros_like(l), np.zeros_like(l), np.zeros_like(l), -l])


# def B_func(W, W_int, g, r):
#     l = (g/2)*(W_int[0, 1, 1:] + W_int[0, 0, :-1]
#                + W_int[2, 1, 1:] + W_int[2, 0, :-1])*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
#     return np.array([np.zeros_like(l), l, np.zeros_like(l), -r*l])

# def S_func(W, W_int, g, r):
#     l = (g/2)*(r*(W_int[0, 1, 1:] + W_int[0, 0, :-1])
#                + W_int[2, 1, 1:] + W_int[2, 0, :-1])*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
#     return np.array([np.zeros_like(l), np.zeros_like(l), np.zeros_like(l), -l])


def B_func(W, W_int, g, r):
    l = g*(W[0, 1:-1] + W[2, 1:-1])*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
    return np.array([np.zeros_like(l), l, np.zeros_like(l), -r*l])


def S_func(W, W_int, g, r):
    l = g*(r*W[0, 1:-1] + W[2, 1:-1]) * (W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
    return np.array([np.zeros_like(l), np.zeros_like(l), np.zeros_like(l), -l])


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


def H(Fluxes, A_int, W_int):
    #
    return (A_int[0, :]*Fluxes[1, ...]
            - A_int[1, :]*Fluxes[0, ...]
            + A_int[0, :]*A_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :])) / (A_int[0, :]
                                                                                - A_int[1, :])


def minmod(alpha, beta):
    return (np.sign(alpha) + np.sign(beta))/2 * np.min(np.array([np.abs(alpha), np.abs(beta)]), axis=0)


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


def RHSS_func(B, S, Bpsi_int, Spsi_int, A_int):
    jump_part = (A_int[1, 1:]*(Bpsi_int[:, 1:] + Spsi_int[:, 1:])) / (A_int[0, 1:] - A_int[1, 1:]) - \
        (A_int[0, :-1]*(Bpsi_int[:, :-1] + Spsi_int[:, :-1])) / \
        (A_int[0, :-1] - A_int[1, :-1])
    #
    centered_part = - B - S
    return centered_part + jump_part


def temporalStep(W, g, r, dx, theta):
    # Compute intercell variables
    W_int = Variables_int(W, dx, theta)
    # Compute Local speeds
    A_int, dtmax = LocalSpeeds(W_int, g, dx)
    # Compute Fluxes
    Fluxes = F(W_int, g, r)
    H_int = H(Fluxes, A_int, W_int)
    # Compute sources
    Bpsi_int, Spsi_int = Bpsi_int_func(W_int, g, r), Spsi_int_func(W_int, g, r)
    B, S = B_func(W, W_int, g, r), S_func(W, W_int, g, r)
    RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, A_int)
    # breakpoint()
    # #### Computing right hand side
    # return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
    return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
