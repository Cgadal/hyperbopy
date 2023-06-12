"""
General functions for the spatial discretization

"""

import numpy as np

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

