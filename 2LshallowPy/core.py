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


# dim(U) = (4, Nx)

# dim(U_int) = (2, 2, Nx):
#     - 2: [w, hu]
#     - 2: [pos, min]

# dim(A) = (2, Nx)
"""

import numpy as np

# #### model specific functions

def F(W_int, g, r):
    return np.swapaxes(np.array([W_int[1, ...], 
                                W_int[1, ...]**2/W_int[0, ...] + g*(W_int[0, ...] + W_int[2, ...])*W_int[0, ...],
                                 W_int[3, ...],
                                W_int[3, ...]**2/(W_int[2, ...]- W_int[4, ...]) + (g/2)*(W_int[2, ...]**2 - r*W_int[1, ...]**2) - g*(r*W_int[1, ...] + W_int[2, ...])*W_int[4, ...]
                                ]), 
                        0, 1)


def Bpsi_int_func(W_int, g, r):
    l = (g/2)*(W_int[0, 0, :] + W_int[2, 0, :] + W_int[0, 1, :] + W_int[2, 1, :])*(W_int[1, 0, :] - W_int[1, 1, :])
    return np.array([0, l, 0, -l*r/2])

def Spsi_int_func(W_int, g, r):
    l = (-g/2)*(r*W_int[0, 0, :] + (W_int[2, 0, :]) + r*W_int[0, 1, :] + W_int[2, 1, :])*(W_int[-1, 0, :] - W_int[-1, 1, :])
    return np.array([0, 0, 0, l])

def LocalSpeeds(W_int, g, dx):
    h2_int = W_int[2, ...] - W_int[4, ...]
    um = (W_int[1, ...] + W_int[3, ...])/(W_int[0, ...] + h2_int)
    #
    ap_int = np.row_stack((um + np.sqrt(g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).max(axis=0)
    am_int = np.row_stack((um - np.sqrt(g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).min(axis=0)
    return np.array([ap_int, am_int]), dx/4./np.max([ap_int, -am_int])

def B_func(W, W_int, g, r):
    l = -g*(W[0, :] + W[2, :])*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
    return np.array([0, l, 0, -r*l])

def S_func(W, W_int, g, r):
    l = -g*(r*W[0, :] + W[2, :])*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
    return np.array([0, 0, 0, l])

# #### General functions
def H(Fluxes, A_int, U_int, g, r):
    #
    return (A_int[0, :]*Fluxes[1, ...]
            - A_int[1, :]*Fluxes[0, ...]
            + A_int[0, :]*A_int[1, :]*(U_int[:, 0, :] - U_int[:, 1, :])) / (A_int[0, :]
                                                                        - A_int[1, :])

def minmod(alpha, beta):
    return (np.sign(alpha) + np.sign(beta))/2 * np.min(np.abs(alpha), np.abs(beta), axis=0)

def Variables_int(var, dx):
    alpha = (var[:, 2:] - var[:, 1:-1])/dx
    beta = (var[:, 1:-1] - var[:, :-2])/dx
    var_x = minmod(alpha, beta)
    #
    var_m_int = var[:, :-1] + dx/2*var_x[:, :-1]
    var_p_int = var[:, 1:] - dx/2*var_x[:, 1:]
    return np.swapaxes(np.array([var_p_int, var_m_int]), 0, 1)


def RHSS_func(B, S, Bpsi_int, Spsi_int, A_int):
    jump_part = (A_int[1, 1:]*(Bpsi_int[1:] + Spsi_int[1:]))/ (A_int[0, 1:] - A_int[1, 1:]) - (A_int[0, :-1]*(Bpsi_int[:-1] + Spsi_int[:-1])) / (A_int[0, :-1] - A_int[1, :-1])
    #
    centered_part = - B - S   
    return centered_part + jump_part


def temporalStep(U, W, g, r, dx):
    # Compute intercell variables
    U_int = Variables_int(U)
    W_int = Variables_int(W)
    # Compute Local speeds
    A_int, dtmax = LocalSpeeds(W_int, g, dx)
    # Compute Fluxes
    Fluxes = F(W_int, g, r)
    H_int = H(Fluxes, A_int, U_int, g, r)
    # Compute sources
    Bpsi_int, Spsi_int = Bpsi_int_func(W_int, g, r), Spsi_int_func(W_int, g, r)
    B, S = B_func(W, W_int, g, r), S_func(W, W_int, g, r)
    RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, A_int)
    # #### Computing right hand side
    return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
    
