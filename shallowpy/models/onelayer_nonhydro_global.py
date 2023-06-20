"""
Here we solve the following system of equations:
    - d[w]/dt + d[q]/dx = 0
    - d[q]/dt + a_m*d[M]/dt + d[q**2/(w - Z) + (g/2)*(w - Z)]/dx = -g*(w - B)*(d[Z]/dx) + pa*d[w]/dx - a_n*N

variables:
    - U  = [w := h + Z, q := hu]:
    - W = [w, q, Z]
    - K = [h, u, q]
    
    - M = d[-h**3/3*d[u]/d[x] + (1/2)*q**2/u*d[Z]/d[x]] + d[Z]/d[x]*(-(1/2)*h**2*d[u]/dx + d[Z]/d[x]*q)
    - N = d[ d[h**2]/dt * (hd[u]/dx - d[Z]/d[x] * u)] + 2*d[Z]/d[x]*d[h]/dt*(h*d[u]/dx - d[Z]/d[x] * u) - d2[Z]/d[x,t]*(-(1/2)*h**2*d[u]/dx + d[Z]/d[x]*q)
         
    - U_int: left/right values of U


dim(U) = (2, Nx)
dim(W) = (3, Nx)

dim(W_int) = (3, 2, Nx):
    - 2: [h, q, Z]
    - 2: [pos, min]

REFERENCE: Chertock, A., Kurganov, A., & Liu, Y. (2020). Finite-volume-particle methods for the two-component Camassa-Holm system. Communications in computational physics, 27(2).

"""

import numpy as np
from scipy.linalg import solve_banded

from shallowpy.models.general import H, Variables_int


def F(W_int, g):
    return np.swapaxes(
        np.array([W_int[1, ...],
                  W_int[1, ...]**2/(W_int[0, ...] - W_int[-1, ...])
                  + (g/2)*(W_int[0, ...] - W_int[-1, ...])**2,
                  ]),
        0, 1)


def LocalSpeeds(h_int, u_int, g, dx):
    ap_int = np.row_stack((u_int + np.sqrt(g*h_int),
                          np.zeros_like(u_int[0, :]))).max(axis=0)
    am_int = np.row_stack((u_int - np.sqrt(g*h_int),
                          np.zeros_like(u_int[0, :]))).min(axis=0)
    return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))


def S_func(W_int, g, pa, dx):
    l1 = -(g/2)*(W_int[0, 1, 1:] - W_int[-1, 1, 1:]
                 + W_int[0, 0, :-1] - W_int[-1, 0, :-1])*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1]) \
        + pa*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
    return np.array([np.zeros_like(l1), l1])/dx


def N_func(u, q, uint, hint, dB, dB_int, dq, dh, dx):
    # here qint from centered diff and not limiter as in W_int
    l1 = -(2/dx)*(hint[1:]*((q[2:] - q[1:-1])/dx)*((hint[1:]*(u[2:] - u[1:-1])/dx) - dB_int[1:]*uint[1:])
                  - hint[:-1]*((q[1:-1] - q[:-2])/dx)*((hint[:-1]*(u[2:] - u[1:-1])/dx) - dB_int[1:]*uint[1:])) \
        - 2*dB[1:-1]*dq[1:-1]*(dq[1:-1] - (dh[1:-1] + dB[1:-1])*u[1:-1])

    return np.array([np.zeros_like(l1), l1])


def CheckPositivityH(W_int, W):
    Bint = (W_int[-1, 1, :] + W_int[-1, 1, :])/2
    kfind = np.argwhere(W_int[0, 1, :] < Bint)[:, 0]
    if kfind.size > 0:
        W_int[0, 1, :][kfind] = Bint[kfind]
        if kfind[0] == 0:
            #     print('h zero proche du bord gauche')
            W_int[0, 0, :][kfind[1:]-1] = 2 * \
                W[0, :][kfind[1:]] - Bint[kfind[1:]]
            W_int[0, 0, 0] = W_int[0, 0, 1]
        else:
            W_int[0, 0, :][kfind-1] = 2*W[0, :][kfind] - Bint[kfind]
    #
    kfind = np.argwhere(W_int[0, 0, :] < Bint)[:, 0]
    if kfind.size > 0:
        W_int[0, 0, :][kfind] = Bint[kfind]
        if kfind[-1] == W_int[0, 0, :].size - 1:
            # print('h zero proche du bord droit')
            W_int[0, 1, :][kfind[:-1]+1] = 2 * \
                W[0, :][kfind[:-1]+1] - Bint[kfind[:-1]]
            W_int[0, 1, -1] = W_int[0, 1, -2]
        else:
            W_int[0, 1, :][kfind+1] = 2*W[0, :][kfind+1] - Bint[kfind]


def reconstruct_u(h, q, epsilon):
    return np.sqrt(2)*h*q/np.sqrt(h**4 + np.max([h**4, epsilon*np.ones_like(h)], axis=0))


def Tau_coeffs(h, dh, hint, dB, dB_int, a_M, dx):
    down = a_M*((h[1:-1]*dB[1:-1] - hint[:-1]*dB_int[:-1]) /
                (4*dx) - hint[:-1]**3/(3*h[:-2]*dx**2))
    mid = 1 + a_M*((hint[1:]*dB_int[1:] - hint[:-1]*dB_int[:-1])/4*dx +
                   (hint[1:]**3 + hint[:-1]**3)/(3*h[1:-1]*dx**2) + dB[1:-1]*dh[1:-1]/2 + dB[1:-1]**2)
    up = a_M*((hint[1:]*dB_int[1:] - h[1:-1]*dB[1:-1]) /
              (4*dx) - hint[1:]**3/(3*h[2:]*dx**2))
    return down, mid, up


def reduced_Tau(h, dh, hint, dB, dB_int, a_M, dx):
    down, mid, up = Tau_coeffs(h, dh, hint, dB, dB_int, a_M, dx)
    return np.array([np.hstack([0, up[:-1]]),
                     mid,
                     np.hstack([down[1:], 0])
                     ])


def LHS_func(q, h, dh, hint, dB, dB_int, a_M, dx):
    # here h is full vector but q := q[1:-1]
    down, mid, up = Tau_coeffs(h, dh, hint, dB, dB_int, a_M, dx)
    return np.hstack([0, q[:-1]*down[1:]]) + q*mid + np.hstack([q[1:]*up[:-1], 0])


def minmod_diff(var, dx, theta):
    # 1d vector
    zk = np.array([theta*(var[1:-1] - var[:-2]),
                   (var[2:] - var[:-2])/2,
                   theta*(var[2:] - var[1:-1])])
    A = np.array([np.min(zk, axis=0), np.max(zk, axis=0)])
    var_x = np.concatenate([(var[1:2] - var[0:1]),
                            A[0, ...]*(A[0, ...] > 0) +
                            A[1, ...]*(A[1, ...] < 0),
                            (var[-1:] - var[-2:-1])])/dx
    return var_x


def centered_diff(var, dx):
    return np.concatenate([var[1:2] - var[0:1], (var[2:] - var[:-2])/2, (var[-1:] - var[-2:-1])])/dx


def temporalStep(W, g, r, pa, a_N, a_M, dx, theta, epsilon=1.e-15):
    # Compute intercell variables
    W_int = Variables_int(W, dx, theta)
    CheckPositivityH(W_int, W)
    u_int = reconstruct_u(W_int[0], W_int[1], epsilon)
    W_int[1, ...] = u_int*W_int[0, ...]
    # Compute Local speeds
    a_int, dtmax = LocalSpeeds(W_int[0, ...], u_int, g*(1-r), dx)
    # Compute Fluxes
    Fluxes = F(W_int, g*(1-r))
    H_int = H(Fluxes, a_int, W_int)
    # Compute sources
    S = S_func(W_int, g, pa, dx)
    # Compute non-hydro source
    u = reconstruct_u(W[0], W[1], epsilon)
    h = W[0, :] - W[-1, :]
    uint = 0.5*(u[1:] + u[:-1])
    hint = 0.5*(h[1:] + h[:-1])
    dB = minmod_diff(W[-1], dx, theta)
    dB_int = (dB[1:] - dB[:-1])/2
    dq = centered_diff(W[1], dx)  # dq from centered differences
    dh = minmod_diff(h, dx, theta)  # dh from minmod limiter
    #
    N = N_func(u, W[1, :], uint, hint, dB, dB_int, dq, dh, dx)
    LHS = LHS_func(W[1, 1:-1], h, dh, hint, dB, dB_int, a_M, dx)
    # Tau_n = tridiag_J(h, dh, hint, dB, dB_int, a_M, dx)
    # #### Computing right hand side
    return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]) + S - a_N*N, dtmax, LHS
    # return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]) + S - a_N*N, dtmax, Tau_n


def update_step(temporalStep, W, g, r, dx, theta, dt_fact=0.5, dt=None, pa=0, a_N=0, a_M=0):
    # RHS, dtmax, Tau_n = temporalStep(W, g, r, pa, a_N, a_M, dx, theta)
    RHS, dtmax, LHS = temporalStep(W, g, r, pa, a_N, a_M, dx, theta)
    # breakpoint()
    if dt is None:
        dt = dtmax*dt_fact
    #
    W_next = np.copy(W)
    # ## update h
    W_next[0, 1:-1] = W[0, 1:-1] + dt*RHS[0, :]  # update only w = h + Z
    # apply boundayr conditions on h here to keep dims
    W_next[0, 0], W_next[0, -1] = W_next[0, 1], W_next[0, -2]
    # ## update q
    # coupled_q = Tau_n @ W[1, 1:-1] + dt*RHS[1, :]
    coupled_q = LHS + dt*RHS[1, :]
    #
    h_next = W_next[0, :] - W[-1, :]
    dh_next = minmod_diff(h_next, dx, theta)
    hint_next = 0.5*(h_next[1:] + h_next[:-1])
    dB = minmod_diff(W[-1], dx, theta)
    dB_int = (dB[1:] - dB[:-1])/2
    #
    Tau_next = reduced_Tau(h_next, dh_next, hint_next,
                           dB, dB_int, a_M, dx)
    q_next = solve_banded((1, 1), Tau_next, coupled_q)
    # q_next = solve_banded((1, 1), Tau_next, coupled_q)
    # apply boundayr conditions on q
    W_next[1, 1:-1] = q_next
    W_next[1, 0], W_next[1, -1] = 0, 0
    #
    return W_next, dt
