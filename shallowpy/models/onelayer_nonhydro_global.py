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

from shallowpy.core import default_pars
from shallowpy.spatial_scheme import (centered_diff, minmod_diff,
                                      reconstruct_u, spatial_discretization)
from shallowpy.temporal_schemes import Runge_kutta_step


class SW_1L_nonhydro_global(spatial_discretization):

    def __init__(self, g=None, r=None, pa=None, a_M=None, a_N=None,
                 theta=None, epsilon=None, dt_fact=None):
        self.g = g if g is not None else default_pars['g']
        self.r = r if r is not None else default_pars['r']
        self.pa = pa if pa is not None else default_pars['pa']
        self.a_M = a_M if a_M is not None else default_pars['a_M']
        self.a_N = a_N if a_N is not None else default_pars['a_N']
        self.theta = theta if theta is not None else default_pars['theta']
        self.epsilon = epsilon if epsilon is not None else default_pars['epsilon']
        self.dt_fact = dt_fact if dt_fact is not None else default_pars['dt_fact']
        #
        self.gprime = self.g*(1 - self.r)
        self.vars = ['h', 'u', 'Z']

    # #### temporal discretization functions

    def temporalstep(self, W, dx):
        return Runge_kutta_step(self, W, dx, euler_step=euler_step_nonhydro)

    # #### spatial discretization functions

    def compute_RHS(self, W, dx):
        # Compute intercell variables
        W_int = self.Variables_int(W, dx)
        self.CheckPositivityH(W_int, W)
        # Compute Local speeds
        a_int, dtmax = self.LocalSpeeds(W_int, dx)
        # Compute Fluxes
        Fluxes = self.F(W_int)
        H_int = self.H(Fluxes, a_int, W_int)
        # Compute sources
        S = self.S(W_int, dx)
        # Compute non-hydro source
        u, h, uint, hint, dB, dB_int, dq, dh = self.computeIntermediateVars(
            W, dx)
        #
        N = self.N(u, W[1, :], uint, hint, dB, dB_int, dq, dh, dx)
        LHS = self.LHS(W[1, 1:-1], h, dh, hint, dB, dB_int, dx)
        #
        return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]) + S - self.a_N*N, dtmax, LHS

    def F(self, W_int):
        return np.swapaxes(
            np.array([W_int[1, ...],
                      W_int[1, ...]**2/(W_int[0, ...] - W_int[-1, ...])
                      + (self.gprime/2)*(W_int[0, ...] - W_int[-1, ...])**2,
                      ]),
            0, 1)

    def S(self, W_int, dx):
        l1 = -(self.gprime/2)*(W_int[0, 1, 1:] - W_int[-1, 1, 1:]
                               + W_int[0, 0, :-1] - W_int[-1, 0, :-1])*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1]) \
            + self.pa*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
        return np.array([np.zeros_like(l1), l1])/dx

    def N(self, u, q, uint, hint, dB, dB_int, dq, dh, dx):
        # here qint from centered diff and not limiter as in W_int
        l1 = -(2/dx)*(hint[1:]*((q[2:] - q[1:-1])/dx)*((hint[1:]*(u[2:] - u[1:-1])/dx) - dB_int[1:]*uint[1:])
                      - hint[:-1]*((q[1:-1] - q[:-2])/dx)*((hint[:-1]*(u[2:] - u[1:-1])/dx) - dB_int[1:]*uint[1:])) \
            - 2*dB[1:-1]*dq[1:-1]*(dq[1:-1] - (dh[1:-1] + dB[1:-1])*u[1:-1])

        return np.array([np.zeros_like(l1), l1])

    def Tau_coeffs(self, h, dh, hint, dB, dB_int, dx):
        down = self.a_M*((h[1:-1]*dB[1:-1] - hint[:-1]*dB_int[:-1]) /
                         (4*dx) - hint[:-1]**3/(3*h[:-2]*dx**2))
        mid = 1 + self.a_M*((hint[1:]*dB_int[1:] - hint[:-1]*dB_int[:-1])/4*dx +
                            (hint[1:]**3 + hint[:-1]**3)/(3*h[1:-1]*dx**2) + dB[1:-1]*dh[1:-1]/2 + dB[1:-1]**2)
        up = self.a_M*((hint[1:]*dB_int[1:] - h[1:-1]*dB[1:-1]) /
                       (4*dx) - hint[1:]**3/(3*h[2:]*dx**2))
        return down, mid, up

    def LHS(self, q, h, dh, hint, dB, dB_int, dx):
        # here h is full vector but q := q[1:-1]
        down, mid, up = self.Tau_coeffs(h, dh, hint, dB, dB_int, dx)
        return np.hstack([0, q[:-1]*down[1:]]) + q*mid + np.hstack([q[1:]*up[:-1], 0])

    def LocalSpeeds(self, W_int, dx):
        # reconstruct u
        u_int = reconstruct_u(W_int[0], W_int[1], self.epsilon)
        # ensure consistency among variables
        W_int[1, ...] = u_int*W_int[0, ...]
        ap_int = np.row_stack((u_int + np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(u_int[0, :]))).max(axis=0)
        am_int = np.row_stack((u_int - np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(u_int[0, :]))).min(axis=0)
        return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))

    def reduced_Tau(self, h, dh, hint, dB, dB_int, dx):
        down, mid, up = self.Tau_coeffs(h, dh, hint, dB, dB_int, dx)
        return np.array([np.hstack([0, up[:-1]]),
                        mid,
                        np.hstack([down[1:], 0])
                         ])

    def CheckPositivityH(self, W_int, W):
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

    def computeIntermediateVars(self, W, dx, half=False):
        h = W[0, :] - W[-1, :]
        hint = 0.5*(h[1:] + h[:-1])
        dB = minmod_diff(W[-1], dx, self.theta)
        dB_int = (dB[1:] - dB[:-1])/2
        dh = minmod_diff(h, dx, self.theta)  # dh from minmod limiter
        if half:
            return h, hint, dB, dB_int, dh
        else:
            u = reconstruct_u(W[0], W[1], self.epsilon)
            uint = 0.5*(u[1:] + u[:-1])
            dq = centered_diff(W[1], dx)  # dq from centered differences
            return u, h, uint, hint, dB, dB_int, dq, dh

    # #### Unused methods but must be defined for now

    def B(self):
        ...

    def Ainv_int(self):
        ...

    def Bpsi_int(self):
        ...

    def Spsi_int(self):
        ...


def euler_step_nonhydro(model, W, dx, dt=None, **kwargs):
    RHS, dtmax, LHS = model.compute_RHS(W, dx)
    if dt is None:
        dt = dtmax*model.dt_fact
    #
    W_next = np.copy(W)
    # ### update only h
    W_next[0, 1:-1] = W[0, 1:-1] + dt*RHS[0, :]  # update only w = h + Z
    # apply boundary conditions on h here to keep dims
    W_next[0, 0], W_next[0, -1] = W_next[0, 1], W_next[0, -2]
    # #### update q
    coupled_q = LHS + dt*RHS[1, :]
    #
    h_next, hint_next, dB, dB_int, dh_next = model.computeIntermediateVars(
        W_next, dx, half=True)
    #
    Tau_next = model.reduced_Tau(h_next, dh_next, hint_next, dB, dB_int, dx)
    q_next = solve_banded((1, 1), Tau_next, coupled_q)
    # apply boundayr conditions on q
    W_next[1, 1:-1] = q_next
    W_next[1, 0], W_next[1, -1] = 0, 0
    #
    return W_next, dt
