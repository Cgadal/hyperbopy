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

from shallowpy.core.spatial_scheme import reconstruct_u

from .basemodel import BaseModel


class SW1LNonhydroGlobal(BaseModel):

    name = 'SW1LNonhydroGlobal'

    def __init__(self, g=None, r=None, pa=None, a_M=None, a_N=None,
                 theta=None, epsilon=None, dt_fact=None):
        self.g = g if g is not None else self.GRAVITATIONAL_CONSTANT
        self.r = r if r is not None else self.DENSITY_RATIO
        self.pa = pa if pa is not None else self.EXTERNAL_PRESSURE
        self.a_M = a_M if a_M is not None else self.NONHYDRO_COEFF_M
        self.a_N = a_N if a_N is not None else self.NONHYDRO_COEFF_N
        self.theta = theta if theta is not None else self.THETA
        self.epsilon = epsilon if epsilon is not None else self.EPSILON
        self.dt_fact = dt_fact if dt_fact is not None else self.DT_FACT
        #
        self.gprime = self.g*(1 - self.r)
        self.var_names = ['h', 'u', 'Z']

    # #### spatial discretization functions

    def compute_F(self, W_int):
        return np.swapaxes(
            np.array([W_int[1, ...],
                      W_int[1, ...]**2/(W_int[0, ...] - W_int[-1, ...])
                      + (self.gprime/2)*(W_int[0, ...] - W_int[-1, ...])**2,
                      ]),
            0, 1)

    def compute_S(self, W_int, dx):
        l1 = -(self.gprime/2)*(W_int[0, 1, 1:] - W_int[-1, 1, 1:]
                               + W_int[0, 0, :-1] - W_int[-1, 0, :-1])*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1]) \
            + self.pa*(W_int[0, 1, 1:] - W_int[0, 0, :-1])
        return np.array([np.zeros_like(l1), l1])/dx

    def compute_N(self, u, q, uint, hint, dB, dB_int, dq, dh, dx):
        # here qint from centered diff and not limiter as in W_int
        l1 = -(2/dx)*(hint[1:]*((q[2:] - q[1:-1])/dx)*((hint[1:]*(u[2:] - u[1:-1])/dx) - dB_int[1:]*uint[1:])
                      - hint[:-1]*((q[1:-1] - q[:-2])/dx)*((hint[:-1]*(u[2:] - u[1:-1])/dx) - dB_int[1:]*uint[1:])) \
            - 2*dB[1:-1]*dq[1:-1]*(dq[1:-1] - (dh[1:-1] + dB[1:-1])*u[1:-1])

        return np.array([np.zeros_like(l1), l1])

    def compute_Tau_coeffs(self, h, dh, hint, dB, dB_int, dx):
        down = self.a_M*((h[1:-1]*dB[1:-1] - hint[:-1]*dB_int[:-1]) /
                         (4*dx) - hint[:-1]**3/(3*h[:-2]*dx**2))
        mid = 1 + self.a_M*((hint[1:]*dB_int[1:] - hint[:-1]*dB_int[:-1])/4*dx +
                            (hint[1:]**3 + hint[:-1]**3)/(3*h[1:-1]*dx**2) + dB[1:-1]*dh[1:-1]/2 + dB[1:-1]**2)
        up = self.a_M*((hint[1:]*dB_int[1:] - h[1:-1]*dB[1:-1]) /
                       (4*dx) - hint[1:]**3/(3*h[2:]*dx**2))
        return down, mid, up

    def compute_LHS(self, q, h, dh, hint, dB, dB_int, dx):
        # here h is full vector but q := q[1:-1]
        down, mid, up = self.compute_Tau_coeffs(h, dh, hint, dB, dB_int, dx)
        return np.hstack([0, q[:-1]*down[1:]]) + q*mid + np.hstack([q[1:]*up[:-1], 0])

    def compute_local_speeds(self, W_int, dx):
        # reconstruct u
        u_int = reconstruct_u(W_int[0], W_int[1], self.epsilon)
        # ensure consistency among variables
        W_int[1, ...] = u_int*W_int[0, ...]
        ap_int = np.row_stack((u_int + np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(u_int[0, :]))).max(axis=0)
        am_int = np.row_stack((u_int - np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(u_int[0, :]))).min(axis=0)
        return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))

    def compute_reduced_Tau(self, h, dh, hint, dB, dB_int, dx):
        down, mid, up = self.compute_Tau_coeffs(h, dh, hint, dB, dB_int, dx)
        return np.array([np.hstack([0, up[:-1]]),
                        mid,
                        np.hstack([down[1:], 0])
                         ])
