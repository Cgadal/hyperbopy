"""
Here we solve the following system of equations:
    - d[h]/dt + d[q]/dx = 0
    - d[q]/dt + d[q**2/h + (g/2)*h**2]/dx = -g*h*(d[Z]/dx)

variables:
    - U  = [h, q]:
    - W = [h, q, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (3, Nx)

dim(W_int) = (3, 2, Nx):
    - 2: [h, q, Z]
    - 2: [pos, min]

REFERENCE: Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

"""

import numpy as np

from shallowpy.core import default_pars
from shallowpy.spatial_scheme import reconstruct_u, spatial_discretization
from shallowpy.temporal_schemes import Runge_kutta_step


class SW_1L_global(spatial_discretization):

    def __init__(self, g=None, r=None, theta=None, epsilon=None, dt_fact=None):
        self.g = g if g is not None else default_pars['g']
        self.r = r if r is not None else default_pars['r']
        self.theta = theta if theta is not None else default_pars['theta']
        self.epsilon = epsilon if epsilon is not None else default_pars['epsilon']
        self.dt_fact = dt_fact if dt_fact is not None else default_pars['dt_fact']
        #
        self.gprime = self.g*(1 - self.r)
        self.vars = ['h', 'q', 'Z']

    # #### temporal discretization functions

    def temporalstep(self, W, dx):
        return Runge_kutta_step(self, W, dx)

    # #### spatial discretization functions

    def F(self, W_int):
        return np.swapaxes(
            np.array([W_int[1, ...],
                      W_int[1, ...]**2/W_int[0, ...] +
                          (self.gprime/2)*W_int[0, ...]**2,
                      ]),
            0, 1)

    def S(self, W, W_int):
        l1 = -self.gprime*W[0, 1:-1]*(W_int[-1, 1, 1:] - W_int[-1, 0, :-1])
        return np.array([np.zeros_like(l1), l1])

    def B(self, W, W_int):
        return np.zeros_like(W[:-1, 1:-1])

    def Spsi_int(self, W, W_int):
        l1 = -(self.gprime/2)*(W_int[0, 0, :] + W_int[0, 1, :]) * \
            (W_int[-1, 0, :] - W_int[-1, 1, :])
        return np.array([np.zeros_like(l1), l1])

    def Bpsi_int(self, W, W_int):
        return np.zeros_like(W_int[:-1, 0, :])

    def Ainv_int(self, W, W_int):
        zero = np.zeros_like(W_int[0, 0, :])
        one = np.ones_like(W_int[0, 0, :])
        #
        l1 = np.array([zero, 2/(self.gprime*W_int[0, 0, :] + W_int[0, 1, :])])
        l2 = np.array([one, zero])
        return np.array([l1, l2])

    def LocalSpeeds(self, W_int, dx):
        # reconstruct u
        u_int = reconstruct_u(W_int, self.epsilon)
        # ensure consistancy among variables
        W_int[1, ...] = u_int*W_int[0, ...]
        #
        ap_int = np.row_stack((u_int + np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(u_int[0, :]))).max(axis=0)
        am_int = np.row_stack((u_int - np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(u_int[0, :]))).min(axis=0)
        return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))


# #### Spatial discretization step


# def temporalStep(W, g, r, dx, theta, epsilon=1.e-15):
#     # Compute intercell variables
#     W_int = Variables_int(W, dx, theta)
#     u_int = reconstruct_u(W_int, epsilon)
#     W_int[1, ...] = u_int*W_int[0, ...]
#     # Compute Local speeds
#     a_int, dtmax = LocalSpeeds(W_int[0, ...], u_int, g*(1-r), dx)
#     # Compute intermediate matrices
#     Ainv_int = Ainv_int_func(W_int, g*(1-r))
#     S, Spsi_int = S_func(W, W_int, g*(1-r)), Spsi_int_func(W_int, g*(1-r))
#     B, Bpsi_int = np.zeros_like(S), np.zeros_like(Spsi_int)
#     # Compute Fluxes
#     Fluxes = F(W_int, g*(1-r))
#     H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
#     # Compute sources
#     RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, a_int)
#     # breakpoint()
#     # #### Computing right hand side
#     # return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
#     return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax
