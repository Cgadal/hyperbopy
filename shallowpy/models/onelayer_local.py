"""
Here we solve the following system of equations:
    - d[h]/dt + d[hu]/dx = 0
    - d[u]/dt + d[u**2/2 + g*(h + Z)]/dx = 0

variables:
    - U  = [h, u]:
    - W = [h, u, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (3, Nx)

dim(W_int) = (3, 2, Nx):
    - 2: [h, u, Z]
    - 2: [pos, min]

REFERENCE: 
    - Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.
    - Fyhn, E. H., Lervåg, K. Y., Ervik, Å., & Wilhelmsen, Ø. (2019). A consistent reduction of the two-layer shallow-water equations to an accurate one-layer spreading model. Physics of Fluids, 31(12), 122103.

"""

import numpy as np

from shallowpy.core import default_pars
from shallowpy.spatial_scheme import spatial_discretization
from shallowpy.temporal_schemes import Runge_kutta_step

# #### model specific functions


class SW_1L_local(spatial_discretization):

    def __init__(self, g=None, r=None, theta=None, epsilon=None, dt_fact=None):
        self.g = g if g is not None else default_pars['g']
        self.r = r if r is not None else default_pars['r']
        self.theta = theta if theta is not None else default_pars['theta']
        self.epsilon = epsilon if r is not None else default_pars['epsilon']
        self.dt_fact = dt_fact if r is not None else default_pars['dt_fact']
        #
        self.gprime = self.g*(1 - self.r)
        self.vars = ['h', 'q', 'Z']

    # #### temporal discretization functions

    def temporalstep(self, W, dx):
        return Runge_kutta_step(self, W, dx)

    # #### spatial discretization functions

    def F(self, W_int):
        return np.swapaxes(
            np.array([W_int[0, ...]*W_int[1, ...],
                      W_int[1, ...]**2/2 + self.g *
                          (W_int[0, ...] + W_int[-1, ...]),
                      ]),
            0, 1)

    def S(self, W, W_int):
        return np.zeros_like(W[:-1, 1:-1])

    def B(self, W, W_int):
        return np.zeros_like(W[:-1, 1:-1])

    def Spsi_int(self, W, W_int):
        return np.zeros_like(W_int[:-1, 0, :])

    def Bpsi_int(self, W, W_int):
        return np.zeros_like(W_int[:-1, 0, :])

    def Ainv_int(self, W, W_int):
        zero = np.zeros_like(W_int[0, 0, :])
        one = np.ones_like(W_int[0, 0, :])
        #
        l1 = np.array([zero, one/self.g])
        l2 = np.array([2/(W_int[0, 0, :] + W_int[0, 1, :]), zero])
        return np.array([l1, l2])

    def LocalSpeeds(self, W_int, dx):
        #
        ap_int = np.row_stack((W_int[1, ...] + np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(W_int[1, ...][0, :]))).max(axis=0)
        am_int = np.row_stack((W_int[1, ...] - np.sqrt(self.gprime*W_int[0, ...]),
                               np.zeros_like(W_int[1, ...][0, :]))).min(axis=0)
        return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))


# #### Spatial discretization step

# def temporalStep(W, g, r, dx, theta, epsilon=1.e-15, path_conservative=False):
#     # Compute intercell variables
#     W_int = Variables_int(W, dx, theta)
#     # Compute Local speeds
#     a_int, dtmax = LocalSpeeds(W_int[0, ...], W_int[1, ...], g*(1-r), dx)
#     # Compute intermediate matrices
#     B, S = np.zeros_like(W[:-1, 1:-1]), np.zeros_like(W[:-1, 1:-1])
#     if path_conservative:
#         Ainv_int = Ainv_int_func(W_int, g*(1-r))
#         Bpsi_int, Spsi_int = np.zeros_like(
#             W_int[:-1, 0, :]), np.zeros_like(W_int[:-1, 0, :])
#     else:
#         Ainv_int, Bpsi_int, Spsi_int = None, None, None
#     # Compute Fluxes
#     Fluxes = F(W_int, g*(1-r))
#     H_int = H(Fluxes, a_int, W_int, Ainv_int, Bpsi_int)
#     # Compute sources
#     # no sources here
#     # #### Computing right hand side
#     return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]), dtmax
