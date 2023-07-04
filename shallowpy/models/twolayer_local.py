"""
Here we solve the following system of equations:
    - d[h1]/dt + d[q1]/dx = 0
    - d[u1]/dt + d[u1**2/2 + g*(h1 + h2 + Z)]/dx = 0
    - d[h2]/dt + d[q2]/dt = 0
    - d[u2]/dt + d[u2**2/2 + g*(r*h1 + h2 + Z)]/dx = 0

with:
    - r = rho1/rho2 

variables:
    - U  = [h1, q1, h2, q2]:
    - W = [h1, u1, h2, u2, Z]: 
    - U_int, W_int: left/right values of U, W


dim(W) = (5, Nx)

dim(W_int) = (5, 2, Nx):
    - 2: [h1, u1, h2, u2, Z]
    - 2: [pos, min]

REFERENCE: Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

"""

import numpy as np

from shallowpy.core import default_pars
from shallowpy.spatial_scheme import spatial_discretization
from shallowpy.temporal_schemes import Runge_kutta_step


class SW_2L_local(spatial_discretization):

    def __init__(self, g=None, r=None, theta=None, epsilon=None, dt_fact=None):
        self.g = g if g is not None else default_pars['g']
        self.r = r if r is not None else default_pars['r']
        self.theta = theta if theta is not None else default_pars['theta']
        self.epsilon = epsilon if epsilon is not None else default_pars['epsilon']
        self.dt_fact = dt_fact if dt_fact is not None else default_pars['dt_fact']
        #
        self.vars = ['h1', 'u1', 'h2', 'u2', 'Z']

    # #### temporal discretization functions

    def temporalstep(self, W, dx):
        return Runge_kutta_step(self, W, dx)

    # #### spatial discretization functions

    def F(self, W_int):
        return np.swapaxes(
            np.array([W_int[0, ...]*W_int[1, ...],
                      W_int[1, ...]**2/2 + self.g *
                      (W_int[0, ...] + W_int[2, ...] + W_int[-1, ...]),
                      W_int[2, ...]*W_int[3, ...],
                      W_int[3, ...]**2/2 + self.g *
                      (self.r*W_int[0, ...] + W_int[2, ...] + W_int[-1, ...]),
                      ]),
            0, 1)

    def S(self, W, W_int):
        return np.zeros_like(W[:-1, 1:-1])

    def B(self, W, W_int):
        return np.zeros_like(W[:-1, 1:-1])

    def Bpsi_int(self, W, W_int):
        return np.zeros_like(W_int[:-1, 0, :])

    def Spsi_int(self, W, W_int):
        return np.zeros_like(W_int[:-1, 0, :])

    def Ainv_int(self, W, W_int):
        zero = np.zeros_like(W_int[0, 0, :])
        one = np.ones_like(W_int[0, 0, :])
        #
        l1 = np.array(
            [zero, one/(self.g*(1-self.r)), 2/((1-self.r)*(W_int[2, 0, :] + W_int[2, 1, :])), one/(self.g*(1-self.r))])
        l2 = np.array([2/(W_int[0, 0, :] + W_int[0, 1, :]), zero, zero, zero])
        l3 = -l1
        l4 = np.array([zero, zero, 2/(W_int[2, 0, :] + W_int[2, 1, :]), zero])
        return np.array([l1, l2, l3, l4])

    def LocalSpeeds(self, W_int, dx):
        h2_int = W_int[2, ...] - W_int[4, ...]
        um = (W_int[1, ...] + W_int[3, ...])/(W_int[0, ...] + h2_int)
        #
        ap_int = np.row_stack(
            (um + np.sqrt(self.g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).max(axis=0)
        am_int = np.row_stack(
            (um - np.sqrt(self.g*(W_int[0, ...] + h2_int)), np.zeros_like(um[0, :]))).min(axis=0)
        return np.array([ap_int, am_int]), dx/(2*np.amax([ap_int, -am_int]))
