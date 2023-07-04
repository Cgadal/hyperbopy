"""
Spatial discretization: central-upwind scheme

"""

import numpy as np
import abc


class spatial_discretization(abc.ABC):

    def compute_RHS(self, W, dx):
        # Compute intercell variables
        W_int = self.Variables_int(W, dx)
        # Compute Local speeds
        a_int, dtmax = self.LocalSpeeds(W_int, dx)
        # Compute intermediate matrices
        Ainv_int = self.Ainv_int(W, W_int)
        Bpsi_int = self.Bpsi_int(W, W_int)
        Spsi_int = self.Bpsi_int(W, W_int)
        B = self.B(W, W_int)
        S = self.S(W, W_int)
        # Compute Fluxes
        Fluxes = self.F(W_int)
        H_int = self.H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
        # Compute sources
        RHSS = self.RHSS_func(B, S, Bpsi_int, Spsi_int, a_int)
        # Computing right hand side
        return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax

    # model-dependant functions
    @abc.abstractmethod
    def LocalSpeeds(self):
        ...

    @abc.abstractmethod
    def Ainv_int(self):
        ...

    @abc.abstractmethod
    def Bpsi_int(self):
        ...

    @abc.abstractmethod
    def Spsi_int(self):
        ...

    @abc.abstractmethod
    def S(self):
        ...

    @abc.abstractmethod
    def B(self):
        ...

    @abc.abstractmethod
    def F(self):
        ...

    # general functions

    def Variables_int(self, var, dx):
        # ##### Minmod limiter for interpolation
        zk = np.array([self.theta*(var[:, 1:-1] - var[:, :-2]),
                       (var[:, 2:] - var[:, :-2])/2,
                       self.theta*(var[:, 2:] - var[:, 1:-1])])
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

    def RHSS_func(self, B, S, Bpsi_int, Spsi_int, a_int):
        jump_part = (a_int[1, 1:]*(Bpsi_int[:, 1:] + Spsi_int[:, 1:])) / (a_int[0, 1:] - a_int[1, 1:]) - \
            (a_int[0, :-1]*(Bpsi_int[:, :-1] + Spsi_int[:, :-1])) / \
            (a_int[0, :-1] - a_int[1, :-1])
        #
        centered_part = - B - S
        return centered_part + jump_part

    def H(self, Fluxes, a_int, W_int, Ainv_int=None, Spsi_int=None):
        if (Ainv_int is None) | (Spsi_int is None):
            return (a_int[0, :]*Fluxes[1, ...]
                    - a_int[1, :]*Fluxes[0, ...]
                    + a_int[0, :]*a_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :])) / (a_int[0, :] - a_int[1, :])
        else:
            return (a_int[0, :]*Fluxes[1, ...]
                    - a_int[1, :]*Fluxes[0, ...]
                    + a_int[0, :]*a_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :]
                                               - np.einsum('ikj,kj -> ij', Ainv_int, Spsi_int))) / (a_int[0, :] - a_int[1, :])


def reconstruct_u(W, epsilon):
    return np.sqrt(2)*W[0, ...]*W[1, ...]/np.sqrt(W[0, ...]**4
                                                  + np.max([W[0, ...]**4, epsilon*np.ones_like(W[0, ...])], axis=0))


# def temporalStep(W, g, r, dx, theta):
#     # Compute intercell variables
#     W_int = Variables_int(W, dx, theta)
#     # Compute Local speeds
#     a_int, dtmax = LocalSpeeds(W_int, g, dx)
#     # Compute intermediate matrices
#     Ainv_int = Ainv_int_func(W_int, g, r)
#     B, S = np.zeros_like(W[:-1, 1:-1]), np.zeros_like(W[:-1, 1:-1])
#     Bpsi_int, Spsi_int = np.zeros_like(
#         W_int[:-1, 0, :]), np.zeros_like(W_int[:-1, 0, :])
#     # Compute Fluxes
#     Fluxes = F(W_int, g, r)
#     H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
#     # Compute sources
#     # no sources
#     # Computing right hand side
#     return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]), dtmax


# def H(Fluxes, a_int, W_int, Ainv_int=None, Spsi_int=None):
#     #
#     if (Ainv_int is None) | (Spsi_int is None):
#         return (a_int[0, :]*Fluxes[1, ...]
#                 - a_int[1, :]*Fluxes[0, ...]
#                 + a_int[0, :]*a_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :])) / (a_int[0, :] - a_int[1, :])
#     else:
#         return (a_int[0, :]*Fluxes[1, ...]
#                 - a_int[1, :]*Fluxes[0, ...]
#                 + a_int[0, :]*a_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :]
#                                            - np.einsum('ikj,kj -> ij', Ainv_int, Spsi_int))) / (a_int[0, :] - a_int[1, :])

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


# def temporalStep(W, dx, phypars, numpars,
#                  F, LocalSpeeds, Ainv_int_func, Bpsi_int_func, Spsi_int_func,
#                  B_func, S_func):
#     # Compute intercell variables
#     W_int = Variables_int(W, dx, numpars['theta'])
#     # Compute Local speeds
#     a_int, dtmax = LocalSpeeds(W_int, dx, **phypars)
#     # Compute intermediate matrices
#     Ainv_int = Ainv_int_func(W_int, **phypars)
#     Bpsi_int = Bpsi_int_func(W_int, **phypars),
#     Spsi_int = Spsi_int_func(W_int, **phypars)
#     B = B_func(W, W_int, **phypars),
#     S = S_func(W, W_int, **phypars)
#     # Compute Fluxes
#     Fluxes = F(W_int, **phypars)
#     H_int = H(Fluxes, a_int, W_int, Ainv_int, Spsi_int)
#     # Compute sources
#     RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, a_int)
#     # Computing right hand side
#     return (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS), dtmax


# #### def model class

# class model:
#     def __init__(self, model_name, phypars_default,
#                  F, Ainv_int, S, B, Spsi_int, Bpsi_int, LocalSpeeds):
#         # properties
#         self.model_name = model_name
#         self.phypars_default = phypars_default
#         # functions
#         self.F = F
#         self.Ainv_int = Ainv_int
#         self.S = S
#         self.B = B
#         self.Spsi_int = Spsi_int
#         self.Bpsi_int = Bpsi_int
#         self.LocalSpeeds = LocalSpeeds
