"""
Spatial discretization

"""

import numpy as np

available_spatial_schemes = {}


def register_spatial_scheme(CustomSpatialScheme):
    global available_spatial_schemes
    available_spatial_schemes[CustomSpatialScheme.name] = CustomSpatialScheme
    return CustomSpatialScheme


class BaseScheme():

    def __init__(self, model):
        self.model = model

    def compute_RHSbase(self, W, dx):
        # Compute intercell variables
        W_int = variables_int(W, dx, self.model.theta)
        # Compute Local speeds
        a_int, dtmax = self.model.compute_local_speeds(W_int, dx)
        # Compute intermediate matrices
        B = self.model.compute_B(W, W_int)
        S = self.model.compute_S(W, W_int)
        # Compute Fluxes
        Fluxes = self.model.compute_F(W_int)
        H_int = self.compute_Hbase(Fluxes, a_int, W_int)
        # Compute sources
        RHSS = self.compute_RHSSbase(B, S)
        # Computing right hand side
        RHS = (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS)
        return RHS, dtmax, W_int, a_int

    def compute_RHSSbase(self, B, S):
        centered_part = - B - S
        return centered_part

    def compute_Hbase(self, Fluxes, a_int, W_int):
        return (a_int[0, :]*Fluxes[1, ...]
                - a_int[1, :]*Fluxes[0, ...]
                + a_int[0, :]*a_int[1, :]*(W_int[:-1, 0, :] - W_int[:-1, 1, :])) / (a_int[0, :] - a_int[1, :])

    # # #### required model definitions
    # @abc.abstractmethod
    # def local_speeds(self):
    #     ...

    # @abc.abstractmethod
    # def compute_B(self):
    #     ...

    # @abc.abstractmethod
    # def compute_S(self):
    #     ...

    # @abc.abstractmethod
    # def compute_F(self):
    #     ...


@register_spatial_scheme
class CentralUpwind(BaseScheme):

    name = 'CentralUpwind'

    def compute_RHS(self, W, dx):
        return self.compute_RHSbase(W, dx)[0: 2]


@register_spatial_scheme
class CentralUpwindPathConservative(BaseScheme):

    def __init__(self, model):
        super().__init__(model=model)

    name = 'CentralUpwindPathConservative'

    def compute_RHS(self, W, dx):
        RHSbase, dtmax, W_int, a_int = self.compute_RHSbase(W, dx)
        # ## compute path_conservative part
        Ainv_int = self.model.compute_Ainv_int(W, W_int)
        Bpsi_int = self.model.compute_Bpsi_int(W, W_int)
        Spsi_int = self.model.compute_Spsi_int(W, W_int)
        # Compute Fluxes
        H_int_pathcons = self.compute_H_pathcons(a_int, Ainv_int, Spsi_int)
        # Compute sources
        RHSS_pathcons = self.compute_RHSS_pathcons(Bpsi_int, Spsi_int, a_int)
        # Computing right hand side
        RHS = RHSbase + \
            (-1/dx)*(H_int_pathcons[:, 1:] -
                     H_int_pathcons[:, :-1] + RHSS_pathcons)
        return RHS, dtmax

    def compute_RHSS_pathcons(self, Bpsi_int, Spsi_int, a_int):
        jump_part = (a_int[1, 1:]*(Bpsi_int[:, 1:] + Spsi_int[:, 1:])) / (a_int[0, 1:] - a_int[1, 1:]) - \
            (a_int[0, :-1]*(Bpsi_int[:, :-1] + Spsi_int[:, :-1])) / \
            (a_int[0, :-1] - a_int[1, :-1])
        #
        return jump_part

    def compute_H_pathcons(self, a_int, Ainv_int, Spsi_int):
        return + a_int[0, :]*a_int[1, :]*(- np.einsum('ikj,kj -> ij', Ainv_int, Spsi_int)) / (a_int[0, :] - a_int[1, :])

    # # #### required model definitions

    # @abc.abstractmethod
    # def compute_Ainv_int(self):
    #     ...

    # @abc.abstractmethod
    # def compute_Bpsi_int(self):
    #     ...

    # @abc.abstractmethod
    # def compute_Spsi_int(self):
    #     ...


@register_spatial_scheme
class CentralUpwindNonHydro(BaseScheme):
    #
    name = 'CentralUpwindPathNoneHydro'

    def __init__(self, model):
        super().__init__(model=model)

    def compute_RHS(self, W, dx):
        # Compute intercell variables
        W_int = variables_int(W, dx, self.model.theta)
        self.check_positivity_h(W_int, W)
        # Compute Local speeds
        a_int, dtmax = self.model.compute_local_speeds(W_int, dx)
        # Compute Fluxes
        Fluxes = self.model.compute_F(W_int)
        H_int = self.compute_Hbase(Fluxes, a_int, W_int)
        # Compute sources
        S = self.model.compute_S(W_int, dx)
        # Compute non-hydro source
        u, h, uint, hint, dB, dB_int, dq, dh = self.computeIntermediateVars(
            W, dx)
        #
        N = self.model.compute_N(
            u, W[1, :], uint, hint, dB, dB_int, dq, dh, dx)
        # Compute LHS
        LHS = self.model.compute_LHS(W[1, 1:-1], h, dh, hint, dB, dB_int, dx)
        # Compute RHS
        RHS = (-1/dx)*(H_int[:, 1:] - H_int[:, :-1]) + \
            S - self.model.a_N*N
        return RHS, dtmax, LHS

    def check_positivity_h(self, W_int, W):
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
        dB = minmod_diff(W[-1], dx, self.model.theta)
        dB_int = (dB[1:] - dB[:-1])/2
        dh = minmod_diff(h, dx, self.model.theta)  # dh from minmod limiter
        if half:
            return h, hint, dB, dB_int, dh
        else:
            u = reconstruct_u(W[0], W[1], self.model.epsilon)
            uint = 0.5*(u[1:] + u[:-1])
            dq = centered_diff(W[1], dx)  # dq from centered differences
            return u, h, uint, hint, dB, dB_int, dq, dh

    # # #### required model definitions

    # @abc.abstractmethod
    # def local_speeds(self):
    #     ...

    # @abc.abstractmethod
    # def compute_S(self):
    #     ...

    # @abc.abstractmethod
    # def compute_F(self):
    #     ...

    # @abc.abstractmethod
    # def compute_N(self):
    #     ...

    # @abc.abstractmethod
    # def compute_LHS(self):
    #     ...

# #### Other usefull functions


def variables_int(var, dx, theta):
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


def reconstruct_u(h, u, epsilon):
    return np.sqrt(2)*h*u/np.sqrt(h**4 + np.max([h**4, epsilon*np.ones_like(h)], axis=0))


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
