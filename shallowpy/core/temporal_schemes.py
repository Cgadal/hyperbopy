"""
Temporal discretization

"""

import numpy as np
from scipy.linalg import solve_banded

available_temporal_schemes = {}


def register_temporal_scheme(CustomTemporalScheme):
    global available_temporal_schemes
    available_temporal_schemes[CustomTemporalScheme.name] = CustomTemporalScheme
    return CustomTemporalScheme


class EulerBase():

    def __init__(self, model, spatial_scheme, dt_fact):
        self.model = model
        self.SpatialScheme = spatial_scheme
        self.dt_fact = dt_fact

    def _euler_step_regular(self, W, dx, dt=None):
        RHS, dtmax = self.SpatialScheme.compute_RHS(W, dx)
        if dt is None:
            dt = dtmax*self.dt_fact
        #
        W_next = np.copy(W)
        W_next[:-1, 1:-1] = W[:-1, 1:-1] + dt*RHS
        # boundary conditions (0 for q or u, reflective for h)
        W_up = np.hstack([np.array([W_next[2*i, 1], 0])
                          for i in range((W_next.shape[0] - 1)//2)])
        W_down = np.hstack([np.array([W_next[2*i, -2], 0])
                            for i in range((W_next.shape[0] - 1)//2)])
        W_next[:-1, 0], W_next[:-1, -1] = W_up, W_down
        return W_next, dt

    def _euler_step_nonhydro(self, W, dx, dt=None):
        RHS, dtmax, LHS = self.SpatialScheme.compute_RHS(W, dx)
        if dt is None:
            dt = dtmax*self.dt_fact
        #
        W_next = np.copy(W)
        # ### update only h
        W_next[0, 1:-1] = W[0, 1:-1] + dt*RHS[0, :]  # update only w = h + Z
        # apply boundary conditions on h here to keep dims
        W_next[0, 0], W_next[0, -1] = W_next[0, 1], W_next[0, -2]
        # #### update q
        coupled_q = LHS + dt*RHS[1, :]
        #
        h_next, hint_next, dB, dB_int, dh_next = self.SpatialScheme.computeIntermediateVars(
            W_next, dx, half=True)
        #
        Tau_next = self.model.compute_reduced_Tau(
            h_next, dh_next, hint_next, dB, dB_int, dx)
        q_next = solve_banded((1, 1), Tau_next, coupled_q)
        # apply boundayr conditions on q
        W_next[1, 1:-1] = q_next
        W_next[1, 0], W_next[1, -1] = 0, 0
        #
        return W_next, dt

    def time_step_base(self, W, dx, dt=None, euler_step='regular'):
        if euler_step == 'regular':
            return self._euler_step_regular(W, dx, dt=dt)
        elif euler_step == 'nonhydro':
            return self._euler_step_nonhydro(W, dx, dt=dt)


@register_temporal_scheme
class Euler(EulerBase):

    name = 'Euler'

    def time_step(self, W, dx, dt=None, euler_step='regular'):
        return self.time_step_base(W, dx, dt=None, euler_step=euler_step)


@register_temporal_scheme
class RungeKutta33(EulerBase):

    name = 'RungeKutta33'

    def time_step(self, W, dx, euler_step='regular'):
        w1, dt = self.time_step_base(W, dx, euler_step=euler_step)
        w2 = (3/4)*W + (1/4)*self.time_step_base(w1,
                                                 dx, dt=dt, euler_step=euler_step)[0]
        w_final = (1/3)*W + (2/3)*self.time_step_base(w2,
                                                      dx, dt=dt, euler_step=euler_step)[0]
        return w_final, dt
