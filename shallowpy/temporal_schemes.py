import numpy as np


def euler_step(model, W, dx, dt=None, **kwargs):
    RHS, dtmax = model.compute_RHS(W, dx)
    if dt is None:
        dt = dtmax*model.dt_fact
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


def Runge_kutta_step(model, W, dx):
    w1, dt = euler_step(model, W, dx)
    w2 = (3/4)*W + (1/4)*euler_step(model, w1, dx, dt=dt)[0]
    w_final = (1/3)*W + (2/3)*euler_step(model, w2, dx, dt=dt)[0]
    return w_final, dt
