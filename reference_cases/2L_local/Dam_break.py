import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model

model = '2L_local'

# ## Domain size
L = 30   # domain length [m]

# ## Grid parameters
tmax = 5  # s  max time
Nx = 1000  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Numerical parameters
theta = 1
hmin = 1e-15

# ## wave properties
# window
l0 = L/2
h0 = 0.5
Htop = 10*h0

# initial conditions
h2 = np.empty_like(x)
h2[x <= l0] = h0
h2[x > l0] = hmin

h1 = Htop*np.ones_like(x) - h2

u1, u2 = np.zeros_like(x), np.zeros_like(x)

Z = 0*np.ones_like(x)

W0 = np.array([h1, u1, h2, u2, Z])

# %% Run model
U, t = run_model(model, W0, tmax, dx, plot_fig=True,
                 dN_fig=50, x=x, Z=Z, theta=theta, dt_fact=0.5)
