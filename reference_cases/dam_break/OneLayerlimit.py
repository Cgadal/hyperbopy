import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model

model = '2L_layerwise'

# Spatial grid
Nx = 500
x = np.linspace(0, 3, Nx)
dx = np.diff(x)[0]
hmin = 1e-10

#  physical parameters
h0 = 0.3
l0 = 0.3
rho_2 = 1005
rho_1 = 1000
g = 9.81
Htop = 10

# initial conditions
h2 = np.empty_like(x)
h2[x <= l0] = h0
h2[x > l0] = hmin

h1 = Htop*np.ones_like(x) - h2

q1, q2 = np.zeros_like(x), np.zeros_like(x)

Z = 0*np.ones_like(x)

W0 = np.array([h1, q1, h2, q2, Z])

# other numerical parameters
theta = 1
tmax = 5

# %% Run model
U, t = run_model(model, W0, tmax, dx, g=9.81, r=rho_1/rho_2, plot_fig=True,
                 dN_fig=100, x=x, Z=Z, theta=theta, dt_fact=0.5)
