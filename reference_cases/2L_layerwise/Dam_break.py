import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model
from shallowpy.models import SW_2L_layerwise

# ## Domain size
L = 10   # domain length [m]

# ## Grid parameters
tmax = 2.5  # s  max time
Nx = 500  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Initial condition
# Bottom topography
Z = 0*np.ones_like(x)

# layers
hmin = 1e-10
l0 = 0.3
h0 = 0.3
Htop = 10

h2 = np.empty_like(x)
h2[x <= l0] = h0
h2[x > l0] = hmin
h1 = Htop*np.ones_like(x) - h2

# velocities
q1, q2 = np.zeros_like(x), np.zeros_like(x)

W0 = np.array([h1, q1, h2, q2, Z])

# ## model instance initialization
model = SW_2L_layerwise()  # with default parameters

# %% Run model
U, t = run_model(model, W0, tmax, dx, plot_fig=True, dN_fig=50, x=x, Z=Z)
