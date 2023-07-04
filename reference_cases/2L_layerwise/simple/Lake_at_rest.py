import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model
from shallowpy.models import SW_2L_layerwise

# ## Domain size
L = 3     # domain length [m]
H = 3   # water height [m]

# ## Grid parameters
tmax = 1  # s  max time
Nx = 500  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Initial condition
# Bottom topography
Z = np.ones_like(x)

# layer 1 (upper)
h1 = H*np.ones_like(x)
q1 = np.zeros_like(x)

# layer 2 (lower)
h2 = np.ones_like(x)
q2 = np.zeros_like(x)

W0 = np.array([h1, q1, h2, q2, Z])

# ## model instance initialization
model = SW_2L_layerwise()  # with default parameters

# %% Run model
U, t = run_model(model, W0, tmax, dx, plot_fig=True, dN_fig=50, x=x, Z=Z)
