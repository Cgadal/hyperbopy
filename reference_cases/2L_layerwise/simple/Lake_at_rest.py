import numpy as np

from shallowpy import Simulation
from shallowpy.models import SW2LLayerwise

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

# ## Initialization
model = SW2LLayerwise()  # model with default parameters
simu = Simulation(
    model, W0, dx)  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)
