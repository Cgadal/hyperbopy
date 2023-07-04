import numpy as np

from shallowpy import run_model
from shallowpy.models import SW_1L_local

# ## Domain size
L = 10   # domain length [m]

# ## Grid parameters
tmax = 2.5  # s  max time
Nx = 500  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Initial condition
# Bottom topography
Z = 0*x

# height
# ## wave properties
l0 = 3
x0 = L/2
h0 = 3
Hlayer = 5

h = Hlayer*np.ones_like(x) + np.where((x >= x0 - l0/2)
                                      & (x <= x0 + l0/2), h0, 0)  # window
# velocity
q = np.zeros_like(x)

W0 = np.array([h, q, Z])

# ## model instance initialization
model = SW_1L_local()  # with default parameters

# %% Run model
U, t = run_model(model, W0, tmax, dx, plot_fig=True, dN_fig=100, x=x, Z=Z)
