import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model
from shallowpy.models import SW_1L_nonhydro_global

# ## Domain size
L = 10   # domain length [m]

# ## Grid parameters
tmax = 2.5  # s  max time
Nx = 1000  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Initial condition
# Bottom topography
Z = 0*x

# layer
hmin = 1e-10
l0 = 5
h0 = 0.5
#
h = hmin*np.ones_like(x) + np.where(x <= l0, h0, 0)  # window

# velocity
q = np.zeros_like(x)

W0 = np.array([h, q, Z])

# ## model instance initialization
model = SW_1L_nonhydro_global(a_N=0.4, a_M=4)  # with default parameters

# %% Run model
U, t = run_model(model, W0, tmax, dx, plot_fig=True, dN_fig=50, x=x, Z=Z)
