import numpy as np

from hyperbopy import Simulation
from hyperbopy.models import SW1LNonhydroGlobal

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

# ## Boundary conditions
BCs = [['symmetry', 'symmetry'], [0, 0]]

# ## Initialization
# model with default parameters
model = SW1LNonhydroGlobal(a_N=0.2, a_M=0.2)
simu = Simulation(model, W0, BCs, dx, temporal_scheme='RungeKutta33',
                  spatial_scheme='CentralUpwindPathNoneHydro')  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)
