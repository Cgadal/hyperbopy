import numpy as np

from shallowpy import Simulation
from shallowpy.models import SW1LNonhydroGlobal

# ## Domain size
L = 10   # domain length [m]

# ## Grid parameters
tmax = 2.5  # s  max time
Nx = 1000  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# injection properties
q0 = 0.5
h0 = 3

# ## Initial condition
# Bottom topography
Z = 0*x

# layer
hmin = 1e-15
h = 0*x + hmin

# velocity
q = np.zeros_like(x)
h[0] = h0
q[0] = q0

W0 = np.array([h, q, Z])

# ## Boundary conditions
BCs = [[h0, 'symmetry'], [q0, 'symmetry']]

# ## Initialization
model = SW1LNonhydroGlobal(a_N=0.05, a_M=0.05)
simu = Simulation(model, W0, BCs, dx, temporal_scheme='RungeKutta33',
                  spatial_scheme='CentralUpwindPathNoneHydro')  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)
