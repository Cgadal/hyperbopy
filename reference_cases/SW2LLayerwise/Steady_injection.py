import numpy as np

from hyperbopy import Simulation
from hyperbopy.models import SW2LLayerwise

# ## Domain size
L = 10   # domain length [m]

# ## Grid parameters
tmax = 2.5  # s  max time
Nx = 1000  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# Injection properties
h0 = 0.5
q0 = 2

# ## Initial condition
# Bottom topography
Z = 0*x

# layer
hmin = 1e-10
Htop = 10

h2 = hmin*np.ones_like(x)
h2[0] = h0
h1 = Htop*np.ones_like(x) - h2

# velocities
q1, q2 = np.zeros_like(x), np.zeros_like(x)
q2[0] = q0

W0 = np.array([h1, q1, h2, q2, Z])

# ## Boundary conditions
BCs = [['symmetry', 'symmetry'], [0, 'symmetry'],
       [h0, 'symmetry'], [q0, 'symmetry']]

# ## Initialization
model = SW2LLayerwise()  # model with default parameters
simu = Simulation(
    model, W0, BCs, dx, spatial_scheme='CentralUpwindPathConservative')  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)
