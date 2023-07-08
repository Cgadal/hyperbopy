import numpy as np

from hyperbopy import Simulation
from hyperbopy.models import SW1LLocal

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

# ## Boundary conditions
BCs = [['symmetry', 'symmetry'], [0, 0]]

# ## Initialization
model = SW1LLocal()  # model with default parameters
simu = Simulation(model, W0, BCs, dx)  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)
