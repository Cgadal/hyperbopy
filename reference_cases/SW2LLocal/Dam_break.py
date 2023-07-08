import numpy as np

from hyperbopy import Simulation
from hyperbopy.models import SW2LLocal

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

# ## Initialization
model = SW2LLocal()  # model with default parameters
simu = Simulation(
    model, W0, dx, spatial_scheme='CentralUpwindPathConservative')  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)
