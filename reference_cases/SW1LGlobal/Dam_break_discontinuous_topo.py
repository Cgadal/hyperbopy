import numpy as np
import matplotlib.pyplot as plt

from hyperbopy import Simulation
from hyperbopy.models import SW1LGlobal

# ## Domain size
L = 1   # domain length [m]

# ## Grid parameters
tmax = 0.1  # s  max time
Nx = 600  # spatial grid points number (evenly spaced)
x = np.linspace(-L/2, L/2, Nx)
dx = L/(Nx - 1)

# ## Initial condition
# Bottom topography
delta = 0.001
Z = 0*x
Z[x <  0.1 - delta] = - 0.5
mask = (x >=  0.1 - delta) & (x <=  0.1 + delta)
Z[mask] = - 0.5 - (0.2/delta)*(x[mask] - 0.1 + delta)
Z[x >  0.1 + delta] = - 0.9

# layer
hmin = 1e-10
#
h = hmin*np.ones_like(x)
h[x <0] = 1 - Z[x < 0]
h[x >=0] = - Z[x >= 0]

# velocity
q = np.zeros_like(x)

W0 = np.array([h, q, Z])

# ## Boundary conditions
BCs = [['symmetry', 'symmetry'], [0, 0]]

# ## Initialization
model = SW1LGlobal(g=9.81, r=0)  # model with default parameters
simu = Simulation(model, W0, BCs, dx, spatial_scheme='CentralUpwind')  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)

plt.figure()
plt.plot(x, Z)
plt.plot(x, U[0, 0, :] + Z)
plt.plot(x, U[50, 0, :] + Z)
plt.plot(x, U[-1, 0, :] + Z)
plt.show()