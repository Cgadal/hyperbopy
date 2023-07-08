import numpy as np
import matplotlib.pyplot as plt

from hyperbopy import Simulation
from hyperbopy.models import SW1LLocal

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
u = np.zeros_like(x)

W0 = np.array([h, u, Z])

# ## Boundary conditions
BCs = [['symmetry', 'symmetry'], [0, 0]]

# ## Initialization
model = SW1LLocal()  # model with default parameters
simu = Simulation(model, W0, BCs, dx)  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)

# %% Compare with theory


def theory(x, t, h0, l0, g=9.81, r=0.95):
    c0 = np.sqrt((1-r)*g*h0)
    x0 = -c0*t
    x1 = (2 - np.sqrt(2))*c0*t/(1 + np.sqrt(2))
    x2 = 2*c0*t/(1 + np.sqrt(2))
    #
    condlist = [x - l0 <= x0, (x - l0 > x0) & (x - l0 <= x1),
                (x - l0 > x1) & (x - l0 <= x2), (x - l0 > x2)]
    #
    funclist = [lambda x: h0,
                lambda x: (h0/9)*(2 - (x-l0)/(c0*t))**2,
                lambda x: 4*h0/(2 + np.sqrt(2))**2,
                lambda x: 0]
    #
    return np.piecewise(x, condlist, funclist)


inds_t = [0, 25, 50, 75]

fig, axarr = plt.subplots(2, 2, layout='constrained')
for i, ax in zip(inds_t, axarr.flatten()):
    ax.plot(x, U[i, 0, :])
    ax.plot(x, theory(x, t[i], h0, l0))

plt.show()
