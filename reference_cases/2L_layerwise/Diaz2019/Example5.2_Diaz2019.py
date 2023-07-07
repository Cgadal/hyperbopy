import matplotlib.pyplot as plt
import numpy as np

from shallowpy import Simulation
from shallowpy.models import SW2LLayerwise

# ## Grid parameters
tmax = 7
Nx = 400
x = np.linspace(-5, 5, Nx)
dx = np.diff(x)[0]

# initial conditions
h1 = np.empty_like(x)
h1[x < 0] = 1.8
h1[x > 0] = 0.2

h2 = np.empty_like(x)
h2[x < 0] = 0.2
h2[x > 0] = 1.8

q1, q2 = np.zeros_like(x), np.zeros_like(x)

Z = -2*np.ones_like(x)

W0 = np.array([h1, q1, h2, q2, Z])

# ## Initialization
model = SW2LLayerwise()  # model with default parameters
simu = Simulation(
    model, W0, dx, spatial_scheme='CentralUpwindPathConservative')  # simulation

# %% Run model
U, t = simu.run_simulation(tmax, plot_fig=True, dN_fig=50, x=x, Z=Z)

# %% plot final figure
fig, ax = plt.subplots(1, 1, layout='constrained')

color = 'tab:blue'
ax.plot(x, Z + U[-1, 0, :] + U[-1, 2, :], color=color)
ax.set_ylabel('water surface [m]', color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.plot(x, Z + U[-1, 2, :], color=color)
ax2.set_ylabel('interface [m]', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax.set_xlabel('x [m]')

plt.show()
