import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model

model = '2L_layerwise'

# Spatial grid
Nx = 1000
x = np.linspace(-1, 1, Nx)
dx = np.diff(x)[0]

# initial conditions
U_R = np.array([0.37002, -0.18684, 1.59310, 0.17416])
U_L = np.array([1.22582, -0.03866, 0.75325, 0.02893])

W0 = np.empty((5, x.size))
W0[:-1, np.argwhere(x < 0).squeeze()] = U_L[:, None]
W0[:-1, np.argwhere(x > 0).squeeze()] = U_R[:, None]
W0[-1, :] = - (U_L[0] + U_L[2])

Z = W0[-1, :]

# other numerical parameters
theta = 1
tmax = 1

# %% Run model
U, t = run_model(model, W0, tmax, dx, g=9.81, r=0.95, plot_fig=True,
                 dN_fig=10, x=x, Z=Z, theta=theta, dt_fact=0.5)

# %% plot final figure
fig, ax = plt.subplots(1, 1, layout='constrained')

color = 'tab:blue'
ax.plot(x, U[-1, 0, :] + U[-1, 2, :], color=color)
ax.set_ylabel('h1 + h2 [m]', color=color)
ax.tick_params(axis='y', labelcolor=color)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.plot(x, U[-1, 2, :], color=color)
ax2.set_ylabel('h2 [m]', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax.set_xlabel('x [m]')

plt.show()
