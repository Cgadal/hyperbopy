import numpy as np

from shallowpy import run_model

model = '1L_non_hydro_global'

# ## Domain size
L = 3   # domain length [m]
Hmax = 5   # water height [m]

# ## Grid parameters
tmax = 2  # s  max time
Nx = 500  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Numerical parameters
theta = 1

# ## wave properties
# gaussian
x0 = L/2
h0 = 1
sigma_0 = 0.2

# window
l0 = 3*sigma_0

# ## Initial condition
# Bottom topography
Z = 0*x

# layer
h = Hmax*np.ones_like(x) + h0*np.exp(-((x - x0)/sigma_0)**2)  # gaussian
# h = Hmax*np.ones_like(x) + np.where((x >= x0 - l0/2)
#                                     & (x <= x0 + l0/2), h0, 0)  # window
q = np.zeros_like(x)

W0 = np.array([h, q, Z])

# %% Run model
U, t = run_model(model, W0, tmax, dx, g=9.81, r=0.95, plot_fig=True,
                 dN_fig=20, x=x, Z=Z, theta=theta, dt_fact=0.5, a_N=0.2, a_M=0.2, pa=0)
