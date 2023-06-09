from twolayerSW.model import run_model
import numpy as np

# ## Domain size
L = 3     # domain length [m]
H = 3   # water height [m]

# ## Grid parameters
tmax = 1  # s  max time
Nx = 500  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Numerical parameters
theta = 1

# ## Initial condition
# Bottom topography
Z = np.ones_like(x)

# layer 1 (upper)
h1 = H*np.ones_like(x)
q1 = np.zeros_like(x)

# layer 2 (lower)
h2 = np.ones_like(x)
q2 = np.zeros_like(x)

w = h2 + Z
W0 = np.array([h1, q1, w, q2, Z])

# %% Run model
U, t = run_model(W0, tmax, dx, g=9.81, r=1.2, plot_fig=True,
                 dN_fig=100, x=x, Z=Z, theta=theta, dt_fact=0.5)
