from twolayerSW.model import run_model
import numpy as np

# ## Domain size
L = 3   # domain length [m]
H = 10   # water height [m]

# ## Grid parameters
Nt = 10000  # time steps number
Nx = 2000  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Numerical parameters
theta = 1

# ## bore properties
# gaussian
x0 = L/2
h0 = 0.3
sigma_0 = 0.2

# window
l0 = 3*sigma_0

# ## Initial condition
# Bottom topography
Z = 0*x

# layer 2 (lower)
h2 = np.ones_like(x) + h0*np.exp(-((x - x0)/sigma_0)**2) - Z # gaussian
# h2 = np.ones_like(x) + np.where((x >= x0 - l0/2) & (x <= x0 + l0/2), h0, 0) - Z # window
q2 = np.zeros_like(x)

# layer 1 (upper) - h2
h1 = H*np.ones_like(x) - h2
q1 = np.zeros_like(x)

w = h2 + Z
W0 = np.array([h1, q1, w, q2, Z])

# %% Run model
run_model(W0, Nt, dx, g=9.81, r=1.005, plot_fig=True,
            dN_fig=200, x=x, Z=Z, theta=theta)
