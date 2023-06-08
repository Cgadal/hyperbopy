from twolayerSW.core import *
from twolayerSW.model import run_model
import numpy as np

# ## Domain size
L = 3   # domain length [m]
Hmax = 5   # water height [m]

# ## Grid parameters
Nt = 5000  # time steps number
Nx = 1000  # spatial grid points number (evenly spaced)
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

# layer 2 (lower)
h2 = np.ones_like(x) - Z
q2 = np.zeros_like(x)

# layer 1 (upper) - h2
h1 = Hmax*np.ones_like(x) - h2 + h0*np.exp(-((x - x0)/sigma_0)**2)  # gaussian
# h1 = Hmax*np.ones_like(x) - h2 + np.where((x >= x0 - l0/2) & (x <= x0 + l0/2), h0, 0)  # window
q1 = np.zeros_like(x)

w = h2 + Z
W0 = np.array([h1, q1, w, q2, Z])

# %% Run model
run_model(W0, Nt, dx, g=9.81, r=0.95, plot_fig=True,
          dN_fig=100, x=x, Z=Z, theta=theta, dt_fact=0.1)


# %% tESTS
g = 9.81
r = 1

# Compute intercell variables
W_int = Variables_int(W0, dx, theta)
# Compute Local speeds
A_int, dtmax = LocalSpeeds(W_int, g, dx)
# Compute Fluxes
Fluxes = F(W_int, g, r)
H_int = H(Fluxes, A_int, W_int)
# Compute sources
Bpsi_int, Spsi_int = Bpsi_int_func(W_int, g, r), Spsi_int_func(W_int, g, r)
B, S = B_func(W0, W_int, g, r), S_func(W0, W_int, g, r)
RHSS = RHSS_func(B, S, Bpsi_int, Spsi_int, A_int)
# breakpoint()
# #### Computing right hand side
bb = (-1/dx)*(H_int[:, 1:] - H_int[:, :-1] + RHSS)
