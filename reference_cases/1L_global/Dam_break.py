import numpy as np
import matplotlib.pyplot as plt

from shallowpy import run_model

model = '1L_global'

# ## Domain size
L = 10   # domain length [m]

# ## Grid parameters
tmax = 2.5  # s  max time
Nx = 1000  # spatial grid points number (evenly spaced)
x = np.linspace(0, L, Nx)
dx = L/(Nx - 1)

# ## Numerical parameters
theta = 1.5

# ## wave properties
# window
l0 = 5
h0 = 0.5

# ## Initial condition
# Bottom topography
Z = 0*x

# layer
hmin = 1e-10
h = hmin*np.ones_like(x) + np.where(x <= l0, h0, 0)  # window
q = np.zeros_like(x)

W0 = np.array([h, q, Z])

# %% Run model
U, t = run_model(model, W0, tmax, dx, plot_fig=True, 
                 dN_fig=50, x=x, Z=Z, theta=theta, dt_fact=0.5)


# %% Compare with theory

def Ritter(x, t, h0, l0, g=9.81, r=0.95):
    c0 = np.sqrt((1-r)*g*h0)
    x0 = -c0*t
    x1 = 2*c0*t
    #
    condlist = [x - l0 <= x0, (x - l0 > x0) & (x - l0 <= x1),
                (x - l0 > x1)]
    #
    funclist = [lambda x: h0, 
                lambda x: (h0/9)*(2 - (x-l0)/(c0*t))**2, 
                lambda x: 0]
    #
    return np.piecewise(x, condlist, funclist)

inds_t = [0, 25, 50, 75]

fig, axarr = plt.subplots (2, 2, layout='constrained')
for i, ax in zip(inds_t, axarr.flatten()):
    ax.plot(x, U[i, 0, :])
    ax.plot(x, Ritter(x, t[i], h0, l0))

plt.show()