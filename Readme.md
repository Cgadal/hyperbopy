# shallowpy

In this project, we solve various types of shallow water equation systems using:
- a well-balanced path-conservative central-upwind scheme for the spatial discretization (unless specified otherwise) [[1]](#1)
- a stage-3 order-3 explicit strong stability preserving Runge-Kutta [eSSPRK 3-3] in time [[2]](#2)

## Models

- $h$: layer height
- $u$: layer velocity
- $q = hu$: layer discharge
- $Z$: bottom topography
- $r = \rho_1/\rho_2$: layer density ratio ($r <=1$)

with subscripts $1$ and $2$ denoting the upper light and lower heavy layers, respectively.

### One-layer models

- #### One-layer shallow water (globally conservative)

  - `model = '1L_global'`

```math
\begin{aligned}

\left[h\right]_{t} + [q]_{x} &= 0, \\
[q]_{t} + \left[\frac{q^{2}}{h} + \frac{g(1-r)}{2}h^{2}\right]_{x} &= -g(1-r) h[Z]_{x}, \\

\end{aligned}

```

- #### One-layer shallow water (locally conservative)

  - `model = '1L_local'`

```math
\begin{aligned}

\left[h\right]_{t} + [hu]_{x} &= 0, \\
[u]_{t} + \left[\frac{u^{2}}{2} + g(1-r)(h + Z) \right]_{x} &= 0, \\

\end{aligned}

```

- #### One-layer non-hydrostatic shallow water (globally conservative)

  - `model = '1L_non_hydro_global'`

```math
\begin{aligned}

\left[h\right]_{t} + [q]_{x} &= 0, \\
[q]_{t} + \alpha_{\rm M} [M]_{t} + \left[\frac{q^{2}}{h} + \frac{g(1-r)}{2}h^{2}\right]_{x} &= -g(1-r) h[Z]_{x} - \alpha_{\rm N} N + p^{a} [h + Z]_{x}, \\

\end{aligned}

```

  where:

```math
\begin{aligned}

M &= \left[-\frac{1}{3}h^{3}[u]_{x} + \frac{1}{2}h^{2}u[Z]_{x}\right]_{x} + [Z]_{x}\left(-\frac{1}{2}h^{2}[u]_{x} + h u [Z]_{x}\right), \\

N &= \left[ [h^{2}]_{t}\left( h [u]_{x} - [Z]_{x} u \right)   \right]_{x} + 2[Z]_{x}[h]_{t}\left( h [u]_{x} - [Z]_{x} u\right) - [Z]_{x, t} \left(-\frac{1}{2}h^{2}[u]_{x} + h u [Z]_{x}\right), \\

\end{aligned}

```
  are the non-hydrostatic terms, and $p^{a}$ is an external constant pressure applied on the surface.


  > **Note**
  > The spatial discretization scheme is here a well-balanced central upwind scheme. For additional details, please refer to [[3]](#3).

### Two-layer models

- #### Two-layer shallow water (layerwise conservative)

  - `model = '2L_layerwise'`

```math
\begin{aligned}

\left[h_{1}\right]_{t} + [q_{1}]_{x} &= 0, \\
[h_{2}]_{t} + [q_{2}]_{x} &= 0, \\
[q_{1}]_{t} + \left[\frac{q_{1}^{2}}{h_{1}} + \frac{g}{2}h_{1}^{2}\right]_{x} &= -g h_{1}[h_{2} + Z]_{x}, \\
[q_{2}]_{t} + \left[\frac{q_{2}^{2}}{h_{2}} + \frac{g}{2}h_{2}^{2}\right]_{x} &= -g h_{2}[r h_{1} + Z]_{x},

\end{aligned}

```

- #### Two-layer shallow water (locally conservative)

  - `model = '2L_layerwise'`

```math
\begin{aligned}

\left[h_{1}\right]_{t} + [h_{1}u_{1}]_{x} &= 0, \\
[h_{2}]_{t} + [h_{2}u_{2}]_{x} &= 0, \\
[u_{1}]_{t} + \left[\frac{u_{1}^{2}}{2} + g(h_{1} + h_{2} + Z)\right]_{x} &= 0, \\
[u_{2}]_{t} + \left[\frac{u_{2}^{2}}{2} + g(rh_{1} + h_{2} + Z)\right]_{x} &= 0, \\

\end{aligned}

```

## Usage

See reference examples.

## Installation

- Clone or download this repository
- `cd shallowpy`
- `pip3 install -e ./` (editable mode installation)


## References

- <a id="1">[1]</a> Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53[3], 959-985.

- <a id="2">[2]</a> Isherwood, L., Grant, Z. J., & Gottlieb, S. (2018). Strong stability preserving integrating factor Runge--Kutta methods. SIAM Journal on Numerical Analysis, 56[6], 3276-3307.

- <a id="3">[3]</a> Chertock, A., & Kurganov, A. (2020). Central-upwind scheme for a non-hydrostatic Saint-Venant system. Hyperbolic Problems: Theory, Numerics, Applications, 10.

## Changelog

- **20/06/02023**:
  - adding non-hydro globally conservative on-layer model

- **13/06/2023**:
  - changing repo organization to be able to select within system of equations
  - adding the locally conservative two-layer model [dam break shows shock solution]
  - adding the globally conservative one-layer model [dam break shows Ritter solution]
  - adding the locally conservative one-layer model [dam break shows shock solution]

- **09/06/2023**: First stable version (two layer globally) with validated reference examples. So far, no shock for dam-break solution, which exhibits the Ritter solution.
  
- **07/06/2023**: First commit