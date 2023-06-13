# shallowpy

In this project, we solve various types of shallow water equation systems using:
- a well-balanced path-conservative central-upwind scheme [[1]](#1) for the spatial discretization
- a stage-3 order-3 explicit strong stability preserving Runge-Kutta (eSSPRK 3-3) [[2]](#2) in time.

<a id="1">[1]</a> Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

<a id="2">[2]</a> Isherwood, L., Grant, Z. J., & Gottlieb, S. (2018). Strong stability preserving integrating factor Runge--Kutta methods. SIAM Journal on Numerical Analysis, 56(6), 3276-3307.

## Models

- $h$: layer height
- $u$: layer velocity
- $q = hu$: layer discharge
- $r = \rho_1/\rho_2$: layer density ratio ($r <=1$)

with subscripts $1$ and $2$ denoting the upper light and lower heavy layers, respectively

### One-layer shallow water (globally conservative)

- `model = '1L_global'`

```math
\begin{aligned}
(h)_{t} + (q)_{x} &= 0, \\
(q)_{t} + \left(\frac{q^{2}}{h} + \frac{g}{2}h^{2}\right)_{x} &= -g(1-r) h(Z)_{x}, \\
\end{aligned}

```

### One-layer shallow water (locally conservative)

- `model = '1L_local'`

```math
\begin{aligned}
(h)_{t} + (hu)_{x} &= 0, \\
(u)_{t} + g(1-r)\left(h + Z \right)_{x} &= -u(u)_{x}, \\
\end{aligned}

```

### Two-layer shallow water (layerwise conservative)

- `model = '2L_layerwise'`

```math
\begin{aligned}
(h_{1})_{t} + (q_{1})_{x} &= 0, \\
(h_{2})_{t} + (q_{2})_{x} &= 0, \\
(q_{1})_{t} + \left(\frac{q_{1}^{2}}{h_{1}} + \frac{g}{2}h_{1}^{2}\right)_{x} &= -g h_{1}(h_{2} + Z)_{x}, \\
(q_{2})_{t} + \left(\frac{q_{2}^{2}}{h_{2}} + \frac{g}{2}h_{2}^{2}\right)_{x} &= -g h_{2}(r h_{1} + Z)_{x},
\end{aligned}

```

## Usage

See reference examples.

## Installation

- Clone or download this repository
- `cd shallowpy`
- `pip3 install -e ./`


## Changelog

- **13/06/2023**:
  - changing repo organization to be able to select within system of equations
  -  adding the globally conservative one-layer model (dam break shows Ritter solution)
  -  adding the locally conservative one-layer model (dam break shows shock solution)

- **09/06/2023**: First stable version with validated reference examples. So far, no shock for dam-break solution, which exhibits the Ritter solution.
  
- **07/06/2023**: First commit