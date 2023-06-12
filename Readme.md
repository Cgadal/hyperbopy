# shallowpy

In this project, we solve various types of shallow water equation systems using a well-balanced path-conservative central-upwind scheme [[1]](#1) for the spatial discretization, and a stage-3 order-3 explicit strong stability preserving Runge-Kutta (eSSPRK 3-3) [[2]](#2) in time.

<a id="1">[1]</a> Diaz, M. J. C., Kurganov, A., & de Luna, T. M. (2019). Path-conservative central-upwind schemes for nonconservative hyperbolic systems. ESAIM: Mathematical Modelling and Numerical Analysis, 53(3), 959-985.

<a id="2">[2]</a> Isherwood, L., Grant, Z. J., & Gottlieb, S. (2018). Strong stability preserving integrating factor Runge--Kutta methods. SIAM Journal on Numerical Analysis, 56(6), 3276-3307.

# Models

## Two-layer shallow water (layerwise conservative)

We solve the two layer shallow water model:

```math
\begin{aligned}
(h_{1})_{t} + (q_{1})_{x} &= 0, \\
(h_{2})_{t} + (q_{2})_{x} &= 0, \\
(q_{1})_{t} + \left(\frac{q_{1}^{2}}{h_{1}} + \frac{g}{2}h_{1}^{2}\right)_{x} &= -g h_{1}(h_{2} + Z)_{x}, \\
(q_{2})_{t} + \left(\frac{q_{2}^{2}}{h_{2}} + \frac{g}{2}h_{2}^{2}\right)_{x} &= -g h_{2}(r h_{1} + Z)_{x},
\end{aligned}

```
with subscripts $1$ and $2$ denoting the upper light and lower heavy layers, respectively, and $r = \rho_1/\rho_2$. 

## Usage

See reference examples.


## Changelog

- 09/06/2023: First stable version with validated reference examples. So far, no shock for dam-break solution, which exhibits the Ritter solution.
  
- 07/06/2023: First commit