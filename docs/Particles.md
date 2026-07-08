# Particles

Covers the `MPI_particles` class in `particles.py`.

Particle positions are stored per MPI rank in `self.coord`, an array of shape `(n_particles, 2d+1)`: the first `d` columns are position, the next `d` are velocity, and the last is mass. Each rank only holds particles currently inside its slab of the domain.

## Moving values between the grid and particles

Both directions use the same cosine-weighted stencil (order 4, i.e. the nearest 4 grid points along each axis), following Yeung & Pope (1988):

$$w(x, x_g) = \frac{1}{4}\left(1 + \cos\left(\frac{\pi(x-x_g)}{2\,\Delta x}\right)\right)$$

with the particle's value built from the product of $w$ across the three axes. Because particles can sit near the edge of a rank's slab, both directions exchange a few stencil points with neighboring ranks first.

- **Interpolation (grid → particle)**: `interp_cosine` reads a field (e.g. velocity, gradients etc) at every particle's exact position. `uinterp_cosine` is a second, independent implementation of only velocity interpolation operation. `uinterp` is an older, Lagrange-polynomial-based interpolator (lower order, not cosine-weighted).
- **Extrapolation (particle → grid)**: `exterp_cosine_scalar` and `exterp_cosine_vector` deposit a per-particle quantity back onto the grid with the same weights, divided by the cell volume so the result is a density. This is how a particle can push something (e.g. a reaction force) back onto the flow.
- **Combined pass**: `interp_exterp_cosine_scalar` and `interp_exterp_cosine_vector` do both directions inside the same stencil loop, so the neighbor-rank communication happens once instead of twice. Used when a particle needs the local flow value for its own equation of motion and must hand a quantity back to the grid in the same step.

Before use, call `to_interp(n)` / `to_exterp(n)` once to allocate buffers for a field with `n` components (3 for velocity, 1 for a scalar).

## One-way vs. two-way coupling

- **One-way coupling**: particles are moved by the flow, but don't affect it. This is what `forced-dns-prtcls.py` currently does — `interp_cosine` reads the velocity at each particle, and $d\mathbf{x}_p/dt = \mathbf{u}$.
- **Two-way coupling**: particles also feed back onto the flow. Each particle deposits a reaction (e.g. the drag force it exerts on the fluid) onto the grid with `exterp_cosine_vector`, and that field is added to the flow's momentum equation (Squires & Eaton, 1990). The extrapolation functions above exist for this; they are not yet called from `forced-dns-prtcls.py`, which only wires up one-way tracer advection.

## Inertial particle equation of motion

For a particle with finite Stokes number (as opposed to a zero-inertia tracer that just follows $\mathbf{u}$), the relevant equation of motion is the Maxey-Riley equation (Maxey & Riley, 1983), whose leading (Stokes-drag) term is: **check coefficients**

$$\frac{d\mathbf{v}_p}{dt} = \frac{\mathbf{u}(\mathbf{x}_p,t) - \mathbf{v}_p}{\tau_p} + \mathbf{g}$$

where $\tau_p$ is the particle's response time, set by its size and density relative to the fluid. `__init__` already computes the coefficients this needs:

- `self.factor`: converts a particle's stored mass to its Stokes number, $St = (m/\text{factor})^{2/3}$, where $\text{factor} = \left(\dfrac{\nu\,\tau_\eta}{2\rho_p}\right)^{3/2}\dfrac{36\pi\rho_p}{M_0}$ ($\tau_\eta$ = Kolmogorov time, $\rho_p$ = particle-to-fluid density ratio, $M_0$ a mass normalization).
- `self.growthfactor` $= \dfrac{9\pi\nu}{2\rho_p}$ and `self.decelerationfactor` $= \sqrt{\dfrac{\rho_p}{8\nu}}$: coefficients for the particle's Stokes response time.

These are defined but not yet read anywhere else in the code. The current `pRHS` sets the velocity-derivative rows of the RHS to zero (pure tracer behavior) rather than using them to compute a drag force. Wiring in two-way coupled inertial particles means: interpolate $\mathbf{u}$ at the particle (`interp_cosine`), use it and `self.st` to evaluate the drag term above for $d\mathbf{v}_p/dt$, and extrapolate the reaction force onto the grid (`exterp_cosine_vector`) to add into `full_RHS`'s right-hand side for the flow.

## Moving particles between ranks

- `particle_exchange(coord)`: reassigns particles to whichever rank now owns their $x$-slab, after a position update may have carried them across a rank boundary.
- `send(x, args)`: the general version — also carries along any auxiliary per-particle arrays (e.g. `interpmat`, `prtclid`) so they stay matched to the right particle after the exchange.
- `update_intrinsic()`: recomputes each particle's Stokes number from its stored mass (`self.st`).

## References

- Yeung, P.K. & Pope, S.B. (1988). [An algorithm for tracking fluid particles in numerical simulations of homogeneous turbulence](https://doi.org/10.1016/0021-9991(88)90022-8). *J. Comput. Phys.* 79, 373–416.
- Maxey, M.R. & Riley, J.J. (1983). [Equation of motion for a small rigid sphere in a nonuniform flow](https://doi.org/10.1063/1.864230). *Phys. Fluids* 26, 883–889.
- Squires, K.D. & Eaton, J.K. (1990). [Particle response and turbulence modification in isotropic turbulence](https://doi.org/10.1063/1.857620). *Phys. Fluids A* 2, 1191–1203.
- Toschi, F. & Bodenschatz, E. (2009). [Lagrangian properties of particles in turbulence](https://doi.org/10.1146/annurev.fluid.010908.165210). *Annu. Rev. Fluid Mech.* 41, 375–404.