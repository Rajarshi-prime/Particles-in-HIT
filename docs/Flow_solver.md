# Flow solver

Covers `forced-dns.py` and the flow-evolution part of `forced-dns-prtcls.py`. Both integrate the incompressible Navier-Stokes equations on a triply-periodic cube using a pseudo-spectral method (Rogallo, 1981), split across MPI ranks by slabs along $x$.

## Governing equation

In Fourier space, for wavevector $\mathbf{k}$:

$$\partial_t \hat u_i = \widehat{(u \times \omega)_i} - i k_i \hat p - \nu k^{2\ell}\hat u_i + \hat f_i$$

- $u \times \omega$ (velocity cross vorticity) is the nonlinear term, computed in physical space and transformed back (the standard pseudo-spectral approach).
- $p$ is pressure, eliminated by projecting onto the divergence-free subspace: $\hat p = i\mathbf{k}\cdot\widehat{(u\times\omega)}/k^2$.
- $\ell$ is the hyperviscosity power (`lp`, passed as a command-line argument). $\ell=1$ is ordinary viscosity; $\ell>1$ concentrates dissipation near the grid cutoff, letting the inertial range extend closer to it.
- $\hat f$ forces the flow: at each step, the code measures the energy in the lowest shell(s) of $|\mathbf{k}|$ and adds just enough energy to replace what viscosity removed, so total energy stays statistically steady rather than decaying.

## Time stepping

RK4, with the viscous term treated implicitly (`viscosity_integrator = "implicit"`) so it doesn't restrict the time step at high hyperviscosity power. An exponential integrator, and explicit implementation is also implemented (commented) as alternatives.

## Dealiasing

Products in the nonlinear term are dealiased by zeroing modes above a spherical cutoff, using a phase-shifted grid (Patterson and Orszag, 1971) rather than the plain $2/3$-rule, since this permits a larger dealiased range.

## Runs and restarts

- Start-up (`forcestart = True`): random-phase velocity field with a prescribed initial spectrum, rescaled to a target energy.
- Restart (`forcestart = False`): loads the last saved velocity field from `savePath` and continues.
- `forced-dns-prtcls.py` always restarts from a field saved by `forced-dns.py` (or a previous particle run) — run the plain flow solver first until the energy trace (printed each save) is statistically steady, then switch to it.

## Output

Every `dt_save` time units: velocity field (`Fields_k_<rank>.npz`, one file per MPI rank, truncated to the dealiased wavenumbers), the energy spectrum, and the energy flux spectrum, all under `savePath/time_<t>/`.

## References

- Patterson and Orszag (1971). [Spectral Calculations of Isotropic Turbulence: Efficient Removal of Aliasing Interactions](https://doi.org/10.1063%2F1.1693365).
