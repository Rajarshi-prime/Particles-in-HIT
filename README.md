# Particles in HIT (tracers by default)


## Generated using Claude but checked by the owner.
Code to simulate passive tracer/inertial particles advected by forced, homogeneous, isotropic turbulence (HIT). Simulations run in parallel using MPI.

For inertial particles, it can also incorporate back-reaction via two-way coupling. For more, see the [docs](docs/). 

## Workflow

1. **Evolve the flow** with `forced-dns.py` until it reaches a statistically stationary state (energy stops growing and fluctuates around a steady mean). Method details: [HIT_3D](https://github.com/Rajarshi-prime/HIT_3D).
2. **Add tracers/ particles** with `forced-dns-prtcls.py`. It loads the last saved velocity field from step 1, seeds particles, and evolves flow + particles together. It calls the `MPI_particles` class in `particles.py` to interpolate the velocity field onto particle positions and to move particles between MPI ranks as they cross domain boundaries.
3. **Analyze output** with `plots.py` (energy and flux spectra).

## Files

| File | Role |
|---|---|
| `forced-dns.py` | Pseudo-spectral DNS solver for forced HIT. |
| `particles.py` | `MPI_particles` class: spectral interpolation of velocity to particle positions, particle exchange across MPI ranks. |
| `forced-dns-prtcls.py` | Continues a saved flow and evolves it together with tracers/particles. |
| `plots.py` | Reads saved output and plots various quantities. |

## Details

- [docs/flow_solver.md](docs/flow_solver.md) — governing equations, hyperviscosity, forcing, dealiasing.
- [docs/particles.md](docs/particles.md) — interpolation/extrapolation between grid and particles, one-way vs. two-way coupling, inertial particle equation of motion.

## Running

```bash
mpirun -n <num_ranks> python3 forced-dns.py <lp>
mpirun -n <num_ranks> python3 forced-dns-prtcls.py <lp>
```
`<lp>` is the hyperviscosity power (`lp = 1` recovers ordinary viscosity). Grid size `N`, run time `T`, save interval `dt_save`, particle count `Nprtcl`, and Stokes numbers `stb_s` are set at the top of each script.

## Physics References

- Rogallo, R.S. (1981). [Numerical experiments in homogeneous turbulence](https://ntrs.nasa.gov/api/citations/19810022965/downloads/19810022965.pdf). NASA TM 81315.
- Toschi, F. & Bodenschatz, E. (2009). [Lagrangian properties of particles in turbulence](https://doi.org/10.1146/annurev.fluid.010908.165210). *Annu. Rev. Fluid Mech.* 41, 375–404.
