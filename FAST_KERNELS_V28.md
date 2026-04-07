# Fast kernels in v28

v28 keeps policy and model resolution on the Python side and moves the free-flight update into Numba kernels.

- `kernel2d_numba.advance_particles_2d_inplace()` handles 2D free-flight updates.
- `kernel3d_numba.advance_particles_3d_inplace()` handles 3D free-flight updates.
- wall-hit refinement, wall-law evaluation and bookkeeping remain in Python for reviewability.

This is a deliberate split:
- high-frequency arithmetic is compiled
- low-frequency branching and policy remain readable

The current solver is a practical high-fidelity reference path for step-aware runs. The next optimization target is collision candidate narrowing and part lookup.
