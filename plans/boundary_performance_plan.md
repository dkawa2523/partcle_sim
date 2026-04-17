# Boundary Performance Notes

Status: active implementation record.

This note keeps the 10k COMSOL runtime issue focused. The goal is to reduce
boundary-check cost without changing `dt`, `t_end`, wall laws, sources, or the
particle physics model.

## Problem

The slow 10k ICP run was dominated by repeated boundary work:

```text
solver_core_s ~= 7612 s
solver_step_count = 5000
edge_prefetch_batch_candidate_count = 48,409,330
primary_hit_count = 592
boundary_edges = 764
```

This is not primarily a visualization or output-size problem. Most particles
are far from walls for most steps, but the old path still paid for exact
geometry and edge checks too often.

## Implemented Direction

The current implementation keeps the solver behavior simple:

1. Add timing fields to `solver_report.json`:
   `freeflight_s`, `collision_classify_s`, `boundary_sdf_prefilter_s`,
   `inside_sdf_prefilter_s`, `inside_check_s`, `edge_prefetch_s`,
   `valid_mask_retry_s`, `collider_resolution_s`, `output_step_summary_s`.
2. Use the existing geometry SDF as a conservative narrow-band filter.
   Particles are allowed to skip exact boundary checks only when start,
   midpoint, and end positions are safely inside the domain.
3. Do not skip if SDF is missing, non-finite, near a wall, outside the SDF
   grid, or the particle has a non-clean valid-mask status.
4. Use SDF strict-inside checks before the expensive polygon inside test.
5. Compile the 2D segment-edge hit loop with Numba and keep exact
   segment-edge intersection as the final hit decision.

This avoids case-specific rescue logic. It does not push particles, repair
fields, change wall reactions, or tune time step settings.

## Validation Snapshot

Short 10k check case: same particle count, shortened to 50 solver steps only
for performance measurement.

```text
solver_core_s = 1.05 s
step_loop_s = 1.05 s
freeflight_s = 0.67 s
collision_classify_s = 0.36 s
boundary_sdf_prefilter_s = 0.14 s
inside_sdf_prefilter_s = 0.05 s
inside_check_s = 0.00 s
edge_prefetch_s = 0.15 s
boundary_far_skip_count = 279,001
boundary_near_check_count = 220,999
edge_prefetch_batch_candidate_count = 220,999
numerical_boundary_stopped_count = 0
invalid_mask_stopped_count = 0
max_hits_reached_count = 0
unresolved_crossing_count = 0
```

Required checks passed:

```text
py -3 -m compileall -q particle_tracer_unified run_from_yaml.py
py -3 -m pytest -q tests\smoke_test.py
py -3 -m pytest -q tests\regression_test.py -k "classify_trial_collisions or boundary"
```

## Remaining Large Items

- Full 10k long run must be repeated only when the user wants the heavy
  verification. The short check shows the boundary bottleneck is removed, but
  it does not replace physical validation over the full `t_end`.
- 3D acceleration is not complete. Do not assume the 2D SDF path solves 3D
  mesh or volume-boundary cases.
- If a future case remains slow after this change, profile first. The next
  likely large target is compiled free-flight or field sampling for dense
  time-varying charged-particle fields.

## Do Not Reintroduce

- Solver-side field clipping, particle push-off, or release-position rescue.
- `dt` or `t_end` tuning mixed into boundary-acceleration validation.
- Case-specific wall/source hacks to make counters look better.
- Old mesh production paths without verified mesh field data.
- Tests that only lock implementation details and do not protect numerical or
  user-facing behavior.
