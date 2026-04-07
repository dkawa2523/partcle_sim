# Numerics Remaining Tasks (2026-04-05)

This note captures the remaining behavior-facing items after the current tranche:

- `IntegratorSpec`, `BoundaryService`, `RuntimeState`
- default-diagnostic plus opt-in authoritative `valid_mask` handling, including collision replay
- 2D boundary-edge topology fail-fast validation
- shared grid/field sampling helpers
- direct `BoundaryHit` flow inside solver collision replay
- shared solver entrypoints with 2D/3D compatibility wrappers

The intent is to keep the next steps behavior-aware and low-complexity.

## Current Safe State

- Public integrator names, YAML, CLI, and output file names are unchanged.
- `valid_mask` defaults to diagnostic-only in the solver hot path, with opt-in `retry_then_stop` available for authoritative handling.
- `retry_then_stop` now stops only on `hard_invalid`; near-wall `mixed_stencil` samples remain diagnostic-only.
- Precomputed fields now get a load-time geometry-aware narrow-band regularization so wall-adjacent support loss is addressed before runtime sampling.
- A 2D triangle-mesh field backend now exists as an opt-in accuracy-path prototype via `providers.field.kind = precomputed_triangle_mesh_npz`.
- 2D nested-hole truth is supported for disjoint loops via even-odd parity.
- 2D boundary-edge inputs with branch/dangling vertices now fail early instead of being silently reconstructed.

## Completed In This Cleanup Line

- `BoundaryService` now flows `BoundaryHit` directly through trial prefetch and collision replay.
- The tuple compatibility helpers were removed from `high_fidelity_common.py`.
- Grid and field sampling rules are centralized in `core/grid_sampling.py` and `core/field_sampling.py`.
- `solver2d.py` and `solver3d.py` remain compatibility wrappers over the shared `solver_entrypoints.py` path.
- `high_fidelity_common.py` is now a thin compatibility hub, while free-flight, collision replay, and runtime loop logic live in dedicated solver modules.
- COMSOL case building can now emit both:
  - `generated/comsol_field_2d.npz` for the rectilinear compatibility path
  - `generated/comsol_field_mesh_2d.npz` for the 2D triangle-mesh backend

## Prioritized Remaining Work

### 1. Decide whether the 2D triangle-mesh backend can become the default accuracy path

Current issue:
- The 2D triangle-mesh backend is implemented, but current 10k rollout measurements still show large `hard_invalid` counts and poor class-match versus the rectilinear reference path.

Why this matters:
- This is the main structural attempt to align field support with wall/collision geometry truth.
- If it cannot meet acceptance on realistic runs, further rectilinear regularization tweaks are unlikely to be the right long-term fix.

Recommended gate before promoting the mesh path:
- Measure `class_match_ratio`, `hard_invalid` counts, and wall-event concentration on target runs.
- Confirm whether remaining mesh `hard_invalid` points are caused by:
  - collision replay offsets leaving the triangle support
  - build-time vertex-value export error
  - lack of support for exact wall-adjacent sampling
- If mesh export from rectilinear bundles remains insufficient, move to a raw FEM/direct-element import under the same provider schema instead of adding more runtime heuristics.

Available tooling:
- `tools/evaluate_valid_mask_rollout.py` now generates `diagnostic` and `retry_then_stop` variants from one base config and writes a rollout summary with the standard checks.

Expected effect:
- A clean path toward matching field support to geometry truth without adding wall-aware runtime heuristics.
- A clear decision point on whether the next phase should be:
  - mesh/direct-element accuracy work
  - or simply keeping rectilinear regularization as the compatibility/default path

### 2. Keep 2D topology support scoped and explicit

Current issue:
- Disjoint closed loops are supported.
- Shared-vertex branching and dangling topologies are rejected.
- More complex topology support would need a deliberate data contract, not silent heuristics.

Why this matters:
- Silent reconstruction of ambiguous geometry is worse than a clear failure.
- Complex CAD-derived cases can otherwise become difficult to diagnose.

Recommended approach:
- Keep the current fail-fast rule for ambiguous edge graphs.
- Only expand beyond disjoint degree-2 loops if the input format or preprocessing pipeline can represent topology unambiguously.

Expected effect:
- More predictable geometry behavior.
- Clearer failure modes for data-preparation issues.

## Suggested Order

1. Evaluate whether the 2D triangle-mesh backend is good enough to keep as the preferred accuracy path.
2. If not, add a raw FEM/direct-element import behind the same provider schema instead of growing runtime heuristics.
3. Revisit richer 2D topology only if a concrete input requirement appears.
