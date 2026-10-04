# Archived former implementation

This directory contains the repository contents that existed before the
clean-room implementation reset on 2026-09-23. Files were moved here to retain
local evidence and recovery options; they are not an active implementation.

- Do not activate `.venv` or reuse the archived `pyproject.toml` / `uv.lock`.
- Do not import `particle_tracer_unified` from the new solver.
- Do not repair or extend archived production paths.
- Inspect this archive only for failure cases, candidate equations, historical
  inputs, and provenance. Independently verify any adopted idea.
- Git history remains at the workspace root and is the recovery mechanism.

Every former Git-tracked implementation file removed from the repository root
was checked against its archive copy before the move was committed. Historical
scratch directories were also moved locally, but generated caches and scratch
content remain ignored.

Producer-native `*.mph` and `*.mphtxt` files are intentionally excluded from
the archive commit by the repository ignore policy. Their prior revisions stay
recoverable from Git history; current producer inputs belong in external
reference storage such as `model_dataset/`, not in the solver source tree.
