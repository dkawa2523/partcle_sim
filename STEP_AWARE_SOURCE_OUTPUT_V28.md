# Step-aware source/output in v28

v28 makes step-aware control part of the normal execution path rather than an external wrapper.

The control chain is:

1. `process_steps.csv` or `recipe_manifest.yaml` defines time windows.
2. `core.process_steps.apply_process_step_controls()` merges run-level defaults and per-step overrides.
3. `core.source_events.compile_source_events()` resolves event timing against steps/transitions.
4. `core.source_materials.apply_source_models()` resolves source law, event gain, release-time shift and source enable flags.
5. `solvers.high_fidelity_common.run_prepared_runtime()` uses the active step to decide physics scales and output policy.

Supported step-aware source fields:
- `source_law_override`
- `source_speed_scale`
- `source_release_time_shift_s`
- `source_event_gain_scale`
- `source_enabled`

Supported step-aware output fields:
- `output_segment_name`
- `output_save_every_override`
- `output_save_positions`
- `output_write_wall_events`
- `output_write_diagnostics`

This means a recipe step can directly change the source law, source gain and output segmentation without changing the solver kernel.
