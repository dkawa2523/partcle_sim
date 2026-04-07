# Stabilization release v29

This release repairs the packaged examples so that the distributed ZIP can be executed immediately after unpacking.

## Fixed

- Added complete input datasets to `examples/minimal_2d`
- Added complete input datasets to `examples/minimal_3d`
- Verified `run_from_yaml.py` end-to-end on both examples
- Verified `tests/smoke_test.py` passes
- Added `particle_tracer_unified/__init__.py` and bumped package version to 29.0.0

## Verified commands

```bash
python -m pytest -q tests/smoke_test.py
python run_from_yaml.py examples/minimal_2d/run_config.yaml --output-dir out_2d
python run_from_yaml.py examples/minimal_3d/run_config.yaml --output-dir out_3d
```

## Next recommended step

Reconnect COMSOL field / geometry providers to the same `PreparedRuntime` path while preserving this package completeness.
