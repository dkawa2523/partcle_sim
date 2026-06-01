# Wall Law Catalog

This catalog documents the wall laws accepted by the production runtime and by
`comsol_faithful` import. Wall laws are resolved once while building the runtime
wall catalog; unknown values must fail instead of falling through to specular
reflection.

The solver does not contain VIGUS-specific wall IDs, part IDs, or case-specific
wall branches. Wall behavior is selected only from explicit wall/material
tables or the global `wall.default_mode` / `wall.mode` setting.

## Status Key

- `supported`: directly represented by current solver behavior.
- `supported with limitations`: represented, but not all COMSOL sub-options are
  implemented.
- `unsupported`: no runtime behavior exists; import/config must fail.
- `forbidden in comsol_faithful unless explicitly mapped`: a COMSOL export must
  be mapped to one of the supported canonical laws before faithful comparison.

## Current Coverage

| COMSOL-style behavior | Canonical runtime law | Status | Notes |
| --- | --- | --- | --- |
| stick | `stick` | supported | Terminal stuck state; particle velocity is zeroed. |
| freeze | `stick` | supported with limitations | Imported as `stick`; there is no separate COMSOL freeze state that remains dynamically active. |
| disappear / absorb | `absorb` | supported | Terminal absorbed state. `disappear` is an alias. |
| escape / open / outflow | `escape` | supported with limitations | Terminal escaped state. `field_support_exit` is an internal exit-style law used for field-support boundaries. |
| pass through | `pass_through` | supported with limitations | Non-colliding boundary in 2D edge hit detection. If a pass-through hit reaches the generic wall step, velocity is left unchanged and the particle advances through the hit; 3D exports should still exclude internal/pass-through surfaces from active collision surfaces. |
| specular bounce / bounce | `specular` | supported | Uses normal reflection at hit-time velocity and configured normal restitution. |
| diffuse reflection / diffuse scattering | `diffuse` | supported with limitations | Samples a diffuse reflection direction at the incoming speed scaled by restitution. COMSOL diffuse temperature / thermal re-emission is not implemented. |
| mixed diffuse/specular | `mixed_specular_diffuse` | supported with limitations | Uses `wall_diffuse_fraction`; otherwise specular. Separate tangential restitution is not supported. |
| general/custom reflection | n/a | unsupported | Production operators may choose an explicit supported law as a modeling approximation, but there is no custom reflection engine and faithful import must not silently approximate it. |
| critical sticking velocity | `critical_sticking_velocity` | supported with limitations | Sticks when impact normal speed is at or below `wall_critical_sticking_velocity_mps`; otherwise reflects through the normal specular path unless other stick probability applies. |
| thermal re-emission | n/a | unsupported | COMSOL `diffuse_temperature` / thermal re-emission metadata is rejected in strict COMSOL import. |

## Fail-Fast Rules

`comsol_faithful`:

- `boundaries.wall_law_file` is required by the manifest.
- Every wall/boundary part in the boundary map must have an explicit wall law.
- Unknown or unsupported COMSOL wall law names fail during import.
- COMSOL thermal re-emission and tangential restitution options fail because
  they are not represented by the current runtime.

Production:

- Normal production runs may use supported wall law aliases in materials,
  `part_walls.csv`, or `wall.default_mode`.
- Unknown wall laws fail during wall catalog construction.
- There is no automatic "closest supported law" approximation. If a production
  case intentionally approximates a custom behavior, the config must name the
  supported law explicitly.
