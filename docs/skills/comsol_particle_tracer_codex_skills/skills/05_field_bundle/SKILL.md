# Skill 05: Field Bundle Export and Validation

## Purpose

COMSOLの定常場・時系列場を、本コードがsampleできるfield bundleへ変換する。力場比較では単位、座標軸、field support、導出量の有無が最重要。

## Inputs

```text
COMSOL field exports
field_mapping.csv template
coordinate system metadata
geometry boundary bundle
```

## Required metadata per field

For every exported field variable record:

```text
source variable name
canonical quantity name
unit
coordinate component direction
dataset / solution index
stationary or time dependent
interpolation grid or mesh type
valid support mask availability
```

## Common quantities

Examples; do not assume names:

```text
flow velocity components
pressure
gas temperature
gas density
gas viscosity
electric field components
electric potential
temperature gradient
grad(E^2)
plasma properties if used
```

## Steps

### 1. Build field mapping

Write `field_mapping.csv`:

```csv
canonical_name,comsol_variable,unit,component_axis,required_for,notes
```

### 2. Export on an appropriate support

Allowed:

```text
regular rectilinear grid
triangle mesh field bundle
explicit samples for comparison probes
```

Do not extrapolate outside COMSOL support to make solver run.

### 3. Validate field support

Generate `field_validation_report.json` containing:

```text
axis ranges
grid spacing or mesh element summary
valid mask counts
mixed stencil counts if known
hard invalid regions
release point field support status
```

### 4. Compare field samples before trajectories

If COMSOL point probe samples exist, run field compare before particle solver.

## Outputs

```text
fields.npz or equivalent provider bundle
field_mapping.csv
field_validation_report.json
optional field_validation_error.csv
```

## Pass criteria

- Units and component axes are explicit.
- Release points lie in clean field support or failure is explained.
- No missing force-required field is silently filled.

## Fail criteria

- Using `x,y` for an RZ model without mapping to `r,z`.
- Treating electric potential as electric field without differentiating/exporting E.
- Filling missing gas properties from defaults in faithful mode without manifest approval.
