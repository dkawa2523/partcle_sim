# Skill 08: Build Solver Case and Run Preflight

## Purpose

COMSOL export資産を本コードのcanonical caseへ変換し、solver実行前にimport/preprocessの失敗を検出する。

## Inputs

```text
comsol_case_manifest.yaml
geometry bundle
field bundle
particles release table
wall law map
force inventory
```

## Steps

### 1. Build case directory

Recommended layout:

```text
examples/<case_name>_from_comsol/
  run_config.yaml
  comsol_case_manifest.yaml
  particles_release_raw.csv
  part_walls.csv
  materials.csv
  fields.npz
  geometry.npz
```

### 2. Select mode

For COMSOL trajectory comparison:

```yaml
mode: comsol_faithful
```

For production surface release:

```yaml
mode: surface_release_production
```

### 3. Run preflight only

Do not solve trajectories until preflight passes.

Typical command:

```powershell
py -3 run_from_yaml.py <case>/run_config.yaml --check-input --output-dir <out_check>
```

### 4. Inspect preflight artifacts

Required:

```text
prepared_runtime_summary.json
input/preflight reports if available
field support counts
source preprocessing summary if production mode
axis/coordinate report
```

### 5. Gate on mode rules

Faithful mode:

```text
no source preprocessing
strict field support
manifest fields complete
```

Production mode:

```text
boundary_release allowed only if explicit
capture tolerance and inward offset separated
```

## Outputs

```text
preflight_report.json
prepared_runtime_summary.json
case_build_report.md
```

## Pass criteria

- Coordinate system is correct.
- Field support is clean for required points.
- Faithful mode has no hidden repair.
- Production boundary release is explicit.

## Fail criteria

- Running solver before field/release preflight passes.
- Fixing preflight failure by broadening solver valid mask silently.
- Mixing faithful release with production boundary snap.
