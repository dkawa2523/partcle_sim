# Skill 02: COMSOL Access and Export Strategy

## Purpose

COMSOL `.mph` から必要情報を抽出する。ただし、COMSOL環境・LiveLink・APIの有無は環境ごとに違うため、Codexはまず利用可能な手段を検出し、読み取り専用exportを行う。

## Supported access routes

Use the first available route:

1. Existing exported CSV / NPZ / MPHTXT files.
2. COMSOL GUI export performed by the operator.
3. COMSOL command line or Java API script in `external/comsol_export/`.
4. LiveLink for MATLAB, if available locally.
5. If none are available, create an export request checklist and stop.

## Hard rule

The solver package must not read `.mph` directly. Extraction belongs under:

```text
external/comsol_export/
tools/
```

not under:

```text
particle_tracer_unified/solvers/
particle_tracer_unified/core/
```

## Environment detection

Codex should inspect, but not assume:

```text
comsol command availability
MATLAB availability
COMSOL installation path
existing external/comsol_export scripts
existing exported data folders
```

Record result in `extraction_manifest.yaml`.

## Export groups

Export only what is required for the selected comparison mode.

### Required for field-only production case

```text
geometry mesh or boundary primitives
boundary/part map
field variables and units
coordinate system and scale
material/wall table if available
```

### Required for COMSOL faithful comparison

```text
all production case exports
COMSOL release table
COMSOL particle trajectory long table
COMSOL output times
wall law / boundary feature mapping
force inventory
particle properties
stochastic settings or statement that stochastic is disabled
```

## Export safety rules

- Export read-only data.
- Do not rerun COMSOL solves unless explicitly requested.
- Do not change model physics.
- Do not overwrite existing exports without writing a new timestamped folder.
- Always record component, study, dataset, and solution index.

## Output layout

Recommended:

```text
comsol_exports/<case_name>/
  extraction_manifest.yaml
  geometry/
  fields/
  particles/
  walls/
  logs/
```

## Pass criteria

- Extraction route is recorded.
- Exported files are listed with roles.
- Missing items are explicit.
- No solver code is changed.

## Fail criteria

- A new model is evaluated using VIGUS-specific variable names without confirmation.
- Field variables are exported without units.
- Particle trajectory references are exported without particle IDs or times.
