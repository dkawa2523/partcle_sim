# Skill 03: Extraction and Case Manifest

## Purpose

COMSOL export資産を、本コードが安全に消費できるmanifestへ整理する。manifestは推測補正の代わりに使う明示契約であり、過剰なschema frameworkではなく、比較に必要な最小情報を保持する。

## Inputs

```text
extraction_manifest.yaml
COMSOL exports
existing case manifest if present
templates/comsol_case_manifest_minimal.yaml
```

## Steps

### 1. Create `comsol_case_manifest.yaml`

Use the template and fill:

```text
comsol version / model / component / study / dataset
coordinate system and scale
geometry export paths
field bundle paths and variable mappings
release table path
particle reference path
boundary map path
wall law path
force inventory
particle property source
stochastic policy
```

### 2. Verify minimum fields

For `comsol_faithful`, fail if missing:

```text
coordinates.coordinate_system
coordinates.coordinate_scale_m_per_model_unit
particles.release_table
fields mappings or bundle
boundaries map and wall laws
forces enabled list
particle result reference or statement that only field/acceleration compare is possible
```

### 3. Mark comparison scope

Set one of:

```text
comparison_scope: field_only
comparison_scope: acceleration_only
comparison_scope: first_step
comparison_scope: trajectory
comparison_scope: boundary_events
comparison_scope: ensemble
```

Do not run a comparison beyond the available scope.

### 4. Record omissions

If a COMSOL model does not contain Particle Tracing, write:

```text
particle_reference_available: false
reason: field-only model
allowed_comparisons: [field, acceleration with supplied particle states]
```

## Outputs

```text
comsol_case_manifest.yaml
manifest_validation_report.json
```

## Pass criteria

- Manifest clearly states what can and cannot be compared.
- Coordinate system and scale are explicit.
- Missing fields fail only the affected comparison layers.

## Fail criteria

- A trajectory comparison is attempted without COMSOL particle IDs and times.
- Release preprocessing is enabled in a faithful manifest.
- Unknown wall laws are silently approximated.
