# Skill 04: Geometry, Boundary, and Part Extraction

## Purpose

COMSOL geometry / mesh / boundary selection informationを、本コードの境界認識に使える形へ変換する。境界ID・パーツID・法線・axis情報を誤ると、wall hit、surface release、near-wall評価がすべて壊れる。

## Inputs

```text
COMSOL mesh export or mphtxt
boundary selections / named selections
part map if available
component coordinate system
templates/boundary_part_map.csv
templates/wall_law_map.csv
```

## Steps

### 1. Identify geometry dimension

Classify:

```text
2D Cartesian
2D axisymmetric RZ
3D Cartesian
unsupported or ambiguous
```

For axisymmetric RZ:

- record radial axis name
- record axial axis name
- identify r=0 axis if present
- do not treat r=0 as ordinary wall unless wall law explicitly says so

### 2. Export explicit boundary primitives

Required by solver:

```text
2D: boundary edges / polyline segments with part_id
3D: boundary triangles with part_id
```

If only field support mask exists, do not pretend it is chamber wall geometry.

### 3. Create boundary-to-part map

Write `boundary_part_map.csv`:

```csv
boundary_id,part_id,boundary_name,selection_name,is_axis_boundary,is_open_boundary,notes
```

### 4. Create wall law table

Write `wall_law_map.csv`:

```csv
part_id,boundary_id,wall_law,stick_probability,restitution,diffuse_fraction,reflectivity,material_id,notes
```

Use explicit values from COMSOL if available. If unavailable, mark `unknown` and fail faithful wall comparison.

### 5. Validate geometry/field axis consistency

Check:

- geometry bounds cover field axes
- release points are in same coordinate units
- boundary primitives align with field support
- axisymmetric RZ has non-negative radial axis

## Outputs

```text
geometry_bundle.npz or mesh-derived boundary primitives
boundary_part_map.csv
wall_law_map.csv
geometry_boundary_report.json
```

## Pass criteria

- Explicit boundary primitives exist.
- Axis and units are known.
- Boundary/part/wall IDs are traceable.

## Fail criteria

- Field support boundary is used as chamber wall without proof.
- Axisymmetric model is treated as Cartesian by default.
- Unknown wall law is accepted in faithful mode.
