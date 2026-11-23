# Selected nodal CSV import

`chamber_particles_tools.csv_import` is the installed, producer-neutral numeric mesh/field client.
It writes a canonical HDF5 bundle; it does not read MPH, solve fields, select particle physics or compare trajectories.
The explicit provider is `csv` for caller-owned data or `comsol` for identified native exports.

```console
uv run --locked python -m chamber_particles inputs import-csv mapping.yaml background
uv run --locked python -m chamber_particles create recipe.yaml -o input
```

After wheel installation the same dispatcher works without a checkout. Python uses
`adapt_from_configuration(configuration_path, output_directory)` from this package.
The fresh directory publishes `bundle.h5` and `report.json` together. Use `background/bundle.h5` in a format-2 input recipe; the single
publisher fills its canonical hash and calls public load/check. Import success alone is not model readiness or accuracy.

## Format 4 and CSV files

One static Cartesian XY or RZ volume domain is selected. All selected fields are continuous nodal scalar/vectors
on the same P1 triangulation. Mesh IDs and explicit domain/side meaning determine the selected rows.
Required input files are:

| File role | Exact CSV header |
|---|---|
| XY vertices | `vertex_id,x_m,y_m` |
| RZ vertices | `vertex_id,r_m,z_m` |
| triangles | `triangle_id,node1_id,node2_id,node3_id,domain_id` |
| quadrilaterals | `quadrilateral_id,node1_id,node2_id,node3_id,node4_id,domain_id` |
| provider csv boundaries | `boundary_edge_id,node1_id,node2_id,boundary_id` |
| provider comsol boundaries | `boundary_edge_id,node1_id,node2_id,mphtxt_boundary_index,comsol_boundary_id` |
| provider csv field dictionary | `column,expression,unit` |
| provider comsol field dictionary | `column,COMSOL_expression,unit` |

Empty triangle/quad files still need their headers. Q1 reference order is `(−1,−1),(+1,−1),(−1,+1),(+1,+1)`,
not cyclic perimeter order. Quads are explicitly triangulated using the supported quality rule; this changes
the field representation from Q1 to P1. `field_values` contains numeric rows without a header, with dictionary
column order; `%` comments are permitted. Expression strings are provenance, never an evaluated DSL.

A complete generic mapping shape is:

```yaml
format_version: 4
provider: csv
source:
  vertices_path: vertices.csv
  triangles_path: triangles.csv
  quadrilaterals_path: quads.csv
  boundary_edges_path: boundaries.csv
  field_values_path: values.csv
  field_columns_path: columns.csv
domain:
  coordinate_system: cartesian_xy
  external_id: 1
  layout_name: background
  quadrilateral_order: q1_reference
  excluded_boundary_ids: []
boundary_groups:
  - {name: wall, external_ids: [1, 2, 3, 4], material_id: 0}
fields:
  coordinate_columns: [x_m, y_m]
  inside_column: inside
  node_id_column: vertex_id
  domain_scope: null
  coordinate_match_tolerance_m: 1.0e-14
  mappings:
    - name: electric_field
      columns: [Ex, Ey]
      components: [x, y]
      stored_basis: cartesian_xy
      association: node
      unit: V/m
      input_unit: V/m
      scale: 1.0
      offset: 0.0
      producer_record:
        producer: analytic input
        time: {representation: static, selection: steady}
        domain_ids: [1]
        side: domain
        recovery: none
        continuity: continuous
        authority: supplied analytic primitive
        topology: fixed
        support: full_selected_domain
        evidence: analytic formula and input-file identities
      axis_transform: null
```

These IDs, tolerance and fields are illustrative, not new-case defaults. `scale` must be positive and finite;
`offset` is finite. Stored components/basis/unit must match the selected model's canonical SI requirements.
No coordinate origin/basis transformation or implicit unit inference is performed.

## Scope, identity and native meaning

Prefer a genuine scoped mesh node ID and verify its coordinates. Without it, set `node_id_column: null` and
prove the selected coordinate bijection within the declared export tolerance. Row/probe indices are not mesh IDs.
`domain_scope` is null or `{column, mode, evidence}` with `equals_domain_id` or `finite`.
The latter needs independent evidence that the anchor column identifies the selected producer domain;
finite values alone do not establish physical support. Surviving duplicate or missing selected rows are rejected.

For `provider: comsol`, every producer record additionally requires dataset, solution and a nonempty member
mapping with the actual selection/indexing. Preserve actual native time, side, recovery, reader/version and
export evidence. Unrecorded historical facts remain unrecorded; a successful import does not authenticate them.
Generic `provider: csv` requires no fabricated native dataset/solution/member.

RZ axis exclusions must be exact axis facets; all remaining perimeter boundary IDs map exactly once to groups.
The symmetry axis is not a particle wall. `axis_transform` defaults to null; an explicit correction declares
component, maximum absolute correction and physical reason. Raw values and changed-node/cell effects are recorded.
No missing field is filled with zero or nearest neighbors. Selected-node finite coverage and off-node transfer
accuracy are separate checks; particle dt refinement cannot remove field-representation error.

Time snapshots, moving/remeshed geometry, discontinuities, cell fields, multi-domain union and a general CAD
repair/importer are outside this client. Other representations require a justified producer and independent checks.

```console
uv run --locked pytest tests/verification/test_csv_adapter.py -q
```
