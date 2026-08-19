"""Public facade for COMSOL mesh parsing, topology, and artifacts."""

from ._mesh_artifacts import write_comsol_entity_maps, write_geometry_npz
from ._mesh_parsing import (
    MeshTypeBlock,
    ParsedMesh,
    parse_comsol_mphtxt,
    scale_mesh_coordinates,
    select_vacuum_domains,
)
from ._mesh_topology import (
    assign_part_ids_from_edge_entities,
    build_precomputed_arrays,
    domain_boundary_edge_vertex_ids,
    surface_triangles_from_mesh,
)

__all__ = (
    "MeshTypeBlock",
    "ParsedMesh",
    "assign_part_ids_from_edge_entities",
    "build_precomputed_arrays",
    "domain_boundary_edge_vertex_ids",
    "parse_comsol_mphtxt",
    "scale_mesh_coordinates",
    "select_vacuum_domains",
    "surface_triangles_from_mesh",
    "write_comsol_entity_maps",
    "write_geometry_npz",
)
