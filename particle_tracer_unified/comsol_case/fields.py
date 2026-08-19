"""Stable entry points for COMSOL field artifacts.

Two storage kinds share one semantic mapping.  ``pack_field_bundle`` writes a
regular grid resampled from the solution; ``pack_mesh_field_bundle`` keeps the
solution on the COMSOL mesh.  Only the builder chooses between them.
"""

from __future__ import annotations

from ._field_mesh import PackedMeshField, pack_mesh_field_bundle
from ._field_profile import build_profile_field_bundle, field_manifest
from ._field_support import PackedField, pack_field_bundle
from .profiles import BuildProfile

__all__ = (
    "BuildProfile",
    "PackedField",
    "PackedMeshField",
    "build_profile_field_bundle",
    "field_manifest",
    "pack_field_bundle",
    "pack_mesh_field_bundle",
)
