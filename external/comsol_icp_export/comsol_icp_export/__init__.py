"""External COMSOL ICP export helpers.

This package is intentionally outside particle_tracer_unified. It may depend on
export artifacts produced by COMSOL, but the solver package must not import it.
"""

from .field_bundle import build_field_bundle_from_table, write_field_bundle

__all__ = ["build_field_bundle_from_table", "write_field_bundle"]
