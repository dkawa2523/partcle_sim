"""Public entry point for one-shot pre-v0.2 case migration."""

from __future__ import annotations

from ._migration.legacy import RemovedSourceGenerationError
from ._migration.service import MigrationResult, migrate_legacy_case

__all__ = [
    "MigrationResult",
    "RemovedSourceGenerationError",
    "migrate_legacy_case",
]
