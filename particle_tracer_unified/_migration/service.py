"""Migration input loading, output writing, and top-level orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from particle_tracer_unified.configuration import RunConfig, dump_run_config

from .legacy import (
    RemovedSourceGenerationError,
    _mapping,
    _read_yaml,
    _resolve,
    _source_generation_findings,
)
from .physics import _canonical_config, _legacy_adapter
from .tables import _canonical_boundaries, _canonical_particles


@dataclass(frozen=True)
class MigrationResult:
    config_path: Path
    particles_path: Path | None
    boundaries_path: Path | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NativeMigrationInputs:
    particles: pd.DataFrame
    walls: pd.DataFrame
    materials: pd.DataFrame | None

    def source_generation_tables(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        tables = [
            ("particles.csv", self.particles),
            ("part_walls.csv", self.walls),
        ]
        if self.materials is not None:
            tables.append(("materials.csv", self.materials))
        return tuple(tables)


def _load_native_migration_inputs(
    *,
    adapter: str,
    source_base: Path,
    paths: Mapping[str, Any],
) -> _NativeMigrationInputs | None:
    if adapter != "native":
        return None
    particles_source = _resolve(
        source_base,
        paths.get("particles_csv"),
        label="paths.particles_csv",
    )
    walls_source = _resolve(
        source_base,
        paths.get("part_walls_csv"),
        label="paths.part_walls_csv",
    )
    particles = pd.read_csv(particles_source)
    # Keep explicit empty text distinct from a missing column for strict migration.
    walls = pd.read_csv(walls_source, keep_default_na=False)

    materials: pd.DataFrame | None = None
    materials_value = paths.get("materials_csv")
    if materials_value is not None and str(materials_value).strip():
        materials_source = _resolve(
            source_base,
            materials_value,
            label="paths.materials_csv",
        )
        materials = pd.read_csv(materials_source, keep_default_na=False)
    return _NativeMigrationInputs(
        particles=particles,
        walls=walls,
        materials=materials,
    )


def _canonical_native_tables(
    inputs: _NativeMigrationInputs | None,
    *,
    config: RunConfig,
    legacy: Mapping[str, Any],
    warnings: list[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if inputs is None:
        return None, None
    particles = _canonical_particles(
        inputs.particles,
        spatial_dim=config.case.spatial_dim,
        coordinate_system=config.case.coordinate_system,
        warnings=warnings,
    )
    boundaries = _canonical_boundaries(
        inputs.walls,
        inputs.materials,
        _mapping(legacy.get("wall")),
    )
    return particles, boundaries


def _migration_targets(
    destination: Path,
    particles: pd.DataFrame | None,
    boundaries: pd.DataFrame | None,
) -> tuple[Path, Path | None, Path | None]:
    config_target = destination / "run_config.yaml"
    particles_target = destination / "particles.csv" if particles is not None else None
    boundaries_target = (
        destination / "boundaries.csv" if boundaries is not None else None
    )
    return config_target, particles_target, boundaries_target


def _reject_existing_targets(
    targets: Sequence[Path | None],
    *,
    overwrite: bool,
) -> None:
    existing = [path for path in targets if path is not None and path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "migration target(s) already exist; use overwrite=True to replace: "
            + ", ".join(str(path) for path in existing)
        )


def _write_migration_outputs(
    *,
    destination: Path,
    config: RunConfig,
    targets: tuple[Path, Path | None, Path | None],
    particles: pd.DataFrame | None,
    boundaries: pd.DataFrame | None,
) -> None:
    config_target, particles_target, boundaries_target = targets
    destination.mkdir(parents=True, exist_ok=True)
    if particles is not None and particles_target is not None:
        particles.to_csv(particles_target, index=False)
    if boundaries is not None and boundaries_target is not None:
        boundaries.to_csv(boundaries_target, index=False)
    dump_run_config(config, config_target)


def migrate_legacy_case(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> MigrationResult:
    """Convert one legacy case without carrying runtime compatibility aliases."""

    source = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    legacy = _read_yaml(source)
    paths = _mapping(legacy.get("paths"))
    adapter = _legacy_adapter(legacy)
    native_inputs = _load_native_migration_inputs(
        adapter=adapter,
        source_base=source.parent,
        paths=paths,
    )
    tables = () if native_inputs is None else native_inputs.source_generation_tables()
    findings = _source_generation_findings(legacy, tables)
    if findings:
        raise RemovedSourceGenerationError(findings)

    warnings: list[str] = []
    config = _canonical_config(
        legacy,
        source_base=source.parent,
        destination_base=destination,
        warnings=warnings,
    )
    particles, boundaries = _canonical_native_tables(
        native_inputs,
        config=config,
        legacy=legacy,
        warnings=warnings,
    )
    targets = _migration_targets(
        destination,
        particles,
        boundaries,
    )
    _reject_existing_targets(
        targets,
        overwrite=overwrite,
    )
    _write_migration_outputs(
        destination=destination,
        config=config,
        targets=targets,
        particles=particles,
        boundaries=boundaries,
    )
    config_target, particles_target, boundaries_target = targets
    return MigrationResult(
        config_path=config_target,
        particles_path=particles_target,
        boundaries_path=boundaries_target,
        warnings=tuple(warnings),
    )
