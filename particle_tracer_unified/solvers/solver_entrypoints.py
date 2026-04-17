from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

from ..core.datamodel import PreparedRuntime
from ..core.input_contract import enforce_initial_particle_field_support
from ..core.provider_contract import enforce_boundary_field_support
from ..io.runtime_builder import build_prepared_runtime_from_yaml
from .high_fidelity_runtime import run_prepared_runtime

__all__ = (
    'build_prepared_runtime_for_dim',
    'build_prepared_runtime_2d',
    'build_prepared_runtime_3d',
    'run_solver_for_dim',
    'run_solver_2d',
    'run_solver_3d',
    'run_solver_from_yaml_for_dim',
    'run_solver_2d_from_yaml',
    'run_solver_3d_from_yaml',
)


def build_prepared_runtime_for_dim(config_or_prepared: Union[str, Path, PreparedRuntime], spatial_dim: int) -> PreparedRuntime:
    if isinstance(config_or_prepared, PreparedRuntime):
        prepared = config_or_prepared
    else:
        prepared = build_prepared_runtime_from_yaml(Path(config_or_prepared))
    if int(prepared.runtime.spatial_dim) != int(spatial_dim):
        raise ValueError(f'{int(spatial_dim)}D solver requires run.spatial_dim={int(spatial_dim)}')
    return prepared


def run_solver_for_dim(prepared: PreparedRuntime, output_dir: Path, spatial_dim: int) -> Mapping[str, object]:
    prepared = build_prepared_runtime_for_dim(prepared, spatial_dim=spatial_dim)
    out = Path(output_dir)
    enforce_boundary_field_support(prepared, out)
    enforce_initial_particle_field_support(prepared, out)
    return run_prepared_runtime(prepared, out, spatial_dim=int(spatial_dim))


def run_solver_from_yaml_for_dim(
    config_path: Union[str, Path],
    *,
    spatial_dim: int,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Mapping[str, object], PreparedRuntime]:
    prepared = build_prepared_runtime_for_dim(config_path, spatial_dim=spatial_dim)
    out = Path(output_dir) if output_dir is not None else (Path(config_path).resolve().parent / f'run_output_{int(spatial_dim)}d')
    enforce_boundary_field_support(prepared, out)
    enforce_initial_particle_field_support(prepared, out)
    report = run_prepared_runtime(prepared, out, spatial_dim=int(spatial_dim))
    return report, prepared


def build_prepared_runtime_2d(config_or_prepared: Union[str, Path, PreparedRuntime]) -> PreparedRuntime:
    return build_prepared_runtime_for_dim(config_or_prepared, spatial_dim=2)


def build_prepared_runtime_3d(config_or_prepared: Union[str, Path, PreparedRuntime]) -> PreparedRuntime:
    return build_prepared_runtime_for_dim(config_or_prepared, spatial_dim=3)


def run_solver_2d(prepared: PreparedRuntime, output_dir: Path) -> Mapping[str, object]:
    return run_solver_for_dim(prepared, Path(output_dir), spatial_dim=2)


def run_solver_3d(prepared: PreparedRuntime, output_dir: Path) -> Mapping[str, object]:
    return run_solver_for_dim(prepared, Path(output_dir), spatial_dim=3)


def run_solver_2d_from_yaml(
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Mapping[str, object], PreparedRuntime]:
    return run_solver_from_yaml_for_dim(config_path, spatial_dim=2, output_dir=output_dir)


def run_solver_3d_from_yaml(
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Mapping[str, object], PreparedRuntime]:
    return run_solver_from_yaml_for_dim(config_path, spatial_dim=3, output_dir=output_dir)
