"""Package COMSOL exports as canonical cases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from particle_tracer_unified.core.coordinate_systems import (
    axisymmetric_rz_geometry_report,
    normalize_coordinate_system,
)
from particle_tracer_unified.io.comsol_boundary_reader import (
    remap_comsol_boundary_entity_ids,
)

from .contracts import (
    FIELD_STORAGE_MESH_NATIVE,
    GeometryOnlyBuild,
    RunnableBuild,
    canonical_boundary_table,
    canonical_release_table,
    copy_explicit_input,
    required_positive_float,
    resolve_force_inventory,
    sha256,
    validate_gas,
    validate_runnable_inputs,
    write_case_contract,
)
from .fields import pack_field_bundle, pack_mesh_field_bundle
from .mesh import (
    ParsedMesh,
    build_precomputed_arrays,
    parse_comsol_mphtxt,
    scale_mesh_coordinates,
    select_vacuum_domains,
    write_comsol_entity_maps,
    write_geometry_npz,
)
from .profiles import (
    BUILD_PROFILES,
    CANONICAL_BOUNDARIES_NAME,
    CANONICAL_PARTICLES_NAME,
    COMSOL_MANIFEST_NAME,
)
from .reporting import build_summary

if TYPE_CHECKING:
    from particle_tracer_unified.force_models import ForceModel


def _axisymmetric_report(
    coordinate_system: str,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    return axisymmetric_rz_geometry_report(
        coordinate_system=coordinate_system,
        spatial_dim=2,
        axes=(axes_x, axes_y),
        boundary_edges=arrays.get("boundary_edges"),
        boundary_edge_part_ids=arrays.get("boundary_part_ids"),
    )


@dataclass(frozen=True, slots=True)
class _RunnableBuildPlan:
    inputs: RunnableBuild
    force_inventory: ForceModel


def _resolve_build_plan(
    inputs: GeometryOnlyBuild | RunnableBuild,
    *,
    enabled_forces: tuple[str, ...],
    force_inventory_path: Path | None,
    coordinate_system: str,
) -> GeometryOnlyBuild | _RunnableBuildPlan:
    if isinstance(inputs, GeometryOnlyBuild):
        return inputs
    force_inventory = resolve_force_inventory(
        drag_law=inputs.drag_law,
        enabled_forces=enabled_forces,
        force_inventory_path=force_inventory_path,
        coordinate_system=coordinate_system,
    )
    return _RunnableBuildPlan(inputs=inputs, force_inventory=force_inventory)


def _prepare_field_artifact(
    plan: GeometryOnlyBuild | _RunnableBuildPlan,
    *,
    field_npz: Path,
    obsolete_field_mesh: Path,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    arrays: Mapping[str, Any],
    geometry_metadata: dict[str, Any],
    mesh: ParsedMesh,
) -> dict[str, Any]:
    if isinstance(plan, GeometryOnlyBuild):
        field_npz.unlink(missing_ok=True)
        obsolete_field_mesh.unlink(missing_ok=True)
        return {"mode": "geometry_only"}
    obsolete_field_mesh.unlink(missing_ok=True)
    if plan.inputs.field_storage == FIELD_STORAGE_MESH_NATIVE:
        node_samples = plan.inputs.field_node_samples_path
        if node_samples is None:
            raise RuntimeError("mesh-native build lost its node sample table")
        packed_mesh = pack_mesh_field_bundle(node_samples, field_npz, mesh=mesh)
        field_summary = dict(packed_mesh.summary)
        field_summary["physical_boundary_edge_count"] = int(
            arrays["boundary_edge_count"]
        )
        # Support is the mesh itself: it ends exactly where the vacuum domain
        # ends, so no diagnostic grid decides where a particle stops.
        geometry_metadata.update(
            source_kind="comsol_selected_vacuum_domain_geometry",
            geometry_mask_applied=False,
            field_support_is_physical_boundary=True,
        )
        return field_summary
    bundle_path = plan.inputs.field_bundle_path
    if bundle_path is None:
        raise RuntimeError("regular-grid build lost its field bundle")
    packed = pack_field_bundle(
        bundle_path,
        field_npz,
        axes_x=axes_x,
        axes_y=axes_y,
        geometry_inside=np.asarray(arrays["inside"], dtype=bool),
        geometry_sdf=np.asarray(arrays["sdf"], dtype=np.float64),
    )
    field_summary = dict(packed.summary)
    field_summary["physical_boundary_edge_count"] = int(arrays["boundary_edge_count"])
    geometry_metadata.update(
        source_kind="comsol_selected_vacuum_domain_geometry",
        geometry_mask_applied=False,
        field_support_is_physical_boundary=False,
    )
    return field_summary


def _apply_boundary_identity(
    plan: GeometryOnlyBuild | _RunnableBuildPlan,
    arrays: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None, dict[int, int]]:
    """Put every runtime boundary array in the canonical solver part-ID space."""

    resolved = dict(arrays)
    if isinstance(plan, GeometryOnlyBuild):
        return resolved, None, {}

    raw_entity_ids = np.unique(
        np.asarray(arrays["boundary_part_ids"], dtype=np.int64)
    ).astype(int)
    boundary_frame = canonical_boundary_table(
        plan.inputs.boundaries_path,
        geometry_entity_ids=raw_entity_ids.tolist(),
    )
    entity_to_part_id = {
        int(entity_id): int(part_id)
        for entity_id, part_id in zip(
            boundary_frame["comsol_entity_id"],
            boundary_frame["part_id"],
            strict=True,
        )
    }
    resolved["boundary_part_ids"] = remap_comsol_boundary_entity_ids(
        np.asarray(arrays["boundary_part_ids"]),
        entity_to_part_id,
        context="COMSOL boundary edges",
    )
    resolved["nearest_boundary_part_id_map"] = remap_comsol_boundary_entity_ids(
        np.asarray(arrays["nearest_boundary_part_id_map"]),
        entity_to_part_id,
        context="COMSOL nearest-boundary map",
    )
    return resolved, boundary_frame, entity_to_part_id


def _write_canonical_tables(
    inputs: RunnableBuild,
    *,
    out_dir: Path,
    boundary_frame: pd.DataFrame,
    coordinate_system: str,
) -> tuple[Path, Path]:
    particle_frame = canonical_release_table(
        inputs.release_table_path, coordinate_system=coordinate_system
    )
    unknown_sources = sorted(
        set(particle_frame["source_part_id"].astype(int))
        - set(boundary_frame["part_id"].astype(int))
    )
    if unknown_sources:
        raise ValueError(
            "particles.csv refers to unregistered source_part_id values: "
            f"{unknown_sources}"
        )
    particles_csv = out_dir / CANONICAL_PARTICLES_NAME
    boundaries_csv = out_dir / CANONICAL_BOUNDARIES_NAME
    copy_explicit_input(inputs.release_table_path, particles_csv)
    copy_explicit_input(inputs.boundaries_path, boundaries_csv)
    return particles_csv, boundaries_csv


def _release_projection_metadata(
    tolerance_m: float | None,
) -> dict[str, dict[str, float]]:
    """Declare how far from a boundary a release still counts as being on it.

    Boundary releases are snapped onto their declared entity and left there,
    which is where COMSOL puts an inlet particle.  Declaring the tolerance is
    what tells the adapter those releases are expected.
    """

    if tolerance_m is None:
        return {}
    return {"release_boundary_projection": {"tolerance_m": float(tolerance_m)}}


def write_case_files(
    mphtxt_path: Path,
    out_dir: Path,
    *,
    field_bundle_path: Path | None = None,
    field_node_samples_path: Path | None = None,
    release_table_path: Path | None = None,
    boundaries_path: Path | None = None,
    geometry_only: bool = False,
    diagnostic_grid_spacing_m: float | None = None,
    coordinate_scale_m_per_model_unit: float | None = None,
    coordinate_system: str = "cartesian_xy",
    profile: str = "generic",
    model_name: str | None = None,
    study: str | None = None,
    dataset: str | None = None,
    solution: str | None = None,
    solution_number: int | None = None,
    drag_law: str | None = None,
    enabled_forces: tuple[str, ...] = (),
    force_inventory_path: Path | None = None,
    vacuum_domain_ids: tuple[int, ...] = (),
    gas_temperature_K: float | None = None,
    gas_dynamic_viscosity_Pas: float | None = None,
    gas_density_kgm3: float | None = None,
    gas_molecular_mass_amu: float | None = None,
    solver_dt_s: float | None = None,
    solver_t_end_s: float | None = None,
    output_mode: str = "standard",
    trajectory_interval_steps: int | None = None,
    release_projection_tolerance_m: float | None = None,
    provenance_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Build either geometry diagnostics or one runnable canonical case."""

    diagnostic_spacing = required_positive_float(
        diagnostic_grid_spacing_m, context="diagnostic_grid_spacing_m"
    )
    coordinate_scale = required_positive_float(
        coordinate_scale_m_per_model_unit,
        context="coordinate_scale_m_per_model_unit",
    )
    if profile not in BUILD_PROFILES:
        raise ValueError(
            f"unknown COMSOL build profile {profile!r}; "
            f"expected {sorted(BUILD_PROFILES)}"
        )
    selected_profile = BUILD_PROFILES[profile]
    if profile != "generic" and coordinate_system != selected_profile.coordinate_system:
        raise ValueError(
            f"profile {profile!r} requires "
            f"coordinate_system={selected_profile.coordinate_system!r}"
        )
    validated_inputs = validate_runnable_inputs(
        geometry_only=geometry_only,
        field_bundle_path=field_bundle_path,
        field_node_samples_path=field_node_samples_path,
        release_table_path=release_table_path,
        boundaries_path=boundaries_path,
        model_name=model_name,
        study=study,
        dataset=dataset,
        solution=solution,
        solution_number=solution_number,
        drag_law=drag_law,
        solver_dt_s=solver_dt_s,
        solver_t_end_s=solver_t_end_s,
    )
    coordinate_system = normalize_coordinate_system(coordinate_system, 2)
    build_plan = _resolve_build_plan(
        validated_inputs,
        enabled_forces=enabled_forces,
        force_inventory_path=force_inventory_path,
        coordinate_system=coordinate_system,
    )

    full_mesh = scale_mesh_coordinates(
        parse_comsol_mphtxt(mphtxt_path), coordinate_scale
    )
    mesh, selected_domains = select_vacuum_domains(full_mesh, vacuum_domain_ids)
    arrays = build_precomputed_arrays(
        mesh, diagnostic_grid_spacing_m=diagnostic_spacing
    )
    arrays, boundary_frame, boundary_identity = _apply_boundary_identity(
        build_plan,
        arrays,
    )
    axes_x = np.asarray(arrays["axes_x"], dtype=np.float64)
    axes_y = np.asarray(arrays["axes_y"], dtype=np.float64)

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = out_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    geometry_npz = generated / "comsol_geometry_2d.npz"
    field_npz = generated / "comsol_field_2d.npz"
    obsolete_field_mesh = generated / "comsol_field_mesh_2d.npz"

    geometry_metadata: dict[str, Any] = {
        "provider_kind": "precomputed_npz",
        "source_kind": "comsol_mphtxt_geometry",
        "requires_field_bundle": True,
        "has_nearest_boundary_part_id_map": True,
        "boundary_region_map_status": "nearest_boundary_part_id_map",
        "diagnostic_grid_spacing_m": diagnostic_spacing,
        "field_ghost_cells": 0,
        "coordinate_unit": "m",
        "coordinate_scale_m_per_model_unit": coordinate_scale,
        "vacuum_domain_ids": list(selected_domains),
        "boundary_part_id_space": (
            "solver_part_id" if boundary_identity else "comsol_entity_id"
        ),
        "boundary_edge_topology": dict(arrays["boundary_edge_topology"]),
        "containment_boundary_edge_count": int(
            arrays["containment_boundary_edge_count"]
        ),
        "internal_interface_edge_count": int(arrays["internal_interface_edge_count"]),
    }
    axisymmetric_report = _axisymmetric_report(
        coordinate_system, axes_x, axes_y, arrays
    )
    if axisymmetric_report:
        geometry_metadata["axisymmetric_rz"] = axisymmetric_report

    field_summary = _prepare_field_artifact(
        build_plan,
        field_npz=field_npz,
        obsolete_field_mesh=obsolete_field_mesh,
        axes_x=axes_x,
        axes_y=axes_y,
        arrays=arrays,
        geometry_metadata=geometry_metadata,
        mesh=mesh,
    )

    write_geometry_npz(
        geometry_npz,
        axes_x=axes_x,
        axes_y=axes_y,
        arrays=arrays,
        mesh=mesh,
        metadata=geometry_metadata,
    )
    boundary_parts = np.unique(arrays["boundary_part_ids"]).astype(int).tolist()
    entity_maps = write_comsol_entity_maps(
        generated,
        full_mesh,
        boundary_parts,
        selected_domains,
        solver_part_id_by_entity_id=boundary_identity or None,
    )
    summary = build_summary(
        mphtxt_path=mphtxt_path,
        out_dir=out_dir,
        mesh=mesh,
        arrays=arrays,
        geometry_npz=geometry_npz,
        entity_map_files=entity_maps,
        geometry_metadata=geometry_metadata,
        field_summary=field_summary,
        diagnostic_spacing=diagnostic_spacing,
        coordinate_scale=coordinate_scale,
        vacuum_domain_ids=selected_domains,
        coordinate_system=coordinate_system,
        profile_name=selected_profile.name,
        axisymmetric_report=axisymmetric_report,
    )

    if isinstance(build_plan, _RunnableBuildPlan):
        inputs = build_plan.inputs
        if boundary_frame is None:
            raise RuntimeError("runnable COMSOL build lost its boundary identity")
        particles_csv, boundaries_csv = _write_canonical_tables(
            inputs,
            out_dir=out_dir,
            boundary_frame=boundary_frame,
            coordinate_system=coordinate_system,
        )
        gas = validate_gas(
            inputs.drag_law,
            {
                "temperature_K": gas_temperature_K,
                "dynamic_viscosity_Pas": gas_dynamic_viscosity_Pas,
                "density_kgm3": gas_density_kgm3,
                "molecular_mass_amu": gas_molecular_mass_amu,
            },
        )
        source_metadata: dict[str, Any] = {
            "source_mphtxt_sha256": sha256(mphtxt_path),
            "source_mphtxt_size_bytes": int(mphtxt_path.stat().st_size),
            "source_coordinate_scale_m_per_model_unit": coordinate_scale,
            "vacuum_domain_ids": list(selected_domains),
            "geometry_source": "explicit_comsol_vacuum_domain_selection",
            "source_solution_number": inputs.solution_number,
            **dict(provenance_metadata or {}),
            **_release_projection_metadata(release_projection_tolerance_m),
        }
        write_case_contract(
            out_dir=out_dir,
            geometry_npz=geometry_npz,
            field_npz=field_npz,
            field_storage=inputs.field_storage,
            particles_csv=particles_csv,
            boundaries_csv=boundaries_csv,
            coordinate_system=coordinate_system,
            profile=selected_profile,
            model_provenance={
                "name": inputs.model_name,
                "study": inputs.study,
                "dataset": inputs.dataset,
                "solution": inputs.solution,
            },
            force_inventory=build_plan.force_inventory,
            gas=gas,
            dt_s=inputs.solver_dt_s,
            t_end_s=inputs.solver_t_end_s,
            output_mode=output_mode,
            trajectory_interval_steps=trajectory_interval_steps,
            source_metadata=source_metadata,
        )
        summary["generated_files"].update(
            particles_csv=CANONICAL_PARTICLES_NAME,
            boundaries_csv=CANONICAL_BOUNDARIES_NAME,
            manifest=COMSOL_MANIFEST_NAME,
            run_config="run_config.yaml",
            field_npz=str(field_npz.relative_to(out_dir)),
        )

    (generated / "comsol_case_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
