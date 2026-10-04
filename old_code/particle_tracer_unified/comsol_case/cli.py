"""CLI adapter for the COMSOL case-building application service."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._raw_export_contract import NODE_SAMPLES_DIGEST_KEY, NODE_SAMPLES_FILENAME
from .builder import write_case_files
from .contracts import (
    load_json_mapping,
    required_positive_float,
    validate_raw_export,
)
from .fields import build_profile_field_bundle
from .profiles import BUILD_PROFILES


@dataclass(frozen=True)
class _CliBuildInputs:
    mphtxt: Path | None
    field_bundle: Path | None
    field_node_samples: Path | None
    coordinate_scale: float | None
    model_name: str | None
    study: str | None
    dataset: str | None
    solution: str | None
    solution_number: int | None
    vacuum_domain_ids: tuple[int, ...]
    provenance: dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="particle-tracer comsol build-case",
        description=(
            "Pack explicit COMSOL exports into the strict particle-tracer "
            "v0.2 contract."
        ),
    )
    parser.add_argument("--profile", choices=tuple(BUILD_PROFILES), default="generic")
    parser.add_argument("--raw-export-dir", type=Path, default=None)
    parser.add_argument("--mphtxt", type=Path, default=None)
    parser.add_argument(
        "--field-bundle",
        type=Path,
        default=None,
        help=(
            "regular-grid field bundle NPZ resampled from the solution; "
            "mutually exclusive with --field-node-samples"
        ),
    )
    parser.add_argument(
        "--field-node-samples",
        type=Path,
        default=None,
        help=(
            "CSV of COMSOL expressions evaluated at mesh vertices, keyed by "
            "node_index; keeps the field on the COMSOL mesh"
        ),
    )
    parser.add_argument("--release-table", type=Path, default=None)
    parser.add_argument("--boundaries", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--diagnostic-grid-spacing-m", type=float, required=True)
    parser.add_argument("--coordinate-scale-m-per-model-unit", type=float, default=None)
    parser.add_argument(
        "--coordinate-system",
        choices=["cartesian_xy", "axisymmetric_rz"],
        default=None,
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--study", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--solution", default=None)
    parser.add_argument("--solution-number", type=int, default=None)
    parser.add_argument(
        "--drag-law",
        choices=[
            "none",
            "stokes",
            "stokes_cunningham",
            "schiller_naumann",
            "epstein",
        ],
        default=None,
    )
    parser.add_argument(
        "--force",
        action="append",
        choices=["electric"],
        default=[],
        help=(
            "Enable coefficient-free electric force; use --force-inventory "
            "for other forces."
        ),
    )
    parser.add_argument(
        "--force-inventory",
        type=Path,
        default=None,
        help="Strict YAML containing typed non-drag force entries under a forces list.",
    )
    parser.add_argument(
        "--vacuum-domain-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Explicit COMSOL 2D domain ID occupied by particles; repeat for a "
            "disconnected vacuum region."
        ),
    )
    parser.add_argument("--gas-temperature-K", type=float, default=None)
    parser.add_argument("--gas-dynamic-viscosity-Pas", type=float, default=None)
    parser.add_argument("--gas-density-kgm3", type=float, default=None)
    parser.add_argument("--gas-molecular-mass-amu", type=float, default=None)
    parser.add_argument("--dt-s", type=float, default=None)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument(
        "--output-mode", choices=["standard", "debug"], default="standard"
    )
    parser.add_argument("--trajectory-interval-steps", type=int, default=None)
    parser.add_argument("--release-projection-tolerance-m", type=float, default=None)
    return parser


def _raw_export_inputs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> _CliBuildInputs:
    """Resolve and verify an exporter directory into ordinary build inputs."""

    profile = BUILD_PROFILES[str(args.profile)]
    if profile.name == "generic":
        parser.error("--raw-export-dir requires a named model profile")
    if args.mphtxt is not None or args.field_bundle is not None:
        parser.error(
            "--raw-export-dir cannot be combined with --mphtxt or --field-bundle"
        )
    duplicate_provenance = [
        name
        for name, value in (
            (
                "--coordinate-scale-m-per-model-unit",
                args.coordinate_scale_m_per_model_unit,
            ),
            ("--model-name", args.model_name),
            ("--study", args.study),
            ("--dataset", args.dataset),
            ("--solution", args.solution),
            ("--solution-number", args.solution_number),
            ("--vacuum-domain-id", args.vacuum_domain_id),
        )
        if value is not None and value != []
    ]
    if duplicate_provenance:
        parser.error(
            "--raw-export-dir takes provenance only from export_manifest.json; "
            f"remove duplicate options: {duplicate_provenance}"
        )
    raw_dir = args.raw_export_dir.resolve()
    manifest_path = raw_dir / "export_manifest.json"
    contract = validate_raw_export(
        raw_dir,
        manifest_path,
        load_json_mapping(manifest_path),
        profile=profile,
    )
    out_dir = args.out_dir.resolve()
    node_samples = (
        raw_dir / NODE_SAMPLES_FILENAME
        if contract.get(NODE_SAMPLES_DIGEST_KEY) is not None
        else None
    )
    field_bundle = (
        None
        if node_samples is not None
        else build_profile_field_bundle(
            raw_dir / "field_samples.csv",
            out_dir / "generated" / f"{profile.name}_field_bundle.npz",
            profile=profile,
            coordinate_scale_m_per_model_unit=float(
                contract["geometry_scale_m_per_model_unit"]
            ),
        )
    )
    provenance = {
        "raw_export_manifest_sha256": contract["manifest_sha256"],
        "raw_export_manifest_size_bytes": contract["manifest_size_bytes"],
        "source_mph_sha256": contract["mph_sha256"],
        "source_mesh_sha256": contract["mesh_sha256"],
        "source_field_samples_sha256": contract["field_samples_sha256"],
        **(
            {"source_field_node_samples_sha256": contract[NODE_SAMPLES_DIGEST_KEY]}
            if node_samples is not None
            else {}
        ),
        "source_export_config_sha256": contract["config_sha256"],
        "source_comsol_version": contract["comsol_version"],
        "source_mesh_tag": contract["mesh_tag"],
        "source_parameter": {
            "name": contract["parameter_name"],
            "value": contract["parameter_value"],
        },
        "source_expression_mapping": contract["expression_mapping"],
        "source_expression_units": contract["expression_units"],
    }
    return _CliBuildInputs(
        mphtxt=raw_dir / "mesh.mphtxt",
        field_bundle=field_bundle,
        field_node_samples=node_samples,
        coordinate_scale=float(contract["geometry_scale_m_per_model_unit"]),
        model_name=str(contract["model_name"]),
        study=str(contract["study"]),
        dataset=str(contract["dataset"]),
        solution=str(contract["solution"]),
        solution_number=int(contract["solution_number"]),
        vacuum_domain_ids=tuple(contract["vacuum_domain_ids"]),
        provenance=provenance,
    )


def _direct_inputs(args: argparse.Namespace) -> _CliBuildInputs:
    return _CliBuildInputs(
        mphtxt=args.mphtxt,
        field_bundle=args.field_bundle,
        field_node_samples=args.field_node_samples,
        coordinate_scale=args.coordinate_scale_m_per_model_unit,
        model_name=args.model_name,
        study=args.study,
        dataset=args.dataset,
        solution=args.solution,
        solution_number=args.solution_number,
        vacuum_domain_ids=tuple(args.vacuum_domain_id),
        provenance={},
    )


def _resolve_build_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> _CliBuildInputs:
    if args.geometry_only and args.raw_export_dir is not None:
        parser.error("--geometry-only cannot be combined with --raw-export-dir")
    if args.geometry_only and args.field_bundle is not None:
        parser.error("--geometry-only cannot be combined with --field-bundle")
    if args.geometry_only and args.field_node_samples is not None:
        parser.error("--geometry-only cannot be combined with --field-node-samples")
    if args.field_bundle is not None and args.field_node_samples is not None:
        parser.error("--field-bundle and --field-node-samples are mutually exclusive")
    if args.raw_export_dir is not None:
        return _raw_export_inputs(args, parser)
    return _direct_inputs(args)


def _validated_diagnostic_spacing(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> float:
    try:
        spacing = required_positive_float(
            args.diagnostic_grid_spacing_m,
            context="--diagnostic-grid-spacing-m",
        )
        if not args.geometry_only:
            required_positive_float(args.dt_s, context="--dt-s")
            required_positive_float(args.t_end_s, context="--t-end-s")
    except ValueError as exc:
        parser.error(str(exc))
    return spacing


def _validated_coordinate_scale(
    inputs: _CliBuildInputs,
    parser: argparse.ArgumentParser,
) -> float:
    if inputs.coordinate_scale is None:
        parser.error(
            "--coordinate-scale-m-per-model-unit is required unless "
            "--raw-export-dir provides geometry_scale_m_per_model_unit"
        )
    try:
        return required_positive_float(
            inputs.coordinate_scale,
            context="--coordinate-scale-m-per-model-unit",
        )
    except ValueError as exc:
        parser.error(str(exc))


def _required_mesh_path(
    inputs: _CliBuildInputs,
    parser: argparse.ArgumentParser,
) -> Path:
    if inputs.mphtxt is None:
        parser.error("--mphtxt is required unless --raw-export-dir is used")
    return inputs.mphtxt.resolve()


def _require_runnable_field(
    args: argparse.Namespace,
    inputs: _CliBuildInputs,
    parser: argparse.ArgumentParser,
) -> None:
    if (
        not args.geometry_only
        and inputs.field_bundle is None
        and inputs.field_node_samples is None
    ):
        parser.error(
            "--field-bundle or --field-node-samples is required for a runnable case"
        )


def _write_resolved_case(
    args: argparse.Namespace,
    *,
    inputs: _CliBuildInputs,
    profile_name: str,
    profile_coordinate_system: str,
    diagnostic_spacing: float,
    coordinate_scale: float,
    mesh_path: Path,
) -> None:
    write_case_files(
        mesh_path,
        args.out_dir.resolve(),
        field_bundle_path=(
            inputs.field_bundle.resolve() if inputs.field_bundle is not None else None
        ),
        field_node_samples_path=(
            inputs.field_node_samples.resolve()
            if inputs.field_node_samples is not None
            else None
        ),
        release_table_path=(
            args.release_table.resolve() if args.release_table is not None else None
        ),
        boundaries_path=(
            args.boundaries.resolve() if args.boundaries is not None else None
        ),
        geometry_only=bool(args.geometry_only),
        diagnostic_grid_spacing_m=diagnostic_spacing,
        coordinate_scale_m_per_model_unit=coordinate_scale,
        coordinate_system=str(args.coordinate_system or profile_coordinate_system),
        profile=profile_name,
        model_name=inputs.model_name,
        study=inputs.study,
        dataset=inputs.dataset,
        solution=inputs.solution,
        solution_number=inputs.solution_number,
        drag_law=args.drag_law,
        enabled_forces=tuple(args.force),
        force_inventory_path=(
            args.force_inventory.resolve() if args.force_inventory is not None else None
        ),
        vacuum_domain_ids=inputs.vacuum_domain_ids,
        gas_temperature_K=args.gas_temperature_K,
        gas_dynamic_viscosity_Pas=args.gas_dynamic_viscosity_Pas,
        gas_density_kgm3=args.gas_density_kgm3,
        gas_molecular_mass_amu=args.gas_molecular_mass_amu,
        solver_dt_s=args.dt_s,
        solver_t_end_s=args.t_end_s,
        output_mode=str(args.output_mode),
        trajectory_interval_steps=args.trajectory_interval_steps,
        release_projection_tolerance_m=args.release_projection_tolerance_m,
        provenance_metadata=inputs.provenance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    profile = BUILD_PROFILES[str(args.profile)]
    diagnostic_spacing = _validated_diagnostic_spacing(args, parser)
    inputs = _resolve_build_inputs(args, parser)
    mesh_path = _required_mesh_path(inputs, parser)
    coordinate_scale = _validated_coordinate_scale(inputs, parser)
    _require_runnable_field(args, inputs, parser)
    _write_resolved_case(
        args,
        inputs=inputs,
        profile_name=profile.name,
        profile_coordinate_system=profile.coordinate_system,
        diagnostic_spacing=diagnostic_spacing,
        coordinate_scale=coordinate_scale,
        mesh_path=mesh_path,
    )
    print(f"Wrote COMSOL-derived case to: {args.out_dir.resolve()}")
    return 0
