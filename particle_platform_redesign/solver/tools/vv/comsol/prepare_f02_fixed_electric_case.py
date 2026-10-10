"""Attach the explicit F02 particle schedule to one completed field bundle.

This is a case-local external V&V fixture step, not a source-distribution
facility in the trajectory solver.  It replaces the historical implicit
``uniform revolved_area`` surface source with 32 durable, per-particle rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from chamber_particles.case_format import RealizedSurfaceSource, read_with_info, write
from chamber_particles.geometry import prepare_geometry

_SOURCE_NAME = "wafer_surface_particles"
_PARTICLE_COUNT = 32
_PARTICLE_ID_START = 1000
_MASS_KG = 1.1519173063162574e-18
_DRAG_DIAMETER_M = 1.0e-7
_ELECTROSTATIC_RADIUS_M = 5.0e-8
_DISPLACED_VOLUME_M3 = 5.235987755982988e-22
_FIXTURE_REVISION = "f02_realized_wafer_schedule_v1"
_PRODUCER = "tools.vv.comsol.prepare_f02_fixed_electric_case"


def prepare_f02_fixed_electric_case(
    input_path: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Write the fixed F02 surface schedule into a new canonical case file."""

    input_path, output_path, report_path = _resolve_paths(input_path, output_path, report_path)
    data, input_info = read_with_info(input_path)
    if data.coordinate_system != "axisymmetric_rz":
        raise ValueError("F02 fixed-electric fixture requires axisymmetric_rz geometry")
    if data.sources:
        raise ValueError("F02 field bundle must not already contain particle sources")
    try:
        wafer_group_id = data.geometry.group_names.index("wafer")
    except ValueError as error:
        raise ValueError("F02 geometry has no wafer boundary group") from error

    geometry = prepare_geometry(data.geometry, data.coordinate_system)
    wafer_facets = np.flatnonzero(geometry.group_id == wafer_group_id).astype("<i8")
    if wafer_facets.size == 0:
        raise ValueError("F02 wafer boundary group has no facets")

    start = geometry.facet_start_m[wafer_facets]
    end = geometry.facet_end_m[wafer_facets]
    # Exact area of each straight R-Z segment revolved about the axis.
    area_m2 = math.pi * (start[:, 0] + end[:, 0]) * geometry.facet_length_m[wafer_facets]
    if not bool(np.isfinite(area_m2).all()) or bool((area_m2 < 0.0).any()):
        raise ValueError("F02 wafer contains an invalid revolved facet area")
    positive = area_m2 > 0.0
    wafer_facets = wafer_facets[positive]
    start = start[positive]
    end = end[positive]
    area_m2 = area_m2[positive]
    total_area_m2 = float(np.sum(area_m2))
    if not math.isfinite(total_area_m2) or total_area_m2 <= 0.0:
        raise ValueError("F02 wafer has zero revolved area")

    # Mid-quantiles give a deterministic equal-area population without RNG.
    target_area_m2 = (np.arange(_PARTICLE_COUNT, dtype=np.float64) + 0.5) * (
        total_area_m2 / _PARTICLE_COUNT
    )
    cumulative_area_m2 = np.cumsum(area_m2)
    local_index = np.searchsorted(cumulative_area_m2, target_area_m2, side="right")
    preceding_area_m2 = np.concatenate((np.asarray([0.0]), cumulative_area_m2[:-1]))
    local_area_fraction = (target_area_m2 - preceding_area_m2[local_index]) / area_m2[local_index]
    r0 = start[local_index, 0]
    r1 = end[local_index, 0]
    radial_integral = 0.5 * (r0 + r1) * local_area_fraction
    discriminant = r0 * r0 + 2.0 * (r1 - r0) * radial_integral
    parameter = 2.0 * radial_integral / (r0 + np.sqrt(discriminant))
    if not bool(np.isfinite(parameter).all()) or bool(
        ((parameter <= 0.0) | (parameter >= 1.0)).any()
    ):
        raise ValueError("F02 equal-area realization reached a facet endpoint")

    facet_id = wafer_facets[local_index]
    velocity_m_s = -geometry.facet_normal[facet_id]
    source = RealizedSurfaceSource(
        name=_SOURCE_NAME,
        particle_id=np.arange(
            _PARTICLE_ID_START,
            _PARTICLE_ID_START + _PARTICLE_COUNT,
            dtype="<i8",
        ),
        release_time_s=np.zeros(_PARTICLE_COUNT, dtype="<f8"),
        facet_id=np.asarray(facet_id, dtype="<i8"),
        facet_parameter=np.asarray(parameter, dtype="<f8"),
        velocity_m_s=np.asarray(velocity_m_s, dtype="<f8"),
        charge_number=np.full(_PARTICLE_COUNT, -1.0, dtype="<f8"),
        mass_kg=np.full(_PARTICLE_COUNT, _MASS_KG, dtype="<f8"),
        drag_diameter_m=np.full(_PARTICLE_COUNT, _DRAG_DIAMETER_M, dtype="<f8"),
        contact_radius_m=np.zeros(_PARTICLE_COUNT, dtype="<f8"),
        electrostatic_radius_m=np.full(
            _PARTICLE_COUNT,
            _ELECTROSTATIC_RADIUS_M,
            dtype="<f8",
        ),
        displaced_volume_m3=np.full(
            _PARTICLE_COUNT,
            _DISPLACED_VOLUME_M3,
            dtype="<f8",
        ),
        model_weight=np.ones(_PARTICLE_COUNT, dtype="<f8"),
        material_id=np.zeros(_PARTICLE_COUNT, dtype="<i4"),
    )
    input_provenance_sha256 = (
        "sha256:" + hashlib.sha256(data.provenance_json.encode("utf-8")).hexdigest()
    )
    input_provenance = json.loads(data.provenance_json)
    provenance = {
        "producer": _PRODUCER,
        "producer_version": _FIXTURE_REVISION,
        "source_sha256": input_info.content_hash,
        "field_semantics_revision": input_provenance["field_semantics_revision"],
        "producer_metadata": {
            "fixture_revision": _FIXTURE_REVISION,
            "input_content_hash": input_info.content_hash,
            "input_provenance_sha256": input_provenance_sha256,
            "source_realization": {
                "boundary_group": "wafer",
                "measure": "revolved_area",
                "quantile_rule": "equal_area_midpoint",
                "particle_count": _PARTICLE_COUNT,
                "particle_id_start": _PARTICLE_ID_START,
                "release_time_s": 0.0,
                "velocity_rule": "negative_outward_unit_normal",
                "charge_number": -1.0,
                "mass_kg": _MASS_KG,
                "drag_diameter_m": _DRAG_DIAMETER_M,
                "electrostatic_radius_m": _ELECTROSTATIC_RADIUS_M,
                "displaced_volume_m3": _DISPLACED_VOLUME_M3,
                "model_weight": 1.0,
                "material_id": 0,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_info = write(
        output_path,
        replace(
            data,
            provenance_json=json.dumps(
                provenance,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            sources=(source,),
        ),
    )
    report: dict[str, object] = {
        "status": "complete",
        "fixture_revision": _FIXTURE_REVISION,
        "input_path": str(input_path),
        "input_content_hash": input_info.content_hash,
        "input_provenance_sha256": input_provenance_sha256,
        "output_path": str(output_path),
        "output_content_hash": output_info.content_hash,
        "source_name": source.name,
        "particle_count": _PARTICLE_COUNT,
        "wafer_facet_count": int(wafer_facets.size),
        "wafer_revolved_area_m2": total_area_m2,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8", errors="strict") as stream:
            stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return report


def _resolve_paths(
    input_path: Path,
    output_path: Path,
    report_path: Path | None,
) -> tuple[Path, Path, Path | None]:
    input_resolved = input_path.expanduser().resolve()
    output_resolved = output_path.expanduser().resolve()
    report_resolved = report_path.expanduser().resolve() if report_path is not None else None
    paths = [input_resolved, output_resolved]
    if report_resolved is not None:
        paths.append(report_resolved)
    if len(set(paths)) != len(paths):
        raise ValueError("input_path, output_path, and report_path must be different")
    if not input_resolved.is_file():
        raise FileNotFoundError(input_resolved)
    if output_resolved.exists():
        raise FileExistsError(output_resolved)
    if report_resolved is not None and report_resolved.exists():
        raise FileExistsError(report_resolved)
    return input_resolved, output_resolved, report_resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare_f02_fixed_electric_case(
        args.input,
        args.output,
        report_path=args.report,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")


if __name__ == "__main__":
    main()
