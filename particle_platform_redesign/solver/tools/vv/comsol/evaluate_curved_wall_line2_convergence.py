"""Evaluate line2 curved-wall reflection against an analytic circle.

This is an external numerical V&V workflow.  It deliberately leaves the
production geometry representation unchanged and quantifies the discretization
error introduced when a smooth wall is supplied as straight facets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    GeometryData,
    RealizedTableSource,
    write,
)

TOOL_REVISION: Final = "curved_wall_line2_convergence_v1"
FACET_COUNTS: Final = (16, 32, 64, 128)
PARTICLE_COUNT: Final = 17
RADIUS_M: Final = 1.0
END_TIME_S: Final = 1.2
MINIMUM_POINT_ORDER: Final = 1.8
MINIMUM_NORMAL_ORDER: Final = 0.8


@dataclass(frozen=True, slots=True)
class LevelMetrics:
    facets: int
    maximum_time_error_s: float
    rms_time_error_s: float
    maximum_point_error_m: float
    rms_point_error_m: float
    maximum_normal_error: float
    rms_normal_error: float
    maximum_reflected_velocity_error_m_s: float
    rms_reflected_velocity_error_m_s: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _angles() -> np.ndarray:
    # The deterministic irrational phase avoids persistent vertex/midpoint
    # alignment while the population RMS removes one-ray phase oscillation.
    index = np.arange(PARTICLE_COUNT, dtype=np.float64)
    return np.mod(0.17320508075688773 + index * 2.399963229728653, 2.0 * math.pi)


def _geometry(facets: int) -> GeometryData:
    angle = 2.0 * math.pi * np.arange(facets, dtype=np.float64) / facets
    boundary_nodes = np.column_stack((np.cos(angle), np.sin(angle))).astype("<f8")
    nodes = np.vstack((np.zeros((1, 2), dtype="<f8"), boundary_nodes))
    current = 1 + np.arange(facets, dtype=np.int64)
    following = 1 + np.mod(np.arange(facets, dtype=np.int64) + 1, facets)
    tri3 = np.column_stack((np.zeros(facets, dtype=np.int64), current, following)).astype("<i8")
    boundary = BoundaryData(
        line2=np.column_stack((current, following)).astype("<i8"),
        boundary_id=np.arange(1000, 1000 + facets, dtype="<i4"),
        group_id=np.zeros(facets, dtype="<i4"),
        material_id=np.zeros(facets, dtype="<i4"),
        owner_cell_type=np.ones(facets, dtype="<u1"),
        owner_cell_local_index=np.arange(facets, dtype="<i8"),
        orientation=np.ones(facets, dtype="<i1"),
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("curved_wall",),
        tri3=tri3,
        tri3_domain_id=np.zeros(facets, dtype="<i4"),
    )


def _source() -> RealizedTableSource:
    angle = _angles()
    velocity = np.column_stack((np.cos(angle), np.sin(angle))).astype("<f8")
    count = angle.size
    return RealizedTableSource(
        name="circle_rays",
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=np.zeros(count, dtype="<f8"),
        position_m=np.zeros((count, 2), dtype="<f8"),
        velocity_m_s=velocity,
        charge_number=np.zeros(count, dtype="<f8"),
        mass_kg=np.ones(count, dtype="<f8"),
        drag_diameter_m=np.full(count, 1.0e-6, dtype="<f8"),
        contact_radius_m=np.zeros(count, dtype="<f8"),
        electrostatic_radius_m=np.full(count, 5.0e-7, dtype="<f8"),
        displaced_volume_m3=np.zeros(count, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.zeros(count, dtype="<i4"),
    )


def _materialize_case(directory: Path, facets: int) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {
        "producer": "tools.vv.comsol.evaluate_curved_wall_line2_convergence",
        "producer_version": TOOL_REVISION,
        "source_sha256": "sha256:"
        + hashlib.sha256(b"analytic-unit-circle-radius-1m-v1").hexdigest(),
        "field_semantics_revision": "no_fields_v1",
        "producer_metadata": {
            "smooth_reference": "circle_radius_1m",
            "line2_facets": facets,
            "contact_radius_m": 0.0,
        },
    }
    bundle = DataBundle(
        coordinate_system="cartesian_xy",
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        geometry=_geometry(facets),
        sources=(_source(),),
    )
    data_path = directory / "case.h5"
    info = write(data_path, bundle)
    document = {
        "format_version": 3,
        "case": {
            "name": f"analytic_circle_line2_{facets}",
            "data_path": data_path.name,
            "expected_content_hash": info.content_hash,
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": 0.0, "end_s": END_TIME_S, "dt_s": END_TIME_S},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 20261009,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 4,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": {"charge": {"model": "fixed"}},
        "sources": [{"name": "rays", "type": "table", "table": "circle_rays"}],
        "boundaries": [{"boundary_group": "curved_wall", "priority": 10, "law": "specular"}],
        "output": {"trajectories": None, "probes": None},
    }
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _level_metrics(result_directory: Path, facets: int) -> LevelMetrics:
    result = open_result(result_directory)
    event = result.read_boundary_events()
    expected_id = np.arange(1, PARTICLE_COUNT + 1, dtype=np.int64)
    if event.particle_id.size != PARTICLE_COUNT:
        raise ValueError(f"{facets}: expected one reflection per particle")
    order = np.argsort(event.particle_id)
    if not np.array_equal(event.particle_id[order], expected_id):
        raise ValueError(f"{facets}: boundary particle identity differs")
    if set(event.law_id.tolist()) != {"specular"} or set(event.outcome.tolist()) != {"reflected"}:
        raise ValueError(f"{facets}: boundary response differs")

    time = event.time_s[order]
    point = event.position_m[order]
    normal = event.normal[order]
    velocity_post = event.velocity_post_m_s[order]
    velocity_pre = _source().velocity_m_s
    analytic_time = np.full(PARTICLE_COUNT, RADIUS_M, dtype=np.float64)
    analytic_point = velocity_pre * RADIUS_M
    analytic_normal = analytic_point / RADIUS_M
    analytic_post = (
        velocity_pre
        - 2.0 * np.sum(velocity_pre * analytic_normal, axis=1)[:, None] * analytic_normal
    )

    time_error = np.abs(time - analytic_time)
    point_error = np.linalg.norm(point - analytic_point, axis=1)
    normal_error = np.linalg.norm(normal - analytic_normal, axis=1)
    velocity_error = np.linalg.norm(velocity_post - analytic_post, axis=1)

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values * values)))

    return LevelMetrics(
        facets=facets,
        maximum_time_error_s=float(np.max(time_error)),
        rms_time_error_s=rms(time_error),
        maximum_point_error_m=float(np.max(point_error)),
        rms_point_error_m=rms(point_error),
        maximum_normal_error=float(np.max(normal_error)),
        rms_normal_error=rms(normal_error),
        maximum_reflected_velocity_error_m_s=float(np.max(velocity_error)),
        rms_reflected_velocity_error_m_s=rms(velocity_error),
    )


def _observed_order(levels: list[LevelMetrics], attribute: str) -> float:
    h = 2.0 * math.pi / np.asarray([level.facets for level in levels], dtype=np.float64)
    error = np.asarray([getattr(level, attribute) for level in levels], dtype=np.float64)
    if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
        raise ValueError(f"cannot estimate order for {attribute}")
    return float(np.polyfit(np.log(h), np.log(error), 1)[0])


def _strictly_decreases(levels: list[LevelMetrics], attribute: str) -> bool:
    error = np.asarray([getattr(level, attribute) for level in levels], dtype=np.float64)
    return bool(np.all(error[1:] < error[:-1]))


def evaluate(output_directory: Path) -> dict[str, object]:
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    levels: list[LevelMetrics] = []
    artifacts: list[dict[str, object]] = []
    for facets in FACET_COUNTS:
        level_root = output_directory / f"facets_{facets}"
        case_path = _materialize_case(level_root / "case", facets)
        result_path = level_root / "result"
        simulate(load_case(case_path), result_path)
        levels.append(_level_metrics(result_path, facets))
        artifacts.extend(
            {
                "path": str(path.relative_to(output_directory)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(level_root.rglob("*"))
            if path.is_file()
        )

    orders = {
        "rms_time": _observed_order(levels, "rms_time_error_s"),
        "rms_point": _observed_order(levels, "rms_point_error_m"),
        "rms_normal": _observed_order(levels, "rms_normal_error"),
        "rms_reflected_velocity": _observed_order(levels, "rms_reflected_velocity_error_m_s"),
    }
    gates = {
        "one_reflection_per_particle": True,
        "rms_errors_decrease_with_refinement": all(
            _strictly_decreases(levels, attribute)
            for attribute in (
                "rms_time_error_s",
                "rms_point_error_m",
                "rms_normal_error",
                "rms_reflected_velocity_error_m_s",
            )
        ),
        "point_order_at_least_1p8": orders["rms_point"] >= MINIMUM_POINT_ORDER,
        "time_order_at_least_1p8": orders["rms_time"] >= MINIMUM_POINT_ORDER,
        "normal_order_at_least_0p8": orders["rms_normal"] >= MINIMUM_NORMAL_ORDER,
        "reflected_velocity_order_at_least_0p8": (
            orders["rms_reflected_velocity"] >= MINIMUM_NORMAL_ORDER
        ),
        "finest_point_error_below_1e_3_m": levels[-1].maximum_point_error_m < 1.0e-3,
        "finest_normal_error_below_3e_2": levels[-1].maximum_normal_error < 3.0e-2,
        "finest_reflected_velocity_error_below_6e_2_m_s": (
            levels[-1].maximum_reflected_velocity_error_m_s < 6.0e-2
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    summary: dict[str, object] = {
        "tool_revision": TOOL_REVISION,
        "scientific_status": status,
        "tested_scope": (
            "Cartesian point particles, force-free straight flight, specular reflection, "
            "unit-circle wall approximated by inscribed line2 facets"
        ),
        "not_tested": [
            "COMSOL curved-geometry equivalence",
            "finite-radius contact",
            "grazing/corner/multiple-hit behavior",
            "curved elements in the production solver",
        ],
        "facets": list(FACET_COUNTS),
        "particle_count_per_level": PARTICLE_COUNT,
        "levels": [asdict(level) for level in levels],
        "observed_orders": orders,
        "gates": gates,
        "artifact_count_before_summary": len(artifacts),
    }
    with (output_directory / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(levels[0])))
        writer.writeheader()
        writer.writerows(asdict(level) for level in levels)
    (output_directory / "comparison_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "artifact_hashes.json").write_text(
        json.dumps(artifacts, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Curved-wall line2 convergence

Status: **{status}**

The public solver API was run for {PARTICLE_COUNT} analytic rays against an
inscribed line2 approximation of a one-metre circle at {list(FACET_COUNTS)}
facets.  First-hit time, hit point, wall normal, and specular post-impact
velocity were compared independently with the smooth-circle analytic result.

Observed RMS orders are `{json.dumps(orders, sort_keys=True)}`.  This validates
mesh convergence of the supported straight-facet representation; it does not
claim exact curved-wall geometry or COMSOL native curved-element equivalence.
"""
    (output_directory / "README.md").write_text(readme, encoding="utf-8")
    if status != "PASS":
        raise RuntimeError("curved-wall line2 convergence gates failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = evaluate(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
