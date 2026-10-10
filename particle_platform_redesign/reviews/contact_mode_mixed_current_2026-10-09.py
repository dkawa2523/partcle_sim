"""One evidence-only H2 row using the existing P14 isolated worker."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tests.performance import p14_matrix as p14
from tests.scenarios.test_contact_geometry_run import _mixed_case

from chamber_particles import open_result


def main() -> None:
    output = Path("../reviews/contact_mode_mixed_current_2026-10-09.json")
    row = p14._row(
        "mixed-center-opening-n10000", "event", 10_000, "regular", "initial", 1, "none", "warm", 1
    )
    started = datetime.now(UTC).isoformat()
    radii = (0.05, 0.15) * 5_000
    observations = []
    oracle = []
    with TemporaryDirectory(prefix="chamber-particles-h2-mixed-evidence-") as temporary:
        root = Path(temporary)
        case_path = _mixed_case(root / "case", radius=radii, frames=False, memory_limit_mb=8192)
        item = p14._MaterializedRow(row, case_path, 1, 0, 0, "linear_exact")
        spec = root / "worker.json"
        spec.write_text(
            json.dumps(p14._worker_spec(item), allow_nan=False, sort_keys=True), encoding="utf-8"
        )
        for repeat in range(3):
            result_root = root / "results" / str(repeat)
            observations.append(
                p14._launch_worker(
                    worker_spec=spec,
                    output_root=result_root,
                    cache_dir=root / "cache" / str(repeat),
                    warmups=1,
                    repeat=repeat,
                )
            )
            result = open_result(result_root / "measured")
            events, final = result.read_boundary_events(), result.read_final()
            order = np.argsort(events.particle_id)
            np.testing.assert_allclose(events.time_s, 0.75, rtol=0, atol=3e-12)
            np.testing.assert_allclose(
                events.position_m, np.tile([1.0, 0.5], (10_000, 1)), rtol=0, atol=3e-12
            )
            np.testing.assert_array_equal(events.contact_radius_m[order], radii)
            np.testing.assert_array_equal(final.contact_radius_m, radii)
            np.testing.assert_array_equal(events.outcome, np.full(10_000, "escaped"))
            assert result.manifest["counts"]["failure_events"] == 0
            assert result.manifest["lifecycle_counts"]["escaped"] == 10_000
            resolved = {
                rule["group"]: rule["contact_geometry"]
                for rule in result.manifest["resolved"]["boundary_laws"]
            }
            assert resolved == {"wall": "particle_surface", "outlet": "particle_center"}
            oracle.append(
                {
                    "repeat": repeat,
                    "event_time_max_abs_error_s": float(np.max(np.abs(events.time_s - 0.75))),
                    "event_position_max_abs_error_m": float(
                        np.max(np.abs(events.position_m - [1.0, 0.5]))
                    ),
                    "retained_body_radii": [0.05, 0.15],
                    "escaped_particles": 10_000,
                    "failure_events": 0,
                    "resolved_group_modes": resolved,
                }
            )
        p14._validate_observations(observations, (item,), 3)
    document = {
        "benchmark": "existing_p14_isolated_worker_h2_mixed_evidence_v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "measurement_started_at_utc": started,
        "non_gating_seconds": True,
        "matrix_scope": "one_additional_current_only_H2_row",
        "conditions": {
            "repeats": 3,
            "warmups_per_isolated_observation": 1,
            "memory_limit_mb": 8192,
            "execution_mode": "single-thread compiled CPU",
            "process_isolation": "fresh process and private initially empty Numba cache per observation",
            "timed_scope": "load_case + simulate + open_result",
            "oracle_checks": "after timed scope",
            "environment_load": "native COMSOL and repository test batches paused; ordinary OS background activity not excluded",
        },
        "profile": {
            "motion": "cartesian_xy",
            "end_s": 1.0,
            "dt_s": 1.0,
            "contact_radius_m": [0.05, 0.15],
            "wall_contact_geometry": "particle_surface",
            "outlet_contact_geometry": "particle_center",
            "law": "escape",
            "expected_event_time_s": 0.75,
            "position_time_oracle_absolute_allowance": 3e-12,
        },
        "machine": p14._p14_machine_metadata(),
        "observations": observations,
        "summaries": p14._summaries(observations),
        "oracle": oracle,
        "identity": {
            "exact_repeats": True,
            "current_only": True,
            "before_baseline_exists_for_this_profile": False,
        },
        "reproduction": {
            "command": "uv run --locked python -c \"import runpy; runpy.run_path('../reviews/contact_mode_mixed_current_2026-10-09.py', run_name='__main__')\"",
            "source_script_sha256": "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
            "case_materializer": "tests.scenarios.test_contact_geometry_run._mixed_case",
            "case_materializer_sha256": "sha256:"
            + sha256(Path("tests/scenarios/test_contact_geometry_run.py").read_bytes()).hexdigest(),
        },
        "interpretation": "Current-only mixed-group cost and integrity evidence; no default/center speedup, portable SLA, full physics performance, or COMSOL timing claim.",
    }
    output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
