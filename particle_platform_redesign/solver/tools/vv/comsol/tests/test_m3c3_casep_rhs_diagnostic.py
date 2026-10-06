import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.vv.comsol import diagnose_m3c3_casep_rhs as diagnostic


def test_read_wide_preserves_particle_then_frame_order(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("% COMSOL diagnostic\n")
        writer = csv.writer(stream, lineterminator="\n")
        for particle in range(1, diagnostic.EXPECTED_PARTICLES + 1):
            row: list[float] = []
            for frame in range(diagnostic.EXPECTED_FRAMES):
                row.extend((float(particle), float(frame)))
            writer.writerow(row)

    values = diagnostic._read_wide(path, ("particle", "frame"))

    assert values["particle"].shape == (diagnostic.EXPECTED_PARTICLES * diagnostic.EXPECTED_FRAMES,)
    assert np.array_equal(
        values["particle"][: diagnostic.EXPECTED_FRAMES],
        np.ones(diagnostic.EXPECTED_FRAMES),
    )
    assert np.array_equal(
        values["frame"][: diagnostic.EXPECTED_FRAMES],
        np.arange(diagnostic.EXPECTED_FRAMES, dtype=np.float64),
    )
    assert values["particle"][-1] == diagnostic.EXPECTED_PARTICLES


def test_metric_uses_predeclared_relative_l2_limit() -> None:
    reference = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    keys = {
        "particle_id": np.asarray([1.0, 1.0, 1.0]),
        "time_s": np.asarray([0.0, 1.0, 2.0]),
    }

    passing = diagnostic._metric(reference * (1.0 + 0.5e-8), reference, keys)
    blocked = diagnostic._metric(reference * (1.0 + 2.0e-8), reference, keys)

    assert passing["passed"] is True
    assert blocked["passed"] is False
    assert passing["worst_key"] == {"particle_id": 1, "time_s": 2.0}


def test_diagnostic_is_external_same_state_analysis() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")

    assert diagnostic.TOOL_REVISION == "m3c3_casep_frozen_rhs_v2"
    assert diagnostic.FORMULA_PARITY_RELATIVE_L2_LIMIT == 1.0e-8
    assert "simulate(" not in source
    assert "production_vs_comsol" in source
    assert "same_state_active_rows_formula_and_runtime_parity" in source
    assert "general COMSOL equivalence" in source


def test_execution_record_binds_existing_runner_outputs(tmp_path: Path) -> None:
    state = tmp_path / "diagnostic_state_raw_wide.csv"
    force = tmp_path / "diagnostic_force_raw_wide.csv"
    state.write_bytes(b"state\n")
    force.write_bytes(b"force\n")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_hash = "a" * 64
    record = tmp_path / "execution_inputs.json"
    record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "LOCKED_FOR_EXECUTION",
                "case_id": "caseP_100nm_three_current",
                "expected_source_mph_sha256": source_hash,
                "source_mph_sha256_before": source_hash,
                "source_mph_sha256_after": source_hash,
                "numerical_run": {
                    "fixed_rk4_step_s": 1.25e-6,
                    "diagnostic_force_export": True,
                    "comparison_scope": "active_rows_only",
                },
                "artifacts": [
                    {
                        "role": "diagnostic_state_raw",
                        "path": state.name,
                        "sha256": digest(state),
                    },
                    {
                        "role": "diagnostic_force_raw",
                        "path": force.name,
                        "sha256": digest(force),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    provenance = diagnostic._validate_execution_record(record, state, force)

    assert provenance["source_mph_sha256"] == source_hash
    assert provenance["source_unchanged"] is True
