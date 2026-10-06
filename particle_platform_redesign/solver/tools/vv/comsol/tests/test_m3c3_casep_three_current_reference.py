from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from tools.vv.comsol.normalize_m3c3_caseP_three_current import normalize
from tools.vv.comsol.prepare_m3c3_caseP_reference_tables import EXPORTS, Export, prepare

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_fragments(
    text: str,
    *,
    present: tuple[str, ...],
    absent: tuple[str, ...] = (),
) -> None:
    for fragment in present:
        assert fragment in text
    for fragment in absent:
        assert fragment not in text


def _release_rows() -> list[tuple[float, ...]]:
    return [
        (particle_id, 0.0, particle_id * 1.0e-7, 0.02, 0.1, -0.05, -500.0)
        for particle_id in range(1, 288)
    ]


def _write_release(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "release_time_s",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number_e",
            )
        )
        writer.writerows(_release_rows())


def _write_canonical(path: Path) -> None:
    text = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as output:
        meta = output.create_group("meta")
        meta.create_dataset("content_hash", data="sha256:test", dtype=text)
        layout = output.create_group("layouts/plasma/unstructured")
        layout.create_dataset("nodes_m", data=[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
        layout.create_dataset("connectivity", data=[[0, 1, 2]])
        layout.create_dataset("cell_support", data=[1])
        fields = output.create_group("fields")
        by_field: dict[str, list[Export]] = {}
        for export in EXPORTS:
            by_field.setdefault(export.field, []).append(export)
        for field_name, exports in by_field.items():
            components = [export.component for export in exports]
            group = fields.create_group(field_name)
            group.create_dataset("association", data="node", dtype=text)
            group.create_dataset("components", data=components, dtype=text)
            group.create_dataset("layout", data="plasma", dtype=text)
            group.create_dataset(
                "stored_basis",
                data="scalar" if components == ["value"] else "axisymmetric_rz",
                dtype=text,
            )
            group.create_dataset("unit", data=exports[0].unit, dtype=text)
            group.create_dataset("values", data=np.ones((3, len(components))))
        source = output.create_group("sources/particles")
        release = np.asarray(_release_rows())
        source.create_dataset("particle_id", data=release[:, 0].astype(np.int64))
        source.create_dataset("release_time_s", data=release[:, 1])
        source.create_dataset("position_m", data=release[:, 2:4])
        source.create_dataset("velocity_m_s", data=release[:, 4:6])
        source.create_dataset("charge_number", data=release[:, 6])


def test_reference_table_preparer_preserves_shared_three_current_z0(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    release = tmp_path / "release.csv"
    output = tmp_path / "reference"
    _write_canonical(candidate)
    _write_release(release)

    receipt = prepare(candidate, release, output)

    assert receipt["status"] == "PASS"
    assert receipt["component_count"] == 27
    assert len(receipt["artifacts"]) == 31  # type: ignore[arg-type]
    assert (output / "m3c3_nn_sectionwise.txt").is_file()
    z0 = np.loadtxt(output / "m3c1_Z0.txt")
    np.testing.assert_array_equal(z0[:, 2], np.full(287, -500.0))
    assert receipt["release"]["initial_charge_recomputed_by_comsol"] is False  # type: ignore[index]


def _configuration_line(step_text: str = "1e-5") -> str:
    values = {
        "case": "caseP_100nm_three_current",
        "step_s": step_text,
        "time_end_s": "0.03",
        "output_times": "121",
        "particle_rows": "287",
        "physics": "fpt",
        "background_study": "std2",
        "background_step": "ftper",
        "background_solution": "sol2",
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "charge_revision": "aggregate_relative_drift_regularized_three_current_v1",
        "ion_drag_revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "drag_revision": "epstein_linear_effective_gas_sensitivity_v1",
        "drag_implementation": "explicit_custom_force",
        "maximum_relative_ion_speed_m_s": "1e6",
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-8",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "release_source": "shared_three_current_release_table",
        "boundary_material": "stick",
        "boundary_37": "freeze_hold",
        "boundary_35": "disappear_escape",
        "source_model": "source_copy.mph",
        "model_saved": "false",
    }
    return "M3C3_CASEP|configuration|" + "|".join(
        f"{name}={value}" for name, value in values.items()
    )


def _times() -> list[float]:
    return (
        [index * 1.0e-5 for index in range(51)]
        + [index * 1.0e-4 for index in range(6, 51)]
        + [index * 1.0e-3 for index in range(6, 31)]
    )


def _write_raw(path: Path) -> None:
    release = _release_rows()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for particle_id, _, r_m, z_m, vr, vz, charge in release:
            row: list[float] = []
            for time_s in _times():
                row.extend((particle_id, time_s, r_m, z_m, vr, vz, charge, 1.0, 1.0, 0.0))
            writer.writerow(row)


def _write_normalizer_inputs(root: Path, step_s: float = 1.0e-5) -> None:
    step_text = {
        1.0e-5: "1e-5",
        5.0e-6: "5e-6",
        2.5e-6: "2.5e-6",
        1.25e-6: "1.25e-6",
    }[step_s]
    numerical_run_role = "baseline" if step_s == 1.0e-5 else "time_step_refinement"
    release = root / "three_current_release_state.csv"
    _write_release(release)
    _write_raw(root / "trajectory_raw_wide.csv")
    log = "\n".join(
        (
            _configuration_line(step_text),
            f"M3C3_CASEP|solve_pass|step_s={step_text}|seconds=1.0|base_solution=solX",
            "M3C3_CASEP|run_pass|case=caseP_100nm_three_current|"
            f"step_s={step_text}|"
            "time_end_s=0.03|output_times=121|particles=287|model_saved=false",
        )
    )
    (root / "comsol_process.log").write_text(log + "\n", encoding="utf-8")
    (root / "comsol_process_metrics.json").write_text(
        json.dumps(
            {
                "wall_time_s": 2.0,
                "peak_rss_bytes": 1000,
                "fixed_rk4_step_s": step_s,
                "numerical_run_role": numerical_run_role,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = {
        "schema_version": 1,
        "status": "LOCKED_FOR_EXECUTION",
        "reference_run_config_sha256": "f" * 64,
        "expected_source_mph_sha256": "source",
        "source_mph_sha256_before": "source",
        "source_mph_sha256_after": "source",
        "numerical_run": {
            "role": numerical_run_role,
            "fixed_rk4_step_s": step_s,
            "base_config_fixed_rk4_step_s": 1.0e-5,
        },
        "artifacts": [
            {
                "role": "release_state",
                "path": release.name,
                "sha256": _sha256(release),
            }
        ],
    }
    (root / "execution_inputs.json").write_text(
        json.dumps(inputs, indent=2) + "\n", encoding="utf-8"
    )


def test_normalizer_emits_fixed_reference_schema_and_receipts(tmp_path: Path) -> None:
    _write_normalizer_inputs(tmp_path)

    summary = normalize(tmp_path)

    assert summary["status"] == "COMPLETE_NORMALIZED_NOT_EVALUATED"
    assert summary["trajectory_rows"] == 287 * 121
    assert summary["fixed_rk4_step_s"] == 1.0e-5
    assert summary["numerical_run_role"] == "baseline"
    assert summary["reference_run_config_sha256"] == "f" * 64
    with (tmp_path / "trajectory_reference.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 287 * 121
    assert {row["lifecycle"] for row in rows} == {"active"}
    receipt = json.loads((tmp_path / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"


def test_normalizer_accepts_explicit_five_microsecond_refinement(tmp_path: Path) -> None:
    _write_normalizer_inputs(tmp_path, 5.0e-6)

    summary = normalize(tmp_path)

    assert summary["fixed_rk4_step_s"] == 5.0e-6
    assert summary["numerical_run_role"] == "time_step_refinement"
    receipt = json.loads((tmp_path / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["configuration"]["step_s"] == "5e-6"
    assert receipt["execution_inputs"]["record"]["numerical_run"] == {
        "role": "time_step_refinement",
        "fixed_rk4_step_s": 5.0e-6,
        "base_config_fixed_rk4_step_s": 1.0e-5,
    }


def test_normalizer_accepts_explicit_two_point_five_microsecond_refinement(
    tmp_path: Path,
) -> None:
    _write_normalizer_inputs(tmp_path, 2.5e-6)

    summary = normalize(tmp_path)

    assert summary["fixed_rk4_step_s"] == 2.5e-6
    assert summary["numerical_run_role"] == "time_step_refinement"
    receipt = json.loads((tmp_path / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["configuration"]["step_s"] == "2.5e-6"


def test_normalizer_accepts_explicit_one_point_two_five_microsecond_refinement(
    tmp_path: Path,
) -> None:
    _write_normalizer_inputs(tmp_path, 1.25e-6)

    summary = normalize(tmp_path)

    assert summary["fixed_rk4_step_s"] == 1.25e-6
    assert summary["numerical_run_role"] == "time_step_refinement"
    receipt = json.loads((tmp_path / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["configuration"]["step_s"] == "1.25e-6"


def test_java_and_powershell_fix_the_casep_campaign_without_core_dispatch() -> None:
    java = (ROOT / "comsol" / "RunM3C3CasePThreeCurrent.java").read_text(encoding="utf-8")
    runner = (ROOT / "run_m3c3_caseP_three_current.ps1").read_text(encoding="utf-8")

    _assert_fragments(
        java,
        present=(
            'ModelUtil.loadCopy("M3C3CasePThreeCurrent", SOURCE)',
            'physics.feature("bf1").active(false)',
            "m3c3_nn",
            "m3c3_unr",
            "m3c3_TnV",
            "aggregate_relative_drift_regularized_three_current_v1",
            "relative_flow_screened_collection_orbital_aggregate_ion_v1",
            'physics.feature("relg1").set("aux0_auxq", "m3c1_Z0(r,z)")',
            'physics.feature("df1").active(false)',
            'private static final String EPSTEIN_TAG = "m3c3Epstein"',
            'physics.create(EPSTEIN_TAG, "Force", 2)',
            "1.3534291735288517",
            '"idf"',
            '"ef1"',
            '"liftfm"',
            '"depf"',
            '"gf1"',
            '"1e-5".equals(value) || "5e-6".equals(value)',
            '|| "2.5e-6".equals(value)',
            '|| "1.25e-6".equals(value)',
        ),
        absent=("ModelUtil.save", "System.getenv"),
    )
    assert java.count("__M3C3_FIXED_STEP_SECONDS__") == 1
    _assert_fragments(
        runner,
        present=(
            "candidate_input_three_current_z0.h5",
            "three_current_release_receipt.json",
            "-nosave",
            "$ExpectedSourceHash",
            "[double]$FixedStepS = 1.0e-5",
            '$StepToken = "__M3C3_FIXED_STEP_SECONDS__"',
            "$StepTokenCount -ne 1",
            "$StagedJavaText.Replace($StepToken, $FixedStepText)",
            "base_config_fixed_rk4_step_s",
            '"reference_dt_2p5us"',
            '"reference_dt_1p25us"',
        ),
    )


def test_java_uses_comsol_safe_one_metre_per_second_regularization() -> None:
    java = (ROOT / "comsol" / "RunM3C3CasePThreeCurrent.java").read_text(encoding="utf-8")

    assert java.count("+(1[m/s])^2") == 2
    assert "+1[m/s]^2" not in java
