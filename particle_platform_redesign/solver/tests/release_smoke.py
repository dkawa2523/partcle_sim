"""Runtime-only wheel smoke for public operations and process execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import fields, replace
from email import policy
from email.parser import BytesParser
from importlib.metadata import Distribution, distribution
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import chamber_particles
from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import RealizedTableSource, write
from tests.verification.microcases import materialize_microcase


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="wheel installed in this runtime")
    parser.add_argument("--json", type=Path, help="save the observed verification receipt")
    arguments = parser.parse_args(argv)
    identity = _wheel_identity(arguments.wheel)
    with tempfile.TemporaryDirectory(prefix="chamber-particles-release-smoke-") as temporary:
        root = Path(temporary)
        static_path = materialize_microcase("C01", root / "static").case_path
        _, static = _run_case(static_path, root / "static-result")
        if (
            static.read_final().particle_id.tolist() != [101]
            or len(tuple(static.iter_frames())) != 5
        ):
            raise RuntimeError("static wheel smoke produced an unexpected particle/frame table")
        time_case, time_result = _run_case(_time_field_case(root / "time"), root / "time-result")
        _check_time_field(time_case, time_result)
        contact_records = [
            _contact_case(root / mode, mode) for mode in ("particle_surface", "particle_center")
        ]
        parallel_path = _parallel_case(root / "parallel")
        _, reference = _run_case(parallel_path, root / "parallel-reference")
        _, parallel = _run_case(parallel_path, root / "parallel-result", workers=2)
        _check_identity(reference, parallel)
        cli = _check_cli(parallel_path, root / "cli", reference)
        payload = {
            "package": "chamber-particles",
            "status": "complete",
            "public_operations": ["load_case", "simulate", "open_result"],
            "wheel_identity": identity,
            "time_field": {"status": "complete", "snapshot_splits_s": [0.15]},
            "contact_geometry": contact_records,
            "process_execution": {
                "status": "complete",
                "workers": 2,
                "particle_count": 32,
                "serial_result_bytes_match": True,
                "live_api_children_after_return": 0,
                "input_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (parallel_path, parallel_path.with_name("parallel.h5"))
                },
            },
            "cli": cli,
            "claim": "Installed-wheel and small analytic input smoke; no application accuracy or performance SLA",
        }
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def _wheel_identity(wheel: Path) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    installed = distribution("chamber-particles")
    package = Path(str(chamber_particles.__file__)).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("release smoke requires the clean-installed wheel, not editable source")
    expected_version = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    if installed.version != expected_version:
        raise RuntimeError("installed wheel version differs from the project metadata")
    source = _python_payload(project / "src/chamber_particles")
    with zipfile.ZipFile(wheel) as archive:
        archived = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("chamber_particles/") and name.endswith(".py")
        }
        metadata_identity = _metadata_identity(project, installed, archive)
    observed = _python_payload(package.parent)
    if not source or source != archived or source != observed:
        raise RuntimeError("source, built wheel and installed Python payload differ")
    return {
        "version": installed.version,
        "python": sys.version,
        "platform": sys.platform,
        "package_path": str(package),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
        "production_python_sha256": observed,
        "all_python_bytes_match": True,
        **metadata_identity,
    }


def _metadata_identity(
    project: Path, installed: Distribution, archive: zipfile.ZipFile
) -> dict[str, Any]:
    names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(names) != 1:
        raise RuntimeError("expected exactly one wheel distribution metadata file")
    archived = archive.read(names[0])
    installed_text = installed.read_text("METADATA")
    if installed_text is None or archived.decode("utf-8").replace(
        "\r\n", "\n"
    ) != installed_text.replace("\r\n", "\n"):
        raise RuntimeError("installed metadata differs from the supplied wheel")
    body = BytesParser(policy=policy.default).parsebytes(archived).get_payload(decode=True)
    readme = project / "README.md"
    if not isinstance(body, bytes) or body.decode("utf-8").replace("\r\n", "\n").rstrip(
        "\n"
    ) != readme.read_text(encoding="utf-8").rstrip("\n"):
        raise RuntimeError("wheel long description differs from the current source README")
    return {
        "readme_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "wheel_metadata_sha256": hashlib.sha256(archived).hexdigest(),
        "installed_metadata_text_matches_wheel": True,
        "long_description_matches_current_readme": True,
    }


def _python_payload(package: Path) -> dict[str, str]:
    return {
        "chamber_particles/" + path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }


def _run_case(case_path: Path, output: Path, *, workers: int = 1) -> tuple[Any, Any]:
    case = load_case(case_path)
    simulate(case, output, workers=workers)
    if multiprocessing.active_children():
        raise RuntimeError("wheel smoke left live API worker processes after simulate returned")
    result = open_result(output)
    if result.manifest.get("status") != "complete" or result.read_failure_events().particle_id.size:
        raise RuntimeError("wheel smoke produced an incomplete or failed particle result")
    if result.manifest["execution"]["worker_count"] != workers:
        raise RuntimeError("wheel smoke did not execute with the requested process count")
    return case, result


def _parallel_case(directory: Path) -> Path:
    paths = materialize_microcase("C01", directory)
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    source = replace(
        source,
        **{
            item.name: np.repeat(value, 32, axis=0)
            for item in fields(source)
            if isinstance(value := getattr(source, item.name), np.ndarray)
        },
    )
    source = replace(source, particle_id=np.arange(101, 133, dtype="<i8"))
    info = write(directory / "parallel.h5", replace(case.data, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path="parallel.h5", expected_content_hash=info.content_hash)
    document["output"]["probes"] = {
        "particle_ids": source.particle_id.tolist(),
        "schedule": {"explicit_times_s": [0.0, 0.05, 0.15, 0.35, 0.8]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return paths.case_path


def _check_identity(reference: Any, actual: Any) -> None:
    for name in (
        "resume_identity",
        "counts",
        "lifecycle_counts",
        "boundary_interactions",
        "failure_reason_counts",
    ):
        if reference.manifest[name] != actual.manifest[name]:
            raise RuntimeError(f"parallel wheel result differs in {name}")
    for reader in (
        "read_final",
        "read_release_events",
        "read_boundary_events",
        "read_failure_events",
        "read_lifecycle_series",
    ):
        _check_record_bytes(getattr(reference, reader)(), getattr(actual, reader)())
    for reader in ("iter_frames", "iter_probes"):
        for first, second in zip(
            getattr(reference, reader)(), getattr(actual, reader)(), strict=True
        ):
            _check_record_bytes(first, second)


def _check_record_bytes(reference: Any, actual: Any) -> None:
    if type(reference) is not type(actual):
        raise RuntimeError("parallel wheel result record type differs")
    for item in fields(reference):
        first, second = getattr(reference, item.name), getattr(actual, item.name)
        if isinstance(first, np.ndarray):
            if (
                not isinstance(second, np.ndarray)
                or first.dtype != second.dtype
                or first.shape != second.shape
                or first.tobytes(order="C") != second.tobytes(order="C")
            ):
                raise RuntimeError(f"parallel wheel result bytes differ in {item.name}")
        elif first != second:
            raise RuntimeError(f"parallel wheel result differs in {item.name}")


def _time_field_case(directory: Path) -> Path:
    paths = materialize_microcase("C04", directory)
    case = load_case(paths.case_path)
    static = case.data.fields[0]
    knots = np.asarray([0.0, 0.15, 0.5], dtype="<f8")
    values = np.zeros((3, static.values.shape[0], 2), dtype="<f8")
    values[:, :, 0] = knots[:, None]
    field = replace(static, values=values, time_s=knots)
    info = write(directory / "time.h5", replace(case.data, fields=(field,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path="time.h5", expected_content_hash=info.content_hash)
    document["time"].update(end_s=0.5, dt_s=0.2)
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 0.25, 0.5]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return paths.case_path


def _check_time_field(case: Any, result: Any) -> None:
    source = case.data.sources[0]
    acceleration_slope = source.charge_number * 1.602176634e-19 / source.mass_kg
    for frame in result.iter_frames():
        expected_x = source.position_m + source.velocity_m_s * frame.time_s
        expected_v = source.velocity_m_s.copy()
        expected_x[:, 0] += acceleration_slope * frame.time_s**3 / 6.0
        expected_v[:, 0] += acceleration_slope * frame.time_s**2 / 2.0
        np.testing.assert_allclose(frame.position_m, expected_x, rtol=0, atol=2e-14)
        np.testing.assert_allclose(frame.velocity_m_s, expected_v, rtol=0, atol=2e-14)
    if result.manifest["time"]["field_snapshot_splits_s"] != [0.15]:
        raise RuntimeError("time-field wheel smoke did not preserve the off-grid snapshot knot")


def _contact_case(directory: Path, mode: str) -> dict[str, Any]:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    source = replace(source, contact_radius_m=np.asarray([0.05], dtype="<f8"))
    info = write(directory / "contact.h5", replace(case.data, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path="contact.h5", expected_content_hash=info.content_hash)
    document["boundaries"][0]["contact_geometry"] = mode
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    _, result = _run_case(paths.case_path, directory / "result")
    event = result.read_boundary_events()
    expected_time = 1.4 if mode == "particle_surface" else 1.5
    np.testing.assert_allclose(event.time_s, [expected_time], rtol=0, atol=3e-12)
    np.testing.assert_allclose(
        event.position_m,
        source.position_m + source.velocity_m_s * expected_time,
        rtol=0,
        atol=3e-12,
    )
    np.testing.assert_array_equal(event.contact_radius_m, [0.05])
    np.testing.assert_array_equal(result.read_final().contact_radius_m, [0.05])
    if result.manifest["resolved"]["boundary_laws"][0]["contact_geometry"] != mode:
        raise RuntimeError("contact mode differs from the requested group semantics")
    return {
        "mode": mode,
        "radius_m": 0.05,
        "hit_time_s": float(event.time_s[0]),
        "status": "complete",
    }


def _check_cli(case_path: Path, output: Path, reference: Any) -> dict[str, Any]:
    environment = {**os.environ, "PYTHONIOENCODING": "ascii"}
    executable = Path(sys.prefix) / (
        "Scripts/chamber-particles.exe" if os.name == "nt" else "bin/chamber-particles"
    )
    commands = (
        ([str(executable), "check", str(case_path)], 0),
        (
            [
                str(executable),
                "run",
                str(case_path),
                "-o",
                str(output / "console"),
                "--workers",
                "2",
            ],
            0,
        ),
        (
            [
                sys.executable,
                "-m",
                "chamber_particles",
                "run",
                str(case_path),
                "-o",
                str(output / "module"),
                "--workers",
                "2",
            ],
            0,
        ),
        ([str(executable), "inspect", str(output / "module")], 0),
        (
            [
                sys.executable,
                "-m",
                "chamber_particles",
                "check",
                str(case_path.parent / "不存在🧪.yaml"),
            ],
            2,
        ),
    )
    for command, expected in commands:
        process = subprocess.run(
            command, env=environment, capture_output=True, check=False, timeout=180
        )
        if process.returncode != expected or (process.stderr if expected == 0 else process.stdout):
            message = process.stderr.decode("ascii", "backslashreplace")
            raise RuntimeError(f"clean-wheel CLI failed ({command!r}): {message}")
        payload = json.loads((process.stdout or process.stderr).decode("ascii"))
        if expected == 2 and payload.get("error_type") != "CaseError":
            raise RuntimeError("clean-wheel CLI did not preserve the public error category")
    for entry in ("console", "module"):
        result = open_result(output / entry)
        if result.manifest["execution"]["worker_count"] != 2:
            raise RuntimeError(f"clean-wheel {entry} CLI did not execute with two workers")
        _check_identity(reference, result)
    return {
        "console_script": "complete",
        "module_entry": "complete",
        "workers": 2,
        "serial_result_bytes_match": True,
        "subprocess_timeout_s": 180,
        "public_error_exit": 2,
    }


if __name__ == "__main__":
    main()
