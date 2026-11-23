"""P14-U process-boundary observations survive a later operational failure."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from tests.performance import p14u_representative as representative


@pytest.mark.parametrize("fault", ["profile", "observation_fsync"])
def test_completed_worker_observations_survive_a_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    # Keep the real driver, specification writer, process decoder and filesystem
    # publisher. Replace only scientific input creation and the process boundary;
    # these sentinel records are observations of this fault test, not timings.
    monkeypatch.setattr(representative, "_convergence_gate", lambda *args: {})
    monkeypatch.setattr(representative, "_rz_gate", lambda *args: {})

    def materialize(directory: Path, **settings: object) -> Path:
        directory.mkdir(parents=True)
        path = directory / "case.yaml"
        path.write_text("synthetic fault-test input", encoding="utf-8")
        return path

    monkeypatch.setattr(representative, "_materialize_xy_case", materialize)
    timed_caches: dict[tuple[int, str, int], Path] = {}
    returned: dict[str, str] = {}
    temporary_roots: set[Path] = set()
    profile_caches: list[Path] = []

    def run_worker(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        specification_path = Path(command[-1])
        specification = json.loads(specification_path.read_text(encoding="utf-8"))
        temporary_roots.add(specification_path.parents[2])
        environment = options["env"]
        assert isinstance(environment, dict)
        cache = Path(environment["NUMBA_CACHE_DIR"])
        if specification["operation"] == "profile":
            profile_caches.append(cache)
            assert cache == timed_caches[(1_000_000, "none", 0)]
            assert (cache / "compiled.marker").read_text(encoding="utf-8") == "1000000:none:0"
            return subprocess.CompletedProcess(command, 1, "", "injected profile failure")

        key = (
            specification["particle_count"],
            specification["output_mode"],
            specification["repeat_index"],
        )
        assert cache not in timed_caches.values()
        assert not cache.exists(), "each timed process requires an initially fresh cache"
        cache.mkdir(parents=True)
        (cache / "compiled.marker").write_text(":".join(map(str, key)), encoding="utf-8")
        timed_caches[key] = cache
        encoded = json.dumps({"fault_test_observation": specification}, sort_keys=True) + "\n"
        returned[specification_path.name] = encoded
        return subprocess.CompletedProcess(command, 0, encoded, "")

    monkeypatch.setattr(representative.subprocess, "run", run_worker)
    real_fsync = representative.os.fsync
    sync_count = 0

    def observe_sync(descriptor: int) -> None:
        nonlocal sync_count
        sync_count += 1
        # The final name must remain absent even while its complete temporary
        # file is being flushed. A direct exclusive write would expose it here.
        current_name = list(returned)[-1]
        assert not (observed_directory / current_name).exists()
        if fault == "observation_fsync" and sync_count == 3:
            raise OSError("injected observation fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(representative.os, "fsync", observe_sync)

    report_path = tmp_path / "release.json"
    observed_directory = report_path.with_name(report_path.name + ".observations")
    arguments = argparse.Namespace(
        suite="release", particles=None, memory_limit_mb=8192, json_path=report_path
    )
    error_type = RuntimeError if fault == "profile" else OSError
    with pytest.raises(error_type, match="injected"):
        representative._driver(arguments)

    assert not report_path.exists(), "an incomplete attempt cannot publish a release report"
    assert temporary_roots and all(not root.exists() for root in temporary_roots)
    retained = {
        path.name: path.read_text(encoding="utf-8") for path in observed_directory.iterdir()
    }
    expected_count = 18 if fault == "profile" else 2
    assert len(retained) == expected_count
    assert retained == dict(list(returned.items())[:expected_count])
    for encoded in retained.values():
        assert set(json.loads(encoded)) == {"fault_test_observation"}
    if fault == "profile":
        assert sync_count == 18
        assert len(timed_caches) == 18
        assert profile_caches == [timed_caches[(1_000_000, "none", 0)]]
    else:
        assert sync_count == 3
        assert len(timed_caches) == 3
        assert not profile_caches


@pytest.mark.parametrize("existing", ["report", "observations"])
def test_existing_attempt_rejected_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    report_path = tmp_path / "release.json"
    observation_directory = report_path.with_name(report_path.name + ".observations")
    if existing == "report":
        report_path.write_text("previous complete report", encoding="utf-8")
    else:
        observation_directory.mkdir()
        (observation_directory / "previous.json").write_text(
            "previous observation", encoding="utf-8"
        )

    def unexpected_measurement(*args: object) -> None:
        pytest.fail("a preexisting attempt must be rejected before measurement")

    monkeypatch.setattr(representative, "_convergence_gate", unexpected_measurement)
    arguments = argparse.Namespace(
        suite="release", particles=None, memory_limit_mb=8192, json_path=report_path
    )
    with pytest.raises(FileExistsError):
        representative._driver(arguments)
    if existing == "report":
        assert report_path.read_text(encoding="utf-8") == "previous complete report"
        assert not observation_directory.exists()
    else:
        assert not report_path.exists()
        assert (observation_directory / "previous.json").read_text(
            encoding="utf-8"
        ) == "previous observation"
