"""Content integrity is independent of scientific values and shape checks."""

import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from chamber_particles import SimulationError, open_result
from chamber_particles.output import (
    ResultWriteError,
    ResultWriter,
    TrajectoryFrame,
)
from tests.verification.test_output import (
    _checkpoint_state,
    _final_particles,
    _identity,
    _initialize_context,
)


def _epochs(output: Path, *, complete: bool) -> None:
    with ResultWriter(output, 3, _identity("content-integrity")) as writer:
        _initialize_context(writer)
        for index in range(4):
            writer.begin_epoch()
            writer.write_frame(
                TrajectoryFrame(
                    index / 4,
                    np.asarray([10], dtype="<i8"),
                    np.asarray([[index / 4, 0.0]], dtype="<f8"),
                    np.asarray([[1.0, 0.0]], dtype="<f8"),
                    np.asarray([0.0], dtype="<f8"),
                    np.asarray([1], dtype="u1"),
                )
            )
            writer.commit_epoch(_checkpoint_state(macro_step_count=index + 1))
        if complete:
            writer.finalize(_final_particles(), {}, macro_step_count=4)


def _finite_edit(path: Path, dataset: str) -> None:
    with h5py.File(path, "r+") as handle:
        handle[dataset][0, 0] += 0.125


def test_old_segment_finite_edits_fail_resume_and_cannot_be_blessed_at_seal(tmp_path: Path) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=False)
    partial = output.with_name("result.partial")
    _finite_edit(partial / "segments/epoch-000000.h5", "frames/position_m")
    with pytest.raises(SimulationError, match="content hash"):
        open_result(output, recovery=True)
    with pytest.raises(ResultWriteError, match="resumable"):
        ResultWriter(output, 3, _identity("content-integrity"))
    assert not output.exists() and not (partial / "_SUCCESS").exists()


def test_finalization_uses_commit_digest_even_in_the_same_writer(tmp_path: Path) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=False)
    with ResultWriter(output, 3, _identity("content-integrity")) as writer:
        partial = output.with_name("result.partial")
        _finite_edit(partial / "segments/epoch-000000.h5", "frames/position_m")
        with pytest.raises(ResultWriteError, match="before finalization"):
            writer.finalize(_final_particles(), {}, macro_step_count=4)
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["final.h5", "segments/epoch-000000.h5", "context.h5"])
def test_public_lazy_reads_recheck_bytes_after_open(tmp_path: Path, artifact: str) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=True)
    result = open_result(output)
    identity = result.source_identity()
    result.verify_contents()
    if artifact == "final.h5":
        _finite_edit(output / artifact, "particles/position_m")
        read = result.read_final
    elif artifact == "context.h5":
        with (output / artifact).open("ab") as stream:
            stream.write(b"changed")
        read = result.read_context
    else:
        _finite_edit(output / artifact, "frames/position_m")

        def read() -> object:
            return list(result.iter_frames())

    with pytest.raises(SimulationError, match="content hash"):
        read()
    assert result.source_identity() == identity


def test_generator_reports_late_missing_file_as_public_error(tmp_path: Path) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=True)
    result = open_result(output)
    frames = result.iter_frames()
    assert next(frames).time_s == 0
    (output / "segments/epoch-000001.h5").unlink()
    with pytest.raises(SimulationError, match="cannot be read"):
        next(frames)
    with pytest.raises(ValueError, match="positive integer"):
        next(result.iter_boundary_event_batches(batch_rows=0))


def test_explicit_verification_reports_full_scope_without_mutating_identity(tmp_path: Path) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=True)
    result = open_result(output)
    identity = result.source_identity()
    assert result.verification_document()["scope"] == "consumed_file_bytes"
    receipt = result.verify_contents()
    assert receipt["scope"] == "all_payload_file_bytes"
    assert len(receipt["content_files"]) == 6
    assert result.source_identity() == identity


@pytest.mark.parametrize("damage", ["old_schema", "missing", "extra", "path_escape", "unbound"])
def test_roster_requires_the_exact_current_committed_payload(tmp_path: Path, damage: str) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=True)
    path = output / "run.json"
    manifest = json.loads(path.read_bytes())
    roster = manifest["content_files"]
    if damage == "old_schema":
        manifest["result_schema_version"] = 4
    elif damage == "missing":
        del roster["segments/epoch-000000.h5"]
    elif damage == "extra":
        roster["uncommitted.h5"] = roster["final.h5"]
    elif damage == "path_escape":
        roster["../outside.h5"] = roster.pop("segments/epoch-000000.h5")
    else:
        roster["segments/epoch-000000.h5"] = "sha256:" + "f" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SimulationError):
        open_result(output)


def test_duplicate_roster_keys_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=True)
    path = output / "run.json"
    encoded = path.read_text(encoding="utf-8")
    encoded = encoded.replace('"content_files": {', '"content_files": {"final.h5": "changed",', 1)
    path.write_text(encoded, encoding="utf-8")
    with pytest.raises(SimulationError):
        open_result(output)


def test_read_uses_the_file_opened_before_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result"
    _epochs(output, complete=True)
    result = open_result(output)
    expected = result.read_final().position_m.copy()
    replacement = tmp_path / "replacement.h5"
    shutil.copyfile(output / "final.h5", replacement)
    _finite_edit(replacement, "particles/position_m")
    original_open = h5py.File

    def replace_before_hdf5_read(*args: object, **kwargs: object) -> h5py.File:
        try:
            os.replace(replacement, output / "final.h5")
        except PermissionError:
            pytest.skip("host denies replacement while the original read handle is open")
        return original_open(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(h5py, "File", replace_before_hdf5_read)
        np.testing.assert_array_equal(result.read_final().position_m, expected)
    with pytest.raises(SimulationError, match="content hash"):
        result.read_final()
