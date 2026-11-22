from __future__ import annotations

import ctypes
import errno
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import chamber_particles.output as output_module
from chamber_particles.output import ResultWriteError, ResultWriter, open_result_store
from tests.verification.test_output import _checkpoint_state, _final_particles

_IDENTITY = {"case": "directory-publication"}


def _committed_writer(output: Path) -> ResultWriter:
    writer = ResultWriter(output, particle_capacity=3, resume_identity=_IDENTITY)
    writer.begin_epoch()
    writer.commit_epoch(_checkpoint_state(macro_step_count=1))
    return writer


def _publication(output: Path, *, resume: bool) -> None:
    if resume:
        with ResultWriter(output, particle_capacity=3, resume_identity=_IDENTITY) as writer:
            assert writer.completed_summary is not None
    else:
        with _committed_writer(output) as writer:
            writer.finalize(_final_particles(), {}, macro_step_count=1)


def _completed_partial(output: Path) -> Path:
    _publication(output, resume=False)
    partial = output.with_name(output.name + ".partial")
    os.rename(output, partial)
    return partial


def _denial(code: int | None) -> PermissionError:
    error = PermissionError(errno.EACCES, "controlled publication denial")
    if code is not None:
        error.winerror = code
    return error


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [0.0]
    monkeypatch.setattr(output_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        output_module.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )
    return clock


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_publication_keeps_complete_science_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool, code: int
) -> None:
    output = tmp_path / "result"
    if resume:
        _completed_partial(output)
    real_rename = os.rename
    attempts = []

    def transient(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        if len(attempts) <= 2:
            raise _denial(code)
        real_rename(source, destination)

    clock = _fake_clock(monkeypatch)
    monkeypatch.setattr(output_module.sys, "platform", "win32")
    monkeypatch.setattr(output_module.os, "rename", transient)
    _publication(output, resume=resume)
    assert len(attempts) == 3 and clock[0] > 0
    assert not output.with_name("result.partial").exists()
    np.testing.assert_array_equal(
        open_result_store(output).read_final().position_m, _final_particles().position_m
    )


@pytest.mark.parametrize("resume", [False, True])
def test_permanent_windows_denial_expires_with_original_error_and_complete_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    output = tmp_path / "result"
    partial = _completed_partial(output) if resume else output.with_name("result.partial")
    errors = []

    def denied(source: Path, destination: Path) -> None:
        error = _denial(5)
        errors.append(error)
        raise error

    clock = _fake_clock(monkeypatch)
    monkeypatch.setattr(output_module.sys, "platform", "win32")
    monkeypatch.setattr(output_module.os, "rename", denied)
    with pytest.raises(PermissionError) as caught:
        _publication(output, resume=resume)
    assert caught.value is errors[0]
    assert 1.0 <= clock[0] <= 1.025 and 1 < len(errors) < 100
    assert not output.exists() and (partial / "_SUCCESS").is_file()
    np.testing.assert_array_equal(
        open_result_store(partial).read_final().charge_number, _final_particles().charge_number
    )


@pytest.mark.parametrize(("platform", "code"), [("linux", 5), ("win32", None), ("win32", 112)])
def test_other_publication_errors_fail_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, code: int | None
) -> None:
    output = tmp_path / "result"
    error = _denial(code)
    attempts = []

    def denied(source: Path, destination: Path) -> None:
        attempts.append(source)
        raise error

    clock = _fake_clock(monkeypatch)
    monkeypatch.setattr(output_module.sys, "platform", platform)
    monkeypatch.setattr(output_module.os, "rename", denied)
    with pytest.raises(PermissionError) as caught:
        _publication(output, resume=False)
    assert caught.value is error and len(attempts) == 1 and clock[0] == 0


@pytest.mark.parametrize("interference", ["target_appears", "source_moves"])
def test_publication_never_overwrites_competitor_or_retries_invalid_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interference: str
) -> None:
    output = tmp_path / "result"
    moved = tmp_path / "moved-complete.partial"
    real_rename = os.rename
    attempts = []

    def changed(source: Path, destination: Path) -> None:
        attempts.append(source)
        assert source.parent.resolve() == moved.parent.resolve() == tmp_path.resolve()
        if interference == "target_appears":
            destination.mkdir()
            (destination / "competitor").write_bytes(b"retain")
        else:
            real_rename(source, moved)
        raise _denial(5)

    _fake_clock(monkeypatch)
    monkeypatch.setattr(output_module.sys, "platform", "win32")
    monkeypatch.setattr(output_module.os, "rename", changed)
    with pytest.raises(ResultWriteError, match=r"appeared|disappeared"):
        _publication(output, resume=False)
    assert len(attempts) == 1
    if interference == "target_appears":
        assert (output / "competitor").read_bytes() == b"retain"
        retained = output.with_name("result.partial")
    else:
        assert not output.exists()
        retained = moved
    assert (retained / "_SUCCESS").is_file()


def _windows_read_lock(path: Path) -> Callable[[], None]:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return lambda: kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="requires real Windows sharing semantics")
@pytest.mark.parametrize("release", [True, False])
def test_real_windows_read_handle_release_and_held_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release: bool
) -> None:
    output = tmp_path / "result"
    real_rename = os.rename
    attempts = []
    close: Callable[[], None] | None = None
    thread: threading.Thread | None = None

    def locked(source: Path, destination: Path) -> None:
        nonlocal close, thread
        attempts.append(source)
        if close is None:
            close = _windows_read_lock(source / "final.h5")
            if release:
                unlock = close

                def later() -> None:
                    time.sleep(0.075)
                    unlock()

                thread = threading.Thread(target=later)
                thread.start()
        real_rename(source, destination)

    monkeypatch.setattr(output_module.os, "rename", locked)
    try:
        if release:
            _publication(output, resume=False)
            assert output.is_dir()
        else:
            with pytest.raises(PermissionError):
                _publication(output, resume=False)
            assert not output.exists()
            assert (output.with_name("result.partial") / "_SUCCESS").is_file()
        assert len(attempts) >= 2
    finally:
        if thread is not None:
            thread.join(timeout=2)
        elif close is not None:
            close()
    monkeypatch.setattr(output_module.os, "rename", real_rename)
    if not release:
        _publication(output, resume=True)
    np.testing.assert_array_equal(
        open_result_store(output).read_final().particle_id, _final_particles().particle_id
    )
