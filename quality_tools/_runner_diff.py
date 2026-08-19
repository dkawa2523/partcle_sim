from __future__ import annotations

import difflib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from ._runner_tools import (
    PRISTINE_DIR,
    PRODUCTION_ROOTS,
    REPORT_DIR,
    ROOT,
    GateFailure,
    _python_files,
    _relative,
    _run,
    _tool,
)


def _git_diff() -> str | None:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        return None
    base = os.environ.get("QUALITY_DIFF_BASE", "").strip()
    if not base or set(base) == {"0"}:
        probe = _run(
            [git, "rev-parse", "--verify", "HEAD^"],
            check=False,
            capture=True,
        )
        if probe.returncode:
            return ""
        base = "HEAD^"
    result = _run(
        [
            git,
            "diff",
            "--unified=0",
            base,
            "HEAD",
            "--",
            "particle_tracer_unified",
            "tools",
            "validation",
        ],
        capture=True,
    )
    return result.stdout


def _snapshot_diff() -> str | None:
    if not PRISTINE_DIR.is_dir():
        return None
    chunks: list[str] = []
    relative_paths = {_relative(path) for path in _python_files(PRODUCTION_ROOTS)} | {
        path.relative_to(PRISTINE_DIR).as_posix()
        for root_name in ("particle_tracer_unified", "tools", "validation")
        for path in (PRISTINE_DIR / root_name).rglob("*.py")
    }
    for relative in sorted(relative_paths):
        before_path = PRISTINE_DIR / relative
        after_path = ROOT / relative
        before = (
            before_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if before_path.is_file()
            else []
        )
        after = (
            after_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if after_path.is_file()
            else []
        )
        file_diff = list(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                n=0,
            )
        )
        if file_diff:
            chunks.append(f"diff --git a/{relative} b/{relative}\n")
            chunks.extend(file_diff)
    return "".join(chunks)


def _run_diff_coverage() -> None:
    diff = _git_diff()
    if diff is None:
        diff = _snapshot_diff()
    if diff is None:
        if os.environ.get("CI"):
            raise GateFailure("CI requires Git history for changed-line coverage")
        print("Changed-line coverage: no Git history or pristine snapshot; skipped")
        return
    if not diff.strip():
        print("Changed-line coverage: no changed production Python lines")
        return
    diff_path = ROOT / ".quality-cache" / "quality.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(diff, encoding="utf-8")
    environment: dict[str, str] | None = None
    if not (ROOT / ".git").exists():
        git = shutil.which("git")
        if git is None:
            raise GateFailure("diff-cover snapshot mode requires Git on PATH")
        metadata = ROOT / ".quality-cache" / "diff-cover.git"
        if not metadata.is_dir():
            _run([git, "init", "--bare", str(metadata)])
        environment = dict(os.environ)
        environment["GIT_DIR"] = str(metadata)
        environment["GIT_WORK_TREE"] = str(ROOT)
    _run(
        [
            _tool("diff-cover"),
            str(REPORT_DIR / "coverage.xml"),
            "--diff-file",
            str(diff_path),
            "--fail-under",
            "90",
            "--format",
            f"json:{REPORT_DIR / 'diff-cover.json'}",
        ],
        environment=environment,
    )


def _changed_path_candidates(explicit: Sequence[str]) -> list[Path]:
    if explicit:
        return [(ROOT / item).resolve() for item in explicit]
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        raise GateFailure(
            "quality-fast requires explicit changed paths when .git is unavailable"
        )
    result = _run(
        [git, "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        capture=True,
    )
    return [(ROOT / line.strip()).resolve() for line in result.stdout.splitlines()]


def _changed_paths(explicit: Sequence[str]) -> list[Path]:
    paths = _changed_path_candidates(explicit)
    python_paths = [path for path in paths if path.is_file() and path.suffix == ".py"]
    outside = [path for path in python_paths if not path.is_relative_to(ROOT)]
    if outside:
        raise GateFailure(f"changed path is outside repository: {outside[0]}")
    return python_paths
