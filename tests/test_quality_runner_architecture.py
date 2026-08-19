from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from quality_tools import _runner_baseline, _runner_diff, _runner_tools, runner


def _completed(
    arguments: Sequence[str] = (),
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        arguments, returncode, stdout=stdout, stderr=stderr
    )


def _static_payload() -> dict[str, Any]:
    return {
        "ruff": {"count": 0, "diagnostics": []},
        "format": {"count": 0, "files": []},
        "radon": {"over_10": 0, "maximum": 1, "functions": []},
        "vulture": {"count": 0, "findings": []},
        "suppressions": [],
    }


def test_runner_facade_directly_reexports_owner_seams() -> None:
    assert runner.main.__module__ == runner.__name__
    for owner in (_runner_tools, _runner_baseline, _runner_diff):
        assert runner not in vars(owner).values()
        assert runner.__name__ not in {
            getattr(value, "__module__", None) for value in vars(owner).values()
        }


def test_static_baseline_keeps_only_complexity_exceptions() -> None:
    current = _static_payload()
    current["radon"] = {
        "over_10": 1,
        "maximum": 11,
        "functions": [
            {"path": "simple.py", "qualified_name": "simple", "complexity": 10},
            {"path": "legacy.py", "qualified_name": "legacy", "complexity": 11},
        ],
    }

    baseline = _runner_baseline._baseline_static(current)

    assert baseline["radon"]["functions"] == [
        {"path": "legacy.py", "qualified_name": "legacy", "complexity": 11}
    ]
    assert current["radon"]["functions"][0]["qualified_name"] == "simple"


@pytest.mark.parametrize(
    ("section", "current_value", "message"),
    [
        (
            "ruff",
            {
                "count": 1,
                "diagnostics": [
                    {
                        "code": "F821",
                        "path": "module.py",
                        "scope": "run",
                        "message": "Undefined name",
                        "count": 1,
                    }
                ],
            },
            "new Ruff diagnostics",
        ),
        (
            "radon",
            {
                "over_10": 1,
                "maximum": 11,
                "functions": [
                    {
                        "path": "module.py",
                        "qualified_name": "run",
                        "complexity": 11,
                    }
                ],
            },
            "complexity regressions",
        ),
        (
            "vulture",
            {"count": 1, "findings": ["module.py: unused function 'old'"]},
            "new Vulture candidates",
        ),
    ],
)
def test_static_baseline_rejects_new_quality_debt(
    section: str,
    current_value: object,
    message: str,
) -> None:
    baseline = _static_payload()
    current = _static_payload()
    current[section] = current_value

    with pytest.raises(_runner_tools.GateFailure, match=message):
        _runner_baseline._compare_static(current, baseline)


def test_test_and_coverage_baselines_reject_regressions() -> None:
    with pytest.raises(_runner_tools.GateFailure, match="test node IDs were removed"):
        _runner_baseline._compare_tests(
            {"tests": {"count": 1, "node_ids": ["test_a.py::test_a"]}},
            {
                "tests": {
                    "count": 2,
                    "node_ids": ["test_a.py::test_a", "test_b.py::test_b"],
                }
            },
        )

    baseline = {"coverage": {"covered": 90, "total": 100, "percent": 90.0}}
    current = {"covered": 89, "total": 100, "percent": 89.0}
    with pytest.raises(_runner_tools.GateFailure, match="coverage decreased"):
        _runner_baseline._compare_coverage(current, baseline)


def test_security_failure_is_propagated_before_secret_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _runner_tools.GateFailure("dependency audit failed")

    def fail_security_tools() -> None:
        raise failure

    def unexpected_secret_scan() -> dict[str, Any]:
        raise AssertionError("secret scan must not run after a security tool failure")

    monkeypatch.setattr(
        _runner_tools, "_run_code_and_dependency_security", fail_security_tools
    )
    monkeypatch.setattr(_runner_tools, "_secret_scan", unexpected_secret_scan)

    with pytest.raises(_runner_tools.GateFailure) as error:
        _runner_tools._run_security()

    assert error.value is failure


def test_changed_line_coverage_falls_back_to_pristine_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    reports = root / ".quality" / "reports"
    root.mkdir()
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def run_stub(
        arguments: Sequence[str],
        *,
        environment: dict[str, str] | None = None,
        **_: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((list(arguments), environment))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(_runner_diff, "ROOT", root)
    monkeypatch.setattr(_runner_diff, "REPORT_DIR", reports)
    monkeypatch.setattr(_runner_diff, "_git_diff", lambda: None)
    monkeypatch.setattr(
        _runner_diff,
        "_snapshot_diff",
        lambda: "diff --git a/module.py b/module.py\n@@ -1 +1 @@\n-old\n+new\n",
    )
    monkeypatch.setattr(_runner_diff, "_tool", lambda name: name)
    monkeypatch.setattr(_runner_diff.shutil, "which", lambda name: name)
    monkeypatch.setattr(_runner_diff, "_run", run_stub)

    _runner_diff._run_diff_coverage()

    assert (
        (root / ".quality-cache" / "quality.diff")
        .read_text(encoding="utf-8")
        .startswith("diff --git")
    )
    assert calls[0][0][:3] == ["git", "init", "--bare"]
    assert "--fail-under" in calls[1][0]
    assert calls[1][0][calls[1][0].index("--fail-under") + 1] == "90"
    assert calls[1][1] is not None


def test_ci_rejects_missing_git_and_snapshot_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_runner_diff, "_git_diff", lambda: None)
    monkeypatch.setattr(_runner_diff, "_snapshot_diff", lambda: None)
    monkeypatch.setenv("CI", "1")

    with pytest.raises(
        _runner_tools.GateFailure,
        match="CI requires Git history for changed-line coverage",
    ):
        _runner_diff._run_diff_coverage()


def test_tool_source_and_scope_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(_runner_tools, "ROOT", tmp_path)
    monkeypatch.setattr(
        _runner_tools.shutil,
        "which",
        lambda name: "on-path" if name == "direct" else None,
    )
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    fallback = scripts / "fallback.exe"
    fallback.write_bytes(b"")
    assert _runner_tools._tool("direct") == "on-path"
    assert _runner_tools._tool("fallback") == str(fallback)
    with pytest.raises(_runner_tools.GateFailure, match="required quality tool"):
        _runner_tools._tool("missing")

    source = tmp_path / "module.py"
    source.write_text(
        "class Owner:\n    def run(self):\n        return 1  " + "# no" + "qa\n\n"
        "async def task():\n    return 2\n",
        encoding="utf-8",
    )
    ignored = tmp_path / ".venv" / "ignored.py"
    ignored.write_text("ignored = True\n", encoding="utf-8")
    assert _runner_tools._python_files((tmp_path, source)) == [source.resolve()]
    assert len(_runner_tools._hash_file(source)) == 64
    assert _runner_tools._scope_for_line(source, 3) == "Owner.run"
    assert _runner_tools._scope_for_line(source, 6) == "task"
    assert _runner_tools._scope_for_line(source, 4) == "<module>"
    monkeypatch.setattr(_runner_tools, "_python_files", lambda: [source])
    suppression_token = _runner_tools._collect_suppressions()[0]["token"]
    assert suppression_token == "# no" + "qa"
    source.write_text("def broken(:\n", encoding="utf-8")
    assert _runner_tools._scope_for_line(source, 1) == "<module>"
    assert _runner_tools._failure_output(_completed()) == ""
    assert _runner_tools._failure_output(_completed(stdout="detail")) == "\ndetail"


def test_collector_failure_paths_and_nested_radon_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(_runner_tools, "ROOT", tmp_path)
    monkeypatch.setattr(_runner_tools, "_tool", lambda name: name)
    results = iter(
        [
            _completed(returncode=1, stderr="no path"),
            _completed(returncode=2, stderr="crashed"),
        ]
    )
    monkeypatch.setattr(_runner_tools, "_run", lambda *args, **kwargs: next(results))
    with pytest.raises(_runner_tools.GateFailure, match="without reporting"):
        _runner_tools._collect_format()
    with pytest.raises(_runner_tools.GateFailure, match="could not run"):
        _runner_tools._collect_format()

    entries = _runner_tools._radon_entries(
        [
            {
                "type": "class",
                "name": "Owner",
                "methods": [
                    {
                        "type": "method",
                        "classname": "Owner",
                        "name": "run",
                        "complexity": 4,
                        "closures": [
                            {"type": "function", "name": "inner", "complexity": 2}
                        ],
                    }
                ],
            }
        ]
    )
    assert [item["qualified_name"] for item in entries] == [
        "Owner.run",
        "Owner.run.inner",
    ]

    duplicate = {
        "module.py": [
            {"type": "function", "name": "run", "complexity": 2},
            {"type": "function", "name": "run", "complexity": 3},
        ]
    }
    monkeypatch.setattr(
        _runner_tools,
        "_run",
        lambda *args, **kwargs: _completed(stdout=json.dumps(duplicate)),
    )
    monkeypatch.setattr(_runner_tools, "_relative", lambda path: str(path))
    with pytest.raises(_runner_tools.GateFailure, match="inconsistent results"):
        _runner_tools._collect_radon()

    results = iter(
        [
            _completed(
                returncode=_runner_tools.VULTURE_FINDINGS_EXIT_CODE,
                stdout="C:\\repo\\a.py:3: unused function 'old'\n\n",
            ),
            _completed(returncode=2, stderr="crashed"),
            _completed(stdout="no tests\n"),
        ]
    )
    monkeypatch.setattr(_runner_tools, "_run", lambda *args, **kwargs: next(results))
    assert _runner_tools._collect_vulture()["findings"] == [
        "repo/a.py: unused function 'old'"
    ]
    with pytest.raises(_runner_tools.GateFailure, match="Vulture could not run"):
        _runner_tools._collect_vulture()
    with pytest.raises(_runner_tools.GateFailure, match="no test node IDs"):
        _runner_tools._collect_tests()

    source = tmp_path / "module.py"
    source.write_text("def run():\n    return missing\n", encoding="utf-8")
    ruff_payload = [
        {
            "filename": str(source),
            "location": {"row": 2},
            "code": "F821",
            "message": "Undefined name `missing`",
        }
    ]
    monkeypatch.setattr(
        _runner_tools,
        "_run",
        lambda *args, **kwargs: _completed(stdout=json.dumps(ruff_payload)),
    )
    monkeypatch.setattr(_runner_tools, "ROOT", tmp_path)
    assert _runner_tools._collect_ruff()["diagnostics"][0]["scope"] == "run"


def test_coverage_security_and_static_report_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    monkeypatch.setattr(_runner_tools, "REPORT_DIR", reports)
    monkeypatch.setattr(_runner_tools, "PIP_AUDIT_CACHE_DIR", tmp_path / "audit")
    monkeypatch.setattr(_runner_tools, "_tool", lambda name: name)
    calls: list[list[str]] = []

    def run_stub(
        arguments: Sequence[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(arguments))
        if "json" in arguments and "coverage" in arguments:
            Path(arguments[arguments.index("-o") + 1]).write_text(
                json.dumps(
                    {
                        "totals": {
                            "covered_lines": 80,
                            "num_statements": 100,
                            "covered_branches": 15,
                            "num_branches": 20,
                        }
                    }
                ),
                encoding="utf-8",
            )
        return _completed(arguments, stdout='{"results": {}}')

    monkeypatch.setattr(_runner_tools, "_run", run_stub)
    coverage = _runner_tools._collect_coverage()
    assert (coverage["covered"], coverage["total"]) == (95, 120)
    assert _runner_tools._secret_scan() == {"results": {}}
    _runner_tools._run_code_and_dependency_security()
    assert any(call[0] == "bandit" for call in calls)
    assert any(call[0] == "pip-audit" for call in calls)

    hash_key = "hashed_" + "secret"
    scan_payload = {"results": {"a": [{"type": "Token", hash_key: "hash"}]}}
    assert _runner_tools._secret_fingerprints(scan_payload) == {("a", "Token", "hash")}
    secret_path = tmp_path / ".secrets.baseline"
    monkeypatch.setattr(_runner_tools, "SECRET_BASELINE_PATH", secret_path)
    monkeypatch.setattr(
        _runner_tools, "_run_code_and_dependency_security", lambda: None
    )
    monkeypatch.setattr(_runner_tools, "_secret_scan", lambda: scan_payload)
    with pytest.raises(_runner_tools.GateFailure, match=r"missing \.secrets\.baseline"):
        _runner_tools._run_security()
    secret_path.write_text('{"results": {}}', encoding="utf-8")
    with pytest.raises(_runner_tools.GateFailure, match="new secret candidates"):
        _runner_tools._run_security()
    secret_path.write_text(json.dumps(scan_payload), encoding="utf-8")
    _runner_tools._run_security()

    monkeypatch.setattr(_runner_tools, "_collect_ruff", lambda: {})
    monkeypatch.setattr(_runner_tools, "_collect_format", lambda: {})
    monkeypatch.setattr(_runner_tools, "_collect_radon", lambda: {"value": 1})
    monkeypatch.setattr(_runner_tools, "_collect_vulture", lambda: {"value": 2})
    monkeypatch.setattr(_runner_tools, "_collect_suppressions", lambda: [])
    assert _runner_tools._collect_static()["radon"] == {"value": 1}
    _runner_tools._run_pyrefly()
    _runner_tools._run_pyrefly(update=True)
    _runner_tools._run_architecture()


def test_remaining_baseline_rejections_and_secret_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = _static_payload()
    current = _static_payload()
    baseline["format"] = {
        "count": 1,
        "files": [{"path": "a.py", "sha256": "before"}],
    }
    current["format"] = {
        "count": 1,
        "files": [{"path": "a.py", "sha256": "after"}],
    }
    with pytest.raises(_runner_tools.GateFailure, match="need Ruff formatting"):
        _runner_baseline._compare_static(current, baseline)
    current["format"] = baseline["format"]
    current["suppressions"] = [{"path": "a.py", "token": "# no" + "qa"}]
    with pytest.raises(_runner_tools.GateFailure, match="new suppressions"):
        _runner_baseline._compare_static(current, baseline)
    with pytest.raises(_runner_tools.GateFailure, match="test count decreased"):
        _runner_baseline._compare_tests(
            {"tests": {"count": 1, "node_ids": ["a::test"]}},
            {"tests": {"count": 2, "node_ids": ["a::test"]}},
        )
    with pytest.raises(_runner_tools.GateFailure, match="initial audit floor"):
        _runner_baseline._compare_coverage(
            {"covered": 1, "total": 100, "percent": 1.0},
            {"coverage": {"covered": 1, "total": 100, "percent": 1.0}},
        )

    secret_path = tmp_path / ".secrets.baseline"
    monkeypatch.setattr(_runner_baseline, "SECRET_BASELINE_PATH", secret_path)
    with pytest.raises(_runner_tools.GateFailure, match="unexpected secret"):
        _runner_baseline._write_secret_baseline(
            {"results": {"other": [{"type": "Token"}]}}
        )
    payload = {"results": {"data/assets.yaml": [{"type": "Hash"}]}}
    _runner_baseline._write_secret_baseline(payload)
    assert (
        json.loads(secret_path.read_text(encoding="utf-8"))["results"][
            "data/assets.yaml"
        ][0]["is_secret"]
        is False
    )


def test_git_snapshot_and_changed_path_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    pristine = tmp_path / "pristine"
    package = root / "particle_tracer_unified"
    old_package = pristine / "particle_tracer_unified"
    package.mkdir(parents=True)
    old_package.mkdir(parents=True)
    (package / "a.py").write_text("new = 1\n", encoding="utf-8")
    (old_package / "a.py").write_text("old = 1\n", encoding="utf-8")
    monkeypatch.setattr(_runner_diff, "ROOT", root)
    monkeypatch.setattr(_runner_diff, "PRISTINE_DIR", pristine)
    monkeypatch.setattr(_runner_diff, "PRODUCTION_ROOTS", (package,))
    monkeypatch.setattr(
        _runner_diff,
        "_relative",
        lambda path: Path(path).relative_to(root).as_posix(),
    )
    assert "a.py" in (_runner_diff._snapshot_diff() or "")
    monkeypatch.setattr(_runner_diff, "PRISTINE_DIR", tmp_path / "missing")
    assert _runner_diff._snapshot_diff() is None

    monkeypatch.setattr(_runner_diff.shutil, "which", lambda name: None)
    assert _runner_diff._git_diff() is None
    with pytest.raises(_runner_tools.GateFailure, match="requires explicit"):
        _runner_diff._changed_path_candidates(())
    text = root / "a.txt"
    text.write_text("text", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    assert _runner_diff._changed_paths(("particle_tracer_unified/a.py", "a.txt")) == [
        package / "a.py"
    ]
    with pytest.raises(_runner_tools.GateFailure, match="outside repository"):
        _runner_diff._changed_paths((str(outside),))

    (root / ".git").mkdir()
    git_calls: list[list[str]] = []

    def git_run(arguments: Sequence[str], **_: Any) -> subprocess.CompletedProcess[str]:
        git_calls.append(list(arguments))
        return _completed(arguments, stdout="production diff")

    monkeypatch.setattr(_runner_diff.shutil, "which", lambda name: name)
    monkeypatch.setattr(_runner_diff, "_run", git_run)
    monkeypatch.delenv("QUALITY_DIFF_BASE", raising=False)
    assert _runner_diff._git_diff() == "production diff"
    assert git_calls[-1][1:4] == ["diff", "--unified=0", "HEAD^"]
    assert _runner_diff._changed_path_candidates(()) == [root / "production diff"]

    monkeypatch.setattr(_runner_diff, "_git_diff", lambda: None)
    monkeypatch.setattr(_runner_diff, "_snapshot_diff", lambda: None)
    monkeypatch.delenv("CI", raising=False)
    _runner_diff._run_diff_coverage()
    monkeypatch.setattr(_runner_diff, "_git_diff", lambda: "\n")
    _runner_diff._run_diff_coverage()
