from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUALITY_DIR = ROOT / ".quality"
REPORT_DIR = QUALITY_DIR / "reports"
BASELINE_PATH = QUALITY_DIR / "baseline.json"
PYREFLY_BASELINE_PATH = QUALITY_DIR / "pyrefly-baseline.json"
SECRET_BASELINE_PATH = ROOT / ".secrets.baseline"
QUALITY_CACHE_DIR = ROOT / ".quality-cache"
PRISTINE_DIR = QUALITY_CACHE_DIR / "pristine"
PIP_AUDIT_CACHE_DIR = QUALITY_CACHE_DIR / "pip-audit"
INITIAL_COVERED = 12_399 + 3_420
INITIAL_TOTAL = 17_341 + 5_948
VULTURE_FINDINGS_EXIT_CODE = 3
PRODUCTION_ROOTS = (
    ROOT / "particle_tracer_unified",
    ROOT / "tools",
    ROOT / "validation",
)
EXCLUDED_PARTS = {
    ".git",
    ".grimp_cache",
    ".hypothesis",
    ".import_linter_cache",
    ".mypy_cache",
    ".nox",
    ".pyrefly_cache",
    ".pytest_cache",
    ".quality-cache",
    ".ruff_cache",
    ".uv-cache",
    ".uv-tools",
    ".venv",
    "mutants",
    "__pycache__",
}
SUPPRESSION_PATTERN = re.compile(
    r"#\s*(?:noqa\b|type:\s*ignore\b|nosec\b|pragma:\s*no cover\b)"
    r"|pytest\.mark\.(?:skip|skipif|xfail)\b"
)


class GateFailure(RuntimeError):
    """Raised when a deterministic quality gate fails."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        executable_name = f"{name}.exe" if os.name == "nt" else name
        candidate = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
        candidate /= executable_name
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise GateFailure(
            f"required quality tool {name!r} is unavailable; run "
            "`uv sync --frozen --group quality`"
        )
    return executable


def _python_module(name: str, *arguments: str) -> list[str]:
    """Run Python tools without platform-specific console launchers."""

    return [sys.executable, "-m", name, *arguments]


def _run(
    arguments: Sequence[str],
    *,
    check: bool = True,
    capture: bool = False,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a tool, keeping captured machine output quiet unless it fails."""
    command = [str(item) for item in arguments]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and completed.returncode:
        details = _failure_output(completed) if capture else ""
        raise GateFailure(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}{details}"
        )
    return completed


def _failure_output(
    completed: subprocess.CompletedProcess[str],
    *,
    line_limit: int = 20,
    character_limit: int = 4_000,
) -> str:
    """Return a bounded, actionable tail for a failed captured command."""
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if not output:
        return ""
    lines = output.splitlines()
    truncated = len(lines) > line_limit or len(output) > character_limit
    tail = "\n".join(lines[-line_limit:])[-character_limit:]
    prefix = "\n... captured output truncated ..." if truncated else ""
    return f"{prefix}\n{tail}"


def _python_files(roots: Iterable[Path] = (ROOT,)) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*.py")
        for path in candidates:
            if path.suffix == ".py" and not any(
                part in EXCLUDED_PARTS for part in path.parts
            ):
                files.append(path.resolve())
    return sorted(set(files))


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scope_for_line(path: Path, line: int) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return "<module>"
    scopes: list[tuple[int, int, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: list[str] = []

        def _visit_scope(self, node: ast.AST, name: str) -> None:
            self.names.append(name)
            end = int(getattr(node, "end_lineno", getattr(node, "lineno", line)))
            scopes.append(
                (
                    int(getattr(node, "lineno", 1)),
                    end,
                    ".".join(self.names),
                )
            )
            self.generic_visit(node)
            self.names.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._visit_scope(node, node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_scope(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_scope(node, node.name)

    Visitor().visit(tree)
    containing = [item for item in scopes if item[0] <= line <= item[1]]
    return max(containing, key=lambda item: item[0])[2] if containing else "<module>"


def _collect_ruff() -> dict[str, Any]:
    result = _run(
        [
            _tool("ruff"),
            "check",
            ".",
            "--output-format",
            "json",
            "--exit-zero",
        ],
        capture=True,
    )
    payload = json.loads(result.stdout or "[]")
    diagnostics: Counter[tuple[str, str, str, str]] = Counter()
    for item in payload:
        filename = Path(str(item["filename"])).resolve()
        path = _relative(filename)
        line = int(item["location"]["row"])
        key = (
            str(item["code"]),
            path,
            _scope_for_line(filename, line),
            str(item["message"]),
        )
        diagnostics[key] += 1
    records = [
        {
            "code": key[0],
            "path": key[1],
            "scope": key[2],
            "message": key[3],
            "count": count,
        }
        for key, count in sorted(diagnostics.items())
    ]
    return {"count": len(payload), "diagnostics": records}


def _collect_format() -> dict[str, Any]:
    result = _run(
        [_tool("ruff"), "format", "--check", "."],
        check=False,
        capture=True,
    )
    paths: set[str] = set()
    for line in f"{result.stdout}\n{result.stderr}".splitlines():
        summary_match = re.search(
            r"(?:Would reformat|would be reformatted):?\s+(.+\.py)$", line
        )
        diagnostic_match = re.match(r"\s*-->\s+(.+\.py):\d+:\d+\s*$", line)
        match = summary_match or diagnostic_match
        if match:
            candidate = Path(match.group(1).strip())
            resolved = candidate if candidate.is_absolute() else ROOT / candidate
            if resolved.is_file():
                paths.add(_relative(resolved))
    if result.returncode not in (0, 1):
        raise GateFailure("Ruff format check could not run" + _failure_output(result))
    if result.returncode and not paths:
        raise GateFailure(
            "Ruff format check failed without reporting affected files"
            + _failure_output(result)
        )
    return {
        "count": len(paths),
        "files": [
            {"path": path, "sha256": _hash_file(ROOT / path)} for path in sorted(paths)
        ],
    }


def _radon_entries(
    items: list[dict[str, Any]], prefix: str = ""
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if prefix:
            name = f"{prefix}.{item['name']}"
        elif item.get("type") == "method" and item.get("classname"):
            name = f"{item['classname']}.{item['name']}"
        else:
            name = str(item["name"])
        if item.get("type") in {"function", "method"}:
            entries.append(
                {"qualified_name": name, "complexity": int(item["complexity"])}
            )
        methods = item.get("methods")
        if isinstance(methods, list):
            entries.extend(_radon_entries(methods, name))
        closures = item.get("closures")
        if isinstance(closures, list):
            entries.extend(_radon_entries(closures, name))
    return entries


def _collect_radon() -> dict[str, Any]:
    result = _run(
        [
            _tool("radon"),
            "cc",
            "--json",
            "--show-complexity",
            *[str(path) for path in PRODUCTION_ROOTS],
        ],
        capture=True,
    )
    payload = json.loads(result.stdout or "{}")
    by_function: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path, items in payload.items():
        path = _relative(Path(raw_path))
        for entry in _radon_entries(items):
            key = (path, str(entry["qualified_name"]))
            existing = by_function.get(key)
            if existing is not None and existing["complexity"] != entry["complexity"]:
                raise GateFailure(f"Radon returned inconsistent results for {key}")
            by_function[key] = {"path": path, **entry}
    functions = sorted(
        by_function.values(), key=lambda item: (item["path"], item["qualified_name"])
    )
    return {
        "over_10": sum(item["complexity"] > 10 for item in functions),
        "maximum": max((item["complexity"] for item in functions), default=0),
        "functions": functions,
    }


def _collect_vulture() -> dict[str, Any]:
    result = _run(
        [
            _tool("vulture"),
            *[str(path) for path in PRODUCTION_ROOTS],
            "--min-confidence",
            "80",
        ],
        check=False,
        capture=True,
    )
    if result.returncode not in (0, VULTURE_FINDINGS_EXIT_CODE):
        raise GateFailure("Vulture could not run" + _failure_output(result))
    findings: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[A-Za-z]:[/\\]", "", line)
        line = re.sub(r":\d+:", ":", line, count=1)
        findings.append(line.replace("\\", "/"))
    return {"count": len(findings), "findings": sorted(findings)}


def _collect_suppressions() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in _python_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in SUPPRESSION_PATTERN.finditer(line):
                inventory.append(
                    {
                        "path": _relative(path),
                        "line": line_number,
                        "token": match.group(0),
                    }
                )
    return inventory


def _collect_tests() -> dict[str, Any]:
    result = _run(_python_module("pytest", "--collect-only", "-q"), capture=True)
    node_ids = sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith("=")
    )
    if not node_ids:
        raise GateFailure("pytest collection returned no test node IDs")
    return {"count": len(node_ids), "node_ids": node_ids}


def _collect_coverage() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _run(_python_module("coverage", "erase"))
    _run(_python_module("coverage", "run", "-m", "pytest"))
    _run(_python_module("coverage", "combine"))
    json_path = REPORT_DIR / "coverage.json"
    xml_path = REPORT_DIR / "coverage.xml"
    _run(_python_module("coverage", "json", "-o", str(json_path)))
    _run(_python_module("coverage", "xml", "-o", str(xml_path)))
    totals = json.loads(json_path.read_text(encoding="utf-8"))["totals"]
    covered = int(totals["covered_lines"]) + int(totals["covered_branches"])
    total = int(totals["num_statements"]) + int(totals["num_branches"])
    return {
        "covered": covered,
        "total": total,
        "percent": 100.0 * covered / total,
        "covered_lines": int(totals["covered_lines"]),
        "statements": int(totals["num_statements"]),
        "covered_branches": int(totals["covered_branches"]),
        "branches": int(totals["num_branches"]),
    }


def _secret_scan() -> dict[str, Any]:
    exclusion = (
        r"(^|[\\/])(\.git|\.grimp_cache|\.venv|\.uv-cache|\.uv-tools|"
        r"\.quality|\.quality-cache|\.ruff_cache|\.mypy_cache|\.pytest_cache|"
        r"\.import_linter_cache|mutants)([\\/]|$)"
        r"|(^|[\\/])\.secrets\.baseline$"
    )
    result = _run(
        [
            _tool("detect-secrets"),
            "scan",
            "--all-files",
            "--exclude-files",
            exclusion,
        ],
        capture=True,
    )
    return json.loads(result.stdout)


def _secret_fingerprints(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    fingerprints: set[tuple[str, str, str]] = set()
    for path, findings in payload.get("results", {}).items():
        for finding in findings:
            fingerprints.add(
                (
                    str(path).replace("\\", "/"),
                    str(finding.get("type", "")),
                    str(finding.get("hashed_secret", "")),
                )
            )
    return fingerprints


def _run_code_and_dependency_security() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PIP_AUDIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _tool("bandit"),
            "-r",
            *[str(path) for path in PRODUCTION_ROOTS],
            "-f",
            "json",
            "-o",
            str(REPORT_DIR / "bandit.json"),
        ]
    )
    _run(
        [
            _tool("pip-audit"),
            "--local",
            "--skip-editable",
            "--progress-spinner",
            "off",
            "--cache-dir",
            str(PIP_AUDIT_CACHE_DIR),
            "--format",
            "json",
            "--output",
            str(REPORT_DIR / "pip-audit.json"),
        ]
    )


def _run_security() -> None:
    _run_code_and_dependency_security()
    current = _secret_scan()
    if not SECRET_BASELINE_PATH.is_file():
        raise GateFailure("missing .secrets.baseline; run quality-baseline explicitly")
    expected = json.loads(SECRET_BASELINE_PATH.read_text(encoding="utf-8"))
    new_findings = _secret_fingerprints(current) - _secret_fingerprints(expected)
    if new_findings:
        raise GateFailure(f"new secret candidates: {sorted(new_findings)}")


def _collect_static() -> dict[str, Any]:
    result = {
        "ruff": _collect_ruff(),
        "format": _collect_format(),
        "radon": _collect_radon(),
        "vulture": _collect_vulture(),
        "suppressions": _collect_suppressions(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("radon", "vulture"):
        (REPORT_DIR / f"{name}.json").write_text(
            json.dumps(result[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _run_pyrefly(*, update: bool = False) -> None:
    arguments = [_tool("pyrefly"), "check"]
    if update:
        arguments.append("--update-baseline")
    _run(arguments)


def _run_architecture() -> None:
    _run([_tool("lint-imports")])
