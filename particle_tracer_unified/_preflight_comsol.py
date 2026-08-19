from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .io.comsol_boundary_reader import (
    read_comsol_boundaries,
    validate_geometry_boundary_coverage,
)
from .io.comsol_manifest import ComsolCaseManifest
from .preflight_types import ValidationIssue


def _config_value(obj: Any, *names: str) -> Any:
    value = obj
    for name in names:
        if isinstance(value, Mapping):
            value = value.get(name)
        else:
            value = getattr(value, name, None)
        if value is None:
            return None
    return value


def _manifest_path(case: Any) -> Path | None:
    configured = _config_value(case.config, "inputs", "comsol_manifest")
    if configured is None or not str(configured).strip():
        return None
    path = Path(str(configured))
    base = Path(case.config_path)
    base = base if base.is_dir() else base.parent
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _is_comsol_case(case: Any) -> bool:
    adapter = _config_value(case.config, "case", "adapter")
    if adapter is not None:
        return str(adapter).strip().lower() == "comsol"
    return _manifest_path(case) is not None


def _field_is_transient(runtime: Any) -> bool:
    """Return whether the field actually varies in time.

    A steady field has one time sample and the same value at every instant, so
    it supports the whole integration window by construction; its declared
    support collapses to a single point only because that is where the sample
    sits.  The runtime already reads it this way, so preflight must too --
    otherwise ``check`` rejects cases that ``run`` integrates without a
    complaint, and a contradicted preflight stops being read.
    """

    field = getattr(getattr(runtime, "field_provider", None), "field", None)
    if field is None:
        return False
    return str(getattr(field, "time_mode", "steady")) == "transient"


def check_time_support(
    case: Any,
    runtime: Any,
    manifest: ComsolCaseManifest,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    support = manifest.time_support_s
    t_end_raw = _config_value(case.config, "time", "t_end")
    if support is None or t_end_raw is None:
        return
    t_end = float(t_end_raw)
    release_times = np.asarray(runtime.particles.release_time, dtype=np.float64)
    active_release = release_times[release_times < t_end]
    required_start = float(np.min(active_release)) if active_release.size else t_end
    transient = _field_is_transient(runtime)
    time_check = {
        "declared_support_s": [float(support[0]), float(support[1])],
        "required_support_s": [required_start, t_end],
        "field_time_mode": "transient" if transient else "steady",
        "passed": bool(
            not transient or (support[0] <= required_start and support[1] >= t_end)
        ),
    }
    checks["comsol_time_support"] = time_check
    if not time_check["passed"]:
        issues.append(
            ValidationIssue(
                "comsol.time_support",
                "COMSOL field time support does not cover every "
                "released-particle integration interval",
                context=time_check,
            )
        )


def check_boundary_coverage(
    runtime: Any,
    manifest: ComsolCaseManifest,
    manifest_errors: list[str],
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    path = manifest.boundaries_path()
    if manifest_errors or path is None:
        return
    boundary_rows, wall_rows = read_comsol_boundaries(path)
    coverage = validate_geometry_boundary_coverage(
        runtime.geometry_provider,
        boundary_rows,
        wall_rows,
        strict=False,
    )
    checks["comsol_boundary_coverage"] = coverage
    if not bool(coverage["passed"]):
        issues.append(
            ValidationIssue(
                "comsol.boundary.coverage",
                "Geometry parts, COMSOL boundary rows, and wall laws must match "
                "exactly",
                context=coverage,
            )
        )


def check_case(
    case: Any,
    runtime: Any,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    if not _is_comsol_case(case):
        return
    path = _manifest_path(case)
    if path is None:
        issues.append(
            ValidationIssue(
                "comsol.manifest.missing",
                "COMSOL case requires inputs.comsol_manifest",
            )
        )
        return
    try:
        manifest = ComsolCaseManifest.load(path)
        manifest_errors = manifest.validate(strict=True, verify_hashes=True)
        checks["comsol_manifest"] = {
            "schema_version": int(manifest.schema_version),
            "path": str(path),
            "artifact_count": len(manifest.artifacts),
            "field_quantity_count": len(manifest.fields),
            "errors": list(manifest_errors),
        }
        issues.extend(
            ValidationIssue("comsol.manifest", message, context={"path": str(path)})
            for message in manifest_errors
        )
        check_time_support(case, runtime, manifest, checks, issues)
        check_boundary_coverage(runtime, manifest, manifest_errors, checks, issues)
    except (OSError, TypeError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                "comsol.manifest",
                str(exc),
                context={"path": str(path)},
            )
        )
