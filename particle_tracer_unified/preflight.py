from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from . import _preflight_boundary as boundary_checks
from . import _preflight_comsol as comsol_checks
from . import _preflight_initial_state as initial_state_checks
from . import _preflight_runtime as runtime_checks
from .drag_validation import initial_drag_regime_report
from .preflight_physics import physics_requirement_issues
from .preflight_types import ValidationIssue, ValidationReport


class SimulationCaseLike(Protocol):
    @property
    def config(self) -> Any: ...

    @property
    def config_path(self) -> Path: ...

    @property
    def _context(self) -> Any: ...


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    keep = {
        "passed",
        "applicable",
        "reason",
        "mode",
        "support_scope",
        "sample_time_scope",
        "particle_count",
        "field_backend_kind",
        "status_counts",
        "field_support_passed",
        "geometry_passed",
        "geometry_status_counts",
        "boundary_release_failed_offset_count",
        "violation_count",
        "scope",
        "model",
        "integrated_particle_count",
        "assessed_particle_count",
        "unassessed_clean_particle_count",
        "relative_mach_assessed_count",
        "dynamic_history_assessed",
        "error_count",
        "warning_count",
        "metrics",
    }
    return {key: value for key, value in report.items() if key in keep}


def _record_report(
    checks: dict[str, Any],
    name: str,
    report: Mapping[str, Any],
    include_details: bool,
) -> None:
    checks[name] = report if include_details else _summary(report)


def _record_boundary_coverage(
    runtime: Any,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    coverage = boundary_checks.runtime_boundary_coverage(runtime)
    checks["boundary_coverage"] = coverage
    if not coverage["passed"]:
        issues.append(
            ValidationIssue(
                "boundary.coverage",
                "Every geometry boundary part must have exactly one explicit "
                "wall model",
                context=coverage,
            )
        )


def _record_experimental_features(
    case: SimulationCaseLike,
    runtime: Any,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    features, findings = runtime_checks.experimental_feature_report(case, runtime)
    checks["experimental_features"] = features
    issues.extend(findings)


def _record_drag_regime(
    runtime: Any,
    include_details: bool,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    try:
        report, findings = initial_drag_regime_report(
            runtime,
            include_violations=include_details,
        )
        _record_report(checks, "drag_regime", report, include_details)
        issues.extend(
            ValidationIssue(
                finding.code,
                finding.message,
                severity=finding.severity,
                context=finding.context,
            )
            for finding in findings
        )
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(ValidationIssue("physics.drag.regime", str(exc)))


def _record_initial_particle_support(
    runtime: Any,
    include_details: bool,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    try:
        report = initial_state_checks.initial_particle_support_report(
            runtime,
            include_violations=include_details,
        )
        _record_report(checks, "initial_particles", report, include_details)
        if not bool(report.get("field_support_passed", False)):
            issues.append(
                ValidationIssue(
                    "input.initial_field_support",
                    "Initial particles do not satisfy the field-support/input contract",
                    context={"status_counts": dict(report.get("status_counts", {}))},
                )
            )
        if not bool(report.get("geometry_passed", False)):
            issues.append(
                ValidationIssue(
                    "input.initial_geometry",
                    "Initial particle positions must be strictly inside the "
                    "authoritative geometry, or on the boundary entity their "
                    "own source_part_id declares",
                    context={
                        "status_counts": dict(report.get("geometry_status_counts", {}))
                    },
                )
            )
    except (TypeError, ValueError) as exc:
        issues.append(ValidationIssue("input.preflight", str(exc)))


def _record_provider_boundary_support(
    runtime: Any,
    include_details: bool,
    checks: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    try:
        report = boundary_checks.boundary_field_support_report(
            runtime,
            include_violations=include_details,
        )
        _record_report(checks, "provider_boundary_support", report, include_details)
        if not bool(report.get("passed", False)):
            issues.append(
                ValidationIssue(
                    "provider.boundary_field_support",
                    "Field provider does not cover the explicit geometry boundary",
                    context={"status_counts": dict(report.get("status_counts", {}))},
                )
            )
    except (TypeError, ValueError) as exc:
        issues.append(ValidationIssue("provider.preflight", str(exc)))


def validate_case_preflight(
    case: SimulationCaseLike,
    *,
    detail: str = "summary",
) -> ValidationReport:
    """Run the single, side-effect-free case preflight."""

    normalized_detail = str(detail).strip().lower()
    if normalized_detail not in {"summary", "full"}:
        raise ValueError("detail must be 'summary' or 'full'")
    include_details = normalized_detail == "full"
    runtime = case._context
    issues = runtime_checks.particle_issues(runtime)
    checks: dict[str, Any] = {}
    _record_boundary_coverage(runtime, checks, issues)
    _record_experimental_features(case, runtime, checks, issues)
    issues.extend(physics_requirement_issues(case, runtime))
    _record_drag_regime(runtime, include_details, checks, issues)
    _record_initial_particle_support(runtime, include_details, checks, issues)
    _record_provider_boundary_support(runtime, include_details, checks, issues)
    comsol_checks.check_case(case, runtime, checks, issues)
    return ValidationReport(
        detail=normalized_detail,
        issues=tuple(issues),
        checks=checks,
    )


__all__ = (
    "SimulationCaseLike",
    "ValidationIssue",
    "ValidationReport",
    "validate_case_preflight",
)
