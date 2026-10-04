from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable preflight finding."""

    code: str
    message: str
    severity: str = "error"
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Stable public representation of preflight results."""

    detail: str
    issues: tuple[ValidationIssue, ...]
    checks: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "validation_report",
            "schema_version": 2,
            "detail": self.detail,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
            "checks": dict(self.checks),
        }


__all__ = ("ValidationIssue", "ValidationReport")
