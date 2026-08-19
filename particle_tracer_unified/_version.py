"""Single source for the source-tree package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "particle-tracer-unified"
PACKAGE_VERSION = "0.2.0"


def distribution_version(distribution: str, *, fallback: str = "unavailable") -> str:
    """Return installed distribution metadata without importing the package."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback


def installed_package_version() -> str:
    """Return installed metadata when available, otherwise the source version."""

    return distribution_version(PACKAGE_NAME, fallback=PACKAGE_VERSION)


__all__ = (
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "distribution_version",
    "installed_package_version",
)
