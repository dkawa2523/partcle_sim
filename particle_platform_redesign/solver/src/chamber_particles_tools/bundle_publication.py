"""Publish a canonical producer bundle and its receipt as one completed input."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def bundle_publication(destination: Path) -> Iterator[Path]:
    """Stage ``bundle.h5`` and ``report.json`` before exposing a fresh directory."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".bundle-publication-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "input"
        staging.mkdir()
        yield staging
        if not (staging / "bundle.h5").is_file() or not (staging / "report.json").is_file():
            raise ValueError("producer publication requires bundle.h5 and report.json")
        staging.rename(destination)
