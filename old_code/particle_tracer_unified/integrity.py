"""Small, streaming integrity helpers shared by input and output edges."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the whole artifact into memory."""

    source = Path(path)
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ("sha256_file",)
