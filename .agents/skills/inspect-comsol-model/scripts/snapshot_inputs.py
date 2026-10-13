"""Hash explicitly listed files without changing inputs or earlier receipts.

This is a disk identity receipt, not a COMSOL inventory or loaded-state check.
Paths are literal and relative to --root; no directories are scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def input_paths(root: Path, names: list[str]) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise ValueError("--root must be an existing directory")
    paths: dict[str, Path] = {}
    for name in names:
        supplied = Path(name)
        if supplied.is_absolute() or supplied.anchor:
            raise ValueError(f"input paths must be relative to --root: {name}")
        portable = Path(os.path.normpath(name)).as_posix()
        if portable in paths:
            raise ValueError(f"duplicate input path: {portable}")
        paths[portable] = (root / supplied).resolve()
    return sorted(paths.items())


def protect_inputs(output: Path, paths: list[tuple[str, Path]], previous: Path | None) -> None:
    protected = {path for _, path in paths}
    if previous is not None:
        protected.add(previous.resolve())
    if output.resolve() in protected:
        raise ValueError("--output must differ from every input and previous receipt")
    if output.exists():
        raise ValueError("--output must be a new file")


def file_identity(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "resolved_path": str(path),
        "status": "present",
        "size_bytes": None,
        "sha256": None,
    }
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            return {**result, "status": "not_file"}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except FileNotFoundError:
        return {**result, "status": "missing"}
    except OSError as error:
        return {**result, "status": "unreadable", "reason": str(error)}
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        return {**result, "status": "changed_during_read"}
    return {**result, "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def read_previous(path: Path) -> tuple[dict[str, tuple[object, ...]], str]:
    payload = path.read_bytes()
    document = json.loads(payload)
    rows = document["files"]
    if not isinstance(rows, list):
        raise ValueError("previous receipt files must be a list")
    values: dict[str, tuple[object, ...]] = {}
    for row in rows:
        key = row["path"]
        if not isinstance(key, str) or key in values:
            raise ValueError("previous receipt needs unique string paths")
        values[key] = (row["status"], row["size_bytes"], row["sha256"])
    return values, hashlib.sha256(payload).hexdigest()


def compare_files(
    current: list[dict[str, object]], previous: dict[str, tuple[object, ...]]
) -> list[str]:
    present = {
        str(row["path"]): (row["status"], row["size_bytes"], row["sha256"]) for row in current
    }
    return sorted(
        key for key in present.keys() | previous.keys() if present.get(key) != previous.get(key)
    )


def create_receipt(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    root = args.root.resolve()
    paths = input_paths(root, args.paths)
    protect_inputs(args.output, paths, args.compare)
    previous = None if args.compare is None else read_previous(args.compare)
    files = [{"path": key, **file_identity(path)} for key, path in paths]
    incomplete = [str(row["path"]) for row in files if row["status"] != "present"]
    changed = [] if previous is None else compare_files(files, previous[0])
    comparison = "initial" if previous is None else "unchanged"
    if changed:
        comparison = "changed"
    receipt = {
        "purpose": "explicit-file-content-receipt",
        "root": str(root),
        "files": files,
        "comparison": comparison,
        "changed_paths": changed,
        "incomplete_paths": incomplete,
        "previous_receipt_sha256": None if previous is None else previous[1],
    }
    return receipt, 2 if incomplete else int(bool(changed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("paths", nargs="+", help="literal relative paths; explicit ../ allowed")
    args = parser.parse_args()
    try:
        receipt, exit_code = create_receipt(args)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"snapshot error: {error}", file=sys.stderr)
        return 2
    print(
        f"{receipt['comparison']}: {len(receipt['files'])} files; "
        f"{len(receipt['changed_paths'])} changed; "
        f"{len(receipt['incomplete_paths'])} incomplete; receipt {args.output}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
