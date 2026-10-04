"""Reject source functions and methods with cyclomatic complexity of 16 or more."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit

_MAX_ALLOWED_COMPLEXITY = 15


def _function_blocks(blocks: Iterable[Any]) -> Iterator[Any]:
    """Yield Radon function and method blocks, including nested definitions."""
    for block in blocks:
        if block.letter in {"F", "M"}:
            yield block
        for attribute in ("methods", "closures", "inner_classes"):
            yield from _function_blocks(getattr(block, attribute, ()))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    source_roots = (project_root / "src" / "chamber_particles", project_root / "tools")
    failures: list[str] = []

    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            blocks = cc_visit(path.read_text(encoding="utf-8"))
            for block in _function_blocks(blocks):
                if block.complexity > _MAX_ALLOWED_COMPLEXITY:
                    relative_path = path.relative_to(project_root)
                    failures.append(
                        f"{relative_path}:{block.lineno} {block.fullname} CC={block.complexity}"
                    )

    if failures:
        print("Cyclomatic complexity must be below 16:")
        print("\n".join(failures))
        return 1

    print("Complexity gate passed: every function and method has CC < 16.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
