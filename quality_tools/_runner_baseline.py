from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ._runner_tools import (
    BASELINE_PATH,
    INITIAL_COVERED,
    INITIAL_TOTAL,
    SECRET_BASELINE_PATH,
    GateFailure,
)


def _load_baseline() -> dict[str, Any] | None:
    if not BASELINE_PATH.is_file():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _baseline_static(current: dict[str, Any]) -> dict[str, Any]:
    """Keep only complexity exceptions; ordinary functions share the CC 10 gate."""

    baseline = dict(current)
    radon = dict(current["radon"])
    radon["functions"] = [
        item for item in radon.get("functions", ()) if int(item["complexity"]) > 10
    ]
    baseline["radon"] = radon
    return baseline


def _records_by_key(
    records: list[dict[str, Any]], keys: Sequence[str]
) -> Counter[tuple[str, ...]]:
    return Counter(
        {
            tuple(str(record[key]) for key in keys): int(record.get("count", 1))
            for record in records
        }
    )


def _reject(regression: object, message: str) -> None:
    if regression:
        raise GateFailure(message)


def _compare_static(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    current_ruff = _records_by_key(
        current["ruff"]["diagnostics"], ("code", "path", "scope", "message")
    )
    baseline_ruff = _records_by_key(
        baseline["ruff"]["diagnostics"], ("code", "path", "scope", "message")
    )
    ruff_regressions = current_ruff - baseline_ruff
    _reject(
        ruff_regressions,
        f"new Ruff diagnostics: {list(ruff_regressions.items())[:20]}",
    )

    baseline_format = {
        item["path"]: item["sha256"] for item in baseline["format"]["files"]
    }
    changed_unformatted = [
        item["path"]
        for item in current["format"]["files"]
        if baseline_format.get(item["path"]) != item["sha256"]
    ]
    _reject(
        changed_unformatted,
        (
            "new or modified files need Ruff formatting: "
            + ", ".join(changed_unformatted[:30])
        ),
    )

    baseline_complexity: dict[tuple[str, str], int] = {}
    for item in baseline["radon"]["functions"]:
        key = (item["path"], item["qualified_name"])
        baseline_complexity[key] = max(
            baseline_complexity.get(key, 0), int(item["complexity"])
        )
    complexity_regressions = []
    for item in current["radon"]["functions"]:
        key = (item["path"], item["qualified_name"])
        previous = baseline_complexity.get(key, 10)
        if int(item["complexity"]) > max(10, previous):
            complexity_regressions.append((*key, previous, item["complexity"]))
    _reject(
        complexity_regressions,
        f"complexity regressions (old/new): {complexity_regressions[:20]}",
    )

    new_vulture = set(current["vulture"]["findings"]) - set(
        baseline["vulture"]["findings"]
    )
    _reject(new_vulture, f"new Vulture candidates: {sorted(new_vulture)}")

    current_suppressions = Counter(
        (item["path"], item["token"]) for item in current["suppressions"]
    )
    baseline_suppressions = Counter(
        (item["path"], item["token"]) for item in baseline["suppressions"]
    )
    new_suppressions = current_suppressions - baseline_suppressions
    _reject(
        new_suppressions,
        f"new suppressions or test skips: {list(new_suppressions.items())[:20]}",
    )


def _compare_tests(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    missing = set(baseline["tests"]["node_ids"]) - set(current["tests"]["node_ids"])
    if missing:
        raise GateFailure(f"test node IDs were removed: {sorted(missing)[:20]}")
    if current["tests"]["count"] < int(baseline["tests"]["count"]):
        raise GateFailure("pytest test count decreased")


def _compare_coverage(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    if current["covered"] * INITIAL_TOTAL < INITIAL_COVERED * current["total"]:
        raise GateFailure(
            "branch coverage fell below the immutable initial audit floor"
        )
    expected = baseline["coverage"]
    if (
        current["covered"] * int(expected["total"])
        < int(expected["covered"]) * current["total"]
    ):
        raise GateFailure(
            f"branch coverage decreased: {current['percent']:.4f}% < "
            f"{expected['percent']:.4f}%"
        )


def _write_secret_baseline(payload: dict[str, Any]) -> None:
    allowed_path = "data/assets.yaml"
    unexpected = [
        (path, finding.get("type"))
        for path, findings in payload.get("results", {}).items()
        for finding in findings
        if path.replace("\\", "/") != allowed_path
    ]
    if unexpected:
        raise GateFailure(
            f"refusing to baseline unexpected secret candidates: {unexpected}"
        )
    for findings in payload.get("results", {}).values():
        for finding in findings:
            finding["is_secret"] = False
    SECRET_BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
