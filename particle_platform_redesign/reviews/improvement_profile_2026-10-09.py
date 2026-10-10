"""Profile two existing P14 presets; this is scoped measurement evidence."""

from __future__ import annotations

import cProfile
import json
import pstats
import tempfile
from pathlib import Path

from tests.performance.p14_matrix import (
    _execute_run,
    _materialize_rows,
    _matrix_rows,
    _worker_spec,
)


def main() -> None:
    selected = {"event-h0-n10000", "event-h20-n10000"}
    rows = tuple(row for row in _matrix_rows("release") if row.row_id in selected)
    report = []
    with tempfile.TemporaryDirectory(prefix="chamber-particles-improvement-profile-") as temporary:
        root = Path(temporary)
        materials = _materialize_rows(root / "cases", rows, suite="release", memory_limit_mb=8192)
        for item in materials:
            spec = _worker_spec(item)
            _execute_run(spec, root / item.row.row_id / "warmup", digest=False)
            profiler = cProfile.Profile()
            with profiler:
                observed = _execute_run(spec, root / item.row.row_id / "measured", digest=True)
            stats = pstats.Stats(profiler)
            staging = []
            top = []
            for (file, line, function), (
                primitive,
                total,
                own_s,
                cumulative_s,
                _,
            ) in stats.stats.items():
                row = {
                    "file": file,
                    "line": line,
                    "function": function,
                    "primitive_calls": primitive,
                    "calls": total,
                    "own_s": own_s,
                    "cumulative_s": cumulative_s,
                }
                if function in {
                    "_allocate_boundary_event_buffer",
                    "_allocate_failure_event_buffer",
                }:
                    staging.append(row)
                top.append(row)
            report.append(
                {
                    "row_id": item.row.row_id,
                    "profiled_public_observation": observed,
                    "profiled_scope_includes_post_timing_scientific_digest": True,
                    "staging_allocations": staging,
                    "largest_own_time": sorted(top, key=lambda row: row["own_s"], reverse=True)[
                        :25
                    ],
                }
            )
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(
        json.dumps(
            {
                "purpose": "F0 hotspot selection; not an accuracy or SLA certification",
                "reuses": "tests.performance.p14_matrix public-run presets and measurements",
                "rows": report,
            },
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
