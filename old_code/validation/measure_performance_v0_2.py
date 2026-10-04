"""Reproduce the warm v0.2 simulation timing and Python allocation samples."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numba
import numpy
import pandas
import yaml

from particle_tracer_unified import load_case, simulate, validate_case

DEFAULT_CASES = (
    Path("examples/v02_minimal/run_config.yaml"),
    Path("examples/v02_minimal_3d/run_config.yaml"),
)


def measure_case(
    path: Path, *, warmups: int, repeats: int, memory_repeats: int
) -> dict[str, object]:
    case = load_case(path)
    report = validate_case(case)
    if not report.passed:
        raise RuntimeError(f"preflight failed for {path}: {report.errors}")

    for _ in range(warmups):
        simulate(case)

    elapsed_s: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = simulate(case)
        elapsed_s.append(time.perf_counter() - started)

    current_bytes: list[int] = []
    peak_bytes: list[int] = []
    for _ in range(memory_repeats):
        gc.collect()
        tracemalloc.start()
        result = simulate(case)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        current_bytes.append(current)
        peak_bytes.append(peak)

    return {
        "config": path.as_posix(),
        "particle_count": result.stats.particle_count,
        "steps": round(case.plan.t_end_s / case.plan.dt_s),
        "wall_time_s": {
            "samples": elapsed_s,
            "min": min(elapsed_s),
            "median": statistics.median(elapsed_s),
            "max": max(elapsed_s),
        },
        "tracemalloc_bytes": {
            "current_samples": current_bytes,
            "peak_samples": peak_bytes,
            "current_median": int(statistics.median(current_bytes)),
            "peak_median": int(statistics.median(peak_bytes)),
            "peak_max": max(peak_bytes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", type=Path, default=list(DEFAULT_CASES))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmups, args.repeats, args.memory_repeats) < 1:
        parser.error("warmups, repeats, and memory-repeats must all be >= 1")

    payload = {
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pyyaml": yaml.__version__,
            "numba": numba.__version__,
        },
        "protocol": {
            "timer": "time.perf_counter",
            "warmups_per_case": args.warmups,
            "timing_repeats_per_case": args.repeats,
            "tracemalloc_repeats_per_case": args.memory_repeats,
            "scope": "simulate(case) only; load_case and validate_case excluded",
        },
        "cases": [
            measure_case(
                path,
                warmups=args.warmups,
                repeats=args.repeats,
                memory_repeats=args.memory_repeats,
            )
            for path in args.cases
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
