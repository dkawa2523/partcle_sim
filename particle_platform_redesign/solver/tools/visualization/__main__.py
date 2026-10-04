"""Render minimal trajectory and boundary-event SVG files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import render_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--particle-id", type=int)
    parser.add_argument("--event-batch-rows", type=int, default=4096)
    parser.add_argument("--max-event-points", type=int, default=5000)
    arguments = parser.parse_args()
    report = render_result(
        arguments.result,
        arguments.output_directory,
        particle_id=arguments.particle_id,
        event_batch_rows=arguments.event_batch_rows,
        max_event_points=arguments.max_event_points,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
