"""Write the minimal v0.1 result summary as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import summarize_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event-batch-rows", type=int, default=4096)
    arguments = parser.parse_args()

    summary = summarize_result(arguments.result, event_batch_rows=arguments.event_batch_rows)
    encoded = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
        return
    source = arguments.result.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output == source or source in output.parents:
        parser.error("analysis output must be outside the source result directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
