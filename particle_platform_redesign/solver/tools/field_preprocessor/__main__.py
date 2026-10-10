"""Create an explicitly selected canonical field cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import preprocess_from_configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--source-time-diagnostic",
        action="store_true",
        help="report saved-knot omission sensitivity without changing cache acceptance",
    )
    arguments = parser.parse_args()
    result = preprocess_from_configuration(
        arguments.configuration,
        arguments.output,
        report_path=arguments.report,
        source_time_diagnostic_requested=arguments.source_time_diagnostic,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
