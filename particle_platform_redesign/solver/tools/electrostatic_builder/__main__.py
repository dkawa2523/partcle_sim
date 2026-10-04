"""Generate reduced electrostatic fields in a canonical case bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import build_from_configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    result = build_from_configuration(
        arguments.configuration,
        arguments.output,
        report_path=arguments.report,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
