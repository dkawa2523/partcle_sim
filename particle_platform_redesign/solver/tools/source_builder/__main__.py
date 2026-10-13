"""Replace one realized internal source in a new copy of a canonical case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .producer import prepare_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="existing canonical case YAML")
    parser.add_argument("rows", type=Path, help="CSV with explicitly mapped initial rows")
    parser.add_argument("configuration", type=Path, help="source conversion JSON, format_version 1")
    parser.add_argument("output", type=Path, help="directory for fresh case.h5 and case.yaml")
    arguments = parser.parse_args()
    receipt = prepare_case(
        arguments.template, arguments.rows, arguments.configuration, arguments.output
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
