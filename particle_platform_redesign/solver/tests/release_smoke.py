"""Runtime-only wheel smoke using the three package-root operations."""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import version
from pathlib import Path

from chamber_particles import load_case, open_result, simulate
from tests.verification.microcases import materialize_microcase


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="chamber-particles-release-smoke-") as temporary:
        root = Path(temporary)
        paths = materialize_microcase("C01", root / "case")
        output = root / "result"
        case = load_case(paths.case_path)
        summary = simulate(case, output)
        result = open_result(output)
        final = result.read_final()
        frames = tuple(result.iter_frames())
        if summary.particle_count != 1 or final.particle_id.tolist() != [101]:
            raise RuntimeError("clean-install smoke produced an unexpected final particle table")
        if len(frames) != 5 or result.manifest.get("status") != "complete":
            raise RuntimeError("clean-install smoke produced an incomplete public result")
        print(
            json.dumps(
                {
                    "package": "chamber-particles",
                    "version": version("chamber-particles"),
                    "public_operations": ["load_case", "simulate", "open_result"],
                    "status": "complete",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
