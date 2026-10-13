"""Execute one registered independent case through the three public operations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import chamber_particles
from chamber_particles import load_case, open_result, simulate

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parent.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_name")
    name = parser.parse_args().case_name
    plan_path = HERE / "scientific_run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    (case_plan,) = [item for item in plan["cases"] if item["case"] == name]
    engine_path = SOLVER / "src/chamber_particles/engine.py"
    engine_sha = digest(engine_path)
    receipts = []
    for item in case_plan["runs"]:
        case_path, output = Path(item["case"]), Path(item["output"])
        if output.exists():
            raise FileExistsError(output)
        print(
            json.dumps({"started": name, "level": item["level"], "dt_s": item["step_s"]}),
            flush=True,
        )
        case = load_case(case_path)
        start = time.perf_counter()
        simulate(case, output, workers=1)
        wall_s = time.perf_counter() - start
        result = open_result(output)
        if not result.complete or digest(engine_path) != engine_sha:
            raise RuntimeError("Incomplete run or changed executor source")
        final = result.read_final()
        if final.particle_id.tolist() != case_plan["source_ids"]:
            raise ValueError("Run identity differs from the registered realized cohort")
        receipt = {
            **item,
            "wall_s": wall_s,
            "completed_utc": dt.datetime.now(dt.UTC).isoformat(),
            "case_sha256": digest(case_path),
            "data_sha256": digest(case.data_path),
            "data_content_hash": case.content_hash,
            "run_sha256": digest(output / "run.json"),
            "status": result.manifest["status"],
            "counts": result.manifest["counts"],
            "maximum_dt_charge_lipschitz": result.manifest["maximum_dt_charge_lipschitz"],
            "maximum_dt_over_tau": result.manifest["maximum_dt_over_tau"],
            "engine_algorithm_revision": result.manifest["engine_algorithm_revision"],
        }
        receipts.append(receipt)
        with (output.parent / f"{item['level']}_execution_receipt.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(receipt, stream, indent=2)
        print(json.dumps(receipt), flush=True)
    overall = {
        "case": name,
        "plan_sha256": digest(plan_path),
        "engine_sha256": engine_sha,
        "imported_package": str(chamber_particles.__file__),
        "python_version": platform.python_version(),
        "uv_lock_sha256": digest(SOLVER / "uv.lock"),
        "runs": receipts,
        "scope": "Fresh publicAPI deterministic finite-speed runs; original COMSOL remains a separately identified reference with different physics.",
    }
    with (HERE / name / "execution_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(overall, stream, indent=2)


if __name__ == "__main__":
    main()
