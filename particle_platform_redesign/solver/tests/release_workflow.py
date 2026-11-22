"""Exercise an installed distribution from an empty cwd without checkout fixtures.

Copy this file outside the checkout and run with the clean runtime interpreter.
Only the standard library and the installed distribution are imported here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

import chamber_particles
import chamber_particles_tools
from chamber_particles import check_case, open_result, simulate
from chamber_particles.case_format import read
from chamber_particles_tools.analysis import export_result, summarize_result
from chamber_particles_tools.csv_import import adapt_from_configuration
from chamber_particles_tools.field_preprocessor.workflow import preprocess_from_configuration
from chamber_particles_tools.inputs import create_case, create_from_configuration, init_example
from chamber_particles_tools.source_builder import read_csv_source
from chamber_particles_tools.visualization import render_result


def _csv(path: Path, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def _own_fields(root: Path) -> Path:
    nodes = [[1, 0, 0], [2, 0.01, 0], [3, 0.01, 0.01], [4, 0, 0.01]]
    _csv(root / "vertices.csv", ["vertex_id", "x_m", "y_m"], nodes)
    _csv(
        root / "triangles.csv",
        ["triangle_id", "node1_id", "node2_id", "node3_id", "domain_id"],
        [[1, 1, 2, 3, 1], [2, 1, 3, 4, 1]],
    )
    _csv(
        root / "quads.csv",
        ["quadrilateral_id", "node1_id", "node2_id", "node3_id", "node4_id", "domain_id"],
        [],
    )
    _csv(
        root / "edges.csv",
        ["boundary_edge_id", "node1_id", "node2_id", "boundary_id"],
        [[1, 1, 2, 1], [2, 2, 3, 2], [3, 3, 4, 3], [4, 4, 1, 4]],
    )
    _csv(
        root / "columns.csv",
        ["column", "expression", "unit"],
        [
            ["x_m", "x", "m"],
            ["y_m", "y", "m"],
            ["inside", "selected", "1"],
            ["vertex_id", "node ID", "1"],
            ["Ex", "1+100*x", "V/m"],
            ["Ey", "0", "V/m"],
        ],
    )
    with (root / "fields.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(
            [[x, y, 1, identifier, 1 + 100 * x, 0] for identifier, x, y in nodes]
        )
    config = {
        "format_version": 4,
        "provider": "csv",
        "source": {
            "vertices_path": "vertices.csv",
            "triangles_path": "triangles.csv",
            "quadrilaterals_path": "quads.csv",
            "boundary_edges_path": "edges.csv",
            "field_values_path": "fields.csv",
            "field_columns_path": "columns.csv",
        },
        "domain": {
            "coordinate_system": "cartesian_xy",
            "external_id": 1,
            "layout_name": "own_fields",
            "quadrilateral_order": "q1_reference",
            "excluded_boundary_ids": [],
        },
        "boundary_groups": [
            {"name": name, "external_ids": [index], "material_id": 0}
            for index, name in enumerate(("bottom", "right", "top", "left"), 1)
        ],
        "fields": {
            "coordinate_columns": ["x_m", "y_m"],
            "inside_column": "inside",
            "node_id_column": "vertex_id",
            "domain_scope": None,
            "coordinate_match_tolerance_m": 1e-14,
            "mappings": [
                {
                    "name": "electric_field",
                    "columns": ["Ex", "Ey"],
                    "components": ["x", "y"],
                    "stored_basis": "cartesian_xy",
                    "association": "node",
                    "unit": "V/m",
                    "input_unit": "V/m",
                    "scale": 1.0,
                    "offset": 0.0,
                    "axis_transform": None,
                    "producer_record": {
                        "producer": "installed analytic CSV example",
                        "time": {"representation": "static", "selection": "analytic"},
                        "domain_ids": [1],
                        "side": "domain",
                        "recovery": "none",
                        "continuity": "continuous",
                        "authority": "analytic primitive",
                        "topology": "fixed",
                        "support": "full_selected_domain",
                        "evidence": "E=(1+100*x,0) SI on a 10 mm square",
                    },
                }
            ],
        },
    }
    path = root / "import.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _first_source_case(root: Path) -> tuple[Path, dict[str, Any]]:
    seed = init_example("xy-ballistic", root / "editable")
    configuration = _own_fields(root)
    background = root / "background.h5"
    adapt_from_configuration(configuration, background)
    assert not read(background).sources
    recipe = yaml.safe_load((seed.case_path.parent / "recipe.yaml").read_text(encoding="utf-8"))
    with (seed.case_path.parent / "particles.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["charge_number"] = "1"
    rows[0]["release_time_s"] = "0.002"
    rows[0]["model_weight"] = "2.5"
    source_csv = root / "particles.csv"
    _csv(source_csv, list(rows[0]), [list(rows[0].values())])
    recipe["input_bundle"] = str(background)
    recipe["sources"][0]["csv"] = str(source_csv)
    recipe["settings"]["case"]["name"] = "own CSV delayed charged source"
    recipe["settings"]["physics"]["electric"] = {
        "model": "coulomb",
        "revision": "electric_coulomb_v1",
        "electric_field": "electric_field",
    }
    path = root / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    case = create_from_configuration(path, root / "own-case")
    check_case(case)
    simulate(case, root / "own-result")
    event = open_result(root / "own-result").read_boundary_events()
    rate = math.sqrt(100 * 1.602176634e-19 / 1e-15)
    low, high = 0.0, 0.02
    for _ in range(64):
        middle = (low + high) / 2
        x = (
            (0.0025 + 0.01) * math.cosh(rate * middle)
            + 0.5 / rate * math.sinh(rate * middle)
            - 0.01
        )
        if x < 0.01:
            low = middle
        else:
            high = middle
    assert abs(float(event.time_s[0]) - (0.002 + (low + high) / 2)) < 3e-11
    assert event.outcome.tolist() == ["stuck"]
    return path, recipe


def _cache_and_surface(root: Path, recipe: dict[str, Any]) -> None:
    config = {
        "format_version": 2,
        "input": {"data_path": "background.h5"},
        "validation": {"memory_limit_mb": 128, "workspace_rows": 32, "max_patch_work": 1000000},
        "target": {
            "kind": "regular",
            "name": "regular_cache",
            "axis0": {"start_m": 0, "stop_m": 0.01, "count": 3},
            "axis1": {"start_m": 0, "stop_m": 0.01, "count": 3},
        },
        "fields": [
            {
                "source": "electric_field",
                "output": "cached_electric_field",
                "limits": {
                    "value_relative_l2": 1e-8,
                    "gradient_relative_l2": 1e-8,
                    "boundary_value_relative_l2": 1e-8,
                },
            }
        ],
    }
    path = root / "cache.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    preprocess_from_configuration(path, root / "cache.h5")
    cached = dict(recipe)
    cached["input_bundle"] = str(root / "cache.h5")
    cached["settings"] = {
        **recipe["settings"],
        "physics": {
            **recipe["settings"]["physics"],
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "cached_electric_field",
            },
        },
    }
    path = root / "cached-recipe.yaml"
    path.write_text(yaml.safe_dump(cached, sort_keys=False), encoding="utf-8")
    case = create_from_configuration(path, root / "cached-case")
    simulate(case, root / "cached-result")
    left = open_result(root / "own-result").read_boundary_events()
    right = open_result(root / "cached-result").read_boundary_events()
    assert abs(float(left.time_s[0] - right.time_s[0])) < 3e-11
    bundle = replace(read(root / "background.h5"), sources=())
    config = dict(recipe["sources"][0]["configuration"])
    config["columns"] = dict(config["columns"])
    config["units"] = dict(config["units"])
    config["kind"] = "surface"
    config["columns"].pop("position_m")
    config["units"].pop("position_m")
    config["columns"].update(facet_id="facet", facet_parameter="parameter")
    with (root / "particles.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    row.update(facet="3", parameter="0.5")
    path = root / "surface.csv"
    _csv(path, list(row), [list(row.values())])
    source = read_csv_source(path, bundle, config)
    settings = {
        **recipe["settings"],
        "sources": [{"name": "release", "type": "surface", "table": "particles"}],
    }
    case = create_case(bundle, settings, (source,), root / "surface-case")
    simulate(case, root / "surface-result")
    assert open_result(root / "surface-result").read_failure_events().particle_id.size == 0


def _clients_without_case(root: Path) -> dict[str, object]:
    shutil.move(root / "own-case", root / "moved-input")
    result = root / "own-result"
    summary = summarize_result(result)
    assert cast(dict[str, Any], summary["population"])["model_weight"] == 2.5
    assert cast(list[dict[str, Any]], summary["deposition"])[0]["particles"] == 1
    assert cast(dict[str, Any], summary["active_curve"])["rows"][0]["active_particles"] == 0
    export_result(result, root / "exports")
    report = render_result(result, root / "view", particle_id=1, detail=True)
    for path in (root / "view").glob("*.svg"):
        drawing = ET.parse(path)
        assert all(
            re.search(r"(?<![a-z])(?:nan|[+-]?inf(?:inity)?)(?![a-z])", value.lower()) is None
            for element in drawing.iter()
            for value in element.attrib.values()
        )
    return {"summary": summary["population"], "view": report}


def _module_and_console(root: Path, *, allow_console_block: bool) -> dict[str, object]:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    commands = [
        (["models"], "complete"),
        (["init", str(root / "cli-example")], "created"),
        (
            ["inputs", "import-csv", str(root / "import.yaml"), str(root / "cli-background.h5")],
            "created",
        ),
        (["inputs", "cache", str(root / "cache.yaml"), str(root / "cli-cache.h5")], "created"),
        (["create", str(root / "recipe.yaml"), "-o", str(root / "cli-case")], "created"),
        (["check", str(root / "cli-case/case.yaml")], "ready"),
        (["run", str(root / "cli-case/case.yaml"), "-o", str(root / "cli-result")], "complete"),
        (["inspect", str(root / "cli-result")], "complete"),
        (["analyze", str(root / "cli-result")], None),
        (["export", str(root / "cli-result"), "-o", str(root / "cli-export")], None),
        (["view", str(root / "cli-result"), "-o", str(root / "cli-view")], None),
        (
            ["inputs", "electrostatic", "check", str(root / "reduced-electrostatic/builder.yaml")],
            "ready",
        ),
        (
            [
                "inputs",
                "electrostatic",
                "build",
                str(root / "reduced-electrostatic/builder.yaml"),
                str(root / "cli-electric.h5"),
            ],
            "created",
        ),
    ]
    executed = []
    for arguments, expected_status in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "chamber_particles", *arguments, "--json"],
            cwd=root,
            env=environment,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        assert isinstance(payload, dict) and payload
        if expected_status is not None:
            assert payload["status"] == expected_status, payload
            assert payload["operation"] == " ".join(
                arguments[:3]
                if arguments[0:2] == ["inputs", "electrostatic"]
                else arguments[:2]
                if arguments[0] == "inputs"
                else arguments[:1]
            )
        executed.append(
            " ".join(
                arguments[:1]
                if arguments[0] != "inputs"
                else arguments[:3]
                if arguments[1] == "electrostatic"
                else arguments[:2]
            )
        )
    missing = subprocess.run(
        [sys.executable, "-m", "chamber_particles", "check", str(root / "missing.yaml"), "--json"],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2 and not missing.stdout
    assert json.loads(missing.stderr)["status"] == "error"
    console = Path(sys.prefix) / (
        "Scripts/chamber-particles.exe" if os.name == "nt" else "bin/chamber-particles"
    )
    try:
        completed = subprocess.run(
            [str(console), "models", "--json"],
            cwd=root,
            env=environment,
            capture_output=True,
            check=True,
        )
    except OSError as error:
        if not allow_console_block or getattr(error, "winerror", None) != 4551:
            raise
        return {
            "module": "passed",
            "commands": executed,
            "console": "host_policy_blocked",
            "winerror": 4551,
        }
    assert json.loads(completed.stdout)["models"]
    return {"module": "passed", "commands": executed, "console": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-console-policy-block", action="store_true")
    arguments = parser.parse_args()
    paths = [
        Path(str(package.__file__)).resolve()
        for package in (chamber_particles, chamber_particles_tools)
    ]
    assert all(path.is_relative_to(Path(sys.prefix).resolve()) for path in paths), paths
    with tempfile.TemporaryDirectory(prefix="installed-particle-workflow-") as temporary:
        root = Path(temporary)
        _, recipe = _first_source_case(root)
        _cache_and_surface(root, recipe)
        for name in ("rz-charged", "xy-time-field", "brownian-wall", "reduced-electrostatic"):
            case = init_example(name, root / name)
            simulate(case, root / f"{name}-result")
            assert open_result(root / f"{name}-result").read_failure_events().particle_id.size == 0
        evidence = _clients_without_case(root)
        entrypoints = _module_and_console(
            root, allow_console_block=arguments.allow_console_policy_block
        )
    payload = {
        "status": "complete",
        "package_paths": list(map(str, paths)),
        "cwd_uses_checkout": False,
        "fixture_imports": [],
        "entrypoints": entrypoints,
        "own_csv_first_source": "analytic field trajectory and hit verified",
        "cache_and_surface": "passed",
        "examples": "five passed",
        **evidence,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "entrypoints": entrypoints}))


if __name__ == "__main__":
    main()
