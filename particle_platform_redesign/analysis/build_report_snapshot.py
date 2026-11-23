"""Reproduce the report's reviewed design snapshot without inventing run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chamber_particles.physics.catalog import describe_models
from chamber_particles_tools.examples import EXAMPLE_NAMES

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "report_app"
QUERY_IDS = (
    "case_audit",
    "trajectory_divergence",
    "design_decisions",
    "physics_scope",
    "module_responsibilities",
    "numerical_contracts",
    "result_layers",
    "implementation_work_packages",
    "quality_findings",
    "roadmap",
)
AUTHORITY_PATHS = (
    "AGENTS.md",
    "product_specification.md",
    "architecture_proposal.md",
    "implementation_plan.md",
    "vv_methodology.md",
    "data_quality_assessment.md",
    "solver/pyproject.toml",
    "solver/README.md",
    "solver/docs/case_format_v3.md",
    "solver/docs/result_format_v4.md",
    "solver/docs/physics_models.md",
    "solver/docs/numerics.md",
    "solver/docs/parallel_execution_plan.md",
    "solver/tests/performance/README.md",
    "solver/tools/vv/comsol/README.md",
    "solver/src/chamber_particles/physics/catalog.py",
    "solver/src/chamber_particles/time_grid.py",
    "solver/src/chamber_particles/lifecycle.py",
    "solver/src/chamber_particles/output.py",
    "solver/src/chamber_particles_tools/inputs.py",
    "solver/src/chamber_particles_tools/bundle_publication.py",
    "solver/src/chamber_particles_tools/selection.py",
)
COMPONENT_IDS = {
    "case_audit": ["dataset-role"],
    "trajectory_divergence": ["dataset-role"],
    "design_decisions": [
        "metric-api",
        "metric-extension",
        "metric-tests",
        "executive-decision",
        "decision-table",
        "coordinates",
        "boundary-numerics",
        "single-loop",
        "complexity",
        "dataset-role",
    ],
    "physics_scope": [
        "executive-decision",
        "governing-equations",
        "physics-table",
        "coordinates",
        "boundary-numerics",
        "start-gate",
    ],
    "module_responsibilities": ["module-table"],
    "numerical_contracts": ["numerics-table", "complexity", "start-gate"],
    "result_layers": ["result-store", "result-table"],
    "implementation_work_packages": ["implementation-sequence", "work-package-table"],
    "quality_findings": ["dataset-role", "quality-table"],
    "roadmap": [
        "metric-stages",
        "implementation-sequence",
        "implementation",
        "roadmap-table",
        "start-gate",
    ],
}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    try:
        name = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        name = path.name
    return {"path": name, "sha256": _digest(path)}


def _query(key: str, rows: list[dict[str, Any]], label: str, files: list[str]) -> dict[str, Any]:
    return {
        "label": label,
        "rows": rows,
        "source": {
            "label": "Current reviewed design; execution evidence is separate",
            "files": files,
            "metricDefinitions": [
                {
                    "label": label,
                    "definition": "Reviewed current contracts or explicitly unavailable evidence; not an inferred execution result.",
                    "componentIds": COMPONENT_IDS[key],
                }
            ],
            "evidenceFlow": [
                {
                    "title": "Reproduce design snapshot",
                    "detail": "Run analysis/build_report_snapshot.py from the locked solver environment; authority and catalog hashes are bound in snapshot.reproducibility.",
                }
            ],
        },
        "methods": [
            {
                "language": "shell",
                "code": "uv run --locked python ../analysis/build_report_snapshot.py --output ../report_app/src/data.json --build-status complete",
            }
        ],
    }


def _unavailable(reason: str) -> list[dict[str, Any]]:
    return [{"status": "unavailable", "reason": reason, "value": None}]


def _design_queries(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    files = list(AUTHORITY_PATHS)
    decisions = [
        {
            "area": "core_api",
            "decision": "load_case / check_case / simulate / open_result",
            "reason": "One execution preparation and engine",
            "count": 4,
        },
        {
            "area": "normal_clients",
            "decision": "source_builder / csv_import / field_preprocessor / electrostatic_builder / analysis / visualization",
            "reason": "Same wheel, CLI and Python; no checkout dependency",
            "count": 6,
        },
        {
            "area": "test_layers",
            "decision": "verification / scenarios / performance",
            "reason": "Independent mathematics, public flows, measured resources",
            "count": 3,
        },
        {
            "area": "publication",
            "decision": "inputs.create_case(bundle, settings, output_directory)",
            "reason": "One bundle owns source tables; atomic case, editable inputs and producer receipts",
        },
        {
            "area": "native_boundary",
            "decision": "Live COMSOL and developer V&V remain external",
            "reason": "Core never imports native adapters or comparisons",
        },
        {
            "area": "native_inputs",
            "decision": "Charge: explicit CSV/current equation; ion force: CSV/current revision and matched or analogous",
            "reason": "Reference meaning and allowance are explicit; no package-folder inference or historical equation aliases",
        },
        {
            "area": "result_reuse",
            "decision": "Result schema 4, nested immutable manifest and reusable ResultCohort",
            "reason": "One source identity and exact selected population across analysis/export/view",
        },
        {
            "area": "resolved_ownership",
            "decision": "Catalog inputs/proof roles; parent clocks and worker numeric settings",
            "reason": "Stage computability, path proof, global bounds and accuracy remain distinct",
        },
    ]
    physics = [
        {
            "capability": f"{item['category']} / {item['model']}",
            "current": item["revision"],
            "limits": "; ".join(item["notes"])
            or "Declared coordinate/integrator and parameter requirements apply",
            "coordinates": list(item["coordinate_systems"]),
            "integrators": list(item["integrators"]),
            "required_parameters": list(item["required_parameters"]),
            "field_parameters": list(item["field_parameters"]),
            "required_fields": item["fields"],
            "certificate_roles": item["certificates"],
        }
        for item in catalog
    ]
    modules = [
        {
            "module": "chamber_particles.case / case_format",
            "owns": "Settings, canonical schema and logical hash",
            "must_not": "Provider interpretation or particle integration",
        },
        {
            "module": "chamber_particles.fields / field_numerics",
            "owns": "Support, stage sampling and one numerical field owner",
            "must_not": "Field production or mesh repair",
        },
        {
            "module": "chamber_particles.physics / integrators / stochastic",
            "owns": "Rates, coupled stages and conditional OU path",
            "must_not": "I/O, field location or wall policy",
        },
        {
            "module": "chamber_particles.geometry / events / boundaries",
            "owns": "First hit, event continuation and explicit post-hit law",
            "must_not": "Endpoint-only passage, nudge or invented event",
        },
        {
            "module": "chamber_particles.time_grid / lifecycle / cpu",
            "owns": "Macro partition/window rules, lifecycle meaning and resident/persistent column declarations",
            "must_not": "Accuracy verdicts, adhesion claims or duplicated state schemas",
        },
        {
            "module": "chamber_particles.engine / transport / execution",
            "owns": "One engine; parent clocks/provenance and typed worker settings/exclusive slabs",
            "must_not": "Producer-specific engines or worker HDF5 writers",
        },
        {
            "module": "chamber_particles.output",
            "owns": "Result, immutable context and durable checkpoint",
            "must_not": "Population interpretation or plotting",
        },
        {
            "module": "chamber_particles_tools.inputs / four input clients",
            "owns": "Import, source conversion, field production and one case publisher",
            "must_not": "Duplicated physics validation or trajectory solver",
        },
        {
            "module": "chamber_particles_tools.analysis / visualization",
            "owns": "Reusable ResultView/cohort consumers and complete export/view publication",
            "must_not": "Private case access or implicit new simulation",
        },
        {
            "module": "tools/vv / native COMSOL",
            "owns": "External native producers/receipts, observation, companions and qualified comparison",
            "must_not": "Normal wheel runtime dependency",
        },
    ]
    numerical = [
        {
            "topic": "macro/window",
            "decision": "Shared MacroGrid merges field knots and exposes the actual selected-window partition",
            "acceptance": "Known-window macro refinement and justified numerical-path refinement are separate prerequisites; nominal dt/global counts certify neither",
        },
        {
            "topic": "primitive/proof",
            "decision": "Catalog finite/positive/nonnegative inputs and selected model/proof roles",
            "acceptance": "Separate global/local enclosure from actual-stage sampling and coupled path proof",
        },
        {
            "topic": "check_case",
            "decision": "Same prepare as simulate; no worker/writer/update",
            "acceptance": "Readiness is not convergence",
        },
        {
            "topic": "x/v/Z",
            "decision": "All forces and charge use actual coupled stages",
            "acceptance": "Independent rate/Jacobian and integration limits",
        },
        {
            "topic": "Brownian",
            "decision": "Joint x/v OU with conditional splits",
            "acceptance": "Covariances and numerical-path resolution; no exact first-passage claim",
        },
        {
            "topic": "wall",
            "decision": "Earliest path hit and remaining-time continuation",
            "acceptance": "Hit geometry, incoming/outgoing state and multi-hit microcases",
        },
        {
            "topic": "identity",
            "decision": "Stable particle IDs, physical RNG streams and observation-independent production steps",
            "acceptance": "Worker/tile/output/resume scientific identity",
        },
        {
            "topic": "stability",
            "decision": "Prepared stability upper bounds use requested dt and global bounds",
            "acceptance": "Not an observed stage maximum or an accuracy estimate",
        },
    ]
    result = [
        {
            "layer": "context.h5",
            "contents": "Geometry, semantic source/boundary labels, topology, units and basis",
            "rule": "Immutable once; bound case/data hashes; no background fields or full source",
        },
        {
            "layer": "ResultView",
            "contents": "Final, interactions, saved frames/probes, context and immutable nested manifest",
            "rule": "manifest_document() copies JSON; source_identity() owns derived receipts; strict result schema 4",
        },
        {
            "layer": "analysis",
            "contents": "ResultCohort, fates, first hit, terminal and repeated interactions",
            "rule": "Reuse final/context and exact selection; computational count/weight, no inferred adhesion/flux",
        },
        {
            "layer": "visualization / report",
            "contents": "Geometry, bounded views and reviewed exported evidence",
            "rule": "Complete receipt-bound export/view before atomic publication; no invented missing states",
        },
    ]
    work = [
        {
            "package": "Installed input flow",
            "capability": "Own CSV mesh/field/source to runnable case",
            "deliverable": "Mapping4 + source2 + recipe2 with explicit complete source-table set",
            "gate": "Load/check/run from installed wheel",
        },
        {
            "package": "Reusable results",
            "capability": "Result alone to analysis/export/view",
            "deliverable": "Context and result4",
            "gate": "Move/remove source case and verify consumers",
        },
        {
            "package": "Scientific preservation",
            "capability": "Same engine and equation meaning",
            "deliverable": "Independent mathematical and identity checks",
            "gate": "Root final receipts; no automatic historical PASS inheritance",
        },
        {
            "package": "Distribution",
            "capability": "CLI and Python in one wheel",
            "deliverable": "Five editable installed examples",
            "gate": "Clean installed acceptance with no checkout",
        },
    ]
    roadmap = [
        {
            "phase": "current",
            "name": "0.4 CLI/Python input → result reuse",
            "exit": "Bound final quality and installed-wheel receipts",
            "example_count": len(EXAMPLE_NAMES),
        },
        {
            "phase": "conditional",
            "name": "Native field/particle correspondence",
            "exit": "Inspect equations, prepare primitives, independent comparison; no fitted defaults",
        },
        {
            "phase": "future",
            "name": "Full materials/floating V/Q, 3D, moving geometry, GPU",
            "exit": "Separate model/design and independent scientific/resource evidence",
        },
    ]
    rows = {
        "case_audit": _unavailable(
            "No current assessment input supplied; prior source-bound evidence is not current platform qualification."
        ),
        "trajectory_divergence": _unavailable(
            "No current trajectory comparison input supplied; independent Brownian paths are not paired errors."
        ),
        "design_decisions": decisions,
        "physics_scope": physics,
        "module_responsibilities": modules,
        "numerical_contracts": numerical,
        "result_layers": result,
        "implementation_work_packages": work,
        "quality_findings": [
            {
                "severity": "unavailable",
                "finding": "Current quality/performance receipts not supplied",
                "evidence": None,
                "disposition": "Keep design capability separate from measured execution qualification.",
            }
        ],
        "roadmap": roadmap,
    }
    return {key: _query(key, rows[key], key.replace("_", " "), files) for key in QUERY_IDS}


def _reviewed_input(path: Path, allowed: set[str]) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"queries"}
        or not isinstance(document["queries"], dict)
    ):
        raise ValueError("reviewed evidence input requires exactly a queries mapping")
    if not set(document["queries"]).issubset(allowed):
        raise ValueError("reviewed evidence input contains an unrequested query ID")
    for key, query in document["queries"].items():
        if (
            not isinstance(query, dict)
            or not isinstance(query.get("rows"), list)
            or not isinstance(query.get("source"), dict)
        ):
            raise ValueError(f"{key} requires reviewed rows and exact source metadata")
        if not query["source"].get("files"):
            raise ValueError(f"{key} requires source.files")
    return document["queries"]


def build_snapshot(
    existing: Path,
    *,
    assessment: Path | None = None,
    quality: Path | None = None,
    performance: Path | None = None,
    build_status: str = "updating",
) -> dict[str, Any]:
    """Preserve app identity/presentation while refreshing ten reviewed datasets."""
    if build_status not in {"updating", "paused", "complete"}:
        raise ValueError("build_status must be updating, paused or complete")
    snapshot = json.loads(existing.read_text(encoding="utf-8"))
    if not isinstance(snapshot.get("id"), str) or not snapshot["id"] or not snapshot.get("title"):
        raise ValueError(
            "existing app id and title are required; this exporter never creates an app"
        )
    catalog = [asdict(item) for item in describe_models()]
    catalog_hash = (
        "sha256:" + hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
    )
    queries = _design_queries(catalog)
    inputs = []
    for path, allowed in (
        (assessment, {"case_audit", "trajectory_divergence"}),
        (quality, {"quality_findings"}),
        (performance, {"implementation_work_packages"}),
    ):
        if path is not None:
            queries.update(_reviewed_input(path, allowed))
            inputs.append(_record(path))
    snapshot.update(
        surface="report",
        buildStatus=build_status,
        generatedAt=datetime.now(UTC).isoformat(),
        queries=queries,
    )
    snapshot["report"] = dict(snapshot.get("report", {}))
    snapshot["report"].update(
        asOf=None,
        evidenceScope="Current source-bound design snapshot; optional execution receipts only when explicitly supplied",
    )
    authorities = [_record(ROOT / name) for name in AUTHORITY_PATHS]
    for query in queries.values():
        query["source"].setdefault("evidenceFlow", []).append(
            {
                "title": "Bound authority snapshot and catalog",
                "detail": json.dumps(
                    {
                        "catalog_sha256": catalog_hash,
                        "authorities": authorities,
                        "optional_inputs": inputs,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    snapshot["reproducibility"] = {
        "snapshot_revision": "platform_design_report_v1",
        "producer": _record(Path(__file__)),
        "authorities": authorities,
        "catalog_sha256": catalog_hash,
        "optional_inputs": inputs,
        "optional_evidence_policy": "Missing is unavailable, never zero; snapshot generation does not run scientific calculations, and imported receipts retain their declared execution scope",
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=APP / "src/data.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--performance", type=Path)
    parser.add_argument(
        "--build-status", choices=("updating", "paused", "complete"), default="updating"
    )
    args = parser.parse_args()
    snapshot = build_snapshot(
        args.existing,
        assessment=args.assessment,
        quality=args.quality,
        performance=args.performance,
        build_status=args.build_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queries": len(snapshot["queries"]),
                "buildStatus": snapshot["buildStatus"],
            }
        )
    )


if __name__ == "__main__":
    main()
