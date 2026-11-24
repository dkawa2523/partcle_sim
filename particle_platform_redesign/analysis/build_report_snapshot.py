"""Reproduce the report's reviewed design snapshot without inventing run evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
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
    "README.md",
    "product_specification.md",
    "architecture_proposal.md",
    "implementation_plan.md",
    "vv_methodology.md",
    "data_quality_assessment.md",
    "quality_tooling_plan.md",
    "technical_research.md",
    "solver/pyproject.toml",
    "solver/README.md",
    "solver/docs/case_format_v3.md",
    "solver/docs/result_format_v5.md",
    "solver/docs/physics_models.md",
    "solver/docs/numerics.md",
    "solver/docs/decisions.md",
    "solver/docs/support_and_errors.md",
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
    "solver/src/chamber_particles_tools/field_generation.py",
    "report_app/README.md",
    "report_app/src/content/report/ReportContent.jsx",
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


def _current_binding_errors(records: Any, root: Path) -> list[str]:
    if not isinstance(records, list):
        return ["report: current authorities must be a list"]
    names = [item.get("path") for item in records if isinstance(item, dict)]
    if len(names) != len(records) or any(not isinstance(name, str) for name in names):
        return ["report: current authorities require path/sha256 records"]
    if len(set(names)) != len(names) or set(names) != set(AUTHORITY_PATHS):
        return ["report: current authority roster differs; regenerate the snapshot"]
    return [
        f"report: current authority missing or hash drift: {item['path']}"
        for item in records
        if not (root / item["path"]).is_file() or item.get("sha256") != _digest(root / item["path"])
    ]


def validate_current_authorities(snapshot: dict[str, Any], root: Path) -> list[str]:
    """Own exact current design bindings; historical execution sources stay historical."""
    reproducibility = snapshot.get("reproducibility")
    if not isinstance(reproducibility, dict):
        return ["report: reproducibility and current authority bindings are required"]
    errors = _current_binding_errors(reproducibility.get("authorities"), root)
    producer = reproducibility.get("producer")
    expected = root / "analysis/build_report_snapshot.py"
    if (
        not isinstance(producer, dict)
        or producer.get("path") != "analysis/build_report_snapshot.py"
        or not expected.is_file()
        or producer.get("sha256") != _digest(expected)
    ):
        errors.append("report: current snapshot producer missing or hash drift")
    return errors


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
        "methods": [],
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
            "decision": "Result schema 5, nested immutable manifest and reusable ResultCohort",
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
            "module": "chamber_particles_tools.inputs / field_generation / four input clients",
            "owns": "Import, source conversion, field production, declared dependency freshness and one case publisher",
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
            "rule": "manifest_document() copies JSON; source_identity() owns derived receipts; strict result schema 5",
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
            "deliverable": "Context, result5 and committed payload identity",
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
            "name": "0.5 CLI/Python input → result reuse",
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


@dataclass(frozen=True)
class EvidenceReference:
    location: str
    availability: str
    sha256: str | None
    execution_source: str | None
    scope: str


@dataclass(frozen=True)
class ReviewedProjection:
    location: str
    document_sha256: str
    queries: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]


_ROW_FIELDS = {
    "design_decisions": ("area", "decision", "reason"),
    "physics_scope": ("capability", "current", "limits"),
    "module_responsibilities": ("module", "owns", "must_not"),
    "numerical_contracts": ("topic", "decision", "acceptance"),
    "result_layers": ("layer", "contents", "rule"),
    "implementation_work_packages": ("package", "capability", "deliverable", "gate"),
    "quality_findings": ("severity", "finding", "disposition"),
    "roadmap": ("phase", "name", "exit"),
}


def reviewed_query_errors(name: str, query: Any) -> list[str]:
    """Own minimum reviewed shape; consumers separately check UI bindings."""
    if not isinstance(query, dict) or not isinstance(query.get("rows"), list):
        return [f"report query {name}: reviewed rows must be a list"]
    source = query.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("files"), list)
        or not source["files"]
    ):
        return [f"report query {name}: source.files must be a nonempty list"]
    for key, entries in (
        ("source.evidenceFlow", source.get("evidenceFlow", [])),
        ("methods", query.get("methods", [])),
    ):
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            return [f"report query {name}: {key} must be a list of objects"]
    errors = _reviewed_row_errors(name, query["rows"])
    errors.extend(
        f"report query {name}: source.files entries must be references"
        for item in source["files"]
        if not isinstance(item, str | dict) or not item
    )
    return errors


def _reviewed_row_errors(name: str, rows: list[Any]) -> list[str]:
    errors = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append(f"report query {name}: reviewed rows must be objects")
        elif row.get("status") == "unavailable":
            if row.get("value") is not None:
                errors.append(
                    f"report query {name}: unavailable value must be null, not a measurement"
                )
        elif any(not isinstance(row.get(key), str) for key in _ROW_FIELDS.get(name, ())):
            errors.append(
                f"report query {name}: displayed row fields must be text: {_ROW_FIELDS[name]}"
            )
    return errors


def _evidence_reference(value: Any, scope: str) -> tuple[EvidenceReference, Any]:
    if isinstance(value, str):
        return EvidenceReference(value, "external_reference", None, None, scope), value
    reference = _structured_reference(value, scope)
    _verify_evidence(reference)
    return reference, _reference_wire(value, reference)


def _structured_reference(value: dict[str, Any], scope: str) -> EvidenceReference:
    location = value.get(
        "location", value.get("file", value.get("href", value.get("label", value.get("name"))))
    )
    availability = value.get("availability", "external_reference")
    scope = value.get("scope", scope)
    if not isinstance(location, str) or not location or not isinstance(scope, str) or not scope:
        raise ValueError("evidence reference requires nonempty location and scope")
    if not isinstance(availability, str) or availability not in {
        "bound_local",
        "external_reference",
        "historical_unavailable",
    }:
        raise ValueError("evidence reference has an unknown availability")
    digest, execution = value.get("sha256"), value.get("execution_source")
    if digest is not None and (not isinstance(digest, str) or not _valid_digest(digest)):
        raise ValueError("evidence reference sha256 must be lowercase sha256:<64 hex>")
    if execution is not None and not isinstance(execution, str):
        raise ValueError("evidence execution_source must be text")
    return EvidenceReference(location, availability, digest, execution, scope)


def _reference_wire(value: dict[str, Any], reference: EvidenceReference) -> dict[str, str]:
    label = value.get("label", reference.location)
    if not isinstance(label, str) or not label:
        raise ValueError("evidence label must be nonempty text")
    wire = {"label": label}
    if "href" in value:
        if not isinstance(value["href"], str):
            raise ValueError("evidence href must be text")
        wire["href"] = value["href"]
    return wire


def _valid_digest(value: str) -> bool:
    return (
        value.startswith("sha256:")
        and len(value) == 71
        and all(item in "0123456789abcdef" for item in value[7:])
    )


def _verify_evidence(reference: EvidenceReference) -> None:
    if reference.availability != "bound_local":
        return
    if reference.sha256 is None:
        raise ValueError("bound_local evidence requires sha256")
    path = Path(reference.location)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or _digest(path) != reference.sha256:
        raise ValueError(f"bound_local evidence missing or hash mismatch: {reference.location}")


def read_projection(path: Path, allowed: set[str]) -> ReviewedProjection:
    """Parse and identify the same bytes; explicit local bindings are verified."""
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != {"queries"}
        or not isinstance(document["queries"], dict)
    ):
        raise ValueError("reviewed evidence input requires exactly a queries mapping")
    if not set(document["queries"]).issubset(allowed):
        raise ValueError("reviewed evidence input contains an unrequested query ID")
    evidence = []
    for key, query in document["queries"].items():
        errors = reviewed_query_errors(key, query)
        if errors:
            raise ValueError("; ".join(errors))
        source = query["source"]
        scope = source.get("scope", "Citation only; no present execution qualification inferred")
        if not isinstance(scope, str) or not scope:
            raise ValueError("reviewed evidence scope must be nonempty text")
        references, files = [], []
        for item in source["files"]:
            reference, wire = _evidence_reference(item, scope)
            references.append({"query_id": key, **asdict(reference)})
            files.append(wire)
        source["files"] = files
        source.setdefault("evidenceFlow", []).append(
            {
                "title": "Reviewed evidence availability and execution scope",
                "detail": json.dumps(references, ensure_ascii=False),
            }
        )
        evidence.extend(references)
    try:
        location = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        location = str(path.resolve())
    return ReviewedProjection(
        location, "sha256:" + hashlib.sha256(raw).hexdigest(), document["queries"], tuple(evidence)
    )


def publish_snapshot(snapshot: dict[str, Any], destination: Path) -> dict[str, str]:
    """Atomically replace an intentionally refreshed snapshot after serialization."""
    raw = (json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
        "utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        published = True
    finally:
        if not published:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}


def _solver_argument(path: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), ROOT / "solver")).as_posix()
    except ValueError:  # A different Windows drive cannot be made relative.
        return str(path.resolve())


def _reproduction(
    existing: Path,
    output: Path,
    projections: tuple[tuple[str, Path | None], ...],
    build_status: str,
) -> dict[str, Any]:
    arguments = [
        "uv",
        "run",
        "--locked",
        "python",
        "../analysis/build_report_snapshot.py",
        "--existing",
        _solver_argument(existing),
        "--output",
        _solver_argument(output),
    ]
    for name, path in projections:
        if path is not None:
            arguments.extend((f"--{name}", _solver_argument(path)))
    arguments.extend(("--build-status", build_status))
    return {
        "working_directory": "particle_platform_redesign/solver",
        "scope": "declared_projections" if any(path for _, path in projections) else "design_only",
        "arguments": arguments,
        "input_policy": "Reuse the recorded projection bytes and existing presentation; regeneration does not rerun their scientific executions",
    }


def _bind_query_reproduction(query: dict[str, Any], reproduction: dict[str, Any]) -> None:
    """Replace producer-owned instructions while preserving external analysis methods."""
    python_call = f"subprocess.run({reproduction['arguments']!r}, check=True)"
    query["methods"] = [
        method
        for method in query.get("methods", [])
        if method.get("owner") != "build_report_snapshot"
        and method.get("code")
        != "uv run --locked python ../analysis/build_report_snapshot.py --output ../report_app/src/data.json --build-status complete"
    ]
    query["methods"].append(
        {
            "owner": "build_report_snapshot",
            "language": "python",
            "code": "# Run from particle_platform_redesign/solver\nimport subprocess\n"
            + python_call,
        }
    )
    query["source"]["evidenceFlow"] = [
        item
        for item in query["source"].get("evidenceFlow", [])
        if item.get("title")
        not in {
            "Reproduce design snapshot",
            "Reconstruct this report snapshot",
            "Bound authority snapshot and catalog",
        }
    ]
    query["source"]["evidenceFlow"].append(
        {
            "title": "Reconstruct this report snapshot",
            "detail": "Instruction for rebuilding the document, not a new scientific execution. "
            f"Working directory: {reproduction['working_directory']}. "
            f"Scope: {reproduction['scope']}. {reproduction['input_policy']}.\n"
            + "import subprocess; "
            + python_call,
        }
    )


def build_snapshot(
    existing: Path,
    *,
    assessment: Path | None = None,
    quality: Path | None = None,
    performance: Path | None = None,
    build_status: str = "updating",
    output: Path | None = None,
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
    reviewed_evidence = []
    for path, allowed in (
        (assessment, {"case_audit", "trajectory_divergence"}),
        (quality, {"quality_findings"}),
        (performance, {"implementation_work_packages"}),
    ):
        if path is not None:
            projection = read_projection(path, allowed)
            queries.update(projection.queries)
            inputs.append({"path": projection.location, "sha256": projection.document_sha256})
            reviewed_evidence.extend(projection.evidence)
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
    reproduction = _reproduction(
        existing,
        output or APP / "src/data.json",
        (("assessment", assessment), ("quality", quality), ("performance", performance)),
        build_status,
    )
    authorities = [_record(ROOT / name) for name in AUTHORITY_PATHS]
    for query in queries.values():
        _bind_query_reproduction(query, reproduction)
        query["source"].setdefault("evidenceFlow", []).append(
            {
                "title": "Bound authority snapshot and catalog",
                "detail": json.dumps(
                    {
                        "catalog_sha256": catalog_hash,
                        "authorities": authorities,
                        "optional_inputs": inputs,
                        "claim_scope": "Current design references only; imported execution claims retain their own source and scope",
                    },
                    ensure_ascii=False,
                ),
            }
        )
    snapshot["reproducibility"] = {
        "snapshot_revision": "platform_design_report_v2",
        "producer": _record(Path(__file__)),
        "authorities": authorities,
        "catalog_sha256": catalog_hash,
        "optional_inputs": inputs,
        "reviewed_evidence": reviewed_evidence,
        "reproduction": reproduction,
        "optional_evidence_policy": "Missing is unavailable, never zero; snapshot generation does not run scientific calculations, and imported receipts retain their declared execution scope",
    }
    errors = validate_current_authorities(snapshot, ROOT)
    if errors:
        raise ValueError("; ".join(errors))
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
        output=args.output,
    )
    binding = publish_snapshot(snapshot, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "queries": len(snapshot["queries"]),
                "buildStatus": snapshot["buildStatus"],
                "snapshot": binding,
            }
        )
    )


if __name__ == "__main__":
    main()
