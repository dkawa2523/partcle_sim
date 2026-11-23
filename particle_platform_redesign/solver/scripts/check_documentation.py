"""Check maintained documentation, project skills and report data connections."""

from __future__ import annotations

import argparse
import json
import re
import runpy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from chamber_particles.yaml_input import parse_document

_REVIEWED_QUERY_ERRORS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "analysis/build_report_snapshot.py")
)["reviewed_query_errors"]
_SKILLS = (
    "inspect-comsol-model",
    "prepare-comsol-field-case",
    "validate-comsol-trajectories",
    "visualize-particle-trajectories",
)
_AUTHORITIES = (
    "README.md",
    "AGENTS.md",
    "particle_platform_redesign/AGENTS.md",
    "particle_platform_redesign/README.md",
    "particle_platform_redesign/product_specification.md",
    "particle_platform_redesign/architecture_proposal.md",
    "particle_platform_redesign/implementation_plan.md",
    "particle_platform_redesign/quality_tooling_plan.md",
    "particle_platform_redesign/vv_methodology.md",
    "particle_platform_redesign/solver/README.md",
    "particle_platform_redesign/report_app/README.md",
)
_CURRENT_SPECS = (
    "case_format_v3.md",
    "result_format_v5.md",
    "physics_models.md",
    "numerics.md",
    "support_and_errors.md",
    "parallel_execution_plan.md",
)


def check_markdown(path: Path) -> list[str]:
    """Check local link targets, excluding code examples and remote URLs."""
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]|\$\$.*?\$\$|\\\(.*?\\\)", "", text, flags=re.DOTALL)
    errors = []
    for raw in re.findall(r"(?<!\\)\[[^\[\]\r\n]+\]\(([^)]+)\)", text):
        target = raw.strip().strip("<>")
        if target.startswith("#") or urlsplit(target).scheme:
            continue
        target = unquote(target.split("#", 1)[0])
        if target and not (path.parent / target).resolve().exists():
            errors.append(f"{path}: missing local link {raw}")
    return errors


def check_skill(path: Path) -> list[str]:
    """Check the project skill's existing metadata contract with the YAML owner."""
    raw = path.read_text(encoding="utf-8")
    header = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", raw, flags=re.DOTALL)
    if header is None:
        return [f"{path}: missing YAML frontmatter"]
    try:
        metadata = parse_document(header.group(1).encode("utf-8"))
    except ValueError as error:
        return [f"{path}: {error}"]
    if not isinstance(metadata, dict) or metadata.get("name") != path.parent.name:
        return [f"{path}: skill name must match its directory"]
    if not isinstance(metadata.get("description"), str) or not metadata["description"].strip():
        return [f"{path}: skill description is required"]
    return []


def _query_references(content: str) -> set[str]:
    references = set(re.findall(r'(?:reviewedRows\(|queryId=)[{]?"([A-Za-z0-9_]+)"', content))
    for literal in re.findall(r"queryIds=\{(\[[^\]]*\])\}", content):
        references.update(json.loads(literal))
    return references


def _query_errors(name: str, query: Any, component_ids: set[str]) -> list[str]:
    errors = _REVIEWED_QUERY_ERRORS(name, query)
    if errors:
        return errors
    source = query["source"]
    for definition in source.get("metricDefinitions", []):
        errors.extend(
            f"report query {name}: unknown component {component}"
            for component in definition.get("componentIds", [])
            if component not in component_ids
        )
    return errors


def check_report(app: Path) -> list[str]:
    """Check authored data bindings, without compiling or modifying the app."""
    snapshot = json.loads((app / "src/data.json").read_text(encoding="utf-8"))
    content = (app / "src/content/report/ReportContent.jsx").read_text(encoding="utf-8")
    errors = []
    if not snapshot.get("id") or not snapshot.get("title") or snapshot.get("surface") != "report":
        errors.append("report: existing id, title and report surface are required")
    if snapshot.get("buildStatus") not in {"creating", "updating", "paused", "complete"}:
        errors.append("report: invalid authoring buildStatus")
    queries = snapshot.get("queries")
    if not isinstance(queries, dict):
        return [*errors, "report: queries mapping is required"]
    errors.extend(
        f"report: authored query {missing} has no reviewed dataset"
        for missing in sorted(_query_references(content) - queries.keys())
    )
    component_ids = set(re.findall(r'\bid="([^"]+)"', content))
    for name, query in queries.items():
        errors.extend(_query_errors(name, query, component_ids))
    return errors


def validate_documentation(root: Path) -> dict[str, object]:
    """Validate only maintained authorities; immutable historical evidence is excluded."""
    files = {root / path for path in _AUTHORITIES}
    files.update(root / "particle_platform_redesign/solver/docs" / name for name in _CURRENT_SPECS)
    skills = [root / ".agents/skills" / name / "SKILL.md" for name in _SKILLS]
    for skill in skills:
        files.update(skill.parent.rglob("*.md"))
    files.update(
        (root / "particle_platform_redesign/solver/src/chamber_particles_tools").rglob("README.md")
    )
    errors = []
    for path in sorted(files):
        if path.is_file():
            errors.extend(check_markdown(path))
        else:
            errors.append(f"missing maintained document: {path}")
    for skill in skills:
        if skill.is_file():
            errors.extend(check_skill(skill))
    errors.extend(check_report(root / "particle_platform_redesign/report_app"))
    return {
        "status": "invalid" if errors else "valid",
        "documents": len(files),
        "skills": len(skills),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    report = validate_documentation(args.root.resolve())
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return int(report["status"] != "valid")


if __name__ == "__main__":
    raise SystemExit(main())
