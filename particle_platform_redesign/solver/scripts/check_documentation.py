"""Check maintained documentation, project skills and report data connections."""

from __future__ import annotations

import argparse
import html
import json
import re
import runpy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from chamber_particles.yaml_input import parse_document

_REPORT_PRODUCER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "analysis/build_report_snapshot.py")
)
_REVIEWED_QUERY_ERRORS = _REPORT_PRODUCER["reviewed_query_errors"]
_VALIDATE_CURRENT_AUTHORITIES = _REPORT_PRODUCER["validate_current_authorities"]
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
    "particle_platform_redesign/technical_research.md",
    "particle_platform_redesign/data_quality_assessment.md",
    "particle_platform_redesign/solver/README.md",
    "particle_platform_redesign/solver/RELEASE_NOTES.md",
    "particle_platform_redesign/solver/examples/README.md",
    "particle_platform_redesign/solver/tests/performance/README.md",
    "particle_platform_redesign/solver/tests/verification/README.md",
    "particle_platform_redesign/report_app/README.md",
)


def _without_fences(text: str) -> str:
    lines = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if marker is not None:
            token, tail = marker.groups()
            if fence is None:
                fence = token[0], len(token)
                continue
            if token[0] == fence[0] and len(token) >= fence[1] and not tail.strip():
                fence = None
                continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def _heading_slug(heading: str) -> str:
    heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = html.unescape(heading).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading))


def _headings(text: str) -> list[str]:
    lines = text.splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", line)
        if match is not None:
            headings.append(match.group(1))
        elif index and re.fullmatch(r" {0,3}(?:=+|-+)\s*", line) and lines[index - 1].strip():
            headings.append(lines[index - 1])
    return headings


def _document_fragments(path: Path) -> set[str]:
    text = _without_fences(path.read_text(encoding="utf-8"))
    fragments = set(re.findall(r"\b(?:id|name)\s*=\s*['\"]([^'\"]+)['\"]", text))
    if path.suffix.lower() != ".md":
        return fragments
    text = re.sub(r"\A---\n.*?\n---(?:\n|\Z)", "", text, flags=re.DOTALL)
    for heading in _headings(text):
        slug = _heading_slug(heading)
        candidate, ordinal = slug, 0
        while candidate in fragments:
            ordinal += 1
            candidate = f"{slug}-{ordinal}"
        fragments.add(candidate)
    return fragments


def _local_link_errors(path: Path, raw: str) -> list[str]:
    target = raw.strip()
    if not target:
        return []
    target = target[1:].split(">", 1)[0] if target.startswith("<") else target.split()[0]
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return []
    destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
    if not destination.exists():
        return [f"{path}: missing local link {raw}"]
    if not parsed.fragment:
        return []
    if destination.is_dir():
        destination /= "README.md"
    if destination.suffix.lower() not in {".md", ".html", ".svg"}:
        return []  # File viewers and source-line fragments have their own formats.
    if not destination.is_file() or unquote(parsed.fragment) not in _document_fragments(
        destination
    ):
        return [f"{path}: missing local fragment {raw}"]
    return []


def check_markdown(path: Path) -> list[str]:
    """Check local paths and Markdown/HTML/SVG fragments, excluding code and remote URLs."""
    text = _without_fences(path.read_text(encoding="utf-8"))
    text = re.sub(r"(`+).*?\1", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]|\$\$.*?\$\$|\\\(.*?\\\)", "", text, flags=re.DOTALL)
    errors = []
    for raw in re.findall(r"(?<!\\)\[[^\[\]\r\n]+\]\(([^)]+)\)", text):
        errors.extend(_local_link_errors(path, raw))
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


def check_report(app: Path, *, authority_root: Path | None = None) -> list[str]:
    """Check data bindings and, for the maintained app, its current authority snapshot."""
    snapshot = json.loads((app / "src/data.json").read_text(encoding="utf-8"))
    content = (app / "src/content/report/ReportContent.jsx").read_text(encoding="utf-8")
    errors = []
    if authority_root is not None:
        errors.extend(_VALIDATE_CURRENT_AUTHORITIES(snapshot, authority_root))
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
    solver = root / "particle_platform_redesign/solver"
    files.update((solver / "docs").rglob("*.md"))
    files.update((solver / "tools/vv").rglob("README.md"))
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
    errors.extend(
        check_report(
            root / "particle_platform_redesign/report_app",
            authority_root=root / "particle_platform_redesign",
        )
    )
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
