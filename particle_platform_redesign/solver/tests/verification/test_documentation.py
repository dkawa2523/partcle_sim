"""Maintained document checks report broken links and truthful data bindings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import check_documentation
from scripts.check_documentation import check_markdown, check_report, check_skill


def test_markdown_distinguishes_real_links_from_remote_and_example_text(tmp_path: Path) -> None:
    (tmp_path / "exists.md").write_text("# Heading", encoding="utf-8")
    path = tmp_path / "document.md"
    path.write_text(
        "[valid](exists.md#heading) [remote](https://example.org/missing)\n"
        "```markdown\n[example](not-an-actual-link.md)\n```\n"
        r"\[a=\left[G(s)\right](\boldsymbol u-\boldsymbol v)\]"
        "\n[broken](missing.md)",
        encoding="utf-8",
    )
    errors = check_markdown(path)
    assert len(errors) == 1 and "missing.md" in errors[0]


def test_markdown_checks_local_fragments_and_duplicate_headings(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text(
        '# 入力と `ID`\n## Repeat\n## Repeat\n<a id="explicit"></a>\n~~~md\n# Not a heading\n~~~\n',
        encoding="utf-8",
    )
    path = tmp_path / "document.md"
    path.write_text(
        "# Here\n[own](#here) [unicode](target.md#%E5%85%A5%E5%8A%9B%E3%81%A8-id) "
        "[duplicate](target.md#repeat-1) [explicit](target.md#explicit)\n"
        "[bad anchor](target.md#missing) [code heading](target.md#not-a-heading)\n"
        "`[inline example](absent.md)`\n",
        encoding="utf-8",
    )
    errors = check_markdown(path)
    assert len(errors) == 2
    assert all("missing local fragment" in error for error in errors)


def test_maintained_scope_includes_decisions_and_excludes_historical_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(check_documentation, "_AUTHORITIES", ())
    monkeypatch.setattr(check_documentation, "_SKILLS", ())
    monkeypatch.setattr(check_documentation, "check_report", lambda *_args, **_kwargs: [])
    current = tmp_path / "particle_platform_redesign/solver/docs/decisions.md"
    current.parent.mkdir(parents=True)
    current.write_text("[current broken](missing-current.md)", encoding="utf-8")
    for directory in ("reviews/history", "solver/evidence/old"):
        historical = tmp_path / "particle_platform_redesign" / directory / "record.md"
        historical.parent.mkdir(parents=True)
        historical.write_text("[historical citation](not-distributed.md)", encoding="utf-8")
    report = check_documentation.validate_documentation(tmp_path)
    assert report["documents"] == 1
    errors = report["errors"]
    assert isinstance(errors, list)
    assert len(errors) == 1
    assert "missing-current.md" in errors[0]


def test_skill_uses_strict_yaml_and_directory_identity(tmp_path: Path) -> None:
    directory = tmp_path / "example-skill"
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(
        "---\nname: example-skill\ndescription: Clear scope\n---\nBody", encoding="utf-8"
    )
    assert check_skill(path) == []
    path.write_text(
        "---\nname: example-skill\nname: replacement\ndescription: x\n---", encoding="utf-8"
    )
    assert "duplicate" in check_skill(path)[0]


def test_report_rejects_missing_query_and_unavailable_zero(tmp_path: Path) -> None:
    source = tmp_path / "src/content/report"
    source.mkdir(parents=True)
    (source / "ReportContent.jsx").write_text(
        'reviewedRows("missing"); <DataComponent id="table" queryId="quality" />', encoding="utf-8"
    )
    data: dict[str, Any] = {
        "id": "existing-report",
        "title": "Existing title",
        "surface": "report",
        "buildStatus": "complete",
        "queries": {
            "quality": {
                "rows": [{"status": "unavailable", "value": 0}],
                "source": {"files": ["receipt.json"]},
            }
        },
    }
    (tmp_path / "src/data.json").write_text(json.dumps(data), encoding="utf-8")
    errors = check_report(tmp_path)
    assert any("missing" in error for error in errors)
    assert any("must be null" in error for error in errors)
    data["queries"]["quality"]["rows"][0]["value"] = None
    data["queries"]["missing"] = {"rows": [], "source": {"files": ["receipt.json"]}}
    (tmp_path / "src/data.json").write_text(json.dumps(data), encoding="utf-8")
    assert check_report(tmp_path) == []
