"""Maintained document checks report broken links and truthful data bindings."""

from __future__ import annotations

import json
from pathlib import Path

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
    data = {
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
