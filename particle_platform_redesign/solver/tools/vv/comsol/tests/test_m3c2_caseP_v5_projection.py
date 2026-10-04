"""Focused checks for the immutable Case-P policy-v5 migration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from tools.vv.comsol.project_m3c2_caseP_pilot_v5 import (
    PROJECTION_REASON,
    _projected_document,
    _validate_policy_transition,
)


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_v5_preserves_v4_science_threshold_seed_and_observable_blocks() -> None:
    root = _solver_root()
    cases = root / "tools" / "vv" / "comsol" / "cases"
    v4 = _json(cases / "m3c2_caseP_100nm_ensemble_evaluation_v4.json")
    v5 = _json(cases / "m3c2_caseP_100nm_ensemble_evaluation_v5.json")

    _validate_policy_transition(v4, v5)

    changed = copy.deepcopy(v5)
    changed["final"]["terminal_population_gate"]["margin"] = 0.051
    with pytest.raises(ValueError, match="thresholds, seeds, observables"):
        _validate_policy_transition(v4, changed)


def test_candidate_recipe_v2_changes_only_revision_and_policy_reference() -> None:
    cases = _solver_root() / "tools" / "vv" / "comsol" / "cases"
    v1 = _json(cases / "m3c2_caseP_100nm_candidate_macro_pilot_v1.json")
    v2 = _json(cases / "m3c2_caseP_100nm_candidate_macro_pilot_v2.json")

    v1["recipe_revision"] = 2
    v1["evaluation_plan"]["policy"] = v2["evaluation_plan"]["policy"]

    assert v1 == v2


def test_projection_changes_only_policy_hash_and_adds_provenance(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output = tmp_path / "projection"
    source_dir.mkdir()
    source = source_dir / "campaign.json"
    v4_policy = tmp_path / "v4.json"
    v5_policy = tmp_path / "v5.json"
    raw = {
        "manifest_kind": "m3c2_campaign",
        "evaluation_policy_sha256": "placeholder",
        "participants": {"candidate": {"levels": []}, "comsol": {"levels": []}},
    }
    source.write_text(json.dumps(raw), encoding="utf-8")
    v4_policy.write_text("{}\n", encoding="utf-8")
    v5_policy.write_text('{"revision": 5}\n', encoding="utf-8")
    from tools.vv.comsol.project_m3c2_caseP_pilot_v5 import _sha256

    raw["evaluation_policy_sha256"] = _sha256(v4_policy)
    source.write_text(json.dumps(raw), encoding="utf-8")

    projected = _projected_document(source, raw, v4_policy, v5_policy, output)

    projection = cast(dict[str, Any], projected.pop("policy_projection"))
    projected["evaluation_policy_sha256"] = raw["evaluation_policy_sha256"]
    assert projected == raw
    assert projection["reason"] == PROJECTION_REASON
    assert projection["changed_fields"] == ["evaluation_policy_sha256"]
    assert projection["raw_solver_outputs_reused_without_rerun"] is True
