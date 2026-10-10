"""Generic COMSOL meaning preflight stays fail-closed and question-specific."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

TOOL = Path(__file__).resolve().parents[1] / "meaning_preflight.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("comsol_meaning_preflight", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


meaning_preflight = _load_tool()
LAYERS = meaning_preflight.LAYERS
InventoryError = meaning_preflight.InventoryError
build_preflight = meaning_preflight.build_preflight
classify_inventory = meaning_preflight.classify_inventory
require_supported_comparison = meaning_preflight.require_supported_comparison


@pytest.fixture(autouse=True)
def _binding_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    values: dict[str, object] = {
        layer: {"enabled": True, "selector": "canonical", "value": 1.0} for layer in LAYERS
    }
    values["status"] = "PASS"
    for name in ("expected.json", "observed.json"):
        (tmp_path / name).write_text(json.dumps(values), encoding="utf-8")


def _reference(path: str, pointer: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "pointer": pointer,
    }


def _item(
    item_id: str,
    *,
    scope: str = "required",
    mapping: str | None = "direct",
) -> dict[str, object]:
    return {
        "id": item_id,
        "scope": scope,
        "source_meaning": None if scope == "unresolved" else f"COMSOL {item_id}",
        "canonical_meaning": f"canonical {item_id}" if mapping in {"direct", "adapter"} else None,
        "mapping": mapping,
        "adapter_action": f"adapt {item_id}" if mapping == "adapter" else None,
        "evidence": [f"inventory row for {item_id}"],
        "binding": [
            {
                "expected": _reference("expected.json", f"/{item_id}"),
                "observed": _reference("observed.json", f"/{item_id}"),
            }
        ]
        if scope == "required" and mapping == "direct"
        else [],
        "reason": f"classification reason for {item_id}",
    }


def _question(
    *,
    reference_representation: str,
    candidate_representation: str = "canonical",
    reference_identity: str | None = "canonical-sha256:abc",
    candidate_identity: str | None = "canonical-sha256:abc",
    lineage: list[str] | None = None,
    requested: bool = True,
    outcome: str = "NOT_TESTED",
) -> dict[str, object]:
    return {
        "requested": requested,
        "reference_field": {
            "representation": reference_representation,
            "identity": reference_identity,
        },
        "candidate_field": {
            "representation": candidate_representation,
            "identity": candidate_identity,
        },
        "adapter_lineage": [] if lineage is None else lineage,
        "outcome": {
            "status": outcome,
            "summary": "comparison has not run" if outcome == "NOT_TESTED" else "result",
            "evidence": [] if outcome in {"NOT_TESTED", "NOT_APPLICABLE"} else ["result.json"],
        },
    }


def _inventory() -> dict[str, object]:
    return {
        "schema_version": 2,
        "inventory_id": "synthetic-comsol-model",
        "source": {
            "kind": "comsol_extracted_model_inventory",
            "model_sha256": "1" * 64,
            "comsol_version": "6.4",
            "component": "comp1",
            "study": "std1",
            "solution": "sol1",
            "dataset": "dset1",
        },
        "layers": {layer: [_item(layer)] for layer in LAYERS},
        "comparison_conditions": {
            "same_canonical_field_solver_parity": _question(reference_representation="canonical"),
            "native_fe_end_to_end_reproduction": _question(
                reference_representation="native_fe",
                reference_identity="comsol-solution:sol1/dset1",
                lineage=["native sol1/dset1 -> canonical-sha256:abc"],
            ),
        },
    }


def _record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_supported_inventory_keeps_two_comparison_questions_separate() -> None:
    conditions, summary = classify_inventory(_inventory())

    assert conditions["overall_classification"] == "SUPPORTED"
    questions = _record(conditions["comparison_questions"])
    same = _record(questions["same_canonical_field_solver_parity"])
    native = _record(questions["native_fe_end_to_end_reproduction"])
    assert same["classification"] == "SUPPORTED"
    assert native["classification"] == "SUPPORTED"
    assert _record(same["condition"])["includes_field_production_import_or_recovery_error"] is False
    assert (
        _record(native["condition"])["includes_field_production_import_or_recovery_error"] is True
    )
    assert _record(same["outcome"])["status"] == "NOT_TESTED"
    assert _record(native["outcome"])["status"] == "NOT_TESTED"
    assert summary["questions_must_not_share_one_error_number"] is True


def test_all_four_classifications_are_reported_without_tag_specific_rules() -> None:
    inventory = _inventory()
    layers = _record(inventory["layers"])
    layers["formulation"] = [_item("particle_formulation", mapping="adapter")]
    layers["source"] = [_item("unused_release", scope="excluded", mapping=None)]
    layers["boundaries"] = [_item("moving_wall", mapping="unsupported")]
    layers["models"] = [_item("unknown_force", scope="unresolved", mapping="unresolved")]

    conditions, summary = classify_inventory(inventory)

    classified_layers = _record(conditions["layers"])
    assert _record(classified_layers["coordinate_dof"])["classification"] == "SUPPORTED"
    assert _record(classified_layers["formulation"])["classification"] == "ADAPTER_REQUIRED"
    assert _record(classified_layers["source"])["classification"] == "NOT_APPLICABLE"
    assert _record(classified_layers["boundaries"])["classification"] == "NOT_APPLICABLE"
    assert _record(classified_layers["models"])["classification"] == "AMBIGUOUS"
    assert summary["overall_classification"] == "AMBIGUOUS"
    counts = _record(summary["classification_counts"])
    for count in counts.values():
        assert isinstance(count, int)
        assert count > 0


def test_same_field_identity_mismatch_is_not_native_fe_failure() -> None:
    inventory = _inventory()
    comparisons = _record(inventory["comparison_conditions"])
    comparisons["same_canonical_field_solver_parity"] = _question(
        reference_representation="canonical",
        candidate_identity="canonical-sha256:different",
    )

    conditions, _ = classify_inventory(inventory)

    questions = _record(conditions["comparison_questions"])
    same = _record(questions["same_canonical_field_solver_parity"])
    native = _record(questions["native_fe_end_to_end_reproduction"])
    assert same["classification"] == "NOT_APPLICABLE"
    assert same["blockers"] == ["canonical_field_identity_differs"]
    assert native["classification"] == "SUPPORTED"


def test_field_adapter_blocks_native_fe_but_not_identical_canonical_field_parity() -> None:
    inventory = _inventory()
    layers = _record(inventory["layers"])
    layers["field_representation_owner_recovery"] = [
        _item("native_fe_to_canonical", mapping="adapter")
    ]

    conditions, summary = classify_inventory(inventory)

    questions = _record(conditions["comparison_questions"])
    same = _record(questions["same_canonical_field_solver_parity"])
    native = _record(questions["native_fe_end_to_end_reproduction"])
    assert same["classification"] == "SUPPORTED"
    assert same["blockers"] == []
    assert native["classification"] == "ADAPTER_REQUIRED"
    assert native["blockers"] == [
        "semantic_layer_field_representation_owner_recovery_adapter_required"
    ]
    assert summary["overall_classification"] == "ADAPTER_REQUIRED"
    assert summary["requested_questions_ready"] is False


def test_cli_readiness_uses_only_requested_question_layers() -> None:
    inventory = _inventory()
    layers = _record(inventory["layers"])
    layers["field_representation_owner_recovery"] = [
        _item("native_fe_to_canonical", mapping="adapter")
    ]
    comparisons = _record(inventory["comparison_conditions"])
    comparisons["native_fe_end_to_end_reproduction"] = _question(
        reference_representation="native_fe",
        reference_identity="comsol-solution:sol1/dset1",
        lineage=["native sol1/dset1 -> canonical-sha256:abc"],
        requested=False,
        outcome="NOT_APPLICABLE",
    )

    _, summary = classify_inventory(inventory)

    assert summary["overall_classification"] == "ADAPTER_REQUIRED"
    assert summary["requested_questions_ready"] is True


def test_cli_readiness_requires_at_least_one_requested_question() -> None:
    inventory = _inventory()
    comparisons = _record(inventory["comparison_conditions"])
    comparisons["same_canonical_field_solver_parity"] = _question(
        reference_representation="canonical",
        requested=False,
        outcome="NOT_APPLICABLE",
    )
    comparisons["native_fe_end_to_end_reproduction"] = _question(
        reference_representation="native_fe",
        reference_identity="comsol-solution:sol1/dset1",
        lineage=["native sol1/dset1 -> canonical-sha256:abc"],
        requested=False,
        outcome="NOT_APPLICABLE",
    )

    _, summary = classify_inventory(inventory)

    assert summary["requested_questions_ready"] is False


def test_missing_layer_becomes_ambiguous_instead_of_being_assumed_irrelevant() -> None:
    inventory = _inventory()
    layers = _record(inventory["layers"])
    del layers["integration"]

    conditions, summary = classify_inventory(inventory)

    integration = _record(_record(conditions["layers"])["integration"])
    item = _record(cast(list[object], integration["items"])[0])
    assert integration["classification"] == "AMBIGUOUS"
    assert item["id"] == "missing_integration_inventory"
    assert summary["overall_classification"] == "AMBIGUOUS"


def test_pass_cannot_be_attached_to_an_unsupported_question() -> None:
    inventory = _inventory()
    layers = _record(inventory["layers"])
    layers["integration"] = [_item("adaptive_integrator", mapping="unsupported")]
    comparisons = _record(inventory["comparison_conditions"])
    comparisons["same_canonical_field_solver_parity"] = _question(
        reference_representation="canonical", outcome="PASS"
    )

    with pytest.raises(InventoryError, match="PASS is invalid"):
        classify_inventory(inventory)


def test_source_identity_is_required_before_semantic_classification() -> None:
    inventory = _inventory()
    source = _record(inventory["source"])
    del source["dataset"]

    with pytest.raises(InventoryError, match="missing required provenance"):
        classify_inventory(inventory)


def test_cli_writer_is_no_clobber_and_hashes_the_inventory(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(_inventory()), encoding="utf-8")
    output = tmp_path / "preflight"

    summary = build_preflight(inventory_path, output)

    assert summary["overall_classification"] == "SUPPORTED"
    conditions = json.loads((output / "comparison_conditions.json").read_text(encoding="utf-8"))
    written_summary = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
    assert conditions["inventory_sha256"] == written_summary["inventory_sha256"]
    with pytest.raises(FileExistsError):
        build_preflight(inventory_path, output)


def test_declaration_only_mapping_requires_actual_adapter_evidence() -> None:
    inventory = _inventory()
    models = cast(list[dict[str, object]], _record(inventory["layers"])["models"])
    models[0]["binding"] = []
    conditions, summary = classify_inventory(inventory)
    assert _record(_record(conditions["layers"])["models"])["classification"] == "ADAPTER_REQUIRED"
    assert summary["requested_questions_ready"] is False


@pytest.mark.parametrize(
    ("layer", "key", "observed"),
    [
        ("models", "selector", "native"),
        ("coordinate_dof", "selector", "Freeze"),
        ("integration", "value", 1),
        ("models", "enabled", 1),
    ],
)
def test_actual_value_and_json_type_must_match_locked_expectation(
    tmp_path: Path, layer: str, key: str, observed: object
) -> None:
    document = json.loads((tmp_path / "observed.json").read_text(encoding="utf-8"))
    document[layer][key] = observed
    (tmp_path / "observed.json").write_text(json.dumps(document), encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(_inventory()), encoding="utf-8")
    output = tmp_path / "preflight"
    with pytest.raises(InventoryError, match="binding value or type mismatch"):
        build_preflight(inventory, output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["hash", "missing", "pointer", "status", "nonfinite"])
def test_unverifiable_receipt_stops_before_any_output(tmp_path: Path, problem: str) -> None:
    inventory = _inventory()
    row = cast(list[dict[str, object]], _record(inventory["layers"])["models"])[0]
    binding = cast(list[dict[str, object]], row["binding"])[0]
    observed = _record(binding["observed"])
    if problem == "hash":
        observed["sha256"] = "0" * 64
    elif problem == "missing":
        observed["path"] = "missing.json"
    elif problem == "pointer":
        observed["pointer"] = "/missing/value"
    elif problem == "status":
        binding["observed"] = _reference("observed.json", "/status")
        binding["expected"] = _reference("expected.json", "/status")
    else:
        (tmp_path / "observed.json").write_text('{"models": 1e400}', encoding="utf-8")
        binding["observed"] = _reference("observed.json", "/models")
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    output = tmp_path / "preflight"
    with pytest.raises(InventoryError):
        build_preflight(path, output)
    assert not output.exists()


def test_old_inventory_cannot_certify_current_execution() -> None:
    inventory = _inventory()
    inventory["schema_version"] = 1
    with pytest.raises(InventoryError, match="declaration-only v1"):
        classify_inventory(inventory)


@pytest.mark.parametrize(
    "declaration",
    [
        {"receipt": {"status": "SUPPORTED"}},
        {"receipt": {"status": "SUPPORTED", "summary": "all checks passed"}},
        {"receipt": [{"classification": "SUPPORTED"}]},
    ],
)
def test_nested_status_wrapper_cannot_replace_observed_values(
    tmp_path: Path, declaration: object
) -> None:
    inventory = _inventory()
    for name in ("expected.json", "observed.json"):
        (tmp_path / name).write_text(json.dumps(declaration), encoding="utf-8")
    layers = _record(inventory["layers"])
    for rows in layers.values():
        for row in cast(list[dict[str, object]], rows):
            row["binding"] = [
                {
                    "expected": _reference("expected.json", ""),
                    "observed": _reference("observed.json", ""),
                }
            ]
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    output = tmp_path / "preflight"
    with pytest.raises(InventoryError, match="status declaration"):
        build_preflight(path, output)
    assert not output.exists()


def test_artifact_paths_resolve_relative_to_inventory_not_process_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(_inventory()), encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    summary = build_preflight(path, tmp_path / "preflight")
    assert summary["requested_questions_ready"] is True


def test_comparison_consumer_rechecks_inventory_and_selected_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(_inventory()), encoding="utf-8")
    reference = _reference(path.name, "")
    del reference["pointer"]
    assert require_supported_comparison(reference, tmp_path) == reference
    document = json.loads((tmp_path / "observed.json").read_text(encoding="utf-8"))
    document["models"]["selector"] = "native"
    (tmp_path / "observed.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(InventoryError, match="binding artifact SHA-256 mismatch"):
        require_supported_comparison(reference, tmp_path)


@pytest.mark.parametrize("problem", [None, "model", "field", "receipt", "missing_receipt"])
def test_comparison_inventory_must_belong_to_the_participant(
    tmp_path: Path, problem: str | None
) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(_inventory()), encoding="utf-8")
    reference = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    observed_hash = hashlib.sha256((tmp_path / "observed.json").read_bytes()).hexdigest()
    model = "2" * 64 if problem == "model" else "1" * 64
    field = "canonical-sha256:other" if problem == "field" else "canonical-sha256:abc"
    receipts = (
        frozenset()
        if problem == "missing_receipt"
        else frozenset({"0" * 64 if problem == "receipt" else observed_hash})
    )
    if problem is not None:
        with pytest.raises(InventoryError):
            require_supported_comparison(
                reference,
                tmp_path,
                expected_model_sha256=model,
                expected_field_identity=field,
                required_observed_sha256=receipts,
            )
    else:
        assert (
            require_supported_comparison(
                reference,
                tmp_path,
                expected_model_sha256=model,
                expected_field_identity=field,
                required_observed_sha256=receipts,
            )
            == reference
        )


@pytest.mark.parametrize("problem", ["missing", "hash", "declaration_only", "unrequested"])
def test_comparison_consumer_cannot_accept_unverified_conditions(
    tmp_path: Path, problem: str
) -> None:
    inventory = _inventory()
    if problem == "declaration_only":
        cast(list[dict[str, object]], _record(inventory["layers"])["models"])[0]["binding"] = []
    elif problem == "unrequested":
        _record(inventory["comparison_conditions"])["same_canonical_field_solver_parity"] = (
            _question(
                reference_representation="canonical", requested=False, outcome="NOT_APPLICABLE"
            )
        )
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    reference: object = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if problem == "missing":
        reference = None
    elif problem == "hash":
        reference = {"path": path.name, "sha256": "0" * 64}
    with pytest.raises(InventoryError):
        require_supported_comparison(reference, tmp_path)
