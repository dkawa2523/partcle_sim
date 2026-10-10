"""Classify an extracted COMSOL inventory before adapting or comparing it.

The input is a producer-neutral semantic inventory.  A COMSOL extractor may
retain arbitrary tags and expressions in its own raw artifact, but it must map
the selected comparison scope into the seven layers defined here.  This tool
does not import COMSOL or the solver core, guess meanings from feature names,
or execute either solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

TOOL_REVISION: Final = "comsol_meaning_preflight_v2"
CLASSIFICATIONS: Final = (
    "SUPPORTED",
    "ADAPTER_REQUIRED",
    "NOT_APPLICABLE",
    "AMBIGUOUS",
)
LAYERS: Final = (
    "coordinate_dof",
    "formulation",
    "field_representation_owner_recovery",
    "source",
    "boundaries",
    "models",
    "integration",
)
QUESTIONS: Final = (
    "same_canonical_field_solver_parity",
    "native_fe_end_to_end_reproduction",
)
QUESTION_LAYERS: Final = {
    "same_canonical_field_solver_parity": tuple(
        layer for layer in LAYERS if layer != "field_representation_owner_recovery"
    ),
    "native_fe_end_to_end_reproduction": LAYERS,
}
_SCOPES: Final = frozenset({"required", "excluded", "unresolved"})
_MAPPINGS: Final = frozenset({"direct", "adapter", "unsupported", "unresolved"})
_OUTCOMES: Final = frozenset({"PASS", "FAIL", "NOT_TESTED", "NOT_APPLICABLE"})
_REPRESENTATIONS: Final = frozenset({"canonical", "native_fe"})
_SOURCE_KEYS: Final = frozenset(
    {"model_sha256", "comsol_version", "component", "study", "solution", "dataset"}
)

Record = dict[str, object]


class InventoryError(ValueError):
    """The semantic inventory is incomplete or internally inconsistent."""


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InventoryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InventoryError(f"{label} must be a string-keyed mapping")
    return cast(dict[str, object], value)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise InventoryError(f"{label} must be a list")
    return cast(list[object], value)


def _text(value: object, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if nullable else ""
        raise InventoryError(f"{label} must be a nonempty string{suffix}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise InventoryError(f"{label} must be a boolean")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise InventoryError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _text_list(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    rows = _sequence(value, label)
    result: list[str] = []
    for index, item in enumerate(rows):
        result.append(cast(str, _text(item, f"{label}[{index}]")))
    if not result and not allow_empty:
        raise InventoryError(f"{label} must not be empty")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(value: object) -> dict[str, object]:
    source = dict(_mapping(value, "source"))
    missing = sorted(_SOURCE_KEYS - set(source))
    if missing:
        raise InventoryError(f"source is missing required provenance: {missing}")
    digest = cast(str, _text(source["model_sha256"], "source.model_sha256")).casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise InventoryError("source.model_sha256 must be a SHA-256 hex digest")
    source["model_sha256"] = digest
    for key in ("comsol_version", "component", "study", "solution", "dataset"):
        _text(source[key], f"source.{key}")
    try:
        json.dumps(source, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise InventoryError("source must contain finite JSON data") from error
    return source


def load_inventory(path: str | Path) -> dict[str, object]:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    inventory_path = Path(path).expanduser().resolve()

    def reject_constant(token: str) -> object:
        raise InventoryError(f"non-finite JSON constant is forbidden: {token}")

    try:
        raw = json.loads(
            inventory_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read semantic inventory: {error}") from error
    try:
        json.dumps(raw, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise InventoryError("artifact must contain finite JSON data") from error
    return _mapping(raw, "inventory")


def _json_pointer(document: object, pointer: str) -> object:
    if not pointer:
        return document
    if not pointer.startswith("/"):
        raise InventoryError("JSON pointer must be empty or start with /")
    current = document
    for encoded in pointer[1:].split("/"):
        escapes = encoded.replace("~0", "").replace("~1", "")
        if "~" in escapes:
            raise InventoryError("invalid JSON pointer escape")
        token = encoded.replace("~1", "/").replace("~0", "~")
        current = _pointer_child(current, token, pointer)
    return current


def _pointer_child(current: object, token: str, pointer: str) -> object:
    if isinstance(current, dict) and token in current:
        return current[token]
    if isinstance(current, list) and token.isascii() and token.isdecimal():
        index = int(token)
        if str(index) == token and index < len(current):
            return current[index]
    raise InventoryError(f"JSON pointer does not resolve: {pointer}")


def _artifact_value(reference: object, base_directory: Path, label: str) -> object:
    row = _mapping(reference, label)
    _exact_keys(row, {"path", "sha256", "pointer"}, label)
    path = Path(cast(str, _text(row["path"], f"{label}.path")))
    path = (base_directory / path).resolve()
    digest = cast(str, _text(row["sha256"], f"{label}.sha256"))
    try:
        actual_digest = _sha256(path)
    except OSError as error:
        raise InventoryError(f"cannot read binding artifact: {path}") from error
    if digest != actual_digest:
        raise InventoryError(f"binding artifact SHA-256 mismatch: {path}")
    pointer = row["pointer"]
    if not isinstance(pointer, str):
        raise InventoryError(f"{label}.pointer must be a string")
    value = _json_pointer(load_inventory(path), pointer)
    _reject_status_claim(value)
    return value


def _reject_status_claim(value: object) -> None:
    if isinstance(value, str) and value in {*CLASSIFICATIONS, *_OUTCOMES}:
        raise InventoryError("a status declaration is not observed binding evidence")
    if isinstance(value, dict) and set(value).issubset({"status", "classification"}):
        raise InventoryError("a status declaration is not observed binding evidence")
    if isinstance(value, (dict, list)):
        for item in value.values() if isinstance(value, dict) else value:
            _reject_status_claim(item)


def _same_typed_value(expected: object, observed: object) -> bool:
    if type(expected) is not type(observed):
        return False
    if isinstance(expected, dict) and isinstance(observed, dict):
        return set(expected) == set(observed) and all(
            _same_typed_value(value, observed[key]) for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(observed, list):
        return len(expected) == len(observed) and all(
            _same_typed_value(left, right) for left, right in zip(expected, observed, strict=True)
        )
    return expected == observed


def _binding_checks(value: object, base_directory: Path, label: str) -> list[Record]:
    checks = _sequence(value, label)
    verified = []
    for index, check in enumerate(checks):
        location = f"{label}[{index}]"
        row = _mapping(check, location)
        _exact_keys(row, {"expected", "observed"}, location)
        expected = _artifact_value(row["expected"], base_directory, f"{location}.expected")
        observed = _artifact_value(row["observed"], base_directory, f"{location}.observed")
        if not _same_typed_value(expected, observed):
            raise InventoryError(f"binding value or type mismatch: {location}")
        verified.append({**row, "matched_value": observed})
    return verified


def _classify_item(item: dict[str, object], label: str, base_directory: Path) -> Record:
    _exact_keys(
        item,
        {
            "id",
            "scope",
            "source_meaning",
            "canonical_meaning",
            "mapping",
            "adapter_action",
            "evidence",
            "binding",
            "reason",
        },
        label,
    )
    item_id = cast(str, _text(item["id"], f"{label}.id"))
    scope = cast(str, _text(item["scope"], f"{label}.scope"))
    if scope not in _SCOPES:
        raise InventoryError(f"{label}.scope must be one of {sorted(_SCOPES)}")
    source_meaning = _text(item["source_meaning"], f"{label}.source_meaning", nullable=True)
    canonical_meaning = _text(
        item["canonical_meaning"], f"{label}.canonical_meaning", nullable=True
    )
    mapping = _text(item["mapping"], f"{label}.mapping", nullable=True)
    if mapping is not None and mapping not in _MAPPINGS:
        raise InventoryError(f"{label}.mapping must be one of {sorted(_MAPPINGS)} or null")
    adapter_action = _text(item["adapter_action"], f"{label}.adapter_action", nullable=True)
    evidence = _text_list(item["evidence"], f"{label}.evidence")
    reason = cast(str, _text(item["reason"], f"{label}.reason"))

    classification = _item_classification(scope, mapping)
    binding = _binding_checks(item["binding"], base_directory, f"{label}.binding")
    if classification == "SUPPORTED" and not binding:
        classification = "ADAPTER_REQUIRED"
        reason += "; actual binding has not been verified against a locked expectation"
    _validate_item_relation(
        label,
        scope=scope,
        mapping=mapping,
        source_meaning=source_meaning,
        canonical_meaning=canonical_meaning,
        adapter_action=adapter_action,
    )
    return {
        "id": item_id,
        "classification": classification,
        "scope": scope,
        "source_meaning": source_meaning,
        "canonical_meaning": canonical_meaning,
        "mapping": mapping,
        "adapter_action": adapter_action,
        "evidence": evidence,
        "binding": binding,
        "reason": reason,
    }


def _item_classification(scope: str, mapping: str | None) -> str:
    if scope == "excluded":
        return "NOT_APPLICABLE"
    if scope == "unresolved":
        return "AMBIGUOUS"
    return {
        "direct": "SUPPORTED",
        "adapter": "ADAPTER_REQUIRED",
        "unsupported": "NOT_APPLICABLE",
        "unresolved": "AMBIGUOUS",
        None: "AMBIGUOUS",
    }[mapping]


def _validate_item_relation(
    label: str,
    *,
    scope: str,
    mapping: str | None,
    source_meaning: str | None,
    canonical_meaning: str | None,
    adapter_action: str | None,
) -> None:
    if scope == "excluded":
        if mapping is not None or adapter_action is not None:
            raise InventoryError(f"{label}: excluded items require null mapping and adapter_action")
        return
    if scope == "unresolved":
        if mapping not in {None, "unresolved"} or adapter_action is not None:
            raise InventoryError(f"{label}: unresolved scope cannot assert a mapping or adapter")
        return
    if source_meaning is None:
        raise InventoryError(f"{label}: required items need source_meaning")
    if mapping in {"direct", "adapter"} and canonical_meaning is None:
        raise InventoryError(f"{label}: mapped items need canonical_meaning")
    if mapping == "adapter" and adapter_action is None:
        raise InventoryError(f"{label}: adapter mapping needs adapter_action")
    if mapping != "adapter" and adapter_action is not None:
        raise InventoryError(f"{label}: adapter_action is valid only for adapter mapping")


def _missing_layer(layer: str) -> Record:
    return {
        "id": f"missing_{layer}_inventory",
        "classification": "AMBIGUOUS",
        "scope": "unresolved",
        "source_meaning": None,
        "canonical_meaning": None,
        "mapping": None,
        "adapter_action": None,
        "evidence": ["No semantic inventory item was supplied for this required layer."],
        "binding": [],
        "reason": "Absence is not evidence that the layer is irrelevant.",
    }


def _layer_classification(items: list[Record]) -> str:
    required = [item for item in items if item["scope"] != "excluded"]
    if not required:
        return "NOT_APPLICABLE"
    statuses = {cast(str, item["classification"]) for item in required}
    for status in ("AMBIGUOUS", "NOT_APPLICABLE", "ADAPTER_REQUIRED", "SUPPORTED"):
        if status in statuses:
            return status
    raise AssertionError("unreachable classification set")


def _classify_layers(value: object, base_directory: Path) -> dict[str, Record]:
    layers = _mapping(value, "layers")
    unknown = sorted(set(layers) - set(LAYERS))
    if unknown:
        raise InventoryError(f"unknown semantic layers: {unknown}")
    output: dict[str, Record] = {}
    seen_ids: set[str] = set()
    for layer in LAYERS:
        raw_items = _sequence(layers.get(layer, []), f"layers.{layer}")
        items = [
            _classify_item(
                _mapping(raw, f"layers.{layer}[{index}]"),
                f"layers.{layer}[{index}]",
                base_directory,
            )
            for index, raw in enumerate(raw_items)
        ]
        if not items:
            items = [_missing_layer(layer)]
        for item in items:
            item_id = cast(str, item["id"])
            if item_id in seen_ids:
                raise InventoryError(f"duplicate semantic item id: {item_id}")
            seen_ids.add(item_id)
        output[layer] = {
            "classification": _layer_classification(items),
            "in_scope_item_count": sum(item["scope"] != "excluded" for item in items),
            "items": items,
        }
    return output


def _overall_classification(layers: Mapping[str, Record]) -> str:
    statuses = {
        cast(str, layer["classification"])
        for layer in layers.values()
        if cast(int, layer["in_scope_item_count"]) > 0
    }
    if not statuses:
        return "NOT_APPLICABLE"
    for status in ("AMBIGUOUS", "NOT_APPLICABLE", "ADAPTER_REQUIRED", "SUPPORTED"):
        if status in statuses:
            return status
    raise AssertionError("unreachable layer classification set")


def _field(value: object, label: str) -> Record:
    field = _mapping(value, label)
    _exact_keys(field, {"representation", "identity"}, label)
    representation = cast(str, _text(field["representation"], f"{label}.representation"))
    if representation not in _REPRESENTATIONS:
        raise InventoryError(f"{label}.representation must be one of {sorted(_REPRESENTATIONS)}")
    identity = _text(field["identity"], f"{label}.identity", nullable=True)
    return {"representation": representation, "identity": identity}


def _outcome(value: object, label: str) -> Record:
    outcome = _mapping(value, label)
    _exact_keys(outcome, {"status", "summary", "evidence"}, label)
    status = cast(str, _text(outcome["status"], f"{label}.status"))
    if status not in _OUTCOMES:
        raise InventoryError(f"{label}.status must be one of {sorted(_OUTCOMES)}")
    summary = cast(str, _text(outcome["summary"], f"{label}.summary"))
    evidence = _text_list(outcome["evidence"], f"{label}.evidence", allow_empty=True)
    if status in {"PASS", "FAIL"} and not evidence:
        raise InventoryError(f"{label}: PASS/FAIL needs result evidence")
    return {"status": status, "summary": summary, "evidence": evidence}


def _question_definition(question: str) -> Record:
    if question == "same_canonical_field_solver_parity":
        return {
            "reference_field_requirement": "canonical",
            "candidate_field_requirement": "canonical",
            "requires_identical_field_identity": True,
            "includes_field_production_import_or_recovery_error": False,
            "semantic_layers": QUESTION_LAYERS[question],
            "allowed_claim": "solver_parity_on_one_identical_canonical_field",
        }
    return {
        "reference_field_requirement": "native_fe",
        "candidate_field_requirement": "canonical",
        "requires_identical_field_identity": False,
        "includes_field_production_import_or_recovery_error": True,
        "semantic_layers": QUESTION_LAYERS[question],
        "allowed_claim": "end_to_end_reproduction_including_field_representation_error",
    }


def _question_semantics(question: str, layers: Mapping[str, Record]) -> tuple[str, list[str]]:
    selected = {name: layers[name] for name in QUESTION_LAYERS[question]}
    classification = _overall_classification(selected)
    blockers = [
        f"semantic_layer_{name}_{cast(str, layer['classification']).casefold()}"
        for name, layer in selected.items()
        if cast(int, layer["in_scope_item_count"]) > 0 and layer["classification"] != "SUPPORTED"
    ]
    return classification, blockers


def _question_classification(
    question: str,
    *,
    requested: bool,
    reference_field: Record,
    candidate_field: Record,
    adapter_lineage: list[str],
    semantic_classification: str,
    semantic_blockers: list[str],
) -> tuple[str, list[str]]:
    if not requested:
        return "NOT_APPLICABLE", ["question_not_requested"]
    definition = _question_definition(question)
    expected_reference = definition["reference_field_requirement"]
    expected_candidate = definition["candidate_field_requirement"]
    reasons: list[str] = []
    if reference_field["representation"] != expected_reference:
        reasons.append("reference_field_representation_does_not_match_question")
    if candidate_field["representation"] != expected_candidate:
        reasons.append("candidate_field_representation_does_not_match_question")
    if reasons:
        return "NOT_APPLICABLE", reasons
    if reference_field["identity"] is None or candidate_field["identity"] is None:
        return "AMBIGUOUS", ["field_identity_missing"]
    if question == "same_canonical_field_solver_parity":
        if reference_field["identity"] != candidate_field["identity"]:
            return "NOT_APPLICABLE", ["canonical_field_identity_differs"]
    elif not adapter_lineage:
        return "AMBIGUOUS", ["native_to_canonical_adapter_lineage_missing"]
    if semantic_classification == "SUPPORTED":
        return "SUPPORTED", []
    return semantic_classification, semantic_blockers


def _classify_questions(value: object, layers: Mapping[str, Record]) -> dict[str, Record]:
    questions = _mapping(value, "comparison_conditions")
    _exact_keys(questions, set(QUESTIONS), "comparison_conditions")
    output: dict[str, Record] = {}
    for question in QUESTIONS:
        label = f"comparison_conditions.{question}"
        row = _mapping(questions[question], label)
        _exact_keys(
            row,
            {"requested", "reference_field", "candidate_field", "adapter_lineage", "outcome"},
            label,
        )
        requested = _boolean(row["requested"], f"{label}.requested")
        reference_field = _field(row["reference_field"], f"{label}.reference_field")
        candidate_field = _field(row["candidate_field"], f"{label}.candidate_field")
        adapter_lineage = _text_list(
            row["adapter_lineage"], f"{label}.adapter_lineage", allow_empty=True
        )
        outcome = _outcome(row["outcome"], f"{label}.outcome")
        semantic_classification, semantic_blockers = _question_semantics(question, layers)
        classification, blockers = _question_classification(
            question,
            requested=requested,
            reference_field=reference_field,
            candidate_field=candidate_field,
            adapter_lineage=adapter_lineage,
            semantic_classification=semantic_classification,
            semantic_blockers=semantic_blockers,
        )
        if outcome["status"] in {"PASS", "FAIL"} and classification != "SUPPORTED":
            raise InventoryError(
                f"{label}: {outcome['status']} is invalid when condition is {classification}"
            )
        if not requested and outcome["status"] != "NOT_APPLICABLE":
            raise InventoryError(f"{label}: an unrequested question must be NOT_APPLICABLE")
        output[question] = {
            "classification": classification,
            "requested": requested,
            "condition": _question_definition(question),
            "reference_field": reference_field,
            "candidate_field": candidate_field,
            "adapter_lineage": adapter_lineage,
            "blockers": blockers,
            "outcome": outcome,
        }
    return output


def classify_inventory(
    inventory: Mapping[str, object], *, base_directory: Path | None = None
) -> tuple[Record, Record]:
    """Return normalized comparison conditions and a compact summary."""

    document = dict(inventory)
    _exact_keys(
        document,
        {"schema_version", "inventory_id", "source", "layers", "comparison_conditions"},
        "inventory",
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 2:
        raise InventoryError("schema_version must be 2; declaration-only v1 is not certification")
    inventory_id = cast(str, _text(document["inventory_id"], "inventory_id"))
    source = _validate_source(document["source"])
    layers = _classify_layers(document["layers"], base_directory or Path.cwd())
    overall = _overall_classification(layers)
    questions = _classify_questions(document["comparison_conditions"], layers)
    counts = dict.fromkeys(CLASSIFICATIONS, 0)
    blockers: list[Record] = []
    for layer_name, layer in layers.items():
        for item in cast(list[Record], layer["items"]):
            status = cast(str, item["classification"])
            counts[status] += 1
            if item["scope"] != "excluded" and status != "SUPPORTED":
                blockers.append(
                    {
                        "layer": layer_name,
                        "item": item["id"],
                        "classification": status,
                        "reason": item["reason"],
                    }
                )
    conditions: Record = {
        "schema_version": 2,
        "tool_revision": TOOL_REVISION,
        "report_kind": "comsol_meaning_comparison_conditions",
        "inventory_id": inventory_id,
        "source": source,
        "overall_classification": overall,
        "layers": layers,
        "comparison_questions": questions,
    }
    requested_questions = [
        question for question in questions.values() if cast(bool, question["requested"])
    ]
    summary: Record = {
        "schema_version": 2,
        "tool_revision": TOOL_REVISION,
        "report_kind": "comsol_meaning_preflight_summary",
        "inventory_id": inventory_id,
        "overall_classification": overall,
        "classification_counts": counts,
        "blockers": blockers,
        "comparison_questions": {
            name: {
                "classification": question["classification"],
                "outcome": question["outcome"],
                "allowed_claim": cast(Record, question["condition"])["allowed_claim"],
                "includes_field_production_import_or_recovery_error": cast(
                    Record, question["condition"]
                )["includes_field_production_import_or_recovery_error"],
            }
            for name, question in questions.items()
        },
        "requested_questions_ready": bool(requested_questions)
        and all(question["classification"] == "SUPPORTED" for question in requested_questions),
        "questions_must_not_share_one_error_number": True,
        "golden_truth": "NOT_CLAIMED",
    }
    return conditions, summary


def require_supported_comparison(
    reference: object,
    base_directory: Path,
    question: str = "same_canonical_field_solver_parity",
    *,
    expected_model_sha256: str | None = None,
    expected_field_identity: str | None = None,
    required_observed_sha256: frozenset[str] | None = None,
) -> dict[str, str]:
    """Recheck one bound inventory before a consumer accepts its comparison.

    This validates conditions, not a numerical verdict. A previously written
    summary or a producer's status declaration cannot replace the inventory.
    """

    if question not in QUESTIONS:
        raise InventoryError(f"unknown comparison question: {question}")
    record = _mapping(reference, "meaning_preflight_inventory")
    _exact_keys(record, {"path", "sha256"}, "meaning_preflight_inventory")
    path_text = cast(str, _text(record["path"], "meaning_preflight_inventory.path"))
    digest = cast(str, _text(record["sha256"], "meaning_preflight_inventory.sha256"))
    path = (base_directory / path_text).resolve()
    try:
        actual_digest = _sha256(path)
    except OSError as error:
        raise InventoryError(f"cannot read bound meaning inventory: {path}") from error
    if actual_digest != digest:
        raise InventoryError(f"meaning inventory SHA-256 mismatch: {path}")
    conditions, _ = classify_inventory(load_inventory(path), base_directory=path.parent)
    questions = cast(dict[str, Record], conditions["comparison_questions"])
    if questions[question]["classification"] != "SUPPORTED":
        raise InventoryError(f"comparison condition is not SUPPORTED: {question}")
    _require_comparison_context(
        conditions,
        question,
        expected_model_sha256,
        expected_field_identity,
        required_observed_sha256,
    )
    return {"path": path_text, "sha256": actual_digest}


def _require_comparison_context(
    conditions: Record,
    question: str,
    expected_model_sha256: str | None,
    expected_field_identity: str | None,
    required_observed_sha256: frozenset[str] | None,
) -> None:
    source = cast(Record, conditions["source"])
    if expected_model_sha256 is not None and source["model_sha256"] != expected_model_sha256:
        raise InventoryError("meaning inventory refers to a different source model")
    selected = cast(dict[str, Record], conditions["comparison_questions"])[question]
    if expected_field_identity is not None:
        for name in ("reference_field", "candidate_field"):
            field = cast(Record, selected[name])
            if (
                field["representation"] == "canonical"
                and field["identity"] != expected_field_identity
            ):
                raise InventoryError("meaning inventory refers to a different canonical field")
    if required_observed_sha256 is not None:
        observed = {
            cast(Record, binding["observed"])["sha256"]
            for name in QUESTION_LAYERS[question]
            for item in cast(
                list[Record], cast(dict[str, Record], conditions["layers"])[name]["items"]
            )
            if item["scope"] == "required"
            for binding in cast(list[Record], item["binding"])
        }
        if not required_observed_sha256 or not required_observed_sha256.issubset(observed):
            raise InventoryError("meaning inventory is not bound to the actual run readbacks")


def build_preflight(inventory_path: str | Path, output_directory: str | Path) -> Record:
    """Classify one inventory and establish a new no-clobber report directory."""

    source_path = Path(inventory_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    conditions, summary = classify_inventory(
        load_inventory(source_path), base_directory=source_path.parent
    )
    for report in (conditions, summary):
        report["inventory_path"] = str(source_path)
        report["inventory_sha256"] = _sha256(source_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison_conditions.json").write_text(
        json.dumps(conditions, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build_preflight(args.inventory, args.output)
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0 if summary["requested_questions_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
