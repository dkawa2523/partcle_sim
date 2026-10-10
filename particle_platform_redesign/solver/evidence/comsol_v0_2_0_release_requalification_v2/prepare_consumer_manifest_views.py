"""Rebase the two registered saved-native manifests for the existing containment gate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from tools.vv.comsol.meaning_preflight import load_inventory

BASE = Path(__file__).resolve().parent
COMMON = BASE.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, str]:
    return {"path": os.path.relpath(path, BASE).replace("\\", "/"), "sha256": digest(path)}


def save(path: Path, value: object) -> None:
    if path.exists():
        raise ValueError(f"Do not overwrite evidence: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    registration = load_inventory(BASE / "registration/execution_preregistration.json")
    for case in ("A", "P"):
        source = BASE / "saved_native" / f"case{case}" / "comsol_campaign_manifest.json"
        if digest(source) != registration["c2"][case]["native_manifest"]["sha256"]:
            raise ValueError("The registered manifest changed")
        original = load_inventory(source)
        view = copy.deepcopy(original)
        changes: list[dict[str, str]] = []

        def rebase(item: dict[str, str], pointer: str) -> None:
            old_path = item["path"]
            target = (source.parent / old_path).resolve()
            if not target.is_relative_to(COMMON) or digest(target) != item["sha256"]:
                raise ValueError(f"The original referenced input does not match: {pointer}")
            new_path = os.path.relpath(target, COMMON).replace("\\", "/")
            item["path"] = new_path
            changes.append(
                {
                    "pointer": pointer,
                    "original": old_path,
                    "consumer": new_path,
                    "sha256": item["sha256"],
                }
            )

        rebase(view["meaning_preflight_inventory"], "/meaning_preflight_inventory/path")
        for level_index, level in enumerate(view["levels"]):
            for replica_index, replica in enumerate(level["replicas"]):
                for key in ("trajectory", "events", "performance", "actual_run_readback"):
                    if replica.get(key) is not None:
                        rebase(
                            replica[key],
                            f"/levels/{level_index}/replicas/{replica_index}/{key}/path",
                        )
        restored = copy.deepcopy(view)
        for change in changes:
            leaf = restored
            parts = change["pointer"].strip("/").split("/")
            for part in parts[:-1]:
                leaf = leaf[int(part)] if isinstance(leaf, list) else leaf[part]
            leaf[parts[-1]] = change["original"]
        if restored != original:
            raise ValueError("A nonpath value changed")
        target = COMMON / f"comsol_v0_2_0_release_case{case}_consumer_manifest.json"
        save(target, view)
        save(
            BASE / f"case{case}_consumer_manifest_projection_receipt.json",
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "scope": "Post-registration transport-only view after candidate outcomes. Not a preregistration or a new numerical/model observation.",
                "reason": "The existing assembler requires trajectory/event/performance artifacts inside the manifest parent. The solver/evidence common ancestor contains all unchanged registered raw inputs; no runtime containment or meaning gate is weakened.",
                "original_registered_manifest": ref(source),
                "consumer_manifest": ref(target),
                "changed_artifact_pointer_path_basis": "solver/evidence",
                "remaining_metadata_path_basis": "Unchanged original declared basis; pilot authorization remains identical to the candidate authority.",
                "changes": changes,
                "restoring_changed_path_values_is_exact_original": True,
                "all_sha256_pointer_type_value_source_model_field_observation_values_unchanged": True,
                "new_native_solve": False,
                "recipe": ref(Path(__file__).resolve()),
            },
        )
        print(json.dumps({"case": case, "consumer_manifest": ref(target)}))


if __name__ == "__main__":
    main()
