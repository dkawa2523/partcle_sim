from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _config_required(config: dict[str, Any] | None) -> list[str]:
    if not config:
        return []
    raw = config.get("required", [])
    if not isinstance(raw, list):
        raise ValueError("config.required must be a list")
    return [str(v) for v in raw]


def _config_axes(config: dict[str, Any] | None) -> list[str]:
    if not config:
        return []
    raw = config.get("axis_names", [])
    if not isinstance(raw, list):
        raise ValueError("config.axis_names must be a list")
    return [str(v) for v in raw]


def _selected_expressions(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = inventory.get("selected", {})
    if not isinstance(selected, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in selected.items():
        if isinstance(value, dict):
            out[str(key)] = value
    return out


def _config_force_models(config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not config:
        return {}
    raw = config.get("force_models", {})
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config.force_models must be an object")
    out: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if isinstance(value, bool):
            out[str(name)] = {"enabled": bool(value)}
        elif isinstance(value, dict):
            out[str(name)] = dict(value)
        else:
            raise ValueError(f"config.force_models.{name} must be a boolean or object")
    return out


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _force_required_fields(name: str, cfg: dict[str, Any], *, spatial_dim: int) -> list[str]:
    override = cfg.get("required_fields")
    if override is not None:
        if not isinstance(override, list):
            raise ValueError(f"force_models.{name}.required_fields must be a list")
        return [str(v) for v in override]
    dim = int(spatial_dim)
    if name == "thermophoresis":
        return ["T", "rho_g", "mu"]
    if name == "dielectrophoresis":
        return ["E_x", "E_y"] + (["E_z"] if dim == 3 else [])
    if name == "lift":
        return ["ux", "uy"] + (["uz"] if dim == 3 else []) + ["rho_g", "mu"]
    if name == "gravity" and _enabled(cfg.get("buoyancy", cfg.get("include_buoyancy", False))):
        return ["rho_g"]
    return []


def _force_requirements(
    config: dict[str, Any] | None,
    selected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    models = _config_force_models(config)
    spatial_dim = int((config or {}).get("spatial_dim", 2))
    out: dict[str, Any] = {}
    for name, cfg in models.items():
        enabled = _enabled(cfg.get("enabled", False))
        required = _force_required_fields(name, cfg, spatial_dim=spatial_dim)
        missing = [field for field in required if not selected.get(field, {}).get("available", False)]
        out[name] = {
            "enabled": int(enabled),
            "required_fields": required,
            "missing_fields": missing,
            "status": "pass" if not enabled or not missing else "fail",
        }
    return out


def _release_inventory_summary(raw_export_dir: Path) -> dict[str, Any]:
    path = raw_export_dir / "particle_release_inventory.json"
    if not path.exists():
        return {"available": False, "feature_count": 0}
    payload = _read_json(path)
    raw = payload.get("features", [])
    if not isinstance(raw, list):
        return {"available": True, "feature_count": 0, "invalid_shape": True}
    kinds: dict[str, int] = {}
    time_dependent = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("release_kind", "unknown"))
        kinds[kind] = kinds.get(kind, 0) + 1
        settings = item.get("known_settings", {})
        if isinstance(settings, dict) and any(
            key in settings
            for key in ("t", "t0", "t1", "tlist", "times", "release_times", "releaseTime", "trelease", "period", "frequency")
        ):
            time_dependent += 1
    return {
        "available": True,
        "feature_count": int(sum(kinds.values())),
        "by_release_kind": kinds,
        "time_dependent_feature_count": int(time_dependent),
    }


def _axis_summary(frame: pd.DataFrame, axes: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for axis in axes:
        if axis not in frame.columns:
            raise ValueError(f"field_samples.csv missing axis column: {axis}")
        values = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"axis {axis} contains non-finite values")
        unique = np.asarray(sorted(np.unique(values)), dtype=np.float64)
        if unique.size < 2:
            raise ValueError(f"axis {axis} must have at least two unique points")
        if np.any(np.diff(unique) <= 0.0):
            raise ValueError(f"axis {axis} must be strictly increasing")
        summary[axis] = {
            "count": int(unique.size),
            "min": float(unique[0]),
            "max": float(unique[-1]),
        }
    expected = 1
    for item in summary.values():
        expected *= int(item["count"])
    if len(frame) != expected:
        raise ValueError(f"field_samples.csv is not a complete tensor grid: rows={len(frame)}, expected={expected}")
    if frame.duplicated(axes).any():
        raise ValueError("field_samples.csv contains duplicate coordinate rows")
    return summary


def _single_field_stats(frame: pd.DataFrame, name: str, valid: np.ndarray) -> dict[str, Any]:
    if name not in frame.columns:
        raise ValueError(f"field_samples.csv missing field column: {name}")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    finite = values[valid]
    if not np.all(np.isfinite(finite)):
        raise ValueError(f"field {name} is non-finite on valid support")
    return {
        "finite_count": int(np.count_nonzero(np.isfinite(finite))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "variation": float(np.max(finite) - np.min(finite)),
    }


def _optional_field_stats(frame: pd.DataFrame, name: str, valid: np.ndarray) -> dict[str, Any]:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    sample = values[valid]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return {"finite_count": 0, "nonfinite_count": int(sample.size)}
    return {
        "finite_count": int(finite.size),
        "nonfinite_count": int(sample.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "variation": float(np.max(finite) - np.min(finite)),
    }


def _field_summary(frame: pd.DataFrame, axes: list[str], required: list[str], selected: list[str]) -> dict[str, Any]:
    axis_info = _axis_summary(frame, axes)
    if "valid_mask" in frame.columns:
        valid = pd.to_numeric(frame["valid_mask"], errors="coerce").to_numpy(dtype=np.float64) > 0.5
    else:
        valid = np.ones(len(frame), dtype=bool)
    if not np.any(valid):
        raise ValueError("valid_mask has no valid nodes")
    fields: dict[str, Any] = {}
    for name in required:
        fields[name] = _single_field_stats(frame, name, valid)
    selected_fields: dict[str, Any] = {}
    for name in selected:
        if name in fields or name not in frame.columns:
            continue
        selected_fields[name] = _optional_field_stats(frame, name, valid)
    return {
        "axes": axis_info,
        "row_count": int(len(frame)),
        "valid_node_count": int(np.count_nonzero(valid)),
        "required_fields": fields,
        "selected_fields": selected_fields,
    }


def validate_raw_export(raw_export_dir: str | Path, config: str | Path | None = None) -> dict[str, Any]:
    raw = Path(raw_export_dir)
    if not raw.exists():
        raise ValueError(f"raw export directory does not exist: {raw}")
    config_payload = _read_json(Path(config)) if config else None
    required = _config_required(config_payload)
    axes = _config_axes(config_payload)

    required_files = ["model_inventory.json", "export_manifest.json", "expression_inventory.json"]
    missing = [name for name in required_files if not (raw / name).exists()]
    if missing:
        raise ValueError(f"raw export missing required file(s): {', '.join(missing)}")

    manifest = _read_json(raw / "export_manifest.json")
    inventory = _read_json(raw / "expression_inventory.json")
    selected = _selected_expressions(inventory)
    unavailable = [name for name in required if not selected.get(name, {}).get("available", False)]
    if unavailable:
        raise ValueError(f"required expression(s) unavailable: {', '.join(unavailable)}")
    force_requirements = _force_requirements(config_payload, selected)
    force_failures = [
        f"{name}: {', '.join(info.get('missing_fields', []))}"
        for name, info in force_requirements.items()
        if info.get("status") == "fail"
    ]
    if force_failures:
        raise ValueError(f"enabled force model missing required expression(s): {'; '.join(force_failures)}")

    summary: dict[str, Any] = {
        "raw_export_dir": str(raw),
        "case_name": manifest.get("case_name", config_payload.get("case_name") if config_payload else ""),
        "source_kind": manifest.get("source_kind", ""),
        "required": required,
        "force_requirements": force_requirements,
        "particle_release_inventory": _release_inventory_summary(raw),
        "files": {
            name: (raw / name).exists()
            for name in required_files
            + ["physics_feature_inventory.json", "particle_release_inventory.json", "mesh.mphtxt", "field_samples.csv"]
        },
    }

    samples = raw / "field_samples.csv"
    if samples.exists():
        if not axes:
            axis_names = manifest.get("axis_names", [])
            axes = [str(v) for v in axis_names] if isinstance(axis_names, list) else []
        if not axes:
            raise ValueError("axis_names are required to validate field_samples.csv")
        frame = pd.read_csv(samples)
        selected_names = [name for name, info in selected.items() if info.get("available", False)]
        summary["field_samples"] = _field_summary(frame, axes, required, selected_names)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a raw external COMSOL particle export.")
    parser.add_argument("--raw-export-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = validate_raw_export(args.raw_export_dir, args.config)
    text = json.dumps(summary, indent=2)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0
