from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .data_export import _COLUMN_RE, _read_rows


CANONICAL_RELEASE_COLUMNS = (
    "particle_id",
    "release_time",
    "x",
    "y",
    "v_x",
    "v_y",
    "source_entity",
    "source_part_id",
    "diameter",
    "density",
    "mass",
    "charge",
)

CANONICAL_WALL_EVENT_COLUMNS = (
    "particle_id",
    "hit_time_s",
    "comsol_entity_id",
    "outcome",
    "hit_x",
    "hit_y",
    "normal_x",
    "normal_y",
    "v_hit_x",
    "v_hit_y",
)

CANONICAL_PARTICLE_STATUS_COLUMNS = (
    "particle_id",
    "stop_time_s",
    "final_status_code",
    "final_status",
)

PROPERTY_PROBES: dict[str, tuple[str, str, float]] = {
    "fpt.dp": ("diameter", "length", 1.0),
    "fpt.rp": ("diameter", "length", 2.0),
    "fpt.rhop": ("density", "density", 1.0),
    "fpt.mp": ("mass", "mass", 1.0),
    "fpt.qp": ("charge", "charge", 1.0),
    "fpt.source": ("source_entity", "identity", 1.0),
    "fpt.relid": ("source_part_id", "identity", 1.0),
    "fpt.inl": ("source_part_id", "identity", 1.0),
}

WALL_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "particle_id": ("particle_id", "ParticleID", "id", "pid", "particle"),
    "hit_time_s": ("hit_time_s", "time_s", "time", "t", "event_time_s", "wall_time_s"),
    "comsol_entity_id": (
        "comsol_entity_id",
        "comsol_geom_entity_id",
        "entity_id",
        "boundary_id",
        "wall_id",
        "bnd",
        "fpt.bnd",
        "fpt.bid",
        "fpt.wallid",
    ),
    "outcome": ("outcome", "wall_outcome", "status", "fpt.status", "fpt.pstatus"),
    "hit_x": ("hit_x", "x", "x_m", "event_x", "wall_x"),
    "hit_y": ("hit_y", "y", "y_m", "event_y", "wall_y"),
    "normal_x": ("normal_x", "nx", "n_x", "fpt.nx"),
    "normal_y": ("normal_y", "ny", "n_y", "fpt.ny"),
    "v_hit_x": ("v_hit_x", "vx", "v_x", "hit_vx", "hit_v_x", "fpt.vx"),
    "v_hit_y": ("v_hit_y", "vy", "v_y", "hit_vy", "hit_v_y", "fpt.vy"),
}

WALL_EVENT_PROBES: dict[str, tuple[str, str]] = {
    "fpt.bnd": ("comsol_entity_id", "identity"),
    "fpt.bid": ("comsol_entity_id", "identity"),
    "fpt.boundary": ("comsol_entity_id", "identity"),
    "fpt.wall": ("comsol_entity_id", "identity"),
    "fpt.wallid": ("comsol_entity_id", "identity"),
    "fpt.status": ("outcome", "identity"),
    "fpt.pstatus": ("outcome", "identity"),
    "fpt.event": ("outcome", "identity"),
    "fpt.freeze": ("outcome", "identity"),
    "fpt.stick": ("outcome", "identity"),
    "fpt.nx": ("normal_x", "identity"),
    "fpt.ny": ("normal_y", "identity"),
}

STATUS_STOP_TIME_EXPRESSION = "fpt.st"
STATUS_FINAL_STATE_PROBES = ("fpt.fs", "particlestatus", "fpt.particlestatus")
STATUS_PROBES = (STATUS_STOP_TIME_EXPRESSION, *STATUS_FINAL_STATE_PROBES)
STATUS_OUTCOME_MAP = {
    0: "unreleased",
    1: "active",
    2: "frozen",
    3: "stuck",
    4: "disappeared",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _first_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for alias in aliases:
        found = lower.get(str(alias).strip().lower())
        if found is not None:
            return found
    return None


def _as_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _unit_scale(unit: str, kind: str) -> float:
    text = (
        str(unit)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("ﾂｵ", "u")
    )
    if not text:
        return 1.0
    length = {"m": 1.0, "mm": 1.0e-3, "um": 1.0e-6, "micrometer": 1.0e-6, "nm": 1.0e-9}
    mass = {"kg": 1.0, "g": 1.0e-3, "mg": 1.0e-6, "ug": 1.0e-9}
    density = {
        "kg/m^3": 1.0,
        "kg/m3": 1.0,
        "g/cm^3": 1000.0,
        "g/cm3": 1000.0,
    }
    charge = {"c": 1.0, "coulomb": 1.0}
    if kind == "length":
        return length.get(text, 1.0)
    if kind == "mass":
        return mass.get(text, 1.0)
    if kind == "density":
        return density.get(text, 1.0)
    if kind == "charge":
        return charge.get(text, 1.0)
    return 1.0


_QUANTITY_RE = re.compile(r"^\s*(?P<value>[-+0-9.eE]+)\s*(?:\[(?P<unit>[^\]]+)\])?\s*$")


def _parse_quantity(value: Any, *, kind: str) -> float:
    if isinstance(value, (int, float, np.floating)):
        return _as_number(value)
    match = _QUANTITY_RE.match(str(value).strip())
    if not match:
        return math.nan
    number = _as_number(match.group("value"))
    if not math.isfinite(number):
        return math.nan
    return number * _unit_scale(match.group("unit") or "", kind)


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _feature_values(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    values = feature.get("property_values", {})
    if isinstance(values, Mapping):
        return values
    values = feature.get("known_settings", {})
    return values if isinstance(values, Mapping) else {}


def particle_property_defaults(inventory_json: str | Path | None) -> dict[str, Any]:
    """Return COMSOL particle-property defaults with explicit source metadata."""

    payload = _read_json_if_exists(Path(inventory_json) if inventory_json is not None else None)
    features = payload.get("features", [])
    if not isinstance(features, list):
        features = []
    property_features = [
        feature
        for feature in features
        if isinstance(feature, Mapping) and str(feature.get("release_kind", "")) == "particle_properties"
    ]
    if not property_features:
        return {"available": False, "values": {}, "sources": {}, "feature_tag": ""}

    feature = property_features[0]
    values = dict(_feature_values(feature))
    spec = str(values.get("ParticlePropertySpec", "")).strip()
    diameter = _parse_quantity(values.get("dp"), kind="length")
    density = _parse_quantity(values.get("rhop"), kind="density")
    mass_config = _parse_quantity(values.get("mp"), kind="mass")
    mass = mass_config
    mass_source = "mp"
    if spec == "SpecifyDensityAndDiameter" and math.isfinite(diameter) and math.isfinite(density):
        mass = float(math.pi / 6.0 * density * diameter**3)
        mass_source = "derived_from_density_diameter"

    charge = _parse_quantity(values.get("qp", values.get("q")), kind="charge")
    charge_source = "qp"
    if not math.isfinite(charge):
        z = _parse_quantity(values.get("Z"), kind="identity")
        if math.isfinite(z):
            charge = z * 1.602176634e-19
            charge_source = "Z_elementary_charge"

    out_values = {
        "diameter": diameter,
        "density": density,
        "mass": mass,
        "charge": charge,
    }
    sources = {
        "diameter": "dp",
        "density": "rhop",
        "mass": mass_source,
        "charge": charge_source,
    }
    finite_values = {key: value for key, value in out_values.items() if math.isfinite(float(value))}
    finite_sources = {key: sources[key] for key in finite_values}
    return {
        "available": bool(finite_values),
        "feature_tag": str(feature.get("feature_tag", "")),
        "particle_property_spec": spec,
        "values": finite_values,
        "sources": finite_sources,
        "raw_values": values,
    }


def _release_features(inventory_json: str | Path | None) -> list[Mapping[str, Any]]:
    payload = _read_json_if_exists(Path(inventory_json) if inventory_json is not None else None)
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []
    return [
        feature
        for feature in features
        if isinstance(feature, Mapping)
        and str(feature.get("release_kind", "")).strip().lower() in {"release", "release_grid"}
    ]


def _release_source_candidates(
    *,
    inventory_json: str | Path | None,
    boundary_map_csv: str | Path | None,
) -> list[dict[str, Any]]:
    if inventory_json is None or boundary_map_csv is None or not Path(boundary_map_csv).exists():
        return []
    boundary = pd.read_csv(Path(boundary_map_csv))
    candidates: list[dict[str, Any]] = []
    for feature in _release_features(inventory_json):
        for raw_entity in list(feature.get("selection_entities", []) or []):
            entity = _as_number(raw_entity)
            if not math.isfinite(entity):
                continue
            entity_id = int(entity)
            match = pd.DataFrame()
            for col in ("comsol_api_selection_entity_id", "comsol_edge_entity_id", "raw_comsol_edge_entity_index"):
                if col not in boundary.columns:
                    continue
                values = pd.to_numeric(boundary[col], errors="coerce")
                match = boundary[values == entity_id]
                if not match.empty:
                    break
            for _, row in match.iterrows():
                try:
                    candidates.append(
                        {
                            "source_entity": entity_id,
                            "source_part_id": int(row["solver_part_id"]),
                            "feature_tag": str(feature.get("feature_tag", "")),
                            "feature_label": str(feature.get("label", "")),
                            "x_min_m": float(row["x_min_m"]),
                            "x_max_m": float(row["x_max_m"]),
                            "y_min_m": float(row["y_min_m"]),
                            "y_max_m": float(row["y_max_m"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    return candidates


def _bbox_distance_m(x: float, y: float, candidate: Mapping[str, Any]) -> float:
    x_min = float(candidate["x_min_m"])
    x_max = float(candidate["x_max_m"])
    y_min = float(candidate["y_min_m"])
    y_max = float(candidate["y_max_m"])
    dx = max(x_min - x, 0.0, x - x_max)
    dy = max(y_min - y, 0.0, y - y_max)
    return float(math.hypot(dx, dy))


def _fill_release_source_from_inventory(
    frame: pd.DataFrame,
    *,
    inventory_json: str | Path | None,
    boundary_map_csv: str | Path | None,
    tolerance_m: float = 5.0e-4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    candidates = _release_source_candidates(inventory_json=inventory_json, boundary_map_csv=boundary_map_csv)
    report: dict[str, Any] = {
        "available": bool(candidates),
        "method": "release_feature_selection_bbox",
        "boundary_map_csv": str(boundary_map_csv) if boundary_map_csv is not None else "",
        "candidate_count": int(len(candidates)),
        "assigned_count": 0,
        "unassigned_count": 0,
        "ambiguous_count": 0,
        "tolerance_m": float(tolerance_m),
        "assigned_distance_m": {"count": 0},
        "source_part_counts": {},
    }
    if not candidates or "x" not in out.columns or "y" not in out.columns:
        report["unassigned_count"] = int(len(out))
        return out, report

    source_entity = pd.to_numeric(out.get("source_entity", pd.Series(np.nan, index=out.index)), errors="coerce")
    source_part = pd.to_numeric(out.get("source_part_id", pd.Series(np.nan, index=out.index)), errors="coerce")
    needs_source = ~np.isfinite(source_entity.to_numpy(dtype=np.float64)) | ~np.isfinite(source_part.to_numpy(dtype=np.float64))
    assigned_parts: list[int] = []
    assigned_distances: list[float] = []
    ambiguous = 0
    unassigned = 0
    for idx in out.index[needs_source]:
        x = _as_number(out.at[idx, "x"])
        y = _as_number(out.at[idx, "y"])
        if not math.isfinite(x) or not math.isfinite(y):
            unassigned += 1
            continue
        distances = sorted(
            ((_bbox_distance_m(x, y, candidate), candidate) for candidate in candidates),
            key=lambda item: item[0],
        )
        inside = [(distance, candidate) for distance, candidate in distances if distance <= float(tolerance_m)]
        if len(inside) != 1:
            if len(inside) > 1:
                ambiguous += 1
            else:
                unassigned += 1
            continue
        _, candidate = inside[0]
        out.at[idx, "source_entity"] = int(candidate["source_entity"])
        out.at[idx, "source_part_id"] = int(candidate["source_part_id"])
        assigned_parts.append(int(candidate["source_part_id"]))
        assigned_distances.append(float(inside[0][0]))

    report["assigned_count"] = int(len(assigned_parts))
    report["unassigned_count"] = int(unassigned)
    report["ambiguous_count"] = int(ambiguous)
    report["source_part_counts"] = {
        str(part): int(assigned_parts.count(part)) for part in sorted(set(assigned_parts))
    }
    if assigned_distances:
        arr = np.asarray(assigned_distances, dtype=np.float64)
        report["assigned_distance_m"] = {
            "count": int(arr.size),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "p99": float(np.quantile(arr, 0.99)),
        }
    report["exact_parity_safe"] = bool(report["assigned_count"]) and report["unassigned_count"] == 0 and report["ambiguous_count"] == 0
    return out, report


def _read_probe_first_values(path: Path, expression: str, output_name: str, unit_kind: str, multiplier: float) -> pd.DataFrame:
    metadata, header, raw_rows = _read_rows(path)
    del metadata
    columns = []
    for index, label in enumerate(header[1:], start=1):
        match = _COLUMN_RE.match(str(label))
        if not match:
            continue
        if match.group("name").strip().lower() != str(expression).strip().lower():
            continue
        columns.append(
            {
                "column_index": index,
                "unit": (match.group("unit") or "").strip(),
                "time_s": float(match.group("time")),
            }
        )
    if not columns:
        return pd.DataFrame(columns=["particle_id", output_name])
    columns = sorted(columns, key=lambda item: float(item["time_s"]))
    rows: list[dict[str, float | int]] = []
    for raw in raw_rows:
        if not raw:
            continue
        particle_id = _as_number(raw[0])
        if not math.isfinite(particle_id):
            continue
        value = math.nan
        for column in columns:
            idx = int(column["column_index"])
            if idx >= len(raw):
                continue
            candidate = _as_number(raw[idx])
            if math.isfinite(candidate):
                value = candidate * _unit_scale(str(column["unit"]), unit_kind) * float(multiplier)
                break
        if math.isfinite(value):
            rows.append({"particle_id": int(particle_id), output_name: float(value)})
    return pd.DataFrame(rows, columns=["particle_id", output_name])


def _probe_columns(path: Path, expression: str) -> tuple[list[dict[str, Any]], list[list[str]]]:
    _, header, raw_rows = _read_rows(path)
    columns: list[dict[str, Any]] = []
    for index, label in enumerate(header[1:], start=1):
        match = _COLUMN_RE.match(str(label))
        if not match:
            continue
        if match.group("name").strip().lower() != str(expression).strip().lower():
            continue
        columns.append(
            {
                "column_index": index,
                "unit": (match.group("unit") or "").strip(),
                "time_s": float(match.group("time")),
            }
        )
    columns = sorted(columns, key=lambda item: float(item["time_s"]))
    return columns, raw_rows


def _probe_finite_report(path: Path, expression: str) -> dict[str, Any]:
    columns, raw_rows = _probe_columns(path, expression)
    finite_rows = 0
    finite_values = 0
    first_particle_id: int | None = None
    first_time_s: float | None = None
    first_value: float | None = None
    for raw in raw_rows:
        if not raw:
            continue
        particle_id = _as_number(raw[0])
        row_has_value = False
        for column in columns:
            idx = int(column["column_index"])
            if idx >= len(raw):
                continue
            value = _as_number(raw[idx])
            if not math.isfinite(value):
                continue
            finite_values += 1
            row_has_value = True
            if first_particle_id is None:
                first_particle_id = int(particle_id) if math.isfinite(particle_id) else None
                first_time_s = float(column["time_s"])
                first_value = float(value)
        if row_has_value:
            finite_rows += 1
    return {
        "path": str(path),
        "expression": expression,
        "time_column_count": int(len(columns)),
        "particle_row_count": int(len(raw_rows)),
        "finite_row_count": int(finite_rows),
        "finite_value_count": int(finite_values),
        "first_particle_id": first_particle_id,
        "first_time_s": first_time_s,
        "first_value": first_value,
    }


def _probe_expression_from_name(path: Path) -> str | None:
    stem = path.stem
    if stem.startswith("probe_"):
        token = stem.removeprefix("probe_")
        if token.startswith("fpt_"):
            return "fpt." + token.removeprefix("fpt_")
        if token == "particlestatus":
            return "particlestatus"
    return None


def _probe_expressions(path: Path, candidates: Iterable[str]) -> list[str]:
    candidate_by_lower = {str(item).strip().lower(): str(item) for item in candidates}
    found: list[str] = []
    try:
        _, header, _ = _read_rows(path)
        for label in header[1:]:
            match = _COLUMN_RE.match(str(label))
            if not match:
                continue
            expression = candidate_by_lower.get(match.group("name").strip().lower())
            if expression is not None and expression not in found:
                found.append(expression)
    except Exception:  # noqa: BLE001 - filename fallback keeps failed probes reportable
        pass
    named = _probe_expression_from_name(path)
    if named is not None and named in set(candidates) and named not in found:
        found.append(named)
    return found


def _discover_property_probe_tables(reextract_root: Path) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    tables: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    if not reextract_root.exists():
        return tables, reports
    for csv_path in sorted(reextract_root.rglob("*.csv")):
        for expression in _probe_expressions(csv_path, PROPERTY_PROBES.keys()):
            output, kind, multiplier = PROPERTY_PROBES[expression]
            try:
                frame = _read_probe_first_values(csv_path, expression, output, kind, multiplier)
                status = "promoted" if not frame.empty else "no_finite_values"
            except Exception as exc:  # noqa: BLE001 - probe failures are recorded, not fatal
                frame = pd.DataFrame(columns=["particle_id", output])
                status = "failed"
                reports.append({"path": str(csv_path), "expression": expression, "output": output, "status": status, "error": str(exc)})
                continue
            reports.append(
                {
                    "path": str(csv_path),
                    "expression": expression,
                    "output": output,
                    "status": status,
                    "row_count": int(len(frame)),
                }
            )
            if not frame.empty:
                tables.append(frame)
    return tables, reports


def _discover_wall_probe_reports(reextract_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not reextract_root.exists():
        return reports
    for csv_path in sorted(reextract_root.rglob("*.csv")):
        for expression in _probe_expressions(csv_path, WALL_EVENT_PROBES.keys()):
            try:
                report = _probe_finite_report(csv_path, expression)
                report["status"] = "has_finite_values" if int(report.get("finite_value_count", 0)) else "no_finite_values"
            except Exception as exc:  # noqa: BLE001 - probe failures are recorded, not fatal
                report = {
                    "path": str(csv_path),
                    "expression": expression,
                    "status": "failed",
                    "error": str(exc),
                }
            reports.append(report)
    return reports


def _discover_status_probe_reports(reextract_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not reextract_root.exists():
        return reports
    for csv_path in sorted(reextract_root.rglob("*.csv")):
        for expression in _probe_expressions(csv_path, STATUS_PROBES):
            try:
                report = _probe_finite_report(csv_path, str(expression))
                report["status"] = "has_finite_values" if int(report.get("finite_value_count", 0)) else "no_finite_values"
            except Exception as exc:  # noqa: BLE001 - probe failures are recorded, not fatal
                report = {
                    "path": str(csv_path),
                    "expression": expression,
                    "status": "failed",
                    "error": str(exc),
                }
            reports.append(report)
    return reports


def _wide_wall_probe_first_hits(reextract_root: Path) -> pd.DataFrame:
    probe_paths: dict[str, Path] = {}
    for csv_path in sorted(reextract_root.rglob("*.csv")):
        for expression in _probe_expressions(csv_path, WALL_EVENT_PROBES.keys()):
            probe_paths[str(expression)] = csv_path
    if not probe_paths:
        return pd.DataFrame(columns=CANONICAL_WALL_EVENT_COLUMNS)

    first_hits: dict[int, dict[str, Any]] = {}
    for expression, csv_path in probe_paths.items():
        canonical, unit_kind = WALL_EVENT_PROBES[expression]
        columns, raw_rows = _probe_columns(csv_path, expression)
        if not columns:
            continue
        for raw in raw_rows:
            if not raw:
                continue
            particle_id = _as_number(raw[0])
            if not math.isfinite(particle_id):
                continue
            pid = int(particle_id)
            for column in columns:
                idx = int(column["column_index"])
                if idx >= len(raw):
                    continue
                value = _as_number(raw[idx])
                if not math.isfinite(value):
                    continue
                hit = first_hits.setdefault(pid, {"particle_id": pid, "hit_time_s": float(column["time_s"])})
                if float(column["time_s"]) < float(hit["hit_time_s"]):
                    hit.clear()
                    hit.update({"particle_id": pid, "hit_time_s": float(column["time_s"])})
                if abs(float(column["time_s"]) - float(hit["hit_time_s"])) > 1.0e-12:
                    continue
                scaled = value * _unit_scale(str(column["unit"]), unit_kind)
                hit[canonical] = str(scaled) if canonical == "outcome" else float(scaled)
                break

    if not first_hits:
        return pd.DataFrame(columns=CANONICAL_WALL_EVENT_COLUMNS)
    rows = pd.DataFrame(first_hits.values())
    for col in CANONICAL_WALL_EVENT_COLUMNS:
        if col not in rows.columns:
            rows[col] = "" if col == "outcome" else np.nan
    rows["particle_id"] = pd.to_numeric(rows["particle_id"], errors="raise").astype(int)
    rows = rows.sort_values(["particle_id", "hit_time_s"], kind="mergesort").reset_index(drop=True)
    return rows[list(CANONICAL_WALL_EVENT_COLUMNS)]


def _read_status_code_values(path: Path, expression: str) -> pd.DataFrame:
    columns, raw_rows = _probe_columns(path, expression)
    if not columns:
        return pd.DataFrame(columns=["particle_id", "status_code"])
    rows: list[dict[str, float | int]] = []
    for raw in raw_rows:
        if not raw:
            continue
        particle_id = _as_number(raw[0])
        if not math.isfinite(particle_id):
            continue
        finite_values: list[float] = []
        for column in columns:
            idx = int(column["column_index"])
            if idx >= len(raw):
                continue
            value = _as_number(raw[idx])
            if math.isfinite(value):
                finite_values.append(float(value))
        if not finite_values:
            continue
        terminal = [value for value in finite_values if int(round(value)) in {2, 3, 4}]
        status_code = terminal[0] if terminal else finite_values[-1]
        rows.append({"particle_id": int(particle_id), "status_code": float(status_code)})
    return pd.DataFrame(rows, columns=["particle_id", "status_code"])


def _status_stop_time_table(reextract_root: Path) -> pd.DataFrame:
    probe_paths: dict[str, Path] = {}
    for csv_path in sorted(reextract_root.rglob("*.csv")):
        for expression in _probe_expressions(csv_path, STATUS_PROBES):
            probe_paths[str(expression)] = csv_path
    stop_path = probe_paths.get(STATUS_STOP_TIME_EXPRESSION)
    if stop_path is None:
        return pd.DataFrame(columns=CANONICAL_PARTICLE_STATUS_COLUMNS)

    stops = _read_probe_first_values(
        stop_path,
        STATUS_STOP_TIME_EXPRESSION,
        "stop_time_s",
        "identity",
        1.0,
    )
    if stops.empty:
        return pd.DataFrame(columns=CANONICAL_PARTICLE_STATUS_COLUMNS)
    stops = stops[np.isfinite(pd.to_numeric(stops["stop_time_s"], errors="coerce").to_numpy(dtype=np.float64))]
    if stops.empty:
        return pd.DataFrame(columns=CANONICAL_PARTICLE_STATUS_COLUMNS)

    status_values = pd.DataFrame(columns=["particle_id", "status_code"])
    for expression in STATUS_FINAL_STATE_PROBES:
        path = probe_paths.get(expression)
        if path is None:
            continue
        status_values = _read_status_code_values(path, expression)
        if not status_values.empty:
            break

    rows = stops.copy()
    if not status_values.empty:
        rows = rows.merge(status_values.drop_duplicates("particle_id"), on="particle_id", how="left")
        rows["final_status"] = [
            STATUS_OUTCOME_MAP.get(int(round(value)), str(value)) if math.isfinite(_as_number(value)) else ""
            for value in rows["status_code"]
        ]
    else:
        rows["status_code"] = np.nan
        rows["final_status"] = ""
    rows = rows.rename(columns={"status_code": "final_status_code"})
    for col in CANONICAL_PARTICLE_STATUS_COLUMNS:
        if col not in rows.columns:
            rows[col] = "" if col == "final_status" else np.nan
    rows["particle_id"] = pd.to_numeric(rows["particle_id"], errors="raise").astype(int)
    rows = rows.sort_values(["particle_id", "stop_time_s"], kind="mergesort").reset_index(drop=True)
    return rows[list(CANONICAL_PARTICLE_STATUS_COLUMNS)]


def _merge_property_tables(base: pd.DataFrame, tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for table in tables:
        if table.empty or "particle_id" not in table.columns:
            continue
        value_cols = [col for col in table.columns if col != "particle_id"]
        for col in value_cols:
            slim = table[["particle_id", col]].dropna().drop_duplicates("particle_id")
            if col in out.columns:
                out = out.merge(slim.rename(columns={col: f"__probe_{col}"}), on="particle_id", how="left")
                out[col] = out[f"__probe_{col}"].combine_first(out[col])
                out = out.drop(columns=[f"__probe_{col}"])
            else:
                out = out.merge(slim, on="particle_id", how="left")
    return out


def _missing_value_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    missing = []
    for col in columns:
        if col not in frame.columns:
            missing.append(col)
            continue
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                missing.append(col)
        else:
            if not series.fillna("").astype(str).str.strip().ne("").all():
                missing.append(col)
    return missing


def promote_release_truth(
    *,
    baseline_release_csv: str | Path,
    out_csv: str | Path,
    reextract_root: str | Path | None = None,
    particle_release_inventory_json: str | Path | None = None,
    boundary_map_csv: str | Path | None = None,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    baseline = pd.read_csv(Path(baseline_release_csv))
    if "particle_id" not in baseline.columns:
        raise ValueError("baseline release table must contain particle_id")
    work = baseline.copy()
    for col in CANONICAL_RELEASE_COLUMNS:
        if col not in work.columns:
            work[col] = np.nan

    probe_reports: list[dict[str, Any]] = []
    if reextract_root is not None:
        probe_tables, probe_reports = _discover_property_probe_tables(Path(reextract_root))
        work = _merge_property_tables(work, probe_tables)

    defaults = particle_property_defaults(particle_release_inventory_json)
    default_values = defaults.get("values", {}) if isinstance(defaults.get("values", {}), Mapping) else {}
    for col in ("diameter", "density", "mass", "charge"):
        value = default_values.get(col)
        if value is None:
            continue
        numeric = pd.to_numeric(work[col], errors="coerce")
        work[col] = numeric.fillna(float(value))

    source_assignment_report = {"available": False}
    if particle_release_inventory_json is not None and boundary_map_csv is not None:
        work, source_assignment_report = _fill_release_source_from_inventory(
            work,
            inventory_json=particle_release_inventory_json,
            boundary_map_csv=boundary_map_csv,
        )

    work = work[list(CANONICAL_RELEASE_COLUMNS)].copy()
    work = work.sort_values(["particle_id"], kind="mergesort").reset_index(drop=True)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(out, index=False)

    missing = _missing_value_columns(work, CANONICAL_RELEASE_COLUMNS)
    report = {
        "source_kind": "external_comsol_particle_export_promoted_release_truth",
        "baseline_release_csv": str(baseline_release_csv),
        "output_csv": str(out),
        "row_count": int(len(work)),
        "columns": [str(col) for col in work.columns],
        "missing_columns_or_values": missing,
        "exact_parity_ready": not missing,
        "property_defaults": defaults,
        "source_assignment": source_assignment_report,
        "probe_reports": probe_reports,
        "interpretation": "Canonical release truth. Inward-clean solver releases must not be promoted here.",
    }
    if report_json is not None:
        _write_json(Path(report_json), report)
    return report


def is_wall_event_table(path: str | Path) -> bool:
    try:
        frame = pd.read_csv(Path(path), nrows=5)
    except Exception:  # noqa: BLE001 - candidate scan should be tolerant
        return False
    columns = set(str(col) for col in frame.columns)
    required = (
        _first_column(columns, WALL_EVENT_ALIASES["particle_id"]),
        _first_column(columns, WALL_EVENT_ALIASES["hit_time_s"]),
    )
    if any(col is None for col in required):
        return False
    has_event_meaning = any(
        _first_column(columns, WALL_EVENT_ALIASES[name]) is not None
        for name in ("comsol_entity_id", "outcome", "normal_x", "normal_y")
    )
    return bool(has_event_meaning)


def canonicalize_wall_event_table(path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(Path(path))
    if not is_wall_event_table(path):
        raise ValueError(f"CSV does not satisfy wall-event schema: {path}")
    out: dict[str, Any] = {}
    for canonical, aliases in WALL_EVENT_ALIASES.items():
        col = _first_column(source.columns, aliases)
        if col is None:
            out[canonical] = "" if canonical == "outcome" else np.nan
            continue
        if canonical == "outcome":
            out[canonical] = source[col].fillna("").astype(str)
        else:
            out[canonical] = pd.to_numeric(source[col], errors="coerce")
    frame = pd.DataFrame(out)
    frame = frame[np.isfinite(pd.to_numeric(frame["particle_id"], errors="coerce").to_numpy(dtype=np.float64))]
    frame = frame[np.isfinite(pd.to_numeric(frame["hit_time_s"], errors="coerce").to_numpy(dtype=np.float64))]
    frame["particle_id"] = pd.to_numeric(frame["particle_id"], errors="raise").astype(int)
    frame = frame.sort_values(["particle_id", "hit_time_s"], kind="mergesort").reset_index(drop=True)
    return frame[list(CANONICAL_WALL_EVENT_COLUMNS)]


def promote_particle_status_truth(
    *,
    reextract_root: str | Path,
    out_csv: str | Path,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(reextract_root)
    status_probe_reports = _discover_status_probe_reports(root)
    rows = _status_stop_time_table(root) if root.exists() else pd.DataFrame(columns=CANONICAL_PARTICLE_STATUS_COLUMNS)
    promoted = not rows.empty
    if promoted:
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
    report = {
        "source_kind": "external_comsol_particle_export_promoted_particle_status_truth",
        "reextract_root": str(root),
        "output_csv": str(out_csv),
        "promoted": promoted,
        "promotion_kind": "status_stop_time_probe" if promoted else "",
        "status_probe_candidate_count": int(len(status_probe_reports)),
        "status_probe_reports": status_probe_reports,
        "row_count": int(len(rows)),
        "interpretation": (
            "Particle status/stop-time truth from fpt.st/fpt.fs. This is not a "
            "wall-hit entity/normal table and must not be used as direct boundary-event truth."
        ),
    }
    if report_json is not None:
        _write_json(Path(report_json), report)
    return report


def promote_wall_event_truth(
    *,
    reextract_root: str | Path,
    out_csv: str | Path,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(reextract_root)
    candidates = [path for path in sorted(root.rglob("*.csv")) if is_wall_event_table(path)] if root.exists() else []
    promoted = False
    rows = pd.DataFrame(columns=CANONICAL_WALL_EVENT_COLUMNS)
    source_csv = ""
    probe_reports = _discover_wall_probe_reports(root)
    promotion_kind = ""
    if candidates:
        source_csv = str(candidates[0])
        rows = canonicalize_wall_event_table(candidates[0])
        promotion_kind = "schema_table"
        promoted = True
    else:
        rows = _wide_wall_probe_first_hits(root) if root.exists() else rows
        if not rows.empty:
            promotion_kind = "wide_probe_first_hit"
            promoted = True
    if promoted:
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
    else:
        out = Path(out_csv)
        if out.exists():
            out.unlink()
    report = {
        "source_kind": "external_comsol_particle_export_promoted_wall_event_truth",
        "reextract_root": str(root),
        "output_csv": str(out_csv),
        "promoted": promoted,
        "promotion_kind": promotion_kind,
        "source_csv": source_csv,
        "candidate_count": int(len(candidates)),
        "wide_probe_candidate_count": int(len(probe_reports)),
        "wide_probe_reports": probe_reports,
        "row_count": int(len(rows)),
        "required_schema": {
            "required": ["particle_id", "hit_time_s"],
            "event_meaning_one_of": ["comsol_entity_id", "outcome", "normal_x", "normal_y"],
        },
        "interpretation": (
            "Direct wall-event truth requires hit time plus entity/outcome/normal data. "
            "Particle status/stop-time probes are promoted separately."
        ),
    }
    if report_json is not None:
        _write_json(Path(report_json), report)
    return report


def promote_reextract_outputs(
    *,
    reextract_root: str | Path,
    baseline_release_csv: str | Path,
    out_dir: str | Path,
    particle_release_inventory_json: str | Path | None = None,
    boundary_map_csv: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    release_report = promote_release_truth(
        baseline_release_csv=baseline_release_csv,
        reextract_root=reextract_root,
        particle_release_inventory_json=particle_release_inventory_json,
        boundary_map_csv=boundary_map_csv,
        out_csv=out / "comsol_release_particles_canonical.csv",
        report_json=out / "comsol_release_promotion_report.json",
    )
    wall_report = promote_wall_event_truth(
        reextract_root=reextract_root,
        out_csv=out / "comsol_wall_events.csv",
        report_json=out / "comsol_wall_event_promotion_report.json",
    )
    status_report = promote_particle_status_truth(
        reextract_root=reextract_root,
        out_csv=out / "comsol_particle_status.csv",
        report_json=out / "comsol_particle_status_promotion_report.json",
    )
    summary = {
        "source_kind": "external_comsol_particle_export_reextract_promotion",
        "reextract_root": str(reextract_root),
        "out_dir": str(out),
        "release": release_report,
        "wall_events": wall_report,
        "particle_status": status_report,
        "ready_inputs": {
            "release_exact_parity_ready": bool(release_report.get("exact_parity_ready", False)),
            "wall_event_truth_ready": bool(wall_report.get("promoted", False)),
            "particle_status_truth_ready": bool(status_report.get("promoted", False)),
        },
    }
    _write_json(out / "promotion_summary.json", summary)
    return summary


__all__ = (
    "CANONICAL_RELEASE_COLUMNS",
    "CANONICAL_WALL_EVENT_COLUMNS",
    "CANONICAL_PARTICLE_STATUS_COLUMNS",
    "canonicalize_wall_event_table",
    "is_wall_event_table",
    "particle_property_defaults",
    "promote_particle_status_truth",
    "promote_reextract_outputs",
    "promote_release_truth",
    "promote_wall_event_truth",
)
