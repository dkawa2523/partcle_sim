from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .field_bundle import build_field_bundle_from_table, load_json_mapping, write_field_bundle, write_json

DEFAULT_GEOMETRY_MODEL_UNIT = "cm"
DEFAULT_GEOMETRY_SCALE_M_PER_MODEL_UNIT = 0.01
WALL_OVERRIDE_COLUMNS = [
    "part_id",
    "part_name",
    "material_id",
    "material_name",
    "wall_law",
    "wall_restitution",
    "wall_diffuse_fraction",
    "wall_stick_probability",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json_mapping(path)


def _hash_manifest_inputs(raw_export_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("export_manifest.json", "expression_inventory.json"):
        path = raw_export_dir / name
        if path.exists():
            out[name] = str(path.resolve())
    return out


def _uniform_step(axis: np.ndarray, name: str) -> float:
    arr = np.asarray(axis, dtype=np.float64)
    diffs = np.diff(arr)
    if diffs.size == 0:
        raise ValueError(f"{name} axis must contain at least two points")
    step = float(np.median(diffs))
    if not np.allclose(diffs, step, atol=max(abs(step) * 1.0e-9, 1.0e-12), rtol=1.0e-9):
        raise ValueError(f"{name} axis must be uniformly spaced for the current mphtxt builder")
    return step


def _selected_geometry_scale(manifest: Mapping[str, Any], override: float | None) -> float:
    value = float(
        override
        if override is not None
        else manifest.get("geometry_scale_m_per_model_unit", DEFAULT_GEOMETRY_SCALE_M_PER_MODEL_UNIT)
    )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("geometry_scale_m_per_model_unit must be a positive finite value")
    return value


def _infer_diagnostic_spacing(bundle: Mapping[str, np.ndarray]) -> float:
    dr = _uniform_step(np.asarray(bundle["axis_0"], dtype=np.float64), "r")
    dz = _uniform_step(np.asarray(bundle["axis_1"], dtype=np.float64), "z")
    if not np.isclose(dr, dz, atol=max(abs(dr) * 1.0e-8, abs(dz) * 1.0e-8, 1.0e-12), rtol=1.0e-8):
        raise ValueError(
            "current tools/build_comsol_case.py uses one diagnostic-grid-spacing-m for both axes; "
            f"got dr={dr:g}, dz={dz:g}"
        )
    return float(0.5 * (dr + dz))


def _run(cmd: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def _copy_raw_manifests(raw_export_dir: Path, generated_dir: Path) -> None:
    for name in ("export_manifest.json", "expression_inventory.json", "material_inventory.json"):
        src = raw_export_dir / name
        if src.exists():
            shutil.copy2(src, generated_dir / name)


def _material_name(material: Mapping[str, Any]) -> str:
    for key in ("label", "name", "tag"):
        value = str(material.get(key, "")).strip()
        if value:
            return value
    return ""


def _material_entity_ids(inventory: Mapping[str, Any]) -> dict[int, list[str]]:
    names: dict[int, list[str]] = {}
    materials = inventory.get("materials", [])
    if not isinstance(materials, list):
        return {}
    for material in materials:
        if not isinstance(material, Mapping):
            continue
        name = _material_name(material)
        entities = material.get("selection_entities", [])
        if not name or not isinstance(entities, list):
            continue
        for entity in entities:
            try:
                domain_id = int(entity)
            except (TypeError, ValueError):
                continue
            names.setdefault(domain_id, [])
            if name not in names[domain_id]:
                names[domain_id].append(name)
    return names


def _select_entity_offset(raw_ids: set[int], known_ids: set[int]) -> int:
    if not raw_ids or not known_ids:
        return 0
    best_offset = 0
    best_score = -1
    for offset in (0, 1, -1):
        score = len({entity_id + offset for entity_id in raw_ids} & known_ids)
        if score > best_score:
            best_score = score
            best_offset = offset
    return int(best_offset)


def _domain_material_names(inventory: Mapping[str, Any], known_domain_ids: set[int] | None = None) -> tuple[dict[int, str], int]:
    raw = _material_entity_ids(inventory)
    offset = _select_entity_offset(set(raw), set(known_domain_ids or set()))
    shifted: dict[int, list[str]] = {}
    for domain_id, values in raw.items():
        target_id = int(domain_id) + int(offset)
        shifted.setdefault(target_id, [])
        for value in values:
            if value not in shifted[target_id]:
                shifted[target_id].append(value)
    return {domain_id: "|".join(values) for domain_id, values in shifted.items()}, int(offset)


def _adjacent_domain_names(value: Any, domain_names: Mapping[int, str]) -> str:
    out: list[str] = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            domain_id = int(token)
        except ValueError:
            continue
        name = domain_names.get(domain_id, "")
        if name and name not in out:
            out.append(name)
    return "|".join(out)


def apply_material_inventory(generated_dir: Path) -> dict[str, Any]:
    inventory_path = Path(generated_dir) / "material_inventory.json"
    if not inventory_path.exists():
        return {"available": False, "applied": False, "reason": "material_inventory.json not found"}
    inventory = _read_json_if_exists(inventory_path)
    domain_csv = Path(generated_dir) / "comsol_domain_entity_mapping.csv"
    known_domain_ids: set[int] = set()
    if domain_csv.exists():
        domain_probe = pd.read_csv(domain_csv)
        if "comsol_domain_entity_id" in domain_probe.columns:
            known_domain_ids = {int(v) for v in domain_probe["comsol_domain_entity_id"].dropna().tolist()}
    domain_names, entity_offset = _domain_material_names(inventory, known_domain_ids)
    if not domain_names:
        return {"available": True, "applied": False, "reason": "no material domain selections found"}

    updated_files: list[str] = []
    if domain_csv.exists():
        domain_df = pd.read_csv(domain_csv)
        if "comsol_domain_entity_id" in domain_df.columns:
            domain_df["comsol_material_name"] = domain_df["comsol_domain_entity_id"].map(domain_names).fillna(
                domain_df.get("comsol_material_name", "not_exported_from_mphtxt")
            )
            domain_df.to_csv(domain_csv, index=False)
            updated_files.append(domain_csv.name)

    boundary_csv = Path(generated_dir) / "comsol_boundary_entity_mapping.csv"
    if boundary_csv.exists():
        boundary_df = pd.read_csv(boundary_csv)
        if "adjacent_domain_ids" in boundary_df.columns:
            mapped = boundary_df["adjacent_domain_ids"].map(lambda value: _adjacent_domain_names(value, domain_names))
            if "comsol_material_name" in boundary_df.columns:
                fallback = boundary_df["comsol_material_name"].tolist()
            else:
                fallback = ["not_exported_from_mphtxt"] * len(boundary_df)
            boundary_df["comsol_material_name"] = [name if name else old for name, old in zip(mapped, fallback)]
            boundary_df.to_csv(boundary_csv, index=False)
            updated_files.append(boundary_csv.name)

    return {
        "available": True,
        "applied": bool(updated_files),
        "material_domain_count": int(len(domain_names)),
        "selection_entity_offset": int(entity_offset),
        "updated_files": updated_files,
    }


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip() != ""


def _rebuild_materials_from_walls(out_dir: Path) -> None:
    walls_path = Path(out_dir) / "part_walls.csv"
    materials_path = Path(out_dir) / "materials.csv"
    walls = pd.read_csv(walls_path)
    existing = pd.read_csv(materials_path) if materials_path.exists() else pd.DataFrame()
    existing_by_id = {
        int(row.material_id): row._asdict()
        for row in existing.itertuples(index=False)
        if hasattr(row, "material_id")
    }

    rows: list[dict[str, Any]] = []
    for material_id, group in walls.groupby("material_id", sort=True):
        first = group.iloc[0]
        previous = existing_by_id.get(int(material_id), {})
        rows.append(
            {
                "material_id": int(material_id),
                "material_name": str(first["material_name"]),
                "source_law": previous.get("source_law", "explicit_csv"),
                "source_speed_scale": float(previous.get("source_speed_scale", 1.0)),
                "wall_law": str(first["wall_law"]),
                "wall_restitution": float(first["wall_restitution"]),
                "wall_diffuse_fraction": float(first["wall_diffuse_fraction"]),
                "wall_stick_probability": float(first["wall_stick_probability"]),
            }
        )
    pd.DataFrame(rows).to_csv(materials_path, index=False)


def apply_wall_catalog_overrides(out_dir: Path, overrides_csv: Path | None) -> dict[str, Any]:
    if overrides_csv is None:
        return {"available": False, "applied": False, "reason": "wall overrides not requested"}
    overrides_csv = Path(overrides_csv)
    if not overrides_csv.exists():
        return {"available": False, "applied": False, "reason": f"wall overrides not found: {overrides_csv}"}

    walls_path = Path(out_dir) / "part_walls.csv"
    if not walls_path.exists():
        raise FileNotFoundError(f"wall overrides require generated part_walls.csv: {walls_path}")
    overrides = pd.read_csv(overrides_csv)
    if "part_id" not in overrides.columns:
        raise ValueError(f"wall overrides must include part_id: {overrides_csv}")
    walls = pd.read_csv(walls_path)
    walls_by_part = {int(row.part_id): row._asdict() for row in walls.itertuples(index=False)}
    changed_parts: list[int] = []
    added_parts: list[int] = []

    for override in overrides.to_dict(orient="records"):
        pid = int(override["part_id"])
        row = dict(walls_by_part.get(pid, {}))
        if not row:
            missing = [name for name in WALL_OVERRIDE_COLUMNS if not _nonempty(override.get(name))]
            if missing:
                raise ValueError(
                    f"new wall override for part_id={pid} must provide all columns; missing={missing}"
                )
            added_parts.append(pid)
        for column in WALL_OVERRIDE_COLUMNS:
            if column in override and _nonempty(override[column]):
                row[column] = override[column]
        row["part_id"] = pid
        walls_by_part[pid] = row
        changed_parts.append(pid)

    ordered = [walls_by_part[pid] for pid in sorted(walls_by_part)]
    pd.DataFrame(ordered, columns=WALL_OVERRIDE_COLUMNS).to_csv(walls_path, index=False)
    _rebuild_materials_from_walls(out_dir)
    return {
        "available": True,
        "applied": bool(changed_parts),
        "override_path": str(overrides_csv),
        "changed_part_ids": sorted(set(int(v) for v in changed_parts)),
        "added_part_ids": sorted(set(int(v) for v in added_parts)),
    }


def write_wall_catalog_review(out_dir: Path, generated_dir: Path) -> dict[str, Any]:
    walls_path = Path(out_dir) / "part_walls.csv"
    mapping_path = Path(generated_dir) / "comsol_boundary_entity_mapping.csv"
    if not walls_path.exists() or not mapping_path.exists():
        return {"written": False, "reason": "part_walls.csv or comsol_boundary_entity_mapping.csv not found"}
    walls = pd.read_csv(walls_path)
    mapping = pd.read_csv(mapping_path)
    if "solver_part_id" not in mapping.columns or "part_id" not in walls.columns:
        return {"written": False, "reason": "mapping or wall catalog is missing part ID columns"}
    review = walls.merge(mapping, left_on="part_id", right_on="solver_part_id", how="left", suffixes=("", "_mapping"))
    columns = [
        "part_id",
        "part_name",
        "material_id",
        "material_name",
        "wall_law",
        "wall_restitution",
        "wall_diffuse_fraction",
        "wall_stick_probability",
        "active_in_solver_boundary",
        "comsol_edge_entity_id",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "adjacent_domain_ids",
        "comsol_material_name",
    ]
    for column in columns:
        if column not in review.columns:
            review[column] = ""
    out = Path(generated_dir) / "wall_catalog_review.csv"
    review[columns].to_csv(out, index=False)
    return {
        "written": True,
        "path": out.name,
        "row_count": int(len(review)),
        "active_boundary_row_count": int(review["active_in_solver_boundary"].fillna(False).astype(bool).sum()),
    }


def _patch_particles(
    particles_csv: Path,
    *,
    q_ref_c: float,
    m_ref_kg: float,
    diameter_m: float,
    density_kgm3: float,
) -> None:
    particles = pd.read_csv(particles_csv)
    particles["mass"] = float(m_ref_kg)
    particles["charge"] = float(q_ref_c)
    particles["diameter"] = float(diameter_m)
    particles["density"] = float(density_kgm3)
    particles.to_csv(particles_csv, index=False)


def _patch_run_config(config_path: Path, *, t_end: float, dt: float, save_every: int) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid run_config.yaml: {config_path}")
    config.setdefault("source", {}).setdefault("preprocess", {})["enabled"] = False
    config.setdefault("input_contract", {})["initial_particle_field_support"] = "strict"
    provider_contract = config.setdefault("provider_contract", {})
    provider_contract["boundary_field_support"] = "strict"
    provider_contract.setdefault("boundary_offset_cells", 1.0)
    solver = config.setdefault("solver", {})
    solver["integrator"] = "etd2"
    solver["dt"] = float(dt)
    solver["t_end"] = float(t_end)
    solver["save_every"] = int(save_every)
    solver["valid_mask_policy"] = "retry_then_stop"
    solver.setdefault("max_hits_retry_splits", 2)
    solver.setdefault("plot_particle_limit", min(80, int(solver.get("plot_particle_limit", 80))))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _format_part_id_list(part_ids: list[int] | None) -> str | None:
    if not part_ids:
        return None
    return ",".join(str(int(pid)) for pid in part_ids)


def pack_solver_case(
    *,
    raw_export_dir: Path,
    out_dir: Path,
    particle_count: int,
    q_ref_c: float | None = None,
    m_ref_kg: float | None = None,
    particle_diameter_m: float = 1.0e-6,
    particle_density_kgm3: float = 1200.0,
    particle_seed: int = 24680,
    particle_release_span_s: float | None = 0.0,
    particle_min_release_offset_cells: float = 1.0,
    diagnostic_grid_spacing_m: float | None = None,
    field_ghost_cells: int = 8,
    t_end: float = 2.0e-6,
    dt: float = 2.0e-8,
    save_every: int = 10,
    geometry_scale_m_per_model_unit: float | None = None,
    source_part_ids: list[int] | None = None,
    wall_overrides_csv: Path | None = None,
) -> Path:
    raw_export_dir = Path(raw_export_dir).resolve()
    out_dir = Path(out_dir).resolve()
    mphtxt = raw_export_dir / "mesh.mphtxt"
    samples = raw_export_dir / "field_samples.csv"
    if not mphtxt.exists():
        raise FileNotFoundError(f"missing COMSOL mesh export: {mphtxt}")
    if not samples.exists():
        raise FileNotFoundError(f"missing COMSOL field sample export: {samples}")

    manifest = _read_json_if_exists(raw_export_dir / "export_manifest.json")
    selected_q = float(q_ref_c if q_ref_c is not None else manifest.get("q_ref_c", 1.602176634e-19))
    selected_m = float(m_ref_kg if m_ref_kg is not None else manifest.get("m_ref_kg", 6.64215627e-26))
    coordinate_scale = _selected_geometry_scale(manifest, geometry_scale_m_per_model_unit)
    coordinate_model_unit = str(manifest.get("geometry_model_unit", DEFAULT_GEOMETRY_MODEL_UNIT))

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = out_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    bundle_path = generated / "raw_comsol_field_bundle_2d.npz"
    bundle = build_field_bundle_from_table(
        samples,
        q_ref_c=selected_q,
        m_ref_kg=selected_m,
        coordinate_scale_m_per_model_unit=coordinate_scale,
        coordinate_model_unit=coordinate_model_unit,
        metadata={
            "raw_export_dir": str(raw_export_dir),
            "raw_export_manifests": _hash_manifest_inputs(raw_export_dir),
            "raw_coordinate_model_unit": coordinate_model_unit,
            "coordinate_scale_m_per_model_unit": coordinate_scale,
        },
        require_nonuniform=True,
    )
    write_field_bundle(bundle, bundle_path)
    spacing = float(diagnostic_grid_spacing_m) if diagnostic_grid_spacing_m is not None else _infer_diagnostic_spacing(bundle)

    root = _repo_root()
    build_script = root / "tools" / "build_comsol_case.py"
    _run(
        [
            sys.executable,
            str(build_script),
            "--mphtxt",
            str(mphtxt),
            "--out-dir",
            str(out_dir),
            "--field-bundle",
            str(bundle_path),
            "--diagnostic-grid-spacing-m",
            f"{spacing:.17g}",
            "--field-ghost-cells",
            str(int(field_ghost_cells)),
            "--coordinate-scale-m-per-model-unit",
            f"{coordinate_scale:.17g}",
        ],
        cwd=root,
    )
    particles_cmd = [
        sys.executable,
        str(build_script),
        "--mphtxt",
        str(mphtxt),
        "--out-dir",
        str(out_dir),
        "--particles-only",
        "--particle-count",
        str(int(particle_count)),
        "--particle-seed",
        str(int(particle_seed)),
        "--particle-min-release-offset-cells",
        f"{float(particle_min_release_offset_cells):.17g}",
        "--diagnostic-grid-spacing-m",
        f"{spacing:.17g}",
        "--coordinate-scale-m-per-model-unit",
        f"{coordinate_scale:.17g}",
    ]
    if particle_release_span_s is not None:
        particles_cmd.extend(["--particle-release-span-s", f"{float(particle_release_span_s):.17g}"])
    formatted_source_parts = _format_part_id_list(source_part_ids)
    if formatted_source_parts is not None:
        particles_cmd.extend(["--source-part-ids", formatted_source_parts])
    _run(particles_cmd, cwd=root)

    _patch_particles(
        out_dir / "particles.csv",
        q_ref_c=selected_q,
        m_ref_kg=selected_m,
        diameter_m=float(particle_diameter_m),
        density_kgm3=float(particle_density_kgm3),
    )
    _patch_run_config(out_dir / "run_config.yaml", t_end=float(t_end), dt=float(dt), save_every=int(save_every))
    _copy_raw_manifests(raw_export_dir, generated)
    material_inventory_summary = apply_material_inventory(generated)
    wall_overrides_summary = apply_wall_catalog_overrides(out_dir, wall_overrides_csv)
    wall_review_summary = write_wall_catalog_review(out_dir, generated)

    case_manifest = {
        "case_kind": "icp_cf4_o2_comsol_external_export",
        "raw_export_dir": str(raw_export_dir),
        "out_dir": str(out_dir),
        "particle_count": int(particle_count),
        "particle_source_part_ids": [int(pid) for pid in source_part_ids] if source_part_ids else [],
        "q_ref_c": selected_q,
        "m_ref_kg": selected_m,
        "particle_diameter_m": float(particle_diameter_m),
        "particle_density_kgm3": float(particle_density_kgm3),
        "raw_coordinate_model_unit": coordinate_model_unit,
        "geometry_scale_m_per_model_unit": coordinate_scale,
        "diagnostic_grid_spacing_m": spacing,
        "field_ghost_cells": int(field_ghost_cells),
        "material_inventory": material_inventory_summary,
        "wall_overrides": wall_overrides_summary,
        "wall_catalog_review": wall_review_summary,
        "solver_defaults": {
            "integrator": "etd2",
            "dt": float(dt),
            "t_end": float(t_end),
            "save_every": int(save_every),
            "valid_mask_policy": "retry_then_stop",
        },
    }
    write_json(generated / "icp_export_case_manifest.json", case_manifest)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack an external ICP COMSOL export into a solver case.")
    ap.add_argument("--raw-export-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("examples/icp_rf_bias_cf4_o2_si_etching_2d"))
    ap.add_argument("--particle-count", type=int, default=1000)
    ap.add_argument("--q-ref-c", type=float, default=None)
    ap.add_argument("--m-ref-kg", type=float, default=None)
    ap.add_argument("--particle-diameter-m", type=float, default=1.0e-6)
    ap.add_argument("--particle-density-kgm3", type=float, default=1200.0)
    ap.add_argument("--particle-seed", type=int, default=24680)
    ap.add_argument("--particle-release-span-s", type=float, default=0.0)
    ap.add_argument("--particle-min-release-offset-cells", type=float, default=1.0)
    ap.add_argument("--diagnostic-grid-spacing-m", type=float, default=None)
    ap.add_argument("--field-ghost-cells", type=int, default=8)
    ap.add_argument("--t-end", type=float, default=2.0e-6)
    ap.add_argument("--dt", type=float, default=2.0e-8)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--geometry-scale-m-per-model-unit", type=float, default=None)
    ap.add_argument("--source-part-ids", type=int, nargs="*", default=None)
    ap.add_argument("--wall-overrides-csv", type=Path, default=None)
    args = ap.parse_args(argv)

    out = pack_solver_case(
        raw_export_dir=args.raw_export_dir,
        out_dir=args.out_dir,
        particle_count=int(args.particle_count),
        q_ref_c=args.q_ref_c,
        m_ref_kg=args.m_ref_kg,
        particle_diameter_m=float(args.particle_diameter_m),
        particle_density_kgm3=float(args.particle_density_kgm3),
        particle_seed=int(args.particle_seed),
        particle_release_span_s=args.particle_release_span_s,
        particle_min_release_offset_cells=float(args.particle_min_release_offset_cells),
        diagnostic_grid_spacing_m=args.diagnostic_grid_spacing_m,
        field_ghost_cells=int(args.field_ghost_cells),
        t_end=float(args.t_end),
        dt=float(args.dt),
        save_every=int(args.save_every),
        geometry_scale_m_per_model_unit=args.geometry_scale_m_per_model_unit,
        source_part_ids=[int(pid) for pid in args.source_part_ids] if args.source_part_ids else None,
        wall_overrides_csv=args.wall_overrides_csv,
    )
    print(f"Packed ICP COMSOL solver case: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
