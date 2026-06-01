from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _first_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in frame.columns}
    for alias in aliases:
        col = lower.get(alias.lower())
        if col is not None:
            return col
    return None


def _numeric(frame: pd.DataFrame, aliases: tuple[str, ...], *, default: float | None = None) -> np.ndarray:
    col = _first_column(frame, aliases)
    if col is None:
        if default is None:
            raise ValueError(f"missing required column; expected one of {aliases}")
        return np.full(len(frame), float(default), dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)


def _nearest_boundary_part_ids(points: np.ndarray, geometry_npz: Path, *, chunk_size: int = 4096) -> np.ndarray:
    with np.load(geometry_npz, allow_pickle=True) as payload:
        edges = np.asarray(payload["boundary_edges"], dtype=np.float64)
        part_ids = np.asarray(payload["boundary_edge_part_ids"], dtype=np.int64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2):
        raise ValueError("geometry boundary_edges must have shape (n, 2, 2)")
    a = edges[:, 0, :]
    b = edges[:, 1, :]
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1.0e-300)
    out = np.empty(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], int(chunk_size)):
        p = points[start : start + int(chunk_size)]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((p[:, None, :] - closest) ** 2, axis=2)
        out[start : start + p.shape[0]] = part_ids[np.argmin(d2, axis=1)]
    return out


def build_solver_particles_from_release(
    release_csv: str | Path,
    out_csv: str | Path,
    *,
    geometry_npz: str | Path | None = None,
    source_part_blocks: list[tuple[int, int]] | None = None,
    mass_kg: float,
    diameter_m: float,
    density_kgm3: float,
    charge_c: float = 0.0,
    material_id: int = 1,
    stick_probability: float = 0.0,
    source_event_tag: str = "comsol_release",
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    release = pd.read_csv(release_csv)
    particle_id = _numeric(release, ("particle_id", "ParticleID", "id", "pid")).astype(np.int64)
    x = _numeric(release, ("x", "x_m", "r", "r_m"))
    y = _numeric(release, ("y", "y_m", "z", "z_m"))
    vx = _numeric(release, ("v_x", "vx", "v_x0", "vx0"), default=0.0)
    vy = _numeric(release, ("v_y", "vy", "v_y0", "vy0", "v0"), default=0.0)
    release_time = _numeric(release, ("release_time", "release_time_s", "time_s", "time", "t0", "t_release"))
    source_part = _numeric(
        release,
        ("source_part_id", "part_id", "source_entity", "source_boundary_id", "boundary_id"),
        default=np.nan,
    )
    if source_part_blocks:
        assigned = []
        for part_id, count in source_part_blocks:
            assigned.extend([int(part_id)] * int(count))
        if len(assigned) != len(release):
            raise ValueError(
                "source_part_blocks count does not match release table length; "
                f"got {len(assigned)}, expected {len(release)}"
            )
        source_part = np.asarray(assigned, dtype=np.float64)
    if not np.isfinite(source_part).all():
        if geometry_npz is None:
            raise ValueError("release source_part_id is incomplete; pass --geometry-npz to infer nearest boundary part")
        source_part = _nearest_boundary_part_ids(np.column_stack([x, y]), Path(geometry_npz)).astype(np.float64)

    rows = pd.DataFrame(
        {
            "particle_id": particle_id,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "release_time": release_time,
            "mass": float(mass_kg),
            "diameter": float(diameter_m),
            "density": float(density_kgm3),
            "charge": float(charge_c),
            "source_part_id": source_part.astype(np.int64),
            "material_id": int(material_id),
            "source_event_tag": str(source_event_tag),
            "stick_probability": float(stick_probability),
        }
    )
    rows = rows.sort_values("particle_id", kind="mergesort")
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out, index=False)

    counts = rows["source_part_id"].value_counts().sort_index()
    report = {
        "source_kind": "external_comsol_particle_export_solver_particles",
        "release_csv": str(release_csv),
        "out_csv": str(out),
        "particle_count": int(len(rows)),
        "release_time_min_s": float(rows["release_time"].min()),
        "release_time_max_s": float(rows["release_time"].max()),
        "source_part_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "mass_kg": float(mass_kg),
        "diameter_m": float(diameter_m),
        "density_kgm3": float(density_kgm3),
        "charge_c": float(charge_c),
    }
    if report_json is not None:
        Path(report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build solver particles.csv from a canonical COMSOL release table.")
    parser.add_argument("--release-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--geometry-npz", type=Path, default=None)
    parser.add_argument(
        "--source-part-blocks",
        default="",
        help="Optional ordered part_id:count blocks, for example 2:1050,6:1050,13:1050.",
    )
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument("--diameter-m", type=float, required=True)
    parser.add_argument("--density-kgm3", type=float, required=True)
    parser.add_argument("--charge-c", type=float, default=0.0)
    parser.add_argument("--material-id", type=int, default=1)
    parser.add_argument("--stick-probability", type=float, default=0.0)
    parser.add_argument("--source-event-tag", default="comsol_release")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args(argv)
    blocks = []
    if args.source_part_blocks.strip():
        for raw in args.source_part_blocks.split(","):
            part, count = raw.split(":", 1)
            blocks.append((int(part), int(count)))
    report = build_solver_particles_from_release(
        args.release_csv,
        args.out_csv,
        geometry_npz=args.geometry_npz,
        source_part_blocks=blocks or None,
        mass_kg=float(args.mass_kg),
        diameter_m=float(args.diameter_m),
        density_kgm3=float(args.density_kgm3),
        charge_c=float(args.charge_c),
        material_id=int(args.material_id),
        stick_probability=float(args.stick_probability),
        source_event_tag=str(args.source_event_tag),
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
