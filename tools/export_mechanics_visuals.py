from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.visualization_common import (
    STATE_COLORS,
    STATE_ORDER,
    add_mesh_fill,
    add_mesh_scalar,
    draw_boundary_edges,
    ensure_visualization_dirs,
    mesh_polygons,
    require_2d_quantity,
    sample_grid_points,
    state_labels,
)


def export_mechanics_visuals(
    case_dir: Path,
    output_dir: Path,
    sample_trajectories: int = 500,
    quiver_stride: int = 12,
) -> Path:
    geom_npz = case_dir / 'generated' / 'comsol_geometry_2d.npz'
    field_npz = case_dir / 'generated' / 'comsol_field_2d.npz'
    positions_npy = output_dir / 'positions_2d.npy'
    final_csv = output_dir / 'final_particles.csv'

    if not geom_npz.exists():
        raise FileNotFoundError(f'Geometry npz not found: {geom_npz}')
    if not field_npz.exists():
        raise FileNotFoundError(f'Field npz not found: {field_npz}')
    if not positions_npy.exists():
        raise FileNotFoundError(f'positions_2d.npy not found: {positions_npy}')
    if not final_csv.exists():
        raise FileNotFoundError(f'final_particles.csv not found: {final_csv}')

    with np.load(geom_npz) as g:
        x = np.asarray(g['axis_0'], dtype=np.float64)
        y = np.asarray(g['axis_1'], dtype=np.float64)
        sdf = np.asarray(g['sdf'], dtype=np.float64)
        nx = np.asarray(g['normal_0'], dtype=np.float64) if 'normal_0' in g else np.zeros_like(sdf)
        ny = np.asarray(g['normal_1'], dtype=np.float64) if 'normal_1' in g else np.zeros_like(sdf)
        if 'nearest_boundary_part_id_map' in g:
            nearest_boundary_part_id = np.asarray(g['nearest_boundary_part_id_map'], dtype=np.int32)
        elif 'part_id_map' in g:
            nearest_boundary_part_id = np.asarray(g['part_id_map'], dtype=np.int32)
        else:
            nearest_boundary_part_id = np.ones_like(sdf, dtype=np.int32)
        boundary_edges = np.asarray(g['boundary_edges'], dtype=np.float64) if 'boundary_edges' in g else None
        boundary_part_ids = np.asarray(g['boundary_edge_part_ids'], dtype=np.int32) if 'boundary_edge_part_ids' in g else None
        geom_valid_mask = np.asarray(g['valid_mask'], dtype=bool) if 'valid_mask' in g else None
        mesh_vertices = np.asarray(g['mesh_vertices'], dtype=np.float64) if 'mesh_vertices' in g else None
        mesh_quads = np.asarray(g['mesh_quads'], dtype=np.int32) if 'mesh_quads' in g else None

    with np.load(field_npz) as f:
        ux = require_2d_quantity(f, 'ux', 'mechanics visuals')
        uy = require_2d_quantity(f, 'uy', 'mechanics visuals')
        mu = require_2d_quantity(f, 'mu', 'mechanics visuals')
        tauw = require_2d_quantity(f, 'tauw', 'mechanics visuals')
        utau = require_2d_quantity(f, 'u_tau', 'mechanics visuals')
        field_valid_mask = np.asarray(f['valid_mask'], dtype=bool) if 'valid_mask' in f else None

    geom_mask = geom_valid_mask if geom_valid_mask is not None and geom_valid_mask.shape == sdf.shape else (sdf <= 0.0)
    field_mask = field_valid_mask if field_valid_mask is not None and field_valid_mask.shape == sdf.shape else np.ones_like(geom_mask, dtype=bool)
    inside = geom_mask & field_mask
    speed = np.sqrt(ux * ux + uy * uy)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    polygons = mesh_polygons(mesh_vertices, mesh_quads) if mesh_vertices is not None and mesh_quads is not None else None
    centroids = mesh_vertices[mesh_quads].mean(axis=1) if mesh_vertices is not None and mesh_quads is not None else None
    speed_masked = np.where(inside, speed, np.nan)
    tauw_masked = np.where(inside, tauw, np.nan)
    utau_masked = np.where(inside, utau, np.nan)
    mu_masked = np.where(inside, mu, np.nan)
    nearest_boundary_part_id_masked = np.where(inside, nearest_boundary_part_id, 0)

    out = ensure_visualization_dirs(output_dir)['mechanics']

    # Export per-grid distribution table for post analysis
    flat = pd.DataFrame(
        {
            'x': xx.ravel(),
            'y': yy.ravel(),
            'inside': inside.astype(np.int32).ravel(),
            'sdf_m': sdf.ravel(),
            'distance_to_wall_m': np.abs(sdf).ravel(),
            'normal_x': nx.ravel(),
            'normal_y': ny.ravel(),
            'nearest_boundary_part_id': nearest_boundary_part_id_masked.ravel(),
            'ux_mps': np.where(inside, ux, np.nan).ravel(),
            'uy_mps': np.where(inside, uy, np.nan).ravel(),
            'speed_mps': speed_masked.ravel(),
            'mu_Pas': mu_masked.ravel(),
            'tauw_Pa': tauw_masked.ravel(),
            'u_tau_mps': utau_masked.ravel(),
        }
    )
    flat = flat[flat['inside'] == 1].reset_index(drop=True)
    flat.to_csv(out / 'mechanics_distribution_on_geometry.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    if polygons is not None:
        add_mesh_fill(ax, polygons, facecolor='#e8f4ff', alpha=1.0)
    else:
        ax.contourf(xx, yy, inside.astype(float), levels=[-0.5, 0.5, 1.5], colors=['#f2f2f2', '#e8f4ff'])
    if boundary_edges is not None and boundary_part_ids is not None:
        draw_boundary_edges(ax, boundary_edges, boundary_part_ids, linewidth=1.35, alpha=0.95)
        unique_pids = np.unique(boundary_part_ids)
        cmap = plt.get_cmap('tab20')
        norm = plt.Normalize(vmin=float(unique_pids.min()), vmax=float(unique_pids.max()))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('Boundary Part ID')
    else:
        masked_part = np.ma.masked_where(~inside, nearest_boundary_part_id.astype(float))
        c = ax.pcolormesh(xx, yy, masked_part, shading='nearest', cmap='tab20', alpha=0.75)
        cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('Nearest Boundary Part ID')
        ax.contour(xx, yy, sdf, levels=[0.0], colors='k', linewidths=1.1)
    ax.set_title('Boundary Part IDs over Geometry')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.tight_layout()
    fig.savefig(out / 'geometry_layout_part_ids.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    fields = [
        ('Speed |u| [m/s]', speed_masked),
        ('Wall Shear tauw [Pa]', tauw_masked),
        ('Friction Velocity u_tau [m/s]', utau_masked),
        ('Dynamic Viscosity mu [Pa*s]', mu_masked),
    ]
    for ax, (title, arr) in zip(axes.ravel(), fields):
        if polygons is not None and centroids is not None:
            values = sample_grid_points(arr, x, y, centroids)
            add_mesh_scalar(fig, ax, polygons, values, title, cmap='viridis')
        else:
            m = np.ma.masked_where(~inside, arr)
            pcm = ax.pcolormesh(xx, yy, m, shading='nearest', cmap='viridis')
            ax.set_title(title)
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
        if boundary_edges is not None:
            draw_boundary_edges(ax, boundary_edges, None, linewidth=0.85, alpha=0.9)
        else:
            ax.contour(xx, yy, sdf, levels=[0.0], colors='k', linewidths=0.8)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
    fig.savefig(out / 'mechanics_maps_with_geometry.png', dpi=170)
    plt.close(fig)

    positions = np.load(positions_npy)
    _, npart, _ = positions.shape
    rng = np.random.default_rng(20260401)
    pick = rng.choice(npart, size=min(sample_trajectories, npart), replace=False)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    if polygons is not None:
        add_mesh_fill(ax, polygons, facecolor='#eef7ff', alpha=1.0)
    else:
        ax.contourf(xx, yy, inside.astype(float), levels=[-0.5, 0.5, 1.5], colors=['#f8f8f8', '#eef7ff'])
    if boundary_edges is not None:
        draw_boundary_edges(ax, boundary_edges, None, linewidth=1.0, alpha=0.95)
    else:
        ax.contour(xx, yy, sdf, levels=[0.0], colors='#444', linewidths=1.0)
    for i in pick:
        tr = positions[:, i, :]
        ax.plot(tr[:, 0], tr[:, 1], lw=0.7, alpha=0.7)
    sx = slice(None, None, max(1, int(quiver_stride)))
    sy = slice(None, None, max(1, int(quiver_stride)))
    qmask = inside[sx, sy]
    ax.quiver(
        xx[sx, sy][qmask],
        yy[sx, sy][qmask],
        ux[sx, sy][qmask],
        uy[sx, sy][qmask],
        angles='xy',
        scale_units='xy',
        scale=20.0,
        width=0.0018,
        color='black',
        alpha=0.35,
    )
    ax.set_title(f'Trajectories + Geometry + Flow Vectors (sample {len(pick)} / {npart})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.tight_layout()
    fig.savefig(out / 'trajectories_geometry_flow_overlay.png', dpi=170)
    plt.close(fig)

    final_df = pd.read_csv(final_csv)
    final_labels = state_labels(final_df)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    if boundary_edges is not None:
        draw_boundary_edges(ax, boundary_edges, None, linewidth=1.0, alpha=0.95)
    else:
        ax.contour(xx, yy, sdf, levels=[0.0], colors='k', linewidths=1.0)
    for name in STATE_ORDER:
        sub = final_df.loc[final_labels == name]
        if not sub.empty:
            ax.scatter(sub['x'], sub['y'], s=4, c=STATE_COLORS[name], label=f'{name} ({len(sub)})', alpha=0.7)
    ax.set_title('Final Particle States over Geometry')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.tight_layout()
    fig.savefig(out / 'final_states_over_geometry.png', dpi=170)
    plt.close(fig)

    report = {
        'case_dir': str(case_dir.resolve()),
        'output_dir': str(output_dir.resolve()),
        'mechanics_dir': str(out.resolve()),
        'n_particles': int(npart),
        'sample_trajectories': int(len(pick)),
        'domain_region_map_status': 'not_implemented',
        'files': [
            'mechanics_distribution_on_geometry.csv',
            'geometry_layout_part_ids.png',
            'mechanics_maps_with_geometry.png',
            'trajectories_geometry_flow_overlay.png',
            'final_states_over_geometry.png',
        ],
    }
    (out / 'visualization_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Export mechanics-distribution and geometry-aware trajectory visuals.')
    ap.add_argument('--case-dir', type=Path, default=Path('examples/comsol_from_data_2d_10k'))
    ap.add_argument('--output-dir', type=Path, default=Path('demo_output/comsol_from_data_2d_10k'))
    ap.add_argument('--sample-trajectories', type=int, default=500)
    ap.add_argument('--quiver-stride', type=int, default=12)
    args = ap.parse_args()
    from tools.export_visualizations import export_visualizations

    index_path = export_visualizations(
        output_dir=args.output_dir.resolve(),
        case_dir=args.case_dir.resolve(),
        modules=("mechanics",),
        mechanics_sample_trajectories=max(1, int(args.sample_trajectories)),
        mechanics_quiver_stride=max(1, int(args.quiver_stride)),
    )
    print(f'Wrote mechanics visuals via unified pipeline: {index_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
