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
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_parts_by_medium,
    ensure_visualization_dirs,
    filter_display_boundary_geometry,
    sample_grid_points,
    state_labels,
)


def _as_2d_quantity(payload: np.lib.npyio.NpzFile, name: str) -> np.ndarray | None:
    if name not in payload:
        return None
    arr = np.asarray(payload[name], dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    return None


def _masked(arr: np.ndarray | None, inside: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.full(inside.shape, np.nan, dtype=np.float64)
    return np.where(inside & np.isfinite(arr), arr, np.nan)


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
        mesh_triangles = np.asarray(g['mesh_triangles'], dtype=np.int32) if 'mesh_triangles' in g else None
        mesh_triangle_part_ids = np.asarray(g['mesh_triangle_part_ids'], dtype=np.int32) if 'mesh_triangle_part_ids' in g else None
        mesh_quads = np.asarray(g['mesh_quads'], dtype=np.int32) if 'mesh_quads' in g else None
        mesh_quad_part_ids = np.asarray(g['mesh_quad_part_ids'], dtype=np.int32) if 'mesh_quad_part_ids' in g else None
    boundary_edges, boundary_part_ids = filter_display_boundary_geometry(boundary_edges, boundary_part_ids)

    with np.load(field_npz) as f:
        ux = _as_2d_quantity(f, 'ux')
        uy = _as_2d_quantity(f, 'uy')
        mu = _as_2d_quantity(f, 'mu')
        ex = _as_2d_quantity(f, 'E_x')
        ey = _as_2d_quantity(f, 'E_y')
        scalar_fields = {
            name: _as_2d_quantity(f, name)
            for name in ('T', 'p', 'rho_g', 'phi', 'ne', 'Te')
            if _as_2d_quantity(f, name) is not None
        }
        field_valid_mask = np.asarray(f['valid_mask'], dtype=bool) if 'valid_mask' in f else None
    if ux is None or uy is None:
        raise ValueError('mechanics visuals require ux and uy in the field bundle')

    geom_mask = geom_valid_mask if geom_valid_mask is not None and geom_valid_mask.shape == sdf.shape else (sdf <= 0.0)
    field_mask = field_valid_mask if field_valid_mask is not None and field_valid_mask.shape == sdf.shape else np.ones_like(geom_mask, dtype=bool)
    inside = geom_mask & field_mask
    speed = np.sqrt(ux * ux + uy * uy)
    e_mag = np.sqrt(ex * ex + ey * ey) if ex is not None and ey is not None else None
    xx, yy = np.meshgrid(x, y, indexing='ij')
    speed_masked = np.where(inside, speed, np.nan)
    e_mag_masked = _masked(e_mag, inside)
    mu_masked = _masked(mu, inside)
    ex_masked = _masked(ex, inside)
    ey_masked = _masked(ey, inside)
    nearest_boundary_part_id_masked = np.where(inside, nearest_boundary_part_id, 0)

    out = ensure_visualization_dirs(output_dir)['mechanics']
    out.mkdir(parents=True, exist_ok=True)
    medium_summary = domain_part_medium_summary(
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
        x,
        y,
        field_mask,
    )
    if not medium_summary.empty:
        medium_summary.to_csv(out / 'domain_part_medium_summary.csv', index=False)

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
            'electric_field_Vpm': e_mag_masked.ravel(),
        }
    )
    flat = flat[flat['inside'] == 1].reset_index(drop=True)
    flat.to_csv(out / 'mechanics_distribution_on_geometry.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    draw_domain_parts_by_medium(
        ax,
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
        medium_summary=medium_summary,
        alpha=0.48,
        linewidth=0.04,
        label_part_ids=True,
        show_legend=True,
    )
    if boundary_edges is not None and boundary_part_ids is not None:
        draw_boundary_edges(ax, boundary_edges, boundary_part_ids, linewidth=1.35, alpha=0.95, label_part_ids=True)
    else:
        masked_part = np.ma.masked_where(~inside, nearest_boundary_part_id.astype(float))
        c = ax.pcolormesh(xx, yy, masked_part, shading='nearest', cmap='Greys', alpha=0.75)
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

    fields = [('Speed |u| [m/s]', speed_masked)]
    if e_mag is not None:
        fields.append(('Electric Field |E| [V/m]', e_mag_masked))
    if mu is not None:
        fields.append(('Dynamic Viscosity mu [Pa*s]', mu_masked))
    fields.extend((f'{name}', _masked(arr, inside)) for name, arr in scalar_fields.items())
    cols = min(3, max(1, len(fields)))
    rows = int(np.ceil(len(fields) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.9 * rows), constrained_layout=True, squeeze=False)
    for ax, (title, arr) in zip(axes.ravel(), fields):
        draw_domain_parts_by_medium(
            ax,
            mesh_vertices,
            mesh_triangles,
            mesh_triangle_part_ids,
            mesh_quads,
            mesh_quad_part_ids,
            medium_summary=medium_summary,
            alpha=0.22,
            linewidth=0.02,
        )
        m = np.ma.masked_where(~inside, arr)
        pcm = ax.pcolormesh(xx, yy, m, shading='nearest', cmap='viridis', alpha=0.86)
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
        draw_domain_parts_by_medium(
            ax,
            mesh_vertices,
            mesh_triangles,
            mesh_triangle_part_ids,
            mesh_quads,
            mesh_quad_part_ids,
            medium_summary=medium_summary,
            alpha=0.10,
            linewidth=0.04,
            edgecolor="#222222",
        )
        if boundary_edges is not None:
            draw_boundary_edges(ax, boundary_edges, None, linewidth=0.85, alpha=0.9)
        else:
            ax.contour(xx, yy, sdf, levels=[0.0], colors='k', linewidths=0.8)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
    for ax in axes.ravel()[len(fields):]:
        ax.axis('off')
    fig.savefig(out / 'mechanics_maps_with_geometry.png', dpi=170)
    plt.close(fig)

    component_specs = [
        ('ux [m/s]', np.where(inside, ux, np.nan)),
        ('uy [m/s]', np.where(inside, uy, np.nan)),
        ('E_x [V/m]', ex_masked),
        ('E_y [V/m]', ey_masked),
    ]
    component_specs = [(title, arr) for title, arr in component_specs if np.isfinite(arr).any()]
    if component_specs:
        cols = min(3, len(component_specs))
        rows = int(np.ceil(len(component_specs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.9 * rows), constrained_layout=True, squeeze=False)
        for ax, (title, arr) in zip(axes.ravel(), component_specs):
            draw_domain_parts_by_medium(
                ax,
                mesh_vertices,
                mesh_triangles,
                mesh_triangle_part_ids,
                mesh_quads,
                mesh_quad_part_ids,
                medium_summary=medium_summary,
                alpha=0.22,
                linewidth=0.02,
            )
            pcm = ax.pcolormesh(xx, yy, np.ma.masked_invalid(arr), shading='nearest', cmap='coolwarm', alpha=0.86)
            draw_domain_parts_by_medium(
                ax,
                mesh_vertices,
                mesh_triangles,
                mesh_triangle_part_ids,
                mesh_quads,
                mesh_quad_part_ids,
                medium_summary=medium_summary,
                alpha=0.10,
                linewidth=0.04,
                edgecolor="#222222",
            )
            draw_boundary_edges(ax, boundary_edges, boundary_part_ids, linewidth=0.75, alpha=0.9)
            ax.set_title(title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
        for ax in axes.ravel()[len(component_specs):]:
            ax.axis('off')
        fig.savefig(out / 'mechanics_component_maps_with_geometry.png', dpi=170)
        plt.close(fig)

    positions = np.load(positions_npy)
    _, npart, _ = positions.shape
    rng = np.random.default_rng(20260401)
    pick = rng.choice(npart, size=min(sample_trajectories, npart), replace=False)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_domain_parts_by_medium(
        ax,
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
        medium_summary=medium_summary,
        alpha=0.32,
        linewidth=0.03,
    )
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
    final_state_part_summary_rows: list[dict[str, object]] = []
    if {'x', 'y'}.issubset(final_df.columns):
        final_points = final_df.loc[:, ['x', 'y']].to_numpy(dtype=np.float64)
        nearest_final_part_id = np.rint(sample_grid_points(nearest_boundary_part_id, x, y, final_points)).astype(np.int32)
        final_state_part_summary = (
            pd.DataFrame(
                {
                    'nearest_boundary_part_id': nearest_final_part_id,
                    'state': final_labels,
                }
            )
            .groupby(['nearest_boundary_part_id', 'state'], as_index=False)
            .size()
            .rename(columns={'size': 'count'})
            .sort_values(['nearest_boundary_part_id', 'state'])
        )
        final_state_part_summary.to_csv(out / 'final_state_by_nearest_boundary_part.csv', index=False)
        final_state_part_summary_rows = [
            {
                'nearest_boundary_part_id': int(row['nearest_boundary_part_id']),
                'state': str(row['state']),
                'count': int(row['count']),
            }
            for _, row in final_state_part_summary.iterrows()
        ]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_domain_parts_by_medium(
        ax,
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
        medium_summary=medium_summary,
        alpha=0.30,
        linewidth=0.03,
    )
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
        'boundary_region_summary_status': 'computed_from_nearest_boundary_part_id_map',
        'final_state_by_nearest_boundary_part': final_state_part_summary_rows,
        'files': [
            'domain_part_medium_summary.csv',
            'mechanics_distribution_on_geometry.csv',
            'final_state_by_nearest_boundary_part.csv',
            'geometry_layout_part_ids.png',
            'mechanics_maps_with_geometry.png',
            'mechanics_component_maps_with_geometry.png',
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
