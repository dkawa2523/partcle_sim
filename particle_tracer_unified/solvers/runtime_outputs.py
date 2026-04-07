from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..core.datamodel import ParticleTable, PreparedRuntime
from ..core.field_backend import field_backend_report
from ..core.source_materials import write_source_summary
from ..io.runtime_builder import prepared_runtime_summary


@dataclass(frozen=True)
class RuntimeOutputPayload:
    prepared: PreparedRuntime
    spatial_dim: int
    particles: ParticleTable
    release_time: np.ndarray
    positions: np.ndarray
    save_meta: List[Dict[str, object]]
    final_position: np.ndarray
    final_velocity: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    absorbed: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    final_step_name: str
    final_segment_name: str
    wall_rows: List[Dict[str, object]]
    wall_law_counts: Dict[str, int]
    wall_summary_counts: Dict[Tuple[int, str, str], int]
    max_hit_rows: List[Dict[str, object]]
    step_rows: List[Dict[str, object]]
    collision_diagnostics: Dict[str, object]
    base_integrator_name: str
    write_collision_diagnostics: int
    max_wall_hits_per_step: int
    max_hits_retry_splits: int
    max_hits_retry_local_adaptive_enabled: int
    min_remaining_dt_ratio: float
    on_boundary_tol_m: float
    epsilon_offset_m: float
    adaptive_substep_enabled: int
    adaptive_substep_tau_ratio: float
    adaptive_substep_max_splits: int
    plot_limit: int
    valid_mask_policy: str


_INVALID_SEGMENT_FILENAME_TRANSLATION = str.maketrans({ch: '_' for ch in '<>:"/\\|?*'})


def _safe_segment_filename(segment_name: object) -> str:
    raw = str(segment_name).strip() if segment_name is not None else ''
    if not raw:
        return 'run'
    safe = raw.translate(_INVALID_SEGMENT_FILENAME_TRANSLATION)
    safe = ''.join(ch if ch.isprintable() and ch not in {'\r', '\n', '\t'} else '_' for ch in safe)
    safe = safe.strip(' .')
    return safe or 'run'


def _save_segmented_positions(output_dir: Path, positions: np.ndarray, save_meta: List[Dict[str, object]], spatial_dim: int) -> None:
    import pandas as pd

    if positions.size == 0 or not save_meta:
        return
    df = pd.DataFrame(save_meta)
    df.to_csv(output_dir / 'save_frames.csv', index=False)
    segments_dir = output_dir / 'segments'
    segments_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for segment_name, sub in df.groupby('segment_name', dropna=False):
        idx = sub['save_index'].to_numpy(dtype=int)
        arr = positions[idx]
        safe = _safe_segment_filename(segment_name)
        np.save(segments_dir / f'positions_{safe}_{spatial_dim}d.npy', arr)
        rows.append({'segment_name': safe, 'save_count': int(len(idx)), 't_start': float(sub['time_s'].min()), 't_end': float(sub['time_s'].max())})
    pd.DataFrame(rows).to_csv(output_dir / 'segment_summary.csv', index=False)


def _build_final_particles_frame(payload: RuntimeOutputPayload) -> pd.DataFrame:
    import pandas as pd

    final_df = pd.DataFrame(
        {
            'particle_id': payload.particles.particle_id,
            'release_time': payload.release_time,
            'released': payload.released.astype(int),
            'active': payload.active.astype(int),
            'stuck': payload.stuck.astype(int),
            'absorbed': payload.absorbed.astype(int),
            'escaped': payload.escaped.astype(int),
            'invalid_mask_stopped': payload.invalid_mask_stopped.astype(int),
            'final_step_name': payload.final_step_name,
            'final_segment_name': payload.final_segment_name,
            'source_part_id': payload.particles.source_part_id,
            'material_id': payload.particles.material_id,
        }
    )
    for j, name in enumerate(['x', 'y', 'z'][: payload.spatial_dim]):
        final_df[name] = payload.final_position[:, j]
        final_df[f'v_{name}'] = payload.final_velocity[:, j]
    return final_df


def _write_resolved_particles(payload: RuntimeOutputPayload, output_dir: Path) -> None:
    import pandas as pd

    if payload.prepared.source_preprocess is None:
        return
    write_source_summary(payload.prepared.source_preprocess, output_dir)
    cols = {
        'particle_id': payload.particles.particle_id,
        'release_time': payload.release_time,
        'source_part_id': payload.particles.source_part_id,
        'material_id': payload.particles.material_id,
        'source_event_tag': payload.particles.source_event_tag,
    }
    for j, name in enumerate(['x', 'y', 'z'][: payload.spatial_dim]):
        cols[name] = payload.particles.position[:, j]
        cols[f'v{name}'] = payload.particles.velocity[:, j]
    pd.DataFrame(cols).to_csv(output_dir / 'resolved_particles.csv', index=False)


def _build_wall_summary_report(wall_summary_counts: Dict[Tuple[int, str, str], int]) -> Dict[str, object]:
    wall_summary_report: Dict[str, object] = {
        'total_wall_interactions': int(sum(wall_summary_counts.values())),
        'by_part': {},
        'by_outcome': {},
        'by_wall_mode': {},
    }
    by_part = wall_summary_report['by_part']
    by_outcome = wall_summary_report['by_outcome']
    by_wall_mode = wall_summary_report['by_wall_mode']
    for (part_id, outcome, wall_mode), count in wall_summary_counts.items():
        part_bucket = by_part.setdefault(str(int(part_id)), {})
        part_bucket[str(outcome)] = int(part_bucket.get(str(outcome), 0) + int(count))
        by_outcome[str(outcome)] = int(by_outcome.get(str(outcome), 0) + int(count))
        by_wall_mode[str(wall_mode)] = int(by_wall_mode.get(str(wall_mode), 0) + int(count))
    return wall_summary_report


def _build_collision_diag_report(payload: RuntimeOutputPayload) -> Dict[str, object]:
    backend_report = field_backend_report(payload.prepared.runtime.field_provider)
    return {
        **payload.collision_diagnostics,
        'integrator': str(payload.base_integrator_name),
        'valid_mask_policy': str(payload.valid_mask_policy),
        **backend_report,
        'max_wall_hits_per_step': int(payload.max_wall_hits_per_step),
        'max_hits_retry_splits': int(payload.max_hits_retry_splits),
        'max_hits_retry_local_adaptive_enabled': int(payload.max_hits_retry_local_adaptive_enabled),
        'min_remaining_dt_ratio': float(payload.min_remaining_dt_ratio),
        'on_boundary_tol_m': float(payload.on_boundary_tol_m),
        'epsilon_offset_m': float(payload.epsilon_offset_m),
        'adaptive_substep_enabled': int(payload.adaptive_substep_enabled),
        'adaptive_substep_tau_ratio': float(payload.adaptive_substep_tau_ratio),
        'adaptive_substep_max_splits': int(payload.adaptive_substep_max_splits),
    }


def build_runtime_report(payload: RuntimeOutputPayload, *, outputs_written: bool) -> Dict[str, object]:
    backend_report = field_backend_report(payload.prepared.runtime.field_provider)
    valid_mask_violation_count = int(payload.collision_diagnostics.get('valid_mask_violation_count', 0))
    valid_mask_violation_particle_count = int(payload.collision_diagnostics.get('valid_mask_violation_particle_count', 0))
    valid_mask_mixed_stencil_count = int(payload.collision_diagnostics.get('valid_mask_mixed_stencil_count', 0))
    valid_mask_mixed_stencil_particle_count = int(payload.collision_diagnostics.get('valid_mask_mixed_stencil_particle_count', 0))
    valid_mask_hard_invalid_count = int(payload.collision_diagnostics.get('valid_mask_hard_invalid_count', 0))
    valid_mask_hard_invalid_particle_count = int(payload.collision_diagnostics.get('valid_mask_hard_invalid_particle_count', 0))
    extension_band_sample_count = int(payload.collision_diagnostics.get('extension_band_sample_count', 0))
    extension_band_sample_particle_count = int(payload.collision_diagnostics.get('extension_band_sample_particle_count', 0))
    invalid_mask_stopped_count = int(payload.invalid_mask_stopped.sum())
    return {
        'particle_count': int(payload.particles.count),
        'released_count': int(payload.released.sum()),
        'stuck_count': int(payload.stuck.sum()),
        'absorbed_count': int(payload.absorbed.sum()),
        'escaped_count': int(payload.escaped.sum()),
        'invalid_mask_stopped_count': int(invalid_mask_stopped_count),
        'save_frame_count': int(len(payload.save_meta)),
        'outputs_written': int(bool(outputs_written)),
        'positions_file': f'positions_{payload.spatial_dim}d.npy' if bool(outputs_written) else '',
        'integrator': str(payload.base_integrator_name),
        'valid_mask_policy': str(payload.valid_mask_policy),
        'wall_law_counts': payload.wall_law_counts,
        'wall_summary_file': 'wall_summary.json' if bool(outputs_written) else '',
        'wall_summary_by_part_file': 'wall_summary_by_part.csv' if bool(outputs_written) else '',
        'max_hit_events_file': 'max_hit_events.csv' if bool(outputs_written) else '',
        'collision_diagnostics_file': (
            'collision_diagnostics.json'
            if bool(outputs_written) and int(payload.write_collision_diagnostics) != 0
            else ''
        ),
        'runtime_step_summary_file': 'runtime_step_summary.csv' if bool(outputs_written) else '',
        'kernel_backend': f'numba_{payload.spatial_dim}d_freeflight',
        'valid_mask_violation_count': int(valid_mask_violation_count),
        'valid_mask_violation_particle_count': int(valid_mask_violation_particle_count),
        'valid_mask_mixed_stencil_count': int(valid_mask_mixed_stencil_count),
        'valid_mask_mixed_stencil_particle_count': int(valid_mask_mixed_stencil_particle_count),
        'valid_mask_hard_invalid_count': int(valid_mask_hard_invalid_count),
        'valid_mask_hard_invalid_particle_count': int(valid_mask_hard_invalid_particle_count),
        'extension_band_sample_count': int(extension_band_sample_count),
        'extension_band_sample_particle_count': int(extension_band_sample_particle_count),
        **backend_report,
    }


def _write_trajectory_plot(output_dir: Path, positions: np.ndarray, spatial_dim: int, plot_limit: int) -> None:
    import matplotlib.pyplot as plt

    particle_count = int(positions.shape[1]) if positions.ndim == 3 else 0
    if int(plot_limit) <= 0 or particle_count == 0:
        return
    if int(spatial_dim) == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(min(particle_count, int(plot_limit))):
            arr = positions[:, i, :]
            ax.plot(arr[:, 0], arr[:, 1], alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Prepared-runtime trajectories 2D')
        fig.tight_layout()
        fig.savefig(output_dir / 'trajectories.png', dpi=150)
        plt.close(fig)
        return
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(min(particle_count, int(plot_limit))):
        arr = positions[:, i, :]
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Prepared-runtime trajectories 3D')
    fig.tight_layout()
    fig.savefig(output_dir / 'trajectories_3d.png', dpi=150)
    plt.close(fig)


def write_runtime_outputs(payload: RuntimeOutputPayload, output_dir: Path) -> Dict[str, object]:
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f'positions_{payload.spatial_dim}d.npy', payload.positions)
    _save_segmented_positions(output_dir, payload.positions, payload.save_meta, payload.spatial_dim)

    final_df = _build_final_particles_frame(payload)
    final_df.to_csv(output_dir / 'final_particles.csv', index=False)

    wall_cols = ['time_s', 'particle_id', 'part_id', 'step_name', 'segment_name', 'outcome', 'wall_mode', 'alpha_hit', 'material_id', 'material_name']
    pd.DataFrame(payload.wall_rows, columns=wall_cols).to_csv(output_dir / 'wall_events.csv', index=False)

    wall_summary_rows = [
        {
            'part_id': int(part_id),
            'outcome': str(outcome),
            'wall_mode': str(wall_mode),
            'count': int(count),
        }
        for (part_id, outcome, wall_mode), count in sorted(
            payload.wall_summary_counts.items(),
            key=lambda item: (-int(item[1]), int(item[0][0]), str(item[0][1]), str(item[0][2])),
        )
    ]
    pd.DataFrame(wall_summary_rows, columns=['part_id', 'outcome', 'wall_mode', 'count']).to_csv(
        output_dir / 'wall_summary_by_part.csv',
        index=False,
    )

    max_hit_cols = [
        'time_s',
        'particle_id',
        'step_name',
        'segment_name',
        'hits_in_step',
        'remaining_dt_s',
        'last_part_id',
        'part_id_sequence',
        'outcome_sequence',
    ]
    pd.DataFrame(payload.max_hit_rows, columns=max_hit_cols).to_csv(output_dir / 'max_hit_events.csv', index=False)

    step_cols = [
        'time_s',
        'step_name',
        'segment_name',
        'released_count',
        'active_count',
        'stuck_count',
        'absorbed_count',
        'escaped_count',
        'invalid_mask_stopped_count_step',
        'save_positions_enabled',
        'write_wall_events_enabled',
        'write_diagnostics_enabled',
        'valid_mask_violation_count_step',
        'valid_mask_mixed_stencil_count_step',
        'valid_mask_hard_invalid_count_step',
        'extension_band_sample_count_step',
    ]
    pd.DataFrame(payload.step_rows, columns=step_cols).to_csv(output_dir / 'runtime_step_summary.csv', index=False)

    (output_dir / 'prepared_runtime_summary.json').write_text(
        json.dumps(prepared_runtime_summary(payload.prepared), indent=2),
        encoding='utf-8',
    )
    _write_resolved_particles(payload, output_dir)

    wall_summary_report = _build_wall_summary_report(payload.wall_summary_counts)
    (output_dir / 'wall_summary.json').write_text(json.dumps(wall_summary_report, indent=2), encoding='utf-8')

    collision_diag_report = _build_collision_diag_report(payload)
    if int(payload.write_collision_diagnostics) != 0:
        (output_dir / 'collision_diagnostics.json').write_text(json.dumps(collision_diag_report, indent=2), encoding='utf-8')

    report = build_runtime_report(payload, outputs_written=True)
    (output_dir / 'solver_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    _write_trajectory_plot(output_dir, payload.positions, payload.spatial_dim, payload.plot_limit)
    return report
