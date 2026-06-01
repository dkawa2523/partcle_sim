from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .datamodel import (
    ParticleTable,
    ProcessStepTable,
    SourceEventTable,
    SourcePreprocessResult,
    SourceResolutionParameters,
    source_provenance_group,
)
from .source_material_common import (
    burst_factor,
    effective_resuspension_threshold_speed,
    effective_resuspension_threshold_tau,
    event_effect,
    event_matches,
    is_finite,
    orthonormal_tangent_basis,
    resolved_slope_rms,
    sample_positive_normal,
    sample_thermal_velocity,
)
from ..providers.source_adapters import (
    ConstantScalarSampler,
    SourceFlowSampler,
    SourceNormalSampler,
    SourceScalarSampler,
    ZeroFlowSampler,
)


def _canonical_normal_velocity_policy(policy: str) -> str:
    text = str(policy or 'keep').strip().lower()
    aliases = {
        'none': 'keep',
        'as_is': 'keep',
        'preserve': 'keep',
        'outward_only': 'reflect_inward',
        'reflect': 'reflect_inward',
        'zero_inward': 'remove_inward_normal',
        'clip_inward': 'remove_inward_normal',
    }
    text = aliases.get(text, text)
    if text not in {'keep', 'reflect_inward', 'remove_inward_normal'}:
        raise ValueError(
            'source.preprocess.normal_velocity_policy must be keep, reflect_inward, or remove_inward_normal'
        )
    return text


def _normal_velocity_adjusted(
    velocity: np.ndarray,
    normal: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, float, float]:
    out = np.asarray(velocity, dtype=np.float64).copy()
    n = np.asarray(normal, dtype=np.float64)[: out.size]
    vn_before = float(np.dot(out, n))
    if vn_before < 0.0:
        if policy == 'reflect_inward':
            out = out - 2.0 * vn_before * n
        elif policy == 'remove_inward_normal':
            out = out - vn_before * n
    return out, vn_before, float(np.dot(out, n))


def _boundary_release_for_particle(
    classifier: Optional[Any],
    *,
    enabled: bool,
    position: np.ndarray,
    source_part_id: int,
    offset_m: float,
    tolerance_m: float,
) -> Any | None:
    if not bool(enabled) or classifier is None:
        return None
    release = classifier(
        np.asarray(position, dtype=np.float64),
        int(source_part_id),
        float(max(offset_m, 0.0)),
        float(max(tolerance_m, 0.0)),
    )
    return release if bool(getattr(release, 'is_on_boundary', False)) else None


def _boundary_release_diagnostics(
    boundary_release: Any | None,
    *,
    solver_offset_m: float,
    source_offset_m: float,
    capture_tolerance_m: float,
    inward_offset_m: float,
) -> Dict[str, float | int | str]:
    total_offset_m = float(solver_offset_m + source_offset_m) if boundary_release is not None else 0.0
    if boundary_release is None:
        return {
            'source_position_offset_m': float(source_offset_m),
            'boundary_release_solver_offset_m': 0.0,
            'boundary_release_total_offset_m': 0.0,
            'boundary_release_applied': 0,
            'boundary_release_distance_m': float('nan'),
            'boundary_release_part_id': 0,
            'boundary_release_primitive_id': -1,
            'boundary_release_ambiguous': 0,
            'boundary_release_inside_after_offset': 0,
            'boundary_release_capture_tolerance_m': float(capture_tolerance_m),
            'boundary_release_inward_offset_m': float(inward_offset_m),
            'capture_tolerance_m': float(capture_tolerance_m),
            'inward_offset_m': float(inward_offset_m),
            'projection_distance_m': float('nan'),
            'projected_part_id': 0,
            'projected_boundary_id': -1,
            'release_preprocess_mode': 'raw',
        }
    return {
        'source_position_offset_m': float(source_offset_m),
        'boundary_release_solver_offset_m': float(solver_offset_m),
        'boundary_release_total_offset_m': float(total_offset_m),
        'boundary_release_applied': 1,
        'boundary_release_distance_m': float(boundary_release.distance_m),
        'boundary_release_part_id': int(boundary_release.nearest_part_id),
        'boundary_release_primitive_id': int(boundary_release.primitive_id),
        'boundary_release_ambiguous': int(bool(boundary_release.ambiguous)),
        'boundary_release_inside_after_offset': int(bool(boundary_release.inside_after_offset)),
        'boundary_release_capture_tolerance_m': float(capture_tolerance_m),
        'boundary_release_inward_offset_m': float(inward_offset_m),
        'capture_tolerance_m': float(capture_tolerance_m),
        'inward_offset_m': float(inward_offset_m),
        'projection_distance_m': float(boundary_release.distance_m),
        'projected_part_id': int(boundary_release.nearest_part_id),
        'projected_boundary_id': int(boundary_release.primitive_id),
        'release_preprocess_mode': 'projected_inward_offset',
    }


def _position_fields(prefix: str, value: Any | None, spatial_dim: int) -> Dict[str, float]:
    axes = ('x', 'y', 'z')[: int(spatial_dim)]
    if value is None:
        return {f'{prefix}_{axis}': float('nan') for axis in axes}
    arr = np.asarray(value, dtype=np.float64)
    return {
        f'{prefix}_{axis}': float(arr[dim]) if dim < arr.size else float('nan')
        for dim, axis in enumerate(axes)
    }


def apply_source_models(
    particles: ParticleTable,
    resolved: SourceResolutionParameters,
    normal_sampler: SourceNormalSampler,
    flow_sampler: Optional[SourceFlowSampler] = None,
    wall_shear_sampler: Optional[SourceScalarSampler] = None,
    friction_velocity_sampler: Optional[SourceScalarSampler] = None,
    viscosity_sampler: Optional[SourceScalarSampler] = None,
    events: Optional[SourceEventTable] = None,
    process_steps: Optional[ProcessStepTable] = None,
    gas_density_kgm3: float = 1.0,
    seed: int = 12345,
    normal_velocity_policy: str = 'keep',
    release_point_classifier: Optional[Any] = None,
    use_boundary_release: bool = False,
    boundary_classifier_tolerance_m: float = 0.0,
    boundary_solver_offset_m: float = 0.0,
) -> SourcePreprocessResult:
    flow_sampler = flow_sampler or ZeroFlowSampler(particles.spatial_dim)
    wall_shear_sampler = wall_shear_sampler or ConstantScalarSampler(np.nan)
    friction_velocity_sampler = friction_velocity_sampler or ConstantScalarSampler(np.nan)
    viscosity_sampler = viscosity_sampler or ConstantScalarSampler(np.nan)

    pos = np.asarray(particles.position, dtype=np.float64).copy()
    vel = np.asarray(particles.velocity, dtype=np.float64).copy()
    rel = np.asarray(particles.release_time, dtype=np.float64).copy()
    diagnostics = []
    release_enabled = np.ones(particles.count, dtype=bool)
    active_events = events.active_rows() if events is not None else tuple()
    event_counter: Dict[str, int] = {}
    normal_policy = _canonical_normal_velocity_policy(normal_velocity_policy)

    for i in range(particles.count):
        pid = int(particles.particle_id[i])
        rng = np.random.default_rng(seed ^ ((pid + 1) * 0x9E3779B97F4A7C15 & 0xFFFFFFFF))
        law = resolved.resolved_law_name[i]
        original_position = pos[i, : particles.spatial_dim].copy()
        source_part_id = int(particles.source_part_id[i])
        configured_offset = float(resolved.source_position_offset_m[i])
        source_position_offset = float(configured_offset) if np.isfinite(configured_offset) and configured_offset > 0.0 else 0.0
        boundary_solver_offset = float(max(float(boundary_solver_offset_m), 0.0))
        boundary_offset = boundary_solver_offset + source_position_offset
        boundary_release = _boundary_release_for_particle(
            release_point_classifier,
            enabled=bool(use_boundary_release),
            position=original_position,
            source_part_id=source_part_id,
            offset_m=boundary_offset,
            tolerance_m=float(boundary_classifier_tolerance_m),
        )
        if boundary_release is not None:
            n = np.asarray(boundary_release.normal, dtype=np.float64)
        else:
            n = np.asarray(normal_sampler(pos[i], source_part_id), dtype=np.float64)
        n_mag = np.linalg.norm(n)
        if n_mag <= 1e-30:
            n = np.zeros(particles.spatial_dim, dtype=np.float64)
            n[0] = 1.0
        else:
            n = n / n_mag
        t1_3d, t2_3d = orthonormal_tangent_basis(np.pad(n, (0, max(0, 3 - n.size)), constant_values=0.0)[:3])
        t1 = t1_3d[: particles.spatial_dim]
        t2 = t2_3d[: particles.spatial_dim]
        base_v = vel[i, : particles.spatial_dim].copy()
        speed_scale = float(resolved.source_speed_scale[i])
        rel_eff = float(rel[i])

        matched_names = []
        event_factor = 1.0
        gate_ok = True
        for row in active_events:
            if event_matches(
                row,
                pid,
                int(particles.source_part_id[i]),
                int(resolved.resolved_material_id[i]),
                law,
                str(resolved.resolved_event_tag[i]),
            ):
                eff = event_effect(row, rel_eff)
                rel_eff += float(eff['release_time_shift_s'])
                event_factor *= float(eff['factor'])
                gate_ok = gate_ok and bool(eff['gate'])
                matched_names.append(str(row.event_name))
                event_counter[str(row.event_name)] = event_counter.get(str(row.event_name), 0) + 1

        step = process_steps.active_at(rel_eff) if process_steps is not None else None

        diag = {
            'particle_id': pid,
            'source_part_id': source_part_id,
            'input_material_id': int(particles.material_id[i]),
            'resolved_material_id': int(resolved.resolved_material_id[i]),
            'law_name': law,
            'event_tag': str(resolved.resolved_event_tag[i]),
            'matched_events': ';'.join(matched_names),
            'event_factor': float(event_factor),
            'release_enabled': 1,
            'particle_mass_kg': float(particles.mass[i]),
            'particle_diameter_m': float(particles.diameter[i]),
            'original_release_time_s': float(particles.release_time[i]),
            'step_name': step.step_name if step is not None else '',
        }
        for dim, axis in enumerate(('x', 'y', 'z')[: particles.spatial_dim]):
            diag[f'original_{axis}'] = float(original_position[dim])
            diag[f'raw_{axis}'] = float(original_position[dim])

        if not gate_ok:
            release_enabled[i] = False
            rel[i] = np.inf
            vel[i, : particles.spatial_dim] = base_v
            diag.update(
                {
                    'release_enabled': 0,
                    'suppression_reason': 'event_gate',
                    'offset_m': 0.0,
                    'source_delta_speed_mps': 0.0,
                    'final_speed_mps': float(np.linalg.norm(base_v)),
                    'resolved_release_time_s': float('inf'),
                }
            )
            diagnostics.append(diag)
            continue

        apply_offset = True
        source_delta = np.zeros(particles.spatial_dim, dtype=np.float64)

        if law == 'explicit_csv':
            source_delta[:] = 0.0

        elif law == 'flake_normal_escape_material':
            vn = sample_positive_normal(
                rng,
                float(resolved.source_normal_speed_mean_mps[i]) * max(0.0, float(resolved.source_flake_weight[i])) * event_factor,
                float(resolved.source_normal_speed_std_mps[i]) * max(1.0, event_factor),
            )
            vt_scale = max(0.0, float(resolved.source_tangent_speed_std_mps[i])) * max(1.0, math.sqrt(max(event_factor, 0.0)))
            vt1 = float(rng.normal(scale=vt_scale))
            vt2 = float(rng.normal(scale=vt_scale)) if particles.spatial_dim == 3 else 0.0
            source_delta = speed_scale * (vn * n[: particles.spatial_dim] + vt1 * t1 + vt2 * t2)
            diag.update({'vn_mps': vn, 'vt1_mps': vt1, 'vt2_mps': vt2})

        elif law == 'flake_burst_material':
            burst = burst_factor(
                rel_eff,
                float(resolved.source_burst_center_s[i]),
                float(resolved.source_burst_sigma_s[i]),
                float(resolved.source_burst_amplitude[i]),
                float(resolved.source_burst_period_s[i]),
                float(resolved.source_burst_phase_s[i]),
                float(resolved.source_burst_min_factor[i]),
                float(resolved.source_burst_max_factor[i]),
            )
            total_burst = burst * event_factor
            vn = sample_positive_normal(
                rng,
                float(resolved.source_normal_speed_mean_mps[i]) * max(0.0, float(resolved.source_flake_weight[i])) * total_burst,
                float(resolved.source_normal_speed_std_mps[i]) * max(1.0, total_burst),
            )
            vt_scale = max(0.0, float(resolved.source_tangent_speed_std_mps[i])) * max(1.0, math.sqrt(max(total_burst, 0.0)))
            vt1 = float(rng.normal(scale=vt_scale))
            vt2 = float(rng.normal(scale=vt_scale)) if particles.spatial_dim == 3 else 0.0
            source_delta = speed_scale * (vn * n[: particles.spatial_dim] + vt1 * t1 + vt2 * t2)
            diag.update({'burst_factor': burst, 'total_burst_factor': total_burst, 'vn_mps': vn, 'vt1_mps': vt1, 'vt2_mps': vt2})

        elif law == 'resuspension_shear_material':
            u_local = np.asarray(flow_sampler(pos[i, : particles.spatial_dim], rel_eff), dtype=np.float64)[: particles.spatial_dim]
            ut = u_local - float(np.dot(u_local, n[: particles.spatial_dim])) * n[: particles.spatial_dim]
            ut_norm = float(np.linalg.norm(ut))
            tau_direct = float(wall_shear_sampler(pos[i, : particles.spatial_dim], rel_eff, int(particles.source_part_id[i])))
            utau_direct = float(friction_velocity_sampler(pos[i, : particles.spatial_dim], rel_eff, int(particles.source_part_id[i])))
            mu_local = float(viscosity_sampler(pos[i, : particles.spatial_dim], rel_eff, int(particles.source_part_id[i])))
            if not is_finite(mu_local):
                mu_local = float(resolved.source_dynamic_viscosity_Pas[i])
            gas_rho = max(float(gas_density_kgm3), 1e-30)
            corr_len = max(float(resolved.source_roughness_corr_length_m[i]), 0.0)
            slope_rms = resolved_slope_rms(
                float(resolved.source_roughness_rms[i]),
                corr_len,
                float(resolved.source_roughness_slope_rms[i]),
            )
            shear_len = max(float(resolved.source_resuspension_shear_length_m[i]), corr_len, 0.25 * float(particles.diameter[i]), 1e-8)
            if is_finite(tau_direct) and float(tau_direct) >= 0.0:
                tau_proxy = float(tau_direct)
                utau_proxy = math.sqrt(max(tau_proxy, 0.0) / gas_rho)
                shear_source = 'direct_tauw'
            elif is_finite(utau_direct) and float(utau_direct) >= 0.0:
                utau_proxy = float(utau_direct)
                tau_proxy = gas_rho * utau_proxy * utau_proxy
                shear_source = 'direct_utau'
            else:
                tau_proxy = abs(mu_local) * ut_norm / shear_len
                utau_proxy = math.sqrt(max(tau_proxy, 0.0) / gas_rho)
                shear_source = 'mu_ut_over_L'
            threshold_speed = effective_resuspension_threshold_speed(
                float(resolved.source_resuspension_speed_threshold_mps[i]),
                float(resolved.source_roughness_rms[i]),
                float(resolved.source_adhesion_energy_Jm2[i]),
                float(resolved.source_resuspension_roughness_scale[i]),
                float(resolved.source_resuspension_adhesion_scale[i]),
            )
            threshold_tau = effective_resuspension_threshold_tau(
                float(resolved.source_resuspension_tau_threshold_Pa[i]),
                float(resolved.source_roughness_rms[i]),
                corr_len,
                slope_rms,
                float(resolved.source_adhesion_energy_Jm2[i]),
                float(resolved.source_resuspension_tau_roughness_scale[i]),
                float(resolved.source_resuspension_tau_adhesion_scale[i]),
                float(resolved.source_resuspension_tau_slope_scale[i]),
            )
            explicit_utau_thresh = max(0.0, float(resolved.source_resuspension_utau_threshold_mps[i]))
            if explicit_utau_thresh > 0.0:
                threshold_utau = explicit_utau_thresh
            elif threshold_tau > 0.0:
                threshold_utau = math.sqrt(max(threshold_tau, 0.0) / gas_rho)
            else:
                threshold_utau = 0.0
            diag.update(
                {
                    'u_tangent_mps': ut_norm,
                    'tau_proxy_Pa': tau_proxy,
                    'u_tau_proxy_mps': utau_proxy,
                    'tau_threshold_Pa': threshold_tau,
                    'u_tau_threshold_mps': threshold_utau,
                    'speed_threshold_mps': threshold_speed,
                    'roughness_slope_rms': slope_rms,
                    'shear_length_m': shear_len,
                    'shear_source': shear_source,
                }
            )
            suppressed = False
            if threshold_tau > 0.0 and tau_proxy + 1e-18 < threshold_tau:
                suppressed = True
                diag['suppression_reason'] = 'tau_threshold'
            if threshold_utau > 0.0 and utau_proxy + 1e-18 < threshold_utau:
                suppressed = True
                diag['suppression_reason'] = diag.get('suppression_reason', 'u_tau_threshold')
            if threshold_speed > 0.0 and ut_norm + 1e-18 < threshold_speed:
                suppressed = True
                diag['suppression_reason'] = diag.get('suppression_reason', 'speed_threshold')
            if suppressed:
                release_enabled[i] = False
                rel[i] = np.inf
                apply_offset = False
                source_delta[:] = 0.0
                diag.update({'release_enabled': 0})
            else:
                vn = sample_positive_normal(
                    rng,
                    float(resolved.source_resuspension_normal_speed_mean_mps[i]) * max(1.0, event_factor),
                    float(resolved.source_resuspension_normal_speed_std_mps[i]),
                )
                vt_scale = max(0.0, float(resolved.source_tangent_speed_std_mps[i]))
                jitter1 = float(rng.normal(scale=vt_scale))
                jitter2 = float(rng.normal(scale=vt_scale)) if particles.spatial_dim == 3 else 0.0
                source_delta = speed_scale * (
                    float(resolved.source_resuspension_velocity_scale[i]) * event_factor * ut
                    + vn * n[: particles.spatial_dim]
                    + jitter1 * t1
                    + jitter2 * t2
                )
                diag.update({'vn_mps': vn, 'jitter1_mps': jitter1, 'jitter2_mps': jitter2})

        elif law == 'thermal_reemission_source_material':
            vtherm = sample_thermal_velocity(
                rng,
                np.pad(n[: particles.spatial_dim], (0, max(0, 3 - particles.spatial_dim)), constant_values=0.0),
                float(particles.mass[i]),
                float(resolved.source_temperature_K[i]),
                float(resolved.source_thermal_accommodation[i]),
            )[: particles.spatial_dim]
            source_delta = speed_scale * max(event_factor, 0.0) * vtherm
            diag.update({'thermal_speed_mps': float(np.linalg.norm(vtherm)), 'temperature_K': float(resolved.source_temperature_K[i])})
        else:
            raise KeyError(f'Unsupported source law in apply_source_models: {law}')

        offset = 0.0
        if apply_offset:
            if boundary_release is not None:
                offset = float(boundary_offset)
                offset_position = getattr(boundary_release, 'offset_position', None)
                if offset_position is not None:
                    pos[i, : particles.spatial_dim] = np.asarray(offset_position, dtype=np.float64)[: particles.spatial_dim]
                else:
                    pos[i, : particles.spatial_dim] += offset * n[: particles.spatial_dim]
            else:
                offset = configured_offset if np.isfinite(configured_offset) else 0.0
                pos[i, : particles.spatial_dim] += offset * n[: particles.spatial_dim]
        final_v = base_v + source_delta
        normal_before = float(np.dot(final_v, n[: particles.spatial_dim]))
        normal_after = normal_before
        if bool(release_enabled[i]) and normal_policy != 'keep':
            final_v, normal_before, normal_after = _normal_velocity_adjusted(
                final_v,
                n[: particles.spatial_dim],
                normal_policy,
            )
        vel[i, : particles.spatial_dim] = final_v
        rel[i] = rel_eff if np.isfinite(rel_eff) else rel[i]
        diag['offset_m'] = offset
        diag['source_delta_speed_mps'] = float(np.linalg.norm(source_delta))
        diag['normal_velocity_policy'] = normal_policy
        diag['normal_velocity_before_mps'] = normal_before
        diag['normal_velocity_after_mps'] = normal_after
        diag.update(
            _boundary_release_diagnostics(
                boundary_release,
                solver_offset_m=boundary_solver_offset,
                source_offset_m=source_position_offset,
                capture_tolerance_m=float(boundary_classifier_tolerance_m),
                inward_offset_m=boundary_solver_offset,
            )
        )
        diag['source_provenance_group'] = source_provenance_group(
            source_part_id,
            production_generated=int(diag.get('boundary_release_applied', 0)) != 0,
        )
        diag.update(_position_fields('projected', getattr(boundary_release, 'boundary_position', None), particles.spatial_dim))
        diag['final_speed_mps'] = float(np.linalg.norm(vel[i, : particles.spatial_dim]))
        diag['resolved_release_time_s'] = float(rel[i]) if np.isfinite(rel[i]) else float('inf')
        for dim, axis in enumerate(('x', 'y', 'z')[: particles.spatial_dim]):
            diag[f'release_{axis}'] = float(pos[i, dim])
            diag[f'initial_after_preprocess_{axis}'] = float(pos[i, dim])
        diagnostics.append(diag)

    particles_out = ParticleTable(
        spatial_dim=particles.spatial_dim,
        particle_id=particles.particle_id.copy(),
        position=pos,
        velocity=vel,
        release_time=rel,
        mass=particles.mass.copy(),
        diameter=particles.diameter.copy(),
        density=particles.density.copy(),
        charge=particles.charge.copy(),
        source_part_id=particles.source_part_id.copy(),
        material_id=particles.material_id.copy(),
        source_event_tag=particles.source_event_tag.copy(),
        source_law_override=particles.source_law_override.copy(),
        source_speed_scale_override=particles.source_speed_scale_override.copy(),
        stick_probability=particles.stick_probability.copy(),
        dep_particle_rel_permittivity=particles.dep_particle_rel_permittivity.copy(),
        thermophoretic_coeff=particles.thermophoretic_coeff.copy(),
        metadata=dict(particles.metadata),
    )

    law_counts: Dict[str, int] = {}
    provenance_counts: Dict[str, int] = {}
    boundary_release_applied_count = 0
    boundary_release_failed_offset_count = 0
    projection_distances = []
    for row in diagnostics:
        law_counts[str(row.get('law_name', ''))] = law_counts.get(str(row.get('law_name', '')), 0) + 1
        provenance = str(row.get('source_provenance_group', source_provenance_group(int(row.get('source_part_id', 0)))))
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
        if int(row.get('boundary_release_applied', 0)) != 0:
            boundary_release_applied_count += 1
            distance = float(row.get('projection_distance_m', np.nan))
            if np.isfinite(distance):
                projection_distances.append(distance)
            if int(row.get('boundary_release_inside_after_offset', 0)) == 0:
                boundary_release_failed_offset_count += 1
    projection_array = np.asarray(projection_distances, dtype=np.float64)
    summary = {
        'law_usage': law_counts,
        'source_provenance_counts': provenance_counts,
        'particle_count': int(particles.count),
        'suppressed_particle_count': int((~release_enabled).sum()),
        'boundary_release_enabled': int(bool(use_boundary_release)),
        'boundary_release_capture_tolerance_m': float(max(float(boundary_classifier_tolerance_m), 0.0)),
        'boundary_release_inward_offset_m': float(max(float(boundary_solver_offset_m), 0.0)),
        'boundary_release_applied_count': int(boundary_release_applied_count),
        'boundary_release_failed_offset_count': int(boundary_release_failed_offset_count),
        'boundary_release_projection_distance_count': int(projection_array.size),
        'boundary_release_projection_distance_max_m': (
            float(np.max(projection_array)) if projection_array.size else None
        ),
        'boundary_release_projection_distance_mean_m': (
            float(np.mean(projection_array)) if projection_array.size else None
        ),
    }
    event_summary = {'matched_event_counts': event_counter, 'total_matches': int(sum(event_counter.values()))}
    return SourcePreprocessResult(
        particles=particles_out,
        resolved=resolved,
        source_model_summary=summary,
        diagnostics_rows=tuple(diagnostics),
        release_enabled=release_enabled,
        event_summary=event_summary,
    )


__all__ = ('apply_source_models',)
