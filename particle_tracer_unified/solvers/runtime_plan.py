from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from ..core.datamodel import PreparedRuntime
from ..core.integrator_registry import IntegratorSpec, get_integrator_spec
from .integrator_common import drag_model_mode_from_name, drag_model_name_from_mode


OUTPUT_MODE_MINIMAL = 'minimal'
OUTPUT_MODE_STANDARD = 'standard'
OUTPUT_MODE_DEBUG = 'debug'
SUPPORTED_OUTPUT_MODES = (OUTPUT_MODE_MINIMAL, OUTPUT_MODE_STANDARD, OUTPUT_MODE_DEBUG)

VALID_MASK_POLICY_DIAGNOSTIC = 'diagnostic'
VALID_MASK_POLICY_RETRY_THEN_STOP = 'retry_then_stop'
VALID_MASK_POLICY_STRICT_CLEAN = 'strict_clean'
VALID_MASK_POLICIES = (
    VALID_MASK_POLICY_DIAGNOSTIC,
    VALID_MASK_POLICY_RETRY_THEN_STOP,
    VALID_MASK_POLICY_STRICT_CLEAN,
)


@dataclass(frozen=True)
class FieldSamplePlan:
    """Fields that should be sampled together for a solver step.

    This is intentionally small. It is not a provider contract and it should not
    validate exported field support. Existing provider/input checks remain the
    source of truth.
    """

    need_flow: bool = True
    need_electric: bool = False
    need_gas_properties: bool = False
    need_valid_mask: bool = True

    @property
    def needs_gas_properties(self) -> bool:
        return bool(self.need_gas_properties)


@dataclass(frozen=True)
class OutputPlan:
    """Lightweight output policy for large particle runs."""

    mode: str = OUTPUT_MODE_STANDARD
    save_trajectory: bool = False
    write_wall_events: bool = False
    write_step_summary: bool = False
    write_force_contributions: bool = False
    write_collision_diagnostics: bool = False
    save_every: int = 10

    @property
    def is_minimal(self) -> bool:
        return self.mode == OUTPUT_MODE_MINIMAL

    @property
    def is_debug(self) -> bool:
        return self.mode == OUTPUT_MODE_DEBUG


@dataclass(frozen=True)
class ReleaseGracePlan:
    """Short opt-in same-source wall bypass immediately after release."""

    enabled: bool = False
    grace_time_s: float = 0.0
    clearance_m: float = 0.0
    min_outward_normal_speed_mps: float = 0.0


@dataclass(frozen=True)
class SolverPlan:
    """Immutable runtime decisions derived from PreparedRuntime.

    The intent is to move repeated config/string lookups out of the hot solver
    loop without adding a broad new validation layer.
    """

    spatial_dim: int
    dt: float
    t_end: float
    base_save_every: int
    plot_limit: int
    rng_seed: int
    integrator_spec: IntegratorSpec
    valid_mask_policy: str
    max_wall_hits_per_step: int
    min_remaining_dt_ratio: float
    adaptive_substep_enabled: int
    adaptive_substep_tau_ratio: float
    adaptive_substep_max_splits: int
    epsilon_offset_m: float
    on_boundary_tol_m: float
    release_grace: ReleaseGracePlan
    boundary_broad_phase_enabled: bool
    drag_model_mode: int
    drag_model_name: str
    field_sample: FieldSamplePlan
    output: OutputPlan

@dataclass(frozen=True)
class ReleaseSchedule:
    """Sorted release order used to avoid scanning all particles every step."""

    order: np.ndarray
    release_time_s: np.ndarray

    @property
    def count(self) -> int:
        return int(self.order.size)


@dataclass
class ReleaseCursor:
    schedule: ReleaseSchedule
    position: int = 0

    @property
    def done(self) -> bool:
        return int(self.position) >= int(self.schedule.count)

    def next_time(self) -> float:
        if self.done:
            return float('inf')
        index = int(self.schedule.order[int(self.position)])
        return float(self.schedule.release_time_s[index])


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config(prepared: PreparedRuntime) -> Mapping[str, Any]:
    return _as_mapping(getattr(prepared.runtime, 'config_payload', {}))


def _solver_cfg(prepared: PreparedRuntime) -> Mapping[str, Any]:
    config_payload = _config(prepared)
    raw_solver_cfg = _as_mapping(config_payload.get('solver', {}))
    try:
        from .forces import solver_cfg_with_force_overrides

        return solver_cfg_with_force_overrides(raw_solver_cfg, getattr(prepared.runtime, 'force_catalog', None))
    except Exception:
        return raw_solver_cfg


def _bool_flag(mapping: Mapping[str, Any], name: str, default: bool) -> bool:
    value = mapping.get(name, default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'off'}:
            return False
    return bool(value)


def _resolve_valid_mask_policy(value: object) -> str:
    text = str(value if value is not None else VALID_MASK_POLICY_RETRY_THEN_STOP).strip().lower()
    if text not in VALID_MASK_POLICIES:
        allowed = "', '".join(VALID_MASK_POLICIES)
        raise ValueError(f"solver.valid_mask_policy must be one of '{allowed}'")
    return text


def _positive_int(value: object, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(max(1, out))


def _adaptive_tau_ratio(value: object) -> float:
    ratio = float(value)
    if not np.isfinite(ratio):
        ratio = 0.5
    return float(max(ratio, 1.0e-8))


def _min_remaining_dt_ratio(value: object) -> float:
    ratio = float(value)
    if not np.isfinite(ratio):
        ratio = 0.05
    return float(np.clip(ratio, 0.0, 1.0))


def _release_grace_float(value: object, *, name: str, default: float, positive: bool = False) -> float:
    raw = default if value is None else value
    try:
        out = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be numeric') from None
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    if bool(positive) and out <= 0.0:
        raise ValueError(f'{name} must be > 0')
    if not bool(positive) and out < 0.0:
        raise ValueError(f'{name} must be >= 0')
    return float(out)


def _build_release_grace_plan(
    solver_cfg: Mapping[str, Any],
    *,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
) -> ReleaseGracePlan:
    raw_cfg = solver_cfg.get('release_grace', {})
    if raw_cfg is None or raw_cfg == '':
        raw_cfg = {}
    if not isinstance(raw_cfg, Mapping):
        raise ValueError('solver.release_grace must be a mapping')
    grace_cfg = _as_mapping(raw_cfg)
    default_clearance = max(float(epsilon_offset_m), float(on_boundary_tol_m))
    enabled = _bool_flag(grace_cfg, 'enabled', False)
    min_speed = _release_grace_float(
        grace_cfg.get('min_outward_normal_speed_mps', 0.0),
        name='solver.release_grace.min_outward_normal_speed_mps',
        default=0.0,
    )
    clearance_m = _release_grace_float(
        grace_cfg.get('clearance_m', None),
        name='solver.release_grace.clearance_m',
        default=default_clearance,
    )
    if not bool(enabled):
        return ReleaseGracePlan(
            enabled=False,
            grace_time_s=0.0,
            clearance_m=float(clearance_m),
            min_outward_normal_speed_mps=float(min_speed),
        )
    grace_time_s = _release_grace_float(
        grace_cfg.get('grace_time_s', None),
        name='solver.release_grace.grace_time_s',
        default=0.0,
        positive=True,
    )
    return ReleaseGracePlan(
        enabled=True,
        grace_time_s=float(grace_time_s),
        clearance_m=float(clearance_m),
        min_outward_normal_speed_mps=float(min_speed),
    )


def normalize_output_mode(value: object, *, default: str = OUTPUT_MODE_STANDARD) -> str:
    text = str(value if value is not None else default).strip().lower()
    if text == 'full':
        return OUTPUT_MODE_DEBUG
    if text in {'diagnostic', 'diagnostics'}:
        return OUTPUT_MODE_DEBUG
    if text not in SUPPORTED_OUTPUT_MODES:
        raise ValueError(
            "output.mode must be one of 'minimal', 'standard', or 'debug' "
            "(legacy output.artifact_mode also accepts 'full')"
        )
    return text


def _is_comsol_faithful_config_payload(config_payload: Mapping[str, Any]) -> bool:
    mode = str(config_payload.get('mode', '')).strip().lower()
    if mode == 'comsol_faithful':
        return True
    comsol_cfg = _as_mapping(config_payload.get('comsol', {}))
    return bool(comsol_cfg.get('manifest'))


def _output_mode_from_config(config_payload: Mapping[str, Any], *, force_debug: bool = False) -> str:
    output_cfg = _as_mapping(config_payload.get('output', {}))
    if bool(force_debug):
        return OUTPUT_MODE_DEBUG
    if 'artifact_mode' in output_cfg:
        return normalize_output_mode(output_cfg.get('artifact_mode'), default=OUTPUT_MODE_STANDARD)
    if 'mode' in output_cfg:
        return normalize_output_mode(output_cfg.get('mode'), default=OUTPUT_MODE_STANDARD)
    return OUTPUT_MODE_STANDARD


def build_output_plan(
    config_payload: Mapping[str, Any],
    *,
    default_save_every: int,
    force_debug: bool = False,
) -> OutputPlan:
    output_cfg = _as_mapping(config_payload.get('output', {}))
    solver_cfg = _as_mapping(config_payload.get('solver', {}))
    mode = _output_mode_from_config(config_payload, force_debug=force_debug)
    save_every = _positive_int(output_cfg.get('save_every', solver_cfg.get('save_every', default_save_every)), default_save_every)

    if mode == OUTPUT_MODE_MINIMAL:
        return OutputPlan(
            mode=mode,
            save_trajectory=_bool_flag(output_cfg, 'save_trajectory', False),
            write_wall_events=_bool_flag(output_cfg, 'write_wall_events', False),
            write_step_summary=_bool_flag(output_cfg, 'write_step_summary', False),
            write_force_contributions=_bool_flag(output_cfg, 'write_force_contributions', False),
            write_collision_diagnostics=_bool_flag(output_cfg, 'write_collision_diagnostics', False),
            save_every=save_every,
        )
    if mode == OUTPUT_MODE_DEBUG:
        return OutputPlan(
            mode=mode,
            save_trajectory=_bool_flag(output_cfg, 'save_trajectory', True),
            write_wall_events=_bool_flag(output_cfg, 'write_wall_events', True),
            write_step_summary=_bool_flag(output_cfg, 'write_step_summary', True),
            write_force_contributions=_bool_flag(output_cfg, 'write_force_contributions', True),
            write_collision_diagnostics=_bool_flag(output_cfg, 'write_collision_diagnostics', True),
            save_every=save_every,
        )
    return OutputPlan(
        mode=mode,
        save_trajectory=_bool_flag(output_cfg, 'save_trajectory', False),
        write_wall_events=_bool_flag(output_cfg, 'write_wall_events', False),
        write_step_summary=_bool_flag(output_cfg, 'write_step_summary', False),
        write_force_contributions=_bool_flag(output_cfg, 'write_force_contributions', False),
        write_collision_diagnostics=_bool_flag(output_cfg, 'write_collision_diagnostics', False),
        save_every=save_every,
    )


def _enabled_force_names(prepared: PreparedRuntime) -> Tuple[str, ...]:
    catalog = getattr(prepared.runtime, 'force_catalog', None)
    if catalog is None:
        return ()
    try:
        from .forces import force_catalog_summary

        summary = force_catalog_summary(catalog)
    except Exception:
        return ()
    names = summary.get('enabled_forces', ()) if isinstance(summary, Mapping) else ()
    return tuple(str(name).strip().lower() for name in names if str(name).strip())


def build_field_sample_plan(prepared: PreparedRuntime) -> FieldSamplePlan:
    solver_cfg = _solver_cfg(prepared)
    drag_model = str(solver_cfg.get('drag_model', 'stokes')).strip().lower()
    enabled_force_names = _enabled_force_names(prepared)
    enabled = set(enabled_force_names)
    if not enabled:
        enabled_force_names = ('drag',)
        enabled = {'drag'}

    try:
        from .charge_model import parse_charge_model_config
        from .forces import force_runtime_parameters_from_catalog
        from .stochastic_motion import parse_stochastic_motion_config

        charge_cfg = parse_charge_model_config(solver_cfg)
        stochastic_cfg = parse_stochastic_motion_config(
            solver_cfg,
            default_seed=int(solver_cfg.get('seed', 12345)),
        )
        force_runtime = force_runtime_parameters_from_catalog(getattr(prepared.runtime, 'force_catalog', None))
    except Exception:
        charge_cfg = None
        stochastic_cfg = None
        force_runtime = None

    charge_enabled = bool(getattr(charge_cfg, 'enabled', False))
    charge_uses_field_background = bool(
        charge_enabled and str(getattr(charge_cfg, 'background_source', 'field')) == 'field'
    )
    stochastic_enabled = bool(getattr(stochastic_cfg, 'enabled', False))
    stochastic_field_temperature = bool(
        stochastic_enabled and str(getattr(stochastic_cfg, 'temperature_source', 'field_T_then_gas')) == 'field_T_then_gas'
    )

    thermophoresis_enabled = 'thermophoresis' in enabled or bool(getattr(force_runtime, 'thermophoresis_enabled', False))
    dielectrophoresis_enabled = 'dielectrophoresis' in enabled or bool(
        getattr(force_runtime, 'dielectrophoresis_enabled', False)
    )
    lift_enabled = 'lift' in enabled or bool(getattr(force_runtime, 'lift_enabled', False))
    pressure_gradient_enabled = 'pressure_gradient' in enabled or bool(
        getattr(force_runtime, 'pressure_gradient_enabled', False)
    )
    virtual_mass_enabled = 'virtual_mass' in enabled or bool(getattr(force_runtime, 'virtual_mass_enabled', False))
    gravity_buoyancy_enabled = bool(getattr(force_runtime, 'gravity_buoyancy_enabled', False))

    need_flow = 'drag' in enabled or lift_enabled or pressure_gradient_enabled or virtual_mass_enabled or stochastic_enabled
    need_electric = (
        'electric' in enabled
        or dielectrophoresis_enabled
        or charge_enabled
        or _bool_flag(solver_cfg, 'electric_enabled', False)
    )
    drag_uses_nonstokes_gas = drag_model in {'epstein', 'schiller_naumann', 'stokes_cunningham'}
    need_gas_density = bool(
        drag_uses_nonstokes_gas
        or stochastic_enabled
        or pressure_gradient_enabled
        or virtual_mass_enabled
        or gravity_buoyancy_enabled
    )
    need_gas_viscosity = bool('drag' in enabled or stochastic_enabled or thermophoresis_enabled or lift_enabled)
    need_gas_temperature = bool(
        drag_model in {'epstein', 'stokes_cunningham'}
        or stochastic_field_temperature
        or thermophoresis_enabled
        or charge_uses_field_background
    )
    valid_mask_policy = _resolve_valid_mask_policy(
        solver_cfg.get('valid_mask_policy', VALID_MASK_POLICY_RETRY_THEN_STOP)
    )
    need_valid_mask = valid_mask_policy in VALID_MASK_POLICIES

    return FieldSamplePlan(
        need_flow=bool(need_flow),
        need_electric=bool(need_electric),
        need_gas_properties=bool(need_gas_density or need_gas_viscosity or need_gas_temperature),
        need_valid_mask=bool(need_valid_mask),
    )


def build_release_schedule(release_time_s: np.ndarray) -> ReleaseSchedule:
    release_time = np.asarray(release_time_s, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(release_time))
    if finite.size == 0:
        return ReleaseSchedule(order=np.zeros(0, dtype=np.int64), release_time_s=release_time)
    order = finite[np.argsort(release_time[finite], kind='mergesort')].astype(np.int64, copy=False)
    return ReleaseSchedule(order=order, release_time_s=release_time)


def build_solver_plan(prepared: PreparedRuntime, spatial_dim: Optional[int] = None) -> SolverPlan:
    runtime = prepared.runtime
    config_payload = _config(prepared)
    solver_cfg = _solver_cfg(prepared)
    wall_cfg = _as_mapping(config_payload.get('wall', {}))

    dim = int(spatial_dim if spatial_dim is not None else runtime.spatial_dim)
    dt = float(solver_cfg.get('dt', 1.0e-3))
    t_end = float(solver_cfg.get('t_end', 0.1))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError('solver.dt must be finite and > 0')
    if not np.isfinite(t_end) or t_end < 0.0:
        raise ValueError('solver.t_end must be finite and >= 0')

    save_every = _positive_int(solver_cfg.get('save_every', 10), 10)
    integrator_spec = get_integrator_spec(solver_cfg.get('integrator', 'drag_relaxation'))
    drag_model_mode = drag_model_mode_from_name(solver_cfg.get('drag_model', 'stokes'))
    drag_model_name = drag_model_name_from_mode(int(drag_model_mode))
    valid_mask_policy = _resolve_valid_mask_policy(
        solver_cfg.get('valid_mask_policy', VALID_MASK_POLICY_RETRY_THEN_STOP)
    )

    min_remaining_dt_ratio = _min_remaining_dt_ratio(solver_cfg.get('min_remaining_dt_ratio', 0.05))
    adaptive_substep_tau_ratio = _adaptive_tau_ratio(solver_cfg.get('adaptive_substep_tau_ratio', 0.5))
    adaptive_substep_enabled = int(_bool_flag(solver_cfg, 'adaptive_substep_enabled', False))
    adaptive_substep_max_splits = int(max(0, solver_cfg.get('adaptive_substep_max_splits', 4)))

    epsilon_offset_m = float(wall_cfg.get('epsilon_offset_m', 1.0e-6))
    on_boundary_tol_raw = solver_cfg.get('on_boundary_tol_m', None)
    if on_boundary_tol_raw is None:
        on_boundary_tol_m = max(2.0 * epsilon_offset_m, 5.0e-7)
    else:
        on_boundary_tol_m = max(float(on_boundary_tol_raw), 0.0)
    release_grace = _build_release_grace_plan(
        solver_cfg,
        epsilon_offset_m=float(epsilon_offset_m),
        on_boundary_tol_m=float(on_boundary_tol_m),
    )

    output = build_output_plan(
        config_payload,
        default_save_every=save_every,
        force_debug=_is_comsol_faithful_config_payload(config_payload),
    )
    field_sample = build_field_sample_plan(prepared)
    is_comsol_faithful = _is_comsol_faithful_config_payload(config_payload)
    boundary_broad_phase_enabled = bool(_bool_flag(solver_cfg, 'boundary_broad_phase_enabled', False))
    if is_comsol_faithful:
        boundary_broad_phase_enabled = False

    return SolverPlan(
        spatial_dim=dim,
        dt=dt,
        t_end=t_end,
        base_save_every=save_every,
        plot_limit=int(solver_cfg.get('plot_particle_limit', 32)),
        rng_seed=int(solver_cfg.get('seed', 12345)),
        integrator_spec=integrator_spec,
        valid_mask_policy=valid_mask_policy,
        max_wall_hits_per_step=_positive_int(solver_cfg.get('max_wall_hits_per_step', 5), 5),
        min_remaining_dt_ratio=float(min_remaining_dt_ratio),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        epsilon_offset_m=epsilon_offset_m,
        on_boundary_tol_m=on_boundary_tol_m,
        release_grace=release_grace,
        boundary_broad_phase_enabled=bool(boundary_broad_phase_enabled),
        drag_model_mode=int(drag_model_mode),
        drag_model_name=str(drag_model_name),
        field_sample=field_sample,
        output=output,
    )
