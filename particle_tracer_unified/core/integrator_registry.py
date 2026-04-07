from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

DEFAULT_INTEGRATOR = 'drag_relaxation'


@dataclass(frozen=True)
class IntegratorSpec:
    name: str
    mode: int
    order: int
    uses_midpoint_stage: bool
    supports_partial_replay: bool
    stage_point_count: int


INTEGRATOR_SPECS = (
    IntegratorSpec(
        name='drag_relaxation',
        mode=0,
        order=1,
        uses_midpoint_stage=False,
        supports_partial_replay=True,
        stage_point_count=1,
    ),
    IntegratorSpec(
        name='etd',
        mode=1,
        order=1,
        uses_midpoint_stage=False,
        supports_partial_replay=True,
        stage_point_count=1,
    ),
    IntegratorSpec(
        name='etd2',
        mode=2,
        order=2,
        uses_midpoint_stage=True,
        supports_partial_replay=True,
        stage_point_count=2,
    ),
)

INTEGRATOR_NAME_TO_SPEC: Dict[str, IntegratorSpec] = {spec.name: spec for spec in INTEGRATOR_SPECS}
INTEGRATOR_MODE_TO_SPEC: Dict[int, IntegratorSpec] = {int(spec.mode): spec for spec in INTEGRATOR_SPECS}
INTEGRATOR_NAME_TO_MODE: Dict[str, int] = {name: int(spec.mode) for name, spec in INTEGRATOR_NAME_TO_SPEC.items()}
INTEGRATOR_MODE_TO_NAME: Dict[int, str] = {mode: spec.name for mode, spec in INTEGRATOR_MODE_TO_SPEC.items()}
SUPPORTED_INTEGRATORS = tuple(INTEGRATOR_NAME_TO_SPEC.keys())


def normalize_integrator_name(value: object, *, default: str = DEFAULT_INTEGRATOR) -> str:
    name = str(default if value is None else value).strip().lower()
    if not name:
        return str(default)
    return name


def supported_integrators_text() -> str:
    return ', '.join(sorted(SUPPORTED_INTEGRATORS))


def validate_integrator_name(value: object, *, default: str = DEFAULT_INTEGRATOR) -> str:
    integrator = normalize_integrator_name(value, default=default)
    if integrator not in INTEGRATOR_NAME_TO_SPEC:
        raise ValueError(f'Unsupported solver.integrator: {integrator}. Supported values: {supported_integrators_text()}')
    return integrator


def get_integrator_spec(value: object, *, default: str = DEFAULT_INTEGRATOR) -> IntegratorSpec:
    integrator = validate_integrator_name(value, default=default)
    return INTEGRATOR_NAME_TO_SPEC[integrator]


def integrator_spec_from_mode(mode: object, *, default: str = DEFAULT_INTEGRATOR) -> IntegratorSpec:
    try:
        numeric_mode = int(mode)
    except Exception:
        return get_integrator_spec(default, default=default)
    return INTEGRATOR_MODE_TO_SPEC.get(numeric_mode, get_integrator_spec(default, default=default))


def integrator_mode_from_name(value: object, *, default: str = DEFAULT_INTEGRATOR) -> int:
    return int(get_integrator_spec(value, default=default).mode)
