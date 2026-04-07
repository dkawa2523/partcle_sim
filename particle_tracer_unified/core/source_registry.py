from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .source_schema import (
    FLAKE_BURST_SOURCE_PARAMETERS,
    FLAKE_ESCAPE_SOURCE_PARAMETERS,
    RESUSPENSION_SOURCE_PARAMETERS,
    THERMAL_REEMISSION_SOURCE_PARAMETERS,
)


@dataclass(frozen=True)
class SourceLawDef:
    name: str
    code: int
    description: str
    parameters: Tuple[str, ...] = ()


SOURCE_LAWS: Dict[str, SourceLawDef] = {
    'explicit_csv': SourceLawDef(
        'explicit_csv',
        1,
        'Use particle positions and velocities exactly as provided in particles.csv.',
    ),
    'flake_normal_escape_material': SourceLawDef(
        'flake_normal_escape_material',
        2,
        'Generate an outward normal escape component plus tangential scatter. Suitable for flake detachment from coated parts.',
        FLAKE_ESCAPE_SOURCE_PARAMETERS,
    ),

    'flake_burst_material': SourceLawDef(
        'flake_burst_material',
        5,
        'Normal escape source with burst envelope in time, suitable for recipe transitions or crack-flake ejection bursts.',
        FLAKE_BURST_SOURCE_PARAMETERS,
    ),
    'resuspension_shear_material': SourceLawDef(
        'resuspension_shear_material',
        3,
        'Generate initial velocity from local flow/shear plus a small outward normal lift-off component.',
        RESUSPENSION_SOURCE_PARAMETERS,
    ),
    'thermal_reemission_source_material': SourceLawDef(
        'thermal_reemission_source_material',
        4,
        'Sample a half-space thermal distribution from source temperature and accommodation.',
        THERMAL_REEMISSION_SOURCE_PARAMETERS,
    ),
}


def get_source_law(name: str) -> SourceLawDef:
    key = str(name).strip()
    if key not in SOURCE_LAWS:
        raise KeyError(f'Unknown source law: {key}')
    return SOURCE_LAWS[key]
