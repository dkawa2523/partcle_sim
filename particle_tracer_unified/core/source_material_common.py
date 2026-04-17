from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .datamodel import SourceEventRow

KB = 1.380649e-23


def is_finite(value: object) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def pick_str(*values: object, default: str = '') -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() in {'nan', 'none', 'null'}:
            continue
        return text
    return str(default)


def pick_float(*values: object, default: float) -> float:
    for value in values:
        if is_finite(value):
            return float(value)
    return float(default)


def orthonormal_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    if n.size == 2:
        n = np.array([n[0], n[1], 0.0], dtype=np.float64)
    nmag = np.linalg.norm(n)
    if nmag <= 1e-30:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / nmag
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = a - np.dot(a, n) * n
    t1_mag = np.linalg.norm(t1)
    if t1_mag <= 1e-30:
        t1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = t1 - np.dot(t1, n) * n
        t1_mag = np.linalg.norm(t1)
    t1 /= max(t1_mag, 1e-30)
    t2 = np.cross(n, t1)
    t2_mag = np.linalg.norm(t2)
    t2 /= max(t2_mag, 1e-30)
    return t1, t2


def sample_positive_normal(rng: np.random.Generator, mean: float, std: float) -> float:
    if std <= 0.0:
        return max(0.0, mean)
    return max(0.0, mean + std * float(rng.normal()))


def sample_thermal_velocity(
    rng: np.random.Generator,
    normal: np.ndarray,
    mass: float,
    temperature_k: float,
    accommodation: float,
) -> np.ndarray:
    mass_eff = max(float(mass), 1e-30)
    temp_eff = max(float(temperature_k), 1.0)
    acc = float(np.clip(accommodation, 0.0, 1.0))
    sigma = math.sqrt(max(1e-30, KB * temp_eff / mass_eff)) * max(0.2, acc)
    n = np.asarray(normal, dtype=np.float64)
    nmag = np.linalg.norm(n)
    n = np.array([1.0, 0.0, 0.0], dtype=np.float64) if nmag <= 1e-30 else n / nmag
    t1, t2 = orthonormal_tangent_basis(n)
    vn = abs(float(rng.normal(scale=sigma)))
    vt1 = float(rng.normal(scale=sigma))
    vt2 = float(rng.normal(scale=sigma))
    return vn * n + vt1 * t1 + vt2 * t2


def burst_factor(
    t: float,
    center: float,
    sigma: float,
    amplitude: float,
    period: float,
    phase: float,
    min_factor: float,
    max_factor: float,
) -> float:
    sig = max(float(sigma), 1e-12)
    if period > 0.0:
        local_t = (float(t) - float(phase)) % float(period)
        d = abs(local_t - float(center))
        d = min(d, float(period) - d)
    else:
        d = abs(float(t) - float(center))
    envelope = math.exp(-0.5 * (d / sig) ** 2)
    fac = 1.0 + float(amplitude) * envelope
    return max(float(min_factor), min(float(max_factor), fac))


def event_matches(
    row: SourceEventRow,
    particle_id: int,
    source_part_id: int,
    material_id: int,
    law_name: str,
    event_tag: str,
) -> bool:
    if int(row.enabled) == 0:
        return False
    if int(row.applies_to_particle_id) > 0 and int(row.applies_to_particle_id) != int(particle_id):
        return False
    if int(row.applies_to_source_part_id) > 0 and int(row.applies_to_source_part_id) != int(source_part_id):
        return False
    if int(row.applies_to_material_id) > 0 and int(row.applies_to_material_id) != int(material_id):
        return False
    if str(row.applies_to_source_law).strip() and str(row.applies_to_source_law).strip() != str(law_name).strip():
        return False
    if str(row.applies_to_event_tag).strip() and str(row.applies_to_event_tag).strip() != str(event_tag).strip():
        return False
    return True


def event_effect(row: SourceEventRow, t: float) -> Dict[str, Any]:
    kind = str(row.event_kind).strip().lower()
    factor = 1.0
    gate = True
    rt_shift = float(row.release_time_shift_s) if is_finite(row.release_time_shift_s) else 0.0
    if kind in {'gaussian_burst', 'burst'}:
        factor = burst_factor(
            t=t,
            center=float(row.center_s if is_finite(row.center_s) else t),
            sigma=float(row.sigma_s if is_finite(row.sigma_s) else 1e-3),
            amplitude=float(row.amplitude if is_finite(row.amplitude) else 0.0),
            period=float(row.period_s if is_finite(row.period_s) else 0.0),
            phase=float(row.phase_s if is_finite(row.phase_s) else 0.0),
            min_factor=float(row.min_factor if is_finite(row.min_factor) else 0.0),
            max_factor=float(row.max_factor if is_finite(row.max_factor) else 1e6),
        )
        if is_finite(row.gain_multiplier):
            factor *= float(row.gain_multiplier)
    elif kind in {'periodic_burst', 'periodic'}:
        factor = burst_factor(
            t=t,
            center=float(row.center_s if is_finite(row.center_s) else 0.0),
            sigma=float(row.sigma_s if is_finite(row.sigma_s) else 1e-3),
            amplitude=float(row.amplitude if is_finite(row.amplitude) else 0.0),
            period=float(row.period_s if is_finite(row.period_s) else 1.0),
            phase=float(row.phase_s if is_finite(row.phase_s) else 0.0),
            min_factor=float(row.min_factor if is_finite(row.min_factor) else 0.0),
            max_factor=float(row.max_factor if is_finite(row.max_factor) else 1e6),
        )
        if is_finite(row.gain_multiplier):
            factor *= float(row.gain_multiplier)
    elif kind in {'gain', 'multiplier'}:
        factor = float(row.gain_multiplier if is_finite(row.gain_multiplier) else 1.0)
    elif kind in {'window_gate', 'gate'}:
        start = float(row.start_s if is_finite(row.start_s) else -np.inf)
        end = float(row.end_s if is_finite(row.end_s) else np.inf)
        gate = bool(start <= t <= end)
        factor = float(row.gain_multiplier if is_finite(row.gain_multiplier) else 1.0)
    return {'factor': float(factor), 'gate': bool(gate), 'release_time_shift_s': float(rt_shift), 'event_kind': kind}


def effective_resuspension_threshold_speed(
    base_thresh: float,
    roughness: float,
    adhesion: float,
    rough_scale: float,
    adhesion_scale: float,
) -> float:
    thresh = max(0.0, float(base_thresh))
    thresh *= 1.0 + float(rough_scale) * max(0.0, float(roughness))
    thresh *= 1.0 + float(adhesion_scale) * max(0.0, float(adhesion))
    return max(0.0, thresh)


def effective_resuspension_threshold_tau(
    base_tau: float,
    roughness: float,
    corr_length: float,
    slope_rms: float,
    adhesion: float,
    rough_scale: float,
    adhesion_scale: float,
    slope_scale: float,
) -> float:
    tau = max(0.0, float(base_tau))
    tau *= 1.0 + float(rough_scale) * max(0.0, float(roughness))
    tau *= 1.0 + float(adhesion_scale) * max(0.0, float(adhesion))
    tau *= 1.0 + float(slope_scale) * max(0.0, float(slope_rms))
    return max(0.0, tau)


def resolved_slope_rms(roughness_rms: float, corr_length: float, slope_rms: float) -> float:
    if is_finite(slope_rms) and float(slope_rms) >= 0.0:
        return float(slope_rms)
    if corr_length > 0.0:
        return max(0.0, float(roughness_rms)) / max(float(corr_length), 1e-30)
    return 0.0


__all__ = (
    'KB',
    'burst_factor',
    'effective_resuspension_threshold_speed',
    'effective_resuspension_threshold_tau',
    'event_effect',
    'event_matches',
    'is_finite',
    'orthonormal_tangent_basis',
    'pick_float',
    'pick_str',
    'resolved_slope_rms',
    'sample_positive_normal',
    'sample_thermal_velocity',
)
