from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np


def summary_float_or_nan(value: object) -> float:
    if value is None:
        return float('nan')
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return float('nan')
        try:
            return float(text)
        except ValueError:
            return float('nan')
    return float('nan')


def summary_unit_for_key(key: str) -> str:
    name = str(key)
    suffix_units = (
        ('_m3', '1/m^3'),
        ('_kgm3', 'kg/m^3'),
        ('_m2Vs', 'm^2/(V s)'),
        ('_rad_s', 'rad/s'),
        ('_mps', 'm/s'),
        ('_eV', 'eV'),
        ('_Pa', 'Pa'),
        ('_K', 'K'),
        ('_amu', 'amu'),
        ('_kg', 'kg'),
        ('_Sm', 'S/m'),
        ('_s', 's'),
        ('_m2', 'm^2'),
        ('_m', 'm'),
        ('_C', 'C'),
        ('_e', 'e'),
    )
    for suffix, unit in suffix_units:
        if name.endswith(suffix):
            return unit
    return ''


def build_scalar_summary_rows(values: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, value in values.items():
        if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
            continue
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                continue
            value = value.item()
        rows.append(
            {
                'quantity': str(key),
                'value': value,
                'unit': summary_unit_for_key(str(key)),
            }
        )
    return rows
