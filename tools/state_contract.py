from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


STATE_ORDER = (
    'active_free_flight',
    'contact_sliding',
    'contact_endpoint_stopped',
    'invalid_mask_stopped',
    'numerical_boundary_stopped',
    'stuck',
    'absorbed',
    'escaped',
    'inactive',
)


def _bool_column(final_particles: pd.DataFrame, name: str) -> np.ndarray:
    if name not in final_particles:
        return np.zeros(len(final_particles), dtype=bool)
    return final_particles[name].to_numpy(dtype=bool)


def classify_particle_states(final_particles: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(final_particles), 'inactive', dtype=object)
    labels[_bool_column(final_particles, 'active')] = 'active_free_flight'
    labels[_bool_column(final_particles, 'contact_sliding')] = 'contact_sliding'
    labels[_bool_column(final_particles, 'contact_endpoint_stopped')] = 'contact_endpoint_stopped'
    for name in ('invalid_mask_stopped', 'numerical_boundary_stopped', 'stuck', 'absorbed', 'escaped'):
        labels[_bool_column(final_particles, name)] = name
    return labels


def particle_class_frame(final_particles: pd.DataFrame) -> pd.DataFrame:
    if 'particle_id' not in final_particles.columns:
        raise ValueError("final_particles.csv is missing required column: ['particle_id']")
    return pd.DataFrame(
        {
            'particle_id': final_particles['particle_id'].astype(np.int64),
            'particle_class': classify_particle_states(final_particles),
        }
    )


def final_state_counts(final_particles: pd.DataFrame) -> Dict[str, int]:
    labels = classify_particle_states(final_particles)
    return {name: int(np.count_nonzero(labels == name)) for name in STATE_ORDER}
