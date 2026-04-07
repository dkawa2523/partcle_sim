from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


STATE_ORDER = (
    'active',
    'invalid_mask_stopped',
    'stuck',
    'absorbed',
    'escaped',
)


def classify_particle_states(final_particles: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(final_particles), 'inactive', dtype=object)
    for name in STATE_ORDER:
        if name not in final_particles:
            continue
        labels[final_particles[name].to_numpy(dtype=bool)] = name
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
