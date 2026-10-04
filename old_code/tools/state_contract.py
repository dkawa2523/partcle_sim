from __future__ import annotations

import numpy as np
import pandas as pd

STATE_ORDER = (
    "active_free_flight",
    "contact_sliding",
    "contact_endpoint_stopped",
    "invalid_mask_stopped",
    "numerical_boundary_stopped",
    "stuck",
    "absorbed",
    "escaped",
    "inactive",
)


def classify_particle_states(final_particles: pd.DataFrame) -> np.ndarray:
    if "final_state" not in final_particles.columns:
        raise ValueError(
            "final_particles.csv is missing required column: ['final_state']"
        )
    labels = (
        final_particles["final_state"]
        .fillna("inactive")
        .astype(str)
        .to_numpy(dtype=object)
    )
    unknown = sorted(set(labels).difference(STATE_ORDER))
    if unknown:
        raise ValueError(
            f"final_particles.csv contains unsupported final_state values: {unknown}"
        )
    return labels


def particle_class_frame(final_particles: pd.DataFrame) -> pd.DataFrame:
    if "particle_id" not in final_particles.columns:
        raise ValueError(
            "final_particles.csv is missing required column: ['particle_id']"
        )
    return pd.DataFrame(
        {
            "particle_id": final_particles["particle_id"].astype(np.int64),
            "particle_class": classify_particle_states(final_particles),
        }
    )


def final_state_counts(final_particles: pd.DataFrame) -> dict[str, int]:
    labels = classify_particle_states(final_particles)
    return {name: int(np.count_nonzero(labels == name)) for name in STATE_ORDER}
