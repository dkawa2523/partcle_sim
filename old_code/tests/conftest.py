from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

os.environ.setdefault("MPLBACKEND", "Agg")

settings.register_profile(
    "quality",
    max_examples=100,
    deadline=None,
)
settings.register_profile(
    "nightly",
    max_examples=1_000,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quality"))
