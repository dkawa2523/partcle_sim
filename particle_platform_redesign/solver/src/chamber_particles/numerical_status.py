"""Shared row-local numerical status codes for the solver hot path."""

from __future__ import annotations

import numpy as np

NUMERICAL_STATUS_OK = np.uint8(0)
FIELD_NUMERICAL_FAILURE = np.uint8(1)
PHYSICS_NUMERICAL_FAILURE = np.uint8(2)
INTEGRATOR_NUMERICAL_FAILURE = np.uint8(3)
INTEGRATOR_ACCURACY_FAILURE = np.uint8(4)
