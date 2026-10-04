from __future__ import annotations

from dataclasses import dataclass, field

from .charge_model import ChargeModelConfig
from .forces import ForceRuntimeParameters
from .plasma_background import PreparedPlasmaBackground
from .stochastic_motion import StochasticMotionConfig


@dataclass(frozen=True)
class RuntimeOptions:
    """Resolved physics objects that are not scalar execution-plan fields."""

    stochastic_motion: StochasticMotionConfig = field(
        default_factory=StochasticMotionConfig
    )
    charge_model: ChargeModelConfig = field(default_factory=ChargeModelConfig)
    plasma_background: PreparedPlasmaBackground | None = None
    force_runtime: ForceRuntimeParameters = field(
        default_factory=ForceRuntimeParameters
    )
