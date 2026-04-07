from __future__ import annotations

import numpy as np


class SourceNormalSampler:
    def __call__(self, position: np.ndarray, source_part_id: int) -> np.ndarray:
        raise NotImplementedError


class SourceFlowSampler:
    def __call__(self, position: np.ndarray, release_time: float) -> np.ndarray:
        raise NotImplementedError


class SourceScalarSampler:
    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        raise NotImplementedError


class ZeroFlowSampler(SourceFlowSampler):
    def __init__(self, spatial_dim: int):
        self.spatial_dim = int(spatial_dim)

    def __call__(self, position: np.ndarray, release_time: float) -> np.ndarray:
        return np.zeros(self.spatial_dim, dtype=np.float64)


class ConstantScalarSampler(SourceScalarSampler):
    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        return self.value


__all__ = (
    'ConstantScalarSampler',
    'SourceFlowSampler',
    'SourceNormalSampler',
    'SourceScalarSampler',
    'ZeroFlowSampler',
)
