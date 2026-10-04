from __future__ import annotations

from .diagnostics import increment_count
from .segment_motion import (
    SegmentMotionRequest,
    ValidMaskPrefixResolution,
    resolve_valid_mask_prefix,
)
from .stochastic_motion import (
    PiecewiseLangevinPath,
    resolve_piecewise_valid_mask_prefix,
)


def resolve_valid_mask_retry_then_stop(
    request: SegmentMotionRequest,
    *,
    collision_diagnostics: dict[str, object],
    require_clean_prefix: bool = False,
    stochastic_path: PiecewiseLangevinPath | None = None,
    stochastic_offset_s: float = 0.0,
) -> ValidMaskPrefixResolution:
    """Find the longest accepted dyadic prefix of one motion request."""

    if stochastic_path is None:
        resolution = resolve_valid_mask_prefix(
            request,
            max_halving_count=int(request.adaptive_substep_max_splits),
            require_clean_prefix=bool(require_clean_prefix),
        )
    else:
        resolution = resolve_piecewise_valid_mask_prefix(
            request,
            stochastic_path,
            stochastic_offset_s=float(stochastic_offset_s),
            max_halving_count=int(request.adaptive_substep_max_splits),
            require_clean_prefix=bool(require_clean_prefix),
        )
    increment_count(
        collision_diagnostics, "invalid_mask_retry_count", resolution.retry_count
    )
    if not bool(resolution.found_valid_prefix):
        increment_count(collision_diagnostics, "invalid_mask_retry_exhausted_count")
    return resolution


__all__ = ("resolve_valid_mask_retry_then_stop",)
