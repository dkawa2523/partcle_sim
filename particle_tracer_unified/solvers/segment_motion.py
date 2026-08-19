"""Stable API for deterministic scalar and batch segment motion."""

from __future__ import annotations

from ._segment_motion_contracts import (
    SegmentMotionBatchDestination as SegmentMotionBatchDestination,
)
from ._segment_motion_contracts import (
    SegmentMotionBatchRequest as SegmentMotionBatchRequest,
)
from ._segment_motion_contracts import SegmentMotionRequest as SegmentMotionRequest
from ._segment_motion_contracts import (
    ValidMaskPrefixResolution as ValidMaskPrefixResolution,
)
from ._segment_motion_scalar import SegmentMotionTrace as SegmentMotionTrace
from ._segment_motion_scalar import (
    resolve_valid_mask_prefix as resolve_valid_mask_prefix,
)
from ._segment_motion_scalar import trace_motion_segment as trace_motion_segment
from .segment_motion_batch import SegmentMotionBatchTrace as SegmentMotionBatchTrace
from .segment_motion_batch import trace_motion_batch as trace_motion_batch

__all__ = (
    "SegmentMotionBatchDestination",
    "SegmentMotionBatchRequest",
    "SegmentMotionBatchTrace",
    "SegmentMotionRequest",
    "SegmentMotionTrace",
    "ValidMaskPrefixResolution",
    "resolve_valid_mask_prefix",
    "trace_motion_batch",
    "trace_motion_segment",
)
