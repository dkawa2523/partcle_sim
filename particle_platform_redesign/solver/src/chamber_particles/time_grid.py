"""The fixed macro grid with required field knots, independent of transport."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Final

MACRO_GRID_REVISION: Final = "fixed_macro_grid_field_knots_v1"
_GRID_ROUNDOFF_ULPS: Final = 8.0


@dataclass(frozen=True, slots=True)
class MacroGrid:
    """Resolved parent clock; particle roots and observation clocks are separate.

    ``snapshot_splits_s`` contains only interior field knots not already on the
    fixed grid. The indexed construction and end snapping preserve the same
    binary64 boundaries during execution, resume and external inspection.
    """

    start_s: float
    end_s: float
    dt_s: float
    snapshot_splits_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.start_s, self.end_s, self.dt_s)):
            raise ValueError("macro grid times must be finite")
        if self.end_s <= self.start_s or self.dt_s <= 0.0:
            raise ValueError("macro grid requires start < end and positive dt")
        self._validate_representable_step()
        previous = self.start_s
        for knot_s in self.snapshot_splits_s:
            if not math.isfinite(knot_s) or not previous < knot_s < self.end_s:
                raise ValueError("macro snapshot splits must be ordered interior times")
            previous = knot_s

    def _validate_representable_step(self) -> None:
        """Reject a nominal grid that binary64 cannot index or advance."""

        span_s = self.end_s - self.start_s
        requested_intervals = span_s / self.dt_s
        if not math.isfinite(requested_intervals) or requested_intervals > 2**53:
            raise ValueError("macro grid exceeds consecutive binary64 step-index resolution")
        if self.start_s + self.dt_s <= self.start_s or self.end_s - self.dt_s >= self.end_s:
            raise ValueError("macro dt cannot advance representably across the run interval")

    @classmethod
    def resolve(
        cls, start_s: float, end_s: float, dt_s: float, field_knots_s: Iterable[float]
    ) -> MacroGrid:
        """Merge non-grid required-field knots without resetting the fixed grid."""

        fixed = cls(start_s, end_s, dt_s)
        splits: list[float] = []
        for knot_s in sorted(set(map(float, field_knots_s))):
            if not math.isfinite(knot_s):
                raise ValueError("field time knots must be finite")
            if knot_s <= start_s or knot_s >= end_s:
                continue
            nearest_step = round((knot_s - start_s) / dt_s)
            if nearest_step >= 1:
                grid_s, roundoff_s = fixed._boundary(nearest_step)
                if abs(knot_s - grid_s) <= roundoff_s:
                    continue
            splits.append(knot_s)
        return cls(start_s, end_s, dt_s, tuple(splits))

    def next_end(self, completed_steps: int, macro_start_s: float) -> float:
        """Return the next indexed boundary, snapping only roundoff at run end."""

        crossed_splits = bisect_right(self.snapshot_splits_s, macro_start_s)
        regular_completed_steps = completed_steps - crossed_splits
        if regular_completed_steps < 0:
            raise ValueError("field split schedule disagrees with macro-step count")
        nominal_end_s = self._regular_end(regular_completed_steps + 1)
        if crossed_splits < len(self.snapshot_splits_s):
            field_knot_s = self.snapshot_splits_s[crossed_splits]
            if field_knot_s < nominal_end_s:
                return field_knot_s
        return nominal_end_s

    @property
    def interval_count(self) -> int:
        """Count resolved run intervals without materializing the full grid."""

        regular_count = max(1, math.ceil((self.end_s - self.start_s) / self.dt_s))
        while regular_count > 1 and self._regular_end(regular_count - 1) >= self.end_s:
            regular_count -= 1
        while self._regular_end(regular_count) < self.end_s:
            regular_count += 1
        return regular_count + len(self.snapshot_splits_s)

    def intervals(self, start_s: float, end_s: float) -> Iterator[tuple[float, float]]:
        """Yield complete macro intervals intersecting a selected half-open window.

        The first/last boundaries are not clipped to observations: a proposal
        containing an observation can end after that observation. Release may
        shorten the first particle interval; that is the consumer's fact.
        """

        if not self.start_s <= start_s < end_s <= self.end_s:
            raise ValueError("selected window must lie inside the macro grid")
        regular = max(0, math.floor((start_s - self.start_s) / self.dt_s))
        while regular > 0 and self._regular_end(regular) > start_s:
            regular -= 1
        while self._regular_end(regular + 1) <= start_s:
            regular += 1
        crossed = bisect_right(self.snapshot_splits_s, start_s)
        left = self.start_s if regular == 0 else self._regular_end(regular)
        if crossed:
            left = max(left, self.snapshot_splits_s[crossed - 1])
        completed = regular + crossed
        while left < end_s:
            right = self.next_end(completed, left)
            if right <= left:
                raise ValueError("macro-grid time no longer advances at float64 precision")
            yield left, right
            left = right
            completed += 1

    def is_base_interval(self, start_s: float, end_s: float) -> bool:
        """Identify one complete fixed-grid interval by its constructed endpoints.

        Comparing widths with an absolute-time ULP allowance can wrongly call
        a field-knot or release-shortened interval uniform near the arithmetic
        floor. Matching indexed endpoints needs no such tolerance.
        """

        nearest = round((start_s - self.start_s) / self.dt_s)
        return any(
            index >= 0
            and self._boundary(index)[0] == start_s
            and self._boundary(index + 1)[0] == end_s
            for index in (nearest - 1, nearest, nearest + 1)
        )

    def _regular_end(self, step_number: int) -> float:
        boundary_s, roundoff_s = self._boundary(step_number)
        if not math.isfinite(boundary_s) or boundary_s >= self.end_s:
            return self.end_s
        if self.end_s - boundary_s <= roundoff_s:
            return self.end_s
        return boundary_s

    def _boundary(self, step_number: int) -> tuple[float, float]:
        offset_s, residual_s = _product(float(step_number), self.dt_s)
        end_s = math.fsum((self.start_s, offset_s, residual_s))
        if not math.isfinite(end_s):
            return end_s, 0.0
        roundoff_s = _GRID_ROUNDOFF_ULPS * max(
            math.ulp(self.start_s),
            math.ulp(self.end_s),
            math.ulp(self.dt_s) * step_number,
            math.ulp(offset_s),
            math.ulp(end_s),
        )
        return end_s, roundoff_s


def _product(left: float, right: float) -> tuple[float, float]:
    """Return a binary64 product and residual without splitter overflow."""

    product = left * right
    if not math.isfinite(product) or product == 0.0:
        return product, 0.0
    left_fraction, left_exponent = math.frexp(left)
    right_fraction, right_exponent = math.frexp(right)
    scaled_product = left_fraction * right_fraction
    splitter = 134_217_729.0
    split_left = splitter * left_fraction
    left_high = split_left - (split_left - left_fraction)
    left_low = left_fraction - left_high
    split_right = splitter * right_fraction
    right_high = split_right - (split_right - right_fraction)
    right_low = right_fraction - right_high
    scaled_residual = (
        ((left_high * right_high - scaled_product) + left_high * right_low) + left_low * right_high
    ) + left_low * right_low
    exponent = left_exponent + right_exponent
    return math.ldexp(scaled_product, exponent), math.ldexp(scaled_residual, exponent)
