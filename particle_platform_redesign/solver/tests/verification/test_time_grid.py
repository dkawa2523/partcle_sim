"""Independent arithmetic checks of the resolved parent clock."""

from fractions import Fraction
from itertools import pairwise

import pytest

from chamber_particles.time_grid import MacroGrid


@pytest.mark.parametrize(
    ("start", "end", "step", "knots"),
    [
        (Fraction(0), Fraction(1), Fraction(1, 4), [Fraction(3, 16), Fraction(5, 8)]),
        (Fraction(1, 8), Fraction(11, 8), Fraction(1, 4), [Fraction(3, 16), Fraction(3, 8)]),
        (Fraction(0), Fraction(7, 8), Fraction(1, 4), [Fraction(1, 4), Fraction(9, 16)]),
    ],
)
def test_macro_grid_matches_independent_rational_union(
    start: Fraction, end: Fraction, step: Fraction, knots: list[Fraction]
) -> None:
    fixed = [start + index * step for index in range(1, 10) if start + index * step < end]
    boundaries = sorted({start, end, *fixed, *knots})
    expected = [(float(left), float(right)) for left, right in pairwise(boundaries)]
    grid = MacroGrid.resolve(float(start), float(end), float(step), map(float, knots))
    assert list(grid.intervals(float(start), float(end))) == expected
    assert grid.interval_count == len(expected)


def test_decimal_grid_has_no_roundoff_tail_and_does_not_restart_after_a_knot() -> None:
    grid = MacroGrid.resolve(0.0, 0.2, 0.02, [0.035, 0.1 + 0.2])
    intervals = list(grid.intervals(0.0, 0.2))
    assert grid.interval_count == 11
    assert intervals[:4] == [(0.0, 0.02), (0.02, 0.035), (0.035, 0.04), (0.04, 0.06)]
    assert intervals[-1] == (0.18, 0.2)
    assert all(right > left for left, right in intervals)


def test_window_keeps_complete_proposal_intervals_and_seeks_late_windows() -> None:
    grid = MacroGrid.resolve(0.0, 1_000_000.0, 0.125, [0.0625])
    assert list(grid.intervals(0.04, 0.08)) == [(0.0, 0.0625), (0.0625, 0.125)]
    assert list(grid.intervals(999_999.625, 1_000_000.0)) == [
        (999_999.625, 999_999.75),
        (999_999.75, 999_999.875),
        (999_999.875, 1_000_000.0),
    ]
    assert grid.interval_count == 8_000_001


def test_roundoff_aligned_field_knot_does_not_add_an_interval() -> None:
    grid = MacroGrid.resolve(0.0, 1.0, 0.1, [0.3, 0.1 + 0.2])
    assert grid.snapshot_splits_s == ()
    assert grid.interval_count == 10


def test_shortened_intervals_are_not_uniform_near_the_absolute_time_floor() -> None:
    start_s = 1e20
    step_s = 32 * 16384.0
    knot_s = start_s + 16 * 16384.0
    grid = MacroGrid.resolve(start_s, start_s + 4 * step_s, step_s, [knot_s])
    intervals = list(grid.intervals(start_s, grid.end_s))
    assert len(intervals) == 5
    assert [grid.is_base_interval(left, right) for left, right in intervals] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert not grid.is_base_interval(knot_s, start_s + step_s)


@pytest.mark.parametrize(
    ("start_s", "end_s", "dt_s"),
    [(1e20, 1e20 + 1e6, 1e-20), (0.0, 1.0, 1e-20), (float(2**53), float(2**53 + 2), 1.0)],
)
def test_unrepresentable_macro_grid_fails_before_iteration(
    start_s: float, end_s: float, dt_s: float
) -> None:
    with pytest.raises(ValueError, match=r"binary64|representably"):
        MacroGrid(start_s, end_s, dt_s)
