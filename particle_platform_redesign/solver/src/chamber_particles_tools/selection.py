"""Exact particle/source selections shared by result consumers."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import numpy as np
from numpy.typing import NDArray

from chamber_particles import open_result
from chamber_particles.output import FinalParticles, ResultContext, ResultView


@dataclass(frozen=True, slots=True)
class ResultCohort:
    """One exact, read-only population bound to an opened result.

    The final rows and context are read once. Analysis, export and visualization
    can reuse this value without reopening the result or resolving the IDs again.
    """

    result: ResultView
    final: FinalParticles
    context: ResultContext
    selected: NDArray[np.bool_]
    selected_ids: NDArray[np.int64]
    source_ids: tuple[int, ...] | None
    particle_ids: tuple[int, ...] | None


type ResultInput = str | PathLike[str] | ResultView | ResultCohort


def select_result(
    source: ResultInput,
    *,
    source_ids: tuple[int, ...] | None = None,
    particle_ids: tuple[int, ...] | None = None,
) -> ResultCohort:
    """Bind exact IDs once; a supplied cohort is reused without further filters.

    To choose another population, select from its ResultView. Independent path
    calls still pass through the normal complete-result integrity checks.
    """
    if isinstance(source, ResultCohort):
        if source_ids is not None or particle_ids is not None:
            raise ValueError("a ResultCohort already defines its selection; select from its result")
        return source
    result = source if isinstance(source, ResultView) else open_result(source)
    context, final = result.read_context(), result.read_final()
    selected = select_particles(final, context, source_ids=source_ids, particle_ids=particle_ids)
    ids = final.particle_id[selected]
    selected.flags.writeable = False
    ids.flags.writeable = False
    return ResultCohort(result, final, context, selected, ids, source_ids, particle_ids)


def select_particles(
    final: FinalParticles,
    context: ResultContext,
    *,
    source_ids: tuple[int, ...] | None,
    particle_ids: tuple[int, ...] | None,
) -> NDArray[np.bool_]:
    """Intersect explicit ID sets, rejecting unknown or repeated requested IDs."""

    selected = np.ones(final.particle_id.size, dtype=np.bool_)
    if source_ids is not None:
        requested = _requested_ids(source_ids, "source_ids")
        if bool((requested >= len(context.sources)).any()):
            raise ValueError("source_ids contains an unknown ID")
        selected &= np.isin(final.source_id, requested)
    if particle_ids is not None:
        requested = _requested_ids(particle_ids, "particle_ids")
        # The result reader already guarantees sorted, unique final IDs. Reuse
        # that invariant instead of scanning all N rows for each requested ID.
        rows = np.searchsorted(final.particle_id, requested)
        if bool((rows >= final.particle_id.size).any()) or not np.array_equal(
            final.particle_id[rows], requested
        ):
            raise ValueError("particle_ids contains an unknown ID")
        keep = np.zeros(final.particle_id.size, dtype=np.bool_)
        keep[rows] = True
        selected &= keep
    return selected


def _requested_ids(requested: tuple[int, ...], label: str) -> NDArray[np.int64]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in requested):
        raise ValueError(f"{label} must contain integer IDs")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{label} must not repeat an ID")
    if any(value < 0 or value > np.iinfo(np.int64).max for value in requested):
        raise ValueError(f"{label} contains an unknown ID")
    return np.asarray(requested, dtype=np.int64)
