"""Exact particle/source selections shared by result consumers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chamber_particles import open_result
from chamber_particles.output import FinalParticles, ResultContext, ResultView


@dataclass(frozen=True, slots=True)
class _ParticleIDFile:
    path: str
    sha256: str
    rows: int


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
    source_names: tuple[str, ...] | None = None
    particle_ids_file: _ParticleIDFile | None = None

    def selection_document(self) -> dict[str, object]:
        """Return independent JSON metadata for the resolved selection inputs."""
        return {
            "source_ids": self.source_ids,
            "source_names": self.source_names,
            "particle_ids": self.particle_ids,
            "particle_ids_file": None
            if self.particle_ids_file is None
            else asdict(self.particle_ids_file),
        }


type ResultInput = str | PathLike[str] | ResultView | ResultCohort


def select_result(
    source: ResultInput,
    *,
    source_ids: tuple[int, ...] | None = None,
    source_names: tuple[str, ...] | None = None,
    particle_ids: tuple[int, ...] | None = None,
    particle_ids_file: str | PathLike[str] | None = None,
) -> ResultCohort:
    """Bind exact IDs once; a supplied cohort is reused without further filters.

    To choose another population, select from its ResultView. Independent path
    calls still pass through the normal complete-result integrity checks.
    """
    if isinstance(source, ResultCohort):
        if any(
            value is not None
            for value in (source_ids, source_names, particle_ids, particle_ids_file)
        ):
            raise ValueError("a ResultCohort already defines its selection; select from its result")
        return source
    result = source if isinstance(source, ResultView) else open_result(source)
    context, final = result.read_context(), result.read_final()
    resolved_sources = _source_ids(context, source_ids, source_names)
    if particle_ids is not None and particle_ids_file is not None:
        raise ValueError("choose particle_ids or particle_ids_file, not both")
    file_selection = None
    requested = particle_ids
    if particle_ids_file is not None:
        requested, file_selection = _read_particle_ids(particle_ids_file)
    selected = select_particles(final, context, source_ids=resolved_sources, particle_ids=requested)
    ids = final.particle_id[selected]
    selected.flags.writeable = False
    ids.flags.writeable = False
    return ResultCohort(
        result,
        final,
        context,
        selected,
        ids,
        resolved_sources,
        particle_ids,
        source_names,
        file_selection,
    )


def _source_ids(
    context: ResultContext, requested: tuple[int, ...] | None, names: tuple[str, ...] | None
) -> tuple[int, ...] | None:
    if names is None:
        return requested
    if requested is not None:
        raise ValueError("choose source_ids or source_names, not both")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("source_names must contain unique nonempty release names")
    lookup = {source.name: source.source_id for source in context.sources}
    unknown = set(names) - lookup.keys()
    if unknown:
        raise ValueError(f"source_names contains unknown release names: {sorted(unknown)}")
    return tuple(lookup[name] for name in names)


def _read_particle_ids(path: str | PathLike[str]) -> tuple[tuple[int, ...], _ParticleIDFile]:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    try:
        if next(reader, None) != ["particle_id"]:
            raise ValueError("particle ID CSV must have the single header particle_id")
        ids = tuple(_particle_id_row(row, reader.line_num) for row in reader if row)
    except csv.Error as error:
        raise ValueError(f"particle ID CSV is malformed near line {reader.line_num}") from error
    _requested_ids(ids, "particle_ids_file")
    return ids, _ParticleIDFile(
        str(resolved), "sha256:" + hashlib.sha256(raw).hexdigest(), len(ids)
    )


def _particle_id_row(row: list[str], line: int) -> int:
    if len(row) != 1:
        raise ValueError(f"particle ID CSV line {line} must have exactly one cell")
    try:
        value = Decimal(row[0])
    except InvalidOperation as error:
        raise ValueError(f"particle ID CSV line {line} must contain an exact integer") from error
    if (
        not value.is_finite()
        or value != value.to_integral_value()
        or not 0 <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"particle ID CSV line {line} must contain a nonnegative int64 integer")
    return int(value)


def result_labels(context: ResultContext) -> dict[str, object]:
    """Project declared release and boundary mappings without inventing names."""
    geometry = context.geometry
    boundary = geometry.boundary
    triples = np.unique(
        np.column_stack((boundary.boundary_id, boundary.group_id, boundary.material_id)), axis=0
    )
    return {
        "sources": [asdict(source) for source in context.sources],
        "boundaries": [
            {
                "boundary_id": int(identifier),
                "group_name": geometry.group_names[int(group)],
                "material_id": int(material),
            }
            for identifier, group, material in triples
        ],
        "boundary_conditions": [
            {
                "group_name": item.group_name,
                "priority": item.priority,
                "law": item.law,
                "contact_geometry": item.contact_geometry,
                "parameters": json.loads(item.parameters_json),
            }
            for item in context.boundaries
        ],
        "axis_policy": context.axis_policy,
    }


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
