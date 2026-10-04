"""Canonical YAML parsing and cross-file case validation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from os import PathLike
from pathlib import Path
from types import MappingProxyType

import yaml

from .case_format import CanonicalDataFootprint, DataBundle, RealizedTableSource, read_with_info

type _PathInput = str | PathLike[str]

_CONTENT_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_INT64_MAX = 2**63 - 1
CASE_FORMAT_VERSION = 2
_PHYSICS_CATEGORIES = frozenset(
    {
        "charge",
        "dielectrophoresis",
        "drag",
        "electric",
        "gravity_buoyancy",
        "ion_drag",
        "lift",
        "noise",
        "thermophoresis",
    }
)


@dataclass(frozen=True, slots=True)
class TimeSpec:
    """Physical integration interval and macro-step size, all in SI."""

    start_s: float
    end_s: float
    dt_s: float


@dataclass(frozen=True, slots=True)
class MotionSpec:
    """Particle motion coordinates, independent of the stored data coordinates."""

    mode: str


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Scale-aware event localization request and bounded work budget."""

    geometry_rtol: float
    roundoff_ulps: int
    max_refinements: int
    max_interactions_per_step: int
    corner_policy: str


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """Requested numerical method, deterministic seed, and event policy."""

    integrator: str
    backend: str
    seed: int
    event: EventSpec


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Explicit execution resource ceiling."""

    memory_limit_mb: int


@dataclass(frozen=True, slots=True)
class ParticleProperties:
    """Independent authoritative particle properties used by a surface source."""

    charge_number: float
    mass_kg: float
    drag_diameter_m: float
    electrostatic_radius_m: float
    displaced_volume_m3: float
    model_weight: float
    material_id: int


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One named table or surface release configuration."""

    name: str
    kind: str
    particle: ParticleProperties | None
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PhysicsSpec:
    """At most one explicitly selected model per physics category."""

    models: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class BoundarySpec:
    """One boundary group and its post-hit law configuration."""

    boundary_group: str
    priority: int
    law: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Optional trajectory and particle-probe frames."""

    trajectories: TrajectoryOutputSpec | None
    probes: ProbeOutputSpec | None


@dataclass(frozen=True, slots=True)
class TrajectoryOutputSpec:
    """P04 trajectory selection and explicit physical output times."""

    selection: str
    explicit_times_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProbeOutputSpec:
    """Explicit particle IDs and physical times for bounded state probes."""

    particle_ids: tuple[int, ...]
    explicit_times_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SimulationSpec:
    """Human-controlled settings separated from the reusable data bundle."""

    name: str
    motion: MotionSpec
    time: TimeSpec
    solver: SolverSpec
    resources: ResourceSpec
    physics: PhysicsSpec
    sources: tuple[SourceSpec, ...]
    boundaries: tuple[BoundarySpec, ...]
    output: OutputSpec


@dataclass(frozen=True, slots=True)
class SimulationCase:
    """Validated combination of one SimulationSpec and one DataBundle."""

    spec: SimulationSpec
    data: DataBundle
    case_path: Path
    data_path: Path
    content_hash: str
    case_file_hash: str
    data_footprint: CanonicalDataFootprint


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as error:
            raise ValueError("YAML mapping keys must be scalar values") from error
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_case(path: _PathInput) -> SimulationCase:
    """Read one strict v2 YAML file and its referenced canonical HDF5 bundle."""
    case_path = Path(path).expanduser().resolve()
    document, case_file_hash = _read_yaml(case_path)
    root = _mapping(document, "document")
    _exact_keys(
        root,
        {
            "format_version",
            "case",
            "motion",
            "time",
            "solver",
            "resources",
            "physics",
            "sources",
            "boundaries",
            "output",
        },
        "document",
    )
    if _integer(root["format_version"], "format_version") != CASE_FORMAT_VERSION:
        raise ValueError("unsupported YAML format_version")

    name, data_path, expected_hash = _parse_case_reference(root["case"], case_path)
    time = _parse_time(root["time"])
    resources = _parse_resources(root["resources"])
    spec = SimulationSpec(
        name=name,
        motion=_parse_motion(root["motion"]),
        time=time,
        solver=_parse_solver(root["solver"]),
        resources=resources,
        physics=_parse_physics(root["physics"]),
        sources=_parse_sources(root["sources"]),
        boundaries=_parse_boundaries(root["boundaries"]),
        output=_parse_output(root["output"], time),
    )
    memory_limit_bytes = resources.memory_limit_mb * 1024 * 1024
    data, file_info = read_with_info(data_path, numeric_array_limit_bytes=memory_limit_bytes)
    actual_hash = file_info.content_hash
    if actual_hash != expected_hash:
        raise ValueError("case data content hash does not match expected_content_hash")

    _validate_cross_references(spec, data)
    return SimulationCase(
        spec=spec,
        data=data,
        case_path=case_path,
        data_path=data_path,
        content_hash=actual_hash,
        case_file_hash=case_file_hash,
        data_footprint=file_info.footprint,
    )


def _read_yaml(path: Path) -> tuple[object, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        document = yaml.load(text, Loader=_UniqueKeyLoader)
        digest = hashlib.sha256(raw).hexdigest()
        return document, f"sha256:{digest}"
    except UnicodeDecodeError as exc:
        raise ValueError(f"case YAML is not valid UTF-8: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}") from exc


def _parse_case_reference(value: object, case_path: Path) -> tuple[str, Path, str]:
    section = _mapping(value, "case")
    _exact_keys(section, {"name", "data_path", "expected_content_hash"}, "case")
    name = _text(section["name"], "case.name")
    data_reference = _text(section["data_path"], "case.data_path")
    expected_hash = _text(section["expected_content_hash"], "case.expected_content_hash")
    if _CONTENT_HASH.fullmatch(expected_hash) is None:
        raise ValueError("case.expected_content_hash must be sha256:<64 lowercase hex>")
    data_path = Path(data_reference).expanduser()
    if not data_path.is_absolute():
        data_path = case_path.parent / data_path
    return name, data_path.resolve(), expected_hash


def _parse_time(value: object) -> TimeSpec:
    section = _mapping(value, "time")
    _exact_keys(section, {"start_s", "end_s", "dt_s"}, "time")
    start_s = _number(section["start_s"], "time.start_s")
    end_s = _number(section["end_s"], "time.end_s")
    dt_s = _positive_number(section["dt_s"], "time.dt_s")
    if end_s <= start_s:
        raise ValueError("time.end_s must be greater than time.start_s")
    duration_s = end_s - start_s
    if not math.isfinite(duration_s):
        raise ValueError("simulation duration must be finite at float64 precision")
    if dt_s > duration_s:
        raise ValueError("time.dt_s must not exceed the simulation interval")
    return TimeSpec(start_s, end_s, dt_s)


def _parse_motion(value: object) -> MotionSpec:
    section = _mapping(value, "motion")
    _exact_keys(section, {"mode"}, "motion")
    mode = _text(section["mode"], "motion.mode")
    if mode not in ("cartesian_xy", "axisymmetric_rz_meridional"):
        raise ValueError("motion.mode must be 'cartesian_xy' or 'axisymmetric_rz_meridional'")
    return MotionSpec(mode)


def _parse_solver(value: object) -> SolverSpec:
    section = _mapping(value, "solver")
    _exact_keys(section, {"integrator", "backend", "seed", "event"}, "solver")
    seed = _integer(section["seed"], "solver.seed")
    if not 0 <= seed < 2**64:
        raise ValueError("solver.seed must fit in an unsigned 64-bit integer")
    return SolverSpec(
        integrator=_text(section["integrator"], "solver.integrator"),
        backend=_text(section["backend"], "solver.backend"),
        seed=seed,
        event=_parse_event(section["event"]),
    )


def _parse_event(value: object) -> EventSpec:
    section = _mapping(value, "solver.event")
    required = {
        "geometry_rtol",
        "roundoff_ulps",
        "max_refinements",
        "max_interactions_per_step",
        "corner_policy",
    }
    _exact_keys(section, required, "solver.event")
    geometry_rtol = _positive_number(section["geometry_rtol"], "solver.event.geometry_rtol")
    if geometry_rtol >= 1.0:
        raise ValueError("solver.event.geometry_rtol must be less than one")
    return EventSpec(
        geometry_rtol=geometry_rtol,
        roundoff_ulps=_positive_integer(section["roundoff_ulps"], "solver.event.roundoff_ulps"),
        max_refinements=_positive_integer(
            section["max_refinements"], "solver.event.max_refinements"
        ),
        max_interactions_per_step=_positive_integer(
            section["max_interactions_per_step"],
            "solver.event.max_interactions_per_step",
        ),
        corner_policy=_text(section["corner_policy"], "solver.event.corner_policy"),
    )


def _parse_resources(value: object) -> ResourceSpec:
    section = _mapping(value, "resources")
    _exact_keys(section, {"memory_limit_mb"}, "resources")
    memory_limit_mb = _positive_integer(section["memory_limit_mb"], "resources.memory_limit_mb")
    return ResourceSpec(memory_limit_mb)


def _parse_particle(value: object, location: str) -> ParticleProperties:
    section = _mapping(value, location)
    required = {
        "charge_number",
        "mass_kg",
        "drag_diameter_m",
        "electrostatic_radius_m",
        "displaced_volume_m3",
        "model_weight",
        "material_id",
    }
    _exact_keys(section, required, location)
    return ParticleProperties(
        charge_number=_number(section["charge_number"], f"{location}.charge_number"),
        mass_kg=_positive_number(section["mass_kg"], f"{location}.mass_kg"),
        drag_diameter_m=_positive_number(section["drag_diameter_m"], f"{location}.drag_diameter_m"),
        electrostatic_radius_m=_nonnegative_number(
            section["electrostatic_radius_m"], f"{location}.electrostatic_radius_m"
        ),
        displaced_volume_m3=_nonnegative_number(
            section["displaced_volume_m3"], f"{location}.displaced_volume_m3"
        ),
        model_weight=_positive_number(section["model_weight"], f"{location}.model_weight"),
        material_id=_nonnegative_integer(section["material_id"], f"{location}.material_id"),
    )


def _parse_sources(value: object) -> tuple[SourceSpec, ...]:
    items = _sequence(value, "sources")
    if not items:
        raise ValueError("sources must contain at least one particle source")
    sources: list[SourceSpec] = []
    names: set[str] = set()
    for index, item in enumerate(items):
        location = f"sources[{index}]"
        section = _mapping(item, location)
        name = _text(section.get("name"), f"{location}.name")
        kind = _text(section.get("type"), f"{location}.type")
        if name in names:
            raise ValueError(f"duplicate source name: {name}")
        names.add(name)
        if kind == "table":
            _exact_keys(section, {"name", "type", "table"}, location)
            parameters = MappingProxyType({"table": _text(section["table"], f"{location}.table")})
            sources.append(SourceSpec(name, kind, None, parameters))
        elif kind == "surface":
            required = {
                "name",
                "type",
                "boundary_group",
                "count",
                "particle_id_start",
                "particle",
                "position",
                "velocity",
                "release",
            }
            _exact_keys(section, required, location)
            count = _positive_integer(section["count"], f"{location}.count")
            particle_id_start = _nonnegative_integer(
                section["particle_id_start"], f"{location}.particle_id_start"
            )
            if particle_id_start > _INT64_MAX - count + 1:
                raise ValueError(f"{location} particle ID range exceeds signed int64")
            parameters = MappingProxyType(
                {
                    "boundary_group": _text(
                        section["boundary_group"], f"{location}.boundary_group"
                    ),
                    "count": count,
                    "particle_id_start": particle_id_start,
                    "position": _model_mapping(section["position"], f"{location}.position"),
                    "velocity": _model_mapping(section["velocity"], f"{location}.velocity"),
                    "release": _model_mapping(section["release"], f"{location}.release"),
                }
            )
            particle = _parse_particle(section["particle"], f"{location}.particle")
            sources.append(SourceSpec(name, kind, particle, parameters))
        else:
            raise ValueError(f"{location}.type must be 'table' or 'surface'")
    return tuple(sources)


def _parse_physics(value: object) -> PhysicsSpec:
    section = _mapping(value, "physics")
    unknown = set(section) - _PHYSICS_CATEGORIES
    if unknown:
        raise ValueError(f"physics has unsupported categories: {sorted(unknown)}")
    if "charge" not in section:
        raise ValueError(
            "physics.charge must be explicit; use model=fixed with source charge_number=0 "
            "if uncharged"
        )

    models: dict[str, Mapping[str, object]] = {}
    for category, model_value in section.items():
        if model_value is None:
            raise ValueError(f"physics.{category}: omit disabled categories instead of using null")
        model = _model_mapping(model_value, f"physics.{category}")
        if category == "charge" and model["model"] == "fixed" and set(model) != {"model"}:
            raise ValueError(
                "physics.charge fixed owns only dZ/dt=0; initial charge_number belongs to sources"
            )
        models[category] = model
    return PhysicsSpec(MappingProxyType(models))


def _parse_boundaries(value: object) -> tuple[BoundarySpec, ...]:
    items = _sequence(value, "boundaries")
    boundaries: list[BoundarySpec] = []
    groups: set[str] = set()
    for index, item in enumerate(items):
        location = f"boundaries[{index}]"
        section = _mapping(item, location)
        if "boundary_group" not in section or "priority" not in section or "law" not in section:
            raise ValueError(f"{location} requires boundary_group, priority, and law")
        group = _text(section["boundary_group"], f"{location}.boundary_group")
        priority = _nonnegative_integer(section["priority"], f"{location}.priority")
        law = _text(section["law"], f"{location}.law")
        if group in groups:
            raise ValueError(f"duplicate boundary law for group: {group}")
        groups.add(group)
        parameters = _freeze_mapping(
            {
                key: item_value
                for key, item_value in section.items()
                if key not in {"boundary_group", "priority", "law"}
            },
            location,
        )
        boundaries.append(BoundarySpec(group, priority, law, parameters))
    return tuple(boundaries)


def _parse_output(value: object, time: TimeSpec) -> OutputSpec:
    section = _mapping(value, "output")
    unexpected = set(section) - {"trajectories", "probes"}
    if "trajectories" not in section or unexpected:
        raise ValueError(
            "output keys are invalid; "
            f"missing={[] if 'trajectories' in section else ['trajectories']}, "
            f"unexpected={sorted(unexpected)}"
        )
    trajectory_value = section["trajectories"]
    trajectory_spec: TrajectoryOutputSpec | None = None
    if trajectory_value is not None:
        trajectories = _mapping(trajectory_value, "output.trajectories")
        _exact_keys(trajectories, {"selection", "schedule"}, "output.trajectories")
        selection = _text(trajectories["selection"], "output.trajectories.selection")
        if selection != "all":
            raise ValueError("output.trajectories.selection must be 'all'")
        times = _parse_output_times(
            trajectories["schedule"],
            "output.trajectories.schedule",
            time,
        )
        trajectory_spec = TrajectoryOutputSpec(selection, times)

    probe_value = section.get("probes")
    probe_spec: ProbeOutputSpec | None = None
    if probe_value is not None:
        probes = _mapping(probe_value, "output.probes")
        _exact_keys(probes, {"particle_ids", "schedule"}, "output.probes")
        raw_ids = _sequence(probes["particle_ids"], "output.probes.particle_ids")
        particle_ids = tuple(
            _nonnegative_integer(item, f"output.probes.particle_ids[{index}]")
            for index, item in enumerate(raw_ids)
        )
        if not particle_ids:
            raise ValueError("output probe particle IDs must not be empty")
        if particle_ids[-1] > _INT64_MAX:
            raise ValueError("output probe particle IDs must fit signed int64")
        if any(current <= previous for previous, current in pairwise(particle_ids)):
            raise ValueError("output probe particle IDs must be strictly increasing")
        times = _parse_output_times(
            probes["schedule"],
            "output.probes.schedule",
            time,
        )
        probe_spec = ProbeOutputSpec(particle_ids, times)
    return OutputSpec(trajectory_spec, probe_spec)


def _parse_output_times(value: object, location: str, time: TimeSpec) -> tuple[float, ...]:
    schedule = _mapping(value, location)
    _exact_keys(schedule, {"explicit_times_s"}, location)
    items = _sequence(schedule["explicit_times_s"], f"{location}.explicit_times_s")
    times = tuple(
        _number(item, f"{location}.explicit_times_s[{index}]") for index, item in enumerate(items)
    )
    if not times:
        raise ValueError(f"{location} times must not be empty")
    if any(current <= previous for previous, current in pairwise(times)):
        raise ValueError(f"{location} times must be strictly increasing")
    if times[0] < time.start_s or times[-1] > time.end_s:
        raise ValueError(f"{location} times must be inside the closed run interval")
    return times


def _validate_cross_references(spec: SimulationSpec, data: DataBundle) -> None:
    _validate_boundary_references(spec, data)
    _validate_source_references(spec, data)


def _validate_boundary_references(spec: SimulationSpec, data: DataBundle) -> None:
    expected_groups = set(data.geometry.group_names)
    configured_groups = {boundary.boundary_group for boundary in spec.boundaries}
    if configured_groups != expected_groups:
        missing = sorted(expected_groups - configured_groups)
        unknown = sorted(configured_groups - expected_groups)
        raise ValueError(
            f"boundary laws do not match data groups; missing={missing}, unknown={unknown}"
        )


def _validate_source_references(spec: SimulationSpec, data: DataBundle) -> None:
    expected_groups = set(data.geometry.group_names)
    table_sources = {source.name: source for source in data.sources}
    used_tables: set[str] = set()
    active_tables: list[RealizedTableSource] = []
    surface_id_ranges: list[tuple[int, int, str]] = []
    for source in spec.sources:
        if source.kind == "table":
            active_tables.append(
                _validate_table_source_reference(source, table_sources, used_tables, spec.time)
            )
        else:
            _validate_surface_source_reference(source, expected_groups)
            start = source.parameters["particle_id_start"]
            count = source.parameters["count"]
            if not isinstance(start, int) or not isinstance(count, int):
                raise TypeError("validated surface particle identity must be integer")
            surface_id_ranges.append((start, start + count, source.name))
    _validate_surface_particle_ids(surface_id_ranges, active_tables)


def _validate_table_source_reference(
    source: SourceSpec,
    table_sources: Mapping[str, RealizedTableSource],
    used_tables: set[str],
    time: TimeSpec,
) -> RealizedTableSource:
    table_name = source.parameters["table"]
    if not isinstance(table_name, str) or table_name not in table_sources:
        raise ValueError(f"source {source.name} references an unknown table")
    if table_name in used_tables:
        raise ValueError(f"source table {table_name} is referenced more than once")
    used_tables.add(table_name)
    table = table_sources[table_name]
    release_times = table.release_time_s
    if float(release_times.min()) < time.start_s or float(release_times.max()) > time.end_s:
        raise ValueError(f"source {source.name} has release times outside the run interval")
    return table


def _validate_surface_source_reference(source: SourceSpec, expected_groups: set[str]) -> None:
    group = source.parameters["boundary_group"]
    if not isinstance(group, str) or group not in expected_groups:
        raise ValueError(f"source {source.name} references an unknown boundary group")


def _validate_surface_particle_ids(
    ranges: list[tuple[int, int, str]], tables: list[RealizedTableSource]
) -> None:
    ordered = sorted(ranges)
    for previous, current in pairwise(ordered):
        if current[0] < previous[1]:
            raise ValueError(
                f"surface source particle ID ranges overlap: {previous[2]} and {current[2]}"
            )
    for start, stop, source_name in ordered:
        for table in tables:
            ids = table.particle_id
            if bool(((ids >= start) & (ids < stop)).any()):
                raise ValueError(
                    f"surface source {source_name} particle IDs overlap table {table.name}"
                )


def _model_mapping(value: object, location: str) -> Mapping[str, object]:
    section = _mapping(value, location)
    if "model" not in section:
        raise ValueError(f"{location} requires model")
    _text(section["model"], f"{location}.model")
    return _freeze_mapping(section, location)


def _freeze_mapping(value: Mapping[str, object], location: str) -> Mapping[str, object]:
    return MappingProxyType(
        {key: _freeze_value(item, f"{location}.{key}") for key, item in value.items()}
    )


def _freeze_value(value: object, location: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(_mapping(value, location), location)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item, f"{location}[]") for item in value)
    raise ValueError(f"{location} contains an unsupported YAML value")


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{location} keys must be strings")
        result[key] = item
    return result


def _sequence(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{location} must be a sequence")
    return value


def _exact_keys(section: Mapping[str, object], required: set[str], location: str) -> None:
    missing = required - set(section)
    unexpected = set(section) - required
    if missing or unexpected:
        raise ValueError(
            f"{location} keys are invalid; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _positive_number(value: object, location: str) -> float:
    result = _number(value, location)
    if result <= 0.0:
        raise ValueError(f"{location} must be positive")
    return result


def _nonnegative_number(value: object, location: str) -> float:
    result = _number(value, location)
    if result < 0.0:
        raise ValueError(f"{location} must be nonnegative")
    return result


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _positive_integer(value: object, location: str) -> int:
    result = _integer(value, location)
    if result <= 0:
        raise ValueError(f"{location} must be positive")
    return result


def _nonnegative_integer(value: object, location: str) -> int:
    result = _integer(value, location)
    if result < 0:
        raise ValueError(f"{location} must be nonnegative")
    return result
