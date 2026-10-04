"""Canonical input/output workflow for the reduced electrostatic builder."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import yaml

from chamber_particles.case_format import (
    DataBundle,
    FieldData,
    P1TriLayout,
    read_with_info,
    write,
)

from .numerics import (
    ClosureParameters,
    ElectrostaticSolution,
    SolverSettings,
    solve_axisymmetric_p1,
)

type PotentialProfile = ConstantPotential | RadialExponentialPotential

BUILDER_FORMAT_VERSION: Final = 1
BUILDER_REVISION: Final = "reduced_electrostatic_builder_v1"
MODEL_REVISION: Final = "boltzmann_bohm_sheath_c2_v1"
FIELD_SEMANTICS_REVISION: Final = "reduced_electrostatic_fields_v1"
PRODUCER_VERSION: Final = version("chamber-particles")
BOLTZMANN_J_K: Final = 1.380649e-23
ELEMENTARY_CHARGE_C: Final = 1.602176634e-19

_GENERATED_FIELD_NAMES: Final = (
    "electric_potential",
    "electric_field",
    "electron_number_density",
    "positive_ion_number_density",
    "space_charge_density",
    "electron_temperature",
    "positive_ion_temperature",
    "positive_ion_velocity",
    "positive_ion_speed",
    "positive_ion_mass",
)


@dataclass(frozen=True, slots=True)
class InputSpecification:
    """Canonical thermal-flow bundle and primitive field selection."""

    data_path: Path
    layout: str
    gas_velocity_field: str
    gas_temperature_field: str


@dataclass(frozen=True, slots=True)
class IonFluxParameters:
    """Drift-diffusion direction model used for the output ion velocity."""

    mobility_m2_V_s: float
    regularization_speed_m_s: float


@dataclass(frozen=True, slots=True)
class ConstantPotential:
    kind: Literal["constant"]
    value_V: float


@dataclass(frozen=True, slots=True)
class RadialExponentialPotential:
    kind: Literal["radial_exponential"]
    inner_value_V: float
    outer_value_V: float
    start_radius_m: float
    transition_length_m: float


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """One semantic boundary group and its physical potential profile."""

    group: str
    priority: int
    potential: PotentialProfile


@dataclass(frozen=True, slots=True)
class BuilderSpecification:
    """Strict, versioned field-production configuration."""

    input: InputSpecification
    closure: ClosureParameters
    ion_flux: IonFluxParameters
    boundaries: tuple[BoundaryCondition, ...]
    solver: SolverSettings


def build_from_configuration(
    configuration_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Build and atomically publish an augmented canonical ``case.h5``."""

    config_path = Path(configuration_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path is not None else None
    if report == output:
        raise ValueError("report_path and output_path must be different")
    if output.exists():
        raise FileExistsError(output)
    if report is not None and report.exists():
        raise FileExistsError(report)
    raw = config_path.read_bytes()
    specification = parse_configuration(raw, base_directory=config_path.parent)
    config_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    result = _build(specification, config_hash, output)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8", errors="strict") as stream:
            stream.write(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return result


def parse_configuration(raw: bytes, *, base_directory: Path) -> BuilderSpecification:
    """Parse the small strict YAML document used by the builder."""

    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("builder configuration is not valid UTF-8 YAML") from error
    root = _mapping(document, "configuration")
    _exact_keys(
        root,
        {"format_version", "input", "model", "boundaries", "solver"},
        "configuration",
    )
    if _integer(root["format_version"], "format_version") != BUILDER_FORMAT_VERSION:
        raise ValueError(f"format_version must be {BUILDER_FORMAT_VERSION}")
    input_specification = _parse_input(_mapping(root["input"], "input"), base_directory)
    closure, ion_flux = _parse_model(_mapping(root["model"], "model"))
    boundaries = _parse_boundaries(root["boundaries"])
    solver = _parse_solver(_mapping(root["solver"], "solver"))
    return BuilderSpecification(input_specification, closure, ion_flux, boundaries, solver)


def _build(
    specification: BuilderSpecification, config_hash: str, output: Path
) -> dict[str, object]:
    data, source_info = read_with_info(specification.input.data_path)
    layout = _select_layout(data, specification.input.layout)
    gas_velocity, gas_temperature = _select_thermal_fields(data, specification.input, layout)
    collisions = set(_GENERATED_FIELD_NAMES).intersection(field.name for field in data.fields)
    if collisions:
        raise ValueError(f"generated field names already exist: {sorted(collisions)}")
    fixed_ids, target = _resolve_boundaries(data, layout, specification.boundaries)
    solve_started = time.perf_counter()
    solution = solve_axisymmetric_p1(
        layout.nodes_m,
        layout.connectivity,
        fixed_ids,
        target,
        specification.closure,
        specification.solver,
    )
    solve_wall_seconds = time.perf_counter() - solve_started
    generated_fields = _generated_fields(
        layout,
        gas_velocity.values,
        gas_temperature.values[:, 0],
        solution,
        specification,
    )
    metadata = _producer_metadata(
        specification,
        config_hash,
        source_info.content_hash,
        data.provenance_json,
        solution,
    )
    provenance = json.dumps(
        {
            "producer": "chamber_particles.electrostatic_builder",
            "producer_version": PRODUCER_VERSION,
            "source_sha256": source_info.content_hash,
            "field_semantics_revision": FIELD_SEMANTICS_REVISION,
            "producer_metadata": metadata,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_info = write(
        output,
        DataBundle(
            coordinate_system=data.coordinate_system,
            provenance_json=provenance,
            geometry=data.geometry,
            layouts=data.layouts,
            fields=(*data.fields, *generated_fields),
            sources=data.sources,
        ),
    )
    return {
        "status": "complete",
        "builder_revision": BUILDER_REVISION,
        "model_revision": MODEL_REVISION,
        "input_path": str(specification.input.data_path),
        "input_content_hash": source_info.content_hash,
        "configuration_sha256": config_hash,
        "output_path": str(output),
        "output_content_hash": output_info.content_hash,
        "generated_fields": list(_GENERATED_FIELD_NAMES),
        "solve_wall_seconds": solve_wall_seconds,
        "linear_storage": _linear_storage_report(
            layout.nodes_m.shape[0] - fixed_ids.size,
            specification.solver.linear_krylov_dimension,
            specification.solver.max_linear_iterations,
        ),
        "convergence": _solve_report(solution),
        "field_ranges": _field_ranges(generated_fields),
    }


def _linear_storage_report(
    free_node_count: int, configured_dimension: int, maximum_iterations: int
) -> dict[str, int]:
    """Report the dominant configured GMRES storage without claiming process RSS."""

    dimension = min(free_node_count, configured_dimension, maximum_iterations)
    basis = free_node_count * (dimension + 1) * np.dtype("<f8").itemsize
    hessenberg = (dimension + 1) * dimension * np.dtype("<f8").itemsize
    return {
        "free_node_count": free_node_count,
        "effective_krylov_dimension": dimension,
        "basis_bytes": basis,
        "hessenberg_bytes": hessenberg,
        "basis_and_hessenberg_bytes": basis + hessenberg,
    }


def _generated_fields(
    layout: P1TriLayout,
    gas_velocity_m_s: np.ndarray,
    gas_temperature_K: np.ndarray,
    solution: ElectrostaticSolution,
    specification: BuilderSpecification,
) -> tuple[FieldData, ...]:
    ion_velocity = _ion_velocity(
        gas_velocity_m_s,
        gas_temperature_K,
        solution,
        specification,
        layout.nodes_m[:, 0] == 0.0,
    )
    electron_temperature_K = (
        specification.closure.electron_temperature_V * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
    )
    scalar = ("value",)
    return (
        _field(
            "electric_potential",
            layout.name,
            scalar,
            "scalar",
            solution.potential_V[:, None],
            "V",
        ),
        _field(
            "electric_field",
            layout.name,
            ("r", "z"),
            "axisymmetric_rz",
            solution.electric_field_V_m,
            "V/m",
        ),
        _field(
            "electron_number_density",
            layout.name,
            scalar,
            "scalar",
            solution.electron_number_density_m3[:, None],
            "1/m^3",
        ),
        _field(
            "positive_ion_number_density",
            layout.name,
            scalar,
            "scalar",
            solution.positive_ion_number_density_m3[:, None],
            "1/m^3",
        ),
        _field(
            "space_charge_density",
            layout.name,
            scalar,
            "scalar",
            solution.space_charge_density_C_m3[:, None],
            "C/m^3",
        ),
        _field(
            "electron_temperature",
            layout.name,
            scalar,
            "scalar",
            np.full((layout.nodes_m.shape[0], 1), electron_temperature_K),
            "K",
        ),
        _field(
            "positive_ion_temperature",
            layout.name,
            scalar,
            "scalar",
            gas_temperature_K[:, None],
            "K",
        ),
        _field(
            "positive_ion_velocity",
            layout.name,
            ("r", "z"),
            "axisymmetric_rz",
            ion_velocity,
            "m/s",
        ),
        _field(
            "positive_ion_speed",
            layout.name,
            scalar,
            "scalar",
            solution.positive_ion_speed_m_s[:, None],
            "m/s",
        ),
        _field(
            "positive_ion_mass",
            layout.name,
            scalar,
            "scalar",
            np.full(
                (layout.nodes_m.shape[0], 1),
                specification.closure.positive_ion_mass_kg,
            ),
            "kg",
        ),
    )


def _ion_velocity(
    gas_velocity_m_s: np.ndarray,
    gas_temperature_K: np.ndarray,
    solution: ElectrostaticSolution,
    specification: BuilderSpecification,
    axis_nodes: np.ndarray,
) -> np.ndarray:
    density = solution.positive_ion_number_density_m3
    mobility = specification.ion_flux.mobility_m2_V_s
    diffusivity = mobility * BOLTZMANN_J_K * gas_temperature_K / ELEMENTARY_CHARGE_C
    flux = (
        density[:, None] * gas_velocity_m_s
        + mobility * density[:, None] * solution.electric_field_V_m
        - diffusivity[:, None] * solution.positive_ion_density_gradient_m4
    )
    regularization = (
        specification.closure.bulk_number_density_m3
        * specification.ion_flux.regularization_speed_m_s
    )
    magnitude = np.sqrt(np.sum(flux * flux, axis=1) + regularization * regularization)
    velocity = solution.positive_ion_speed_m_s[:, None] * flux / magnitude[:, None]
    velocity[axis_nodes, 0] = 0.0
    if not bool(np.isfinite(velocity).all()):
        raise ValueError("ion drift-diffusion direction produced a non-finite velocity")
    return np.ascontiguousarray(velocity, dtype="<f8")


def _field(
    name: str,
    layout: str,
    components: tuple[str, ...],
    basis: str,
    values: np.ndarray,
    unit: str,
) -> FieldData:
    return FieldData(
        name=name,
        layout=layout,
        association="node",
        components=components,
        stored_basis=basis,
        values=np.ascontiguousarray(values, dtype="<f8"),
        unit=unit,
    )


def _select_layout(data: DataBundle, name: str) -> P1TriLayout:
    if data.coordinate_system != "axisymmetric_rz":
        raise ValueError("builder v1 requires coordinate_system=axisymmetric_rz")
    matches = [layout for layout in data.layouts if layout.name == name]
    if len(matches) != 1 or not isinstance(matches[0], P1TriLayout):
        raise ValueError(f"input.layout {name!r} must select exactly one P1TriLayout")
    layout = cast(P1TriLayout, matches[0])
    geometry = data.geometry
    if geometry.tri3 is None or geometry.quad4 is not None:
        raise ValueError("builder v1 requires a triangle-only canonical geometry")
    if not np.array_equal(geometry.nodes_m, layout.nodes_m) or not np.array_equal(
        geometry.tri3, layout.connectivity
    ):
        raise ValueError("builder layout must exactly match canonical geometry nodes and triangles")
    if not bool(np.all(layout.cell_support == 1)):
        raise ValueError("builder v1 requires every P1 cell to be supported")
    return layout


def _select_thermal_fields(
    data: DataBundle, specification: InputSpecification, layout: P1TriLayout
) -> tuple[FieldData, FieldData]:
    by_name = {field.name: field for field in data.fields}
    try:
        velocity = by_name[specification.gas_velocity_field]
        temperature = by_name[specification.gas_temperature_field]
    except KeyError as error:
        raise ValueError(f"missing thermal-flow field: {error.args[0]}") from error
    _require_field(
        velocity,
        layout,
        components=("r", "z"),
        basis="axisymmetric_rz",
        unit="m/s",
    )
    _require_field(
        temperature,
        layout,
        components=("value",),
        basis="scalar",
        unit="K",
    )
    if bool((temperature.values[:, 0] <= 0.0).any()):
        raise ValueError("gas temperature must be positive")
    axis = layout.nodes_m[:, 0] == 0.0
    if bool((velocity.values[axis, 0] != 0.0).any()):
        raise ValueError("gas radial velocity must be exactly zero on the symmetry axis")
    return velocity, temperature


def _require_field(
    field: FieldData,
    layout: P1TriLayout,
    *,
    components: tuple[str, ...],
    basis: str,
    unit: str,
) -> None:
    expected = (layout.name, "node", components, basis, unit)
    actual = (
        field.layout,
        field.association,
        field.components,
        field.stored_basis,
        field.unit,
    )
    if actual != expected:
        raise ValueError(
            f"field {field.name!r} metadata must be layout/node/{components}/{basis}/{unit}"
        )


def _resolve_boundaries(
    data: DataBundle,
    layout: P1TriLayout,
    conditions: tuple[BoundaryCondition, ...],
) -> tuple[np.ndarray, np.ndarray]:
    names = {name: index for index, name in enumerate(data.geometry.group_names)}
    priority = np.full(layout.nodes_m.shape[0], np.iinfo(np.int64).min, dtype=np.int64)
    selected = np.zeros(layout.nodes_m.shape[0], dtype=np.bool_)
    target = np.zeros(layout.nodes_m.shape[0], dtype="<f8")
    for condition in conditions:
        if condition.group not in names:
            raise ValueError(f"unknown boundary group: {condition.group}")
        rows = data.geometry.boundary.group_id == names[condition.group]
        if not bool(rows.any()):
            raise ValueError(f"boundary group has no facets: {condition.group}")
        nodes = np.unique(data.geometry.boundary.line2[rows].reshape(-1))
        values = _potential_values(condition.potential, layout.nodes_m[nodes, 0])
        _merge_boundary_values(
            nodes,
            values,
            condition.priority,
            priority,
            selected,
            target,
            condition.group,
        )
    ids = np.flatnonzero(selected).astype("<i8", copy=False)
    if ids.size == 0:
        raise ValueError("at least one semantic boundary group must prescribe potential")
    return np.ascontiguousarray(ids), np.ascontiguousarray(target[ids], dtype="<f8")


def _merge_boundary_values(
    nodes: np.ndarray,
    values: np.ndarray,
    new_priority: int,
    priority: np.ndarray,
    selected: np.ndarray,
    target: np.ndarray,
    group: str,
) -> None:
    for local_index, node in enumerate(nodes):
        node_id = int(node)
        value = float(values[local_index])
        if new_priority > int(priority[node_id]):
            priority[node_id] = new_priority
            target[node_id] = value
            selected[node_id] = True
        elif new_priority == int(priority[node_id]) and target[node_id] != value:
            raise ValueError(
                f"equal-priority boundary potentials conflict at node {node_id} in group {group!r}"
            )


def _potential_values(profile: PotentialProfile, radius_m: np.ndarray) -> np.ndarray:
    if isinstance(profile, ConstantPotential):
        return np.full(radius_m.shape, profile.value_V, dtype="<f8")
    distance = np.maximum(0.0, radius_m - profile.start_radius_m)
    result = profile.outer_value_V + (profile.inner_value_V - profile.outer_value_V) * np.exp(
        -distance / profile.transition_length_m
    )
    return np.ascontiguousarray(result, dtype="<f8")


def _producer_metadata(
    specification: BuilderSpecification,
    config_hash: str,
    input_hash: str,
    input_provenance: str,
    solution: ElectrostaticSolution,
) -> dict[str, object]:
    return {
        "builder_revision": BUILDER_REVISION,
        "model_revision": MODEL_REVISION,
        "discretization_revision": solution.report.discretization_revision,
        "configuration_sha256": config_hash,
        "input_content_hash": input_hash,
        "input_provenance_sha256": "sha256:"
        + hashlib.sha256(input_provenance.encode("utf-8")).hexdigest(),
        "layout": specification.input.layout,
        "thermal_fields": {
            "gas_velocity": specification.input.gas_velocity_field,
            "gas_temperature": specification.input.gas_temperature_field,
        },
        "closure": asdict(specification.closure),
        "ion_flux": asdict(specification.ion_flux),
        "boundary_conditions": [_boundary_record(item) for item in specification.boundaries],
        "solver": asdict(specification.solver),
        "convergence": _solve_report(solution),
        "generated_fields": list(_GENERATED_FIELD_NAMES),
        "positive_ion_temperature_model": "gas_temperature_equilibrium_v1",
        "positive_ion_velocity_model": "drift_diffusion_direction_bohm_speed_v1",
    }


def _boundary_record(condition: BoundaryCondition) -> dict[str, object]:
    return {
        "group": condition.group,
        "priority": condition.priority,
        "potential": asdict(condition.potential),
    }


def _solve_report(solution: ElectrostaticSolution) -> dict[str, object]:
    report = solution.report
    return {
        "model_revision": report.model_revision,
        "discretization_revision": report.discretization_revision,
        "node_count": report.node_count,
        "cell_count": report.cell_count,
        "continuation": [asdict(item) for item in report.continuation],
        "free_residual_linf_C": report.free_residual_linf_C,
        "relative_residual": report.relative_residual,
        "total_space_charge_C": report.total_space_charge_C,
        "boundary_reaction_charge_C": report.boundary_reaction_charge_C,
        "charge_balance_error_C": report.charge_balance_error_C,
        "electron_density_floor_nodes": report.electron_density_floor_nodes,
        "positive_ion_density_floor_nodes": report.positive_ion_density_floor_nodes,
        "positive_ion_speed_floor_nodes": report.positive_ion_speed_floor_nodes,
    }


def _field_ranges(fields: tuple[FieldData, ...]) -> dict[str, object]:
    return {
        field.name: {
            "minimum": np.min(field.values, axis=0).tolist(),
            "maximum": np.max(field.values, axis=0).tolist(),
            "unit": field.unit,
        }
        for field in fields
    }


def _parse_input(value: dict[str, object], base_directory: Path) -> InputSpecification:
    _exact_keys(
        value,
        {"data_path", "layout", "gas_velocity_field", "gas_temperature_field"},
        "input",
    )
    raw_path = Path(_text(value["data_path"], "input.data_path"))
    data_path = raw_path if raw_path.is_absolute() else base_directory / raw_path
    return InputSpecification(
        data_path.resolve(),
        _name(value["layout"], "input.layout"),
        _name(value["gas_velocity_field"], "input.gas_velocity_field"),
        _name(value["gas_temperature_field"], "input.gas_temperature_field"),
    )


def _parse_model(value: dict[str, object]) -> tuple[ClosureParameters, IonFluxParameters]:
    required = {
        "revision",
        "bulk_number_density_m3",
        "electron_temperature_V",
        "positive_ion_mass_kg",
        "bulk_potential_V",
        "ion_mobility_m2_V_s",
        "ion_flux_regularization_speed_m_s",
        "sheath_smoothing_V",
        "density_floor_m3",
        "ion_speed_floor_m_s",
    }
    _exact_keys(value, required, "model")
    revision = _text(value["revision"], "model.revision")
    if revision != MODEL_REVISION:
        raise ValueError(f"model.revision must be {MODEL_REVISION}")
    closure = ClosureParameters(
        bulk_number_density_m3=_number(
            value["bulk_number_density_m3"], "model.bulk_number_density_m3"
        ),
        electron_temperature_V=_number(
            value["electron_temperature_V"], "model.electron_temperature_V"
        ),
        positive_ion_mass_kg=_number(value["positive_ion_mass_kg"], "model.positive_ion_mass_kg"),
        bulk_potential_V=_number(value["bulk_potential_V"], "model.bulk_potential_V"),
        sheath_smoothing_V=_number(value["sheath_smoothing_V"], "model.sheath_smoothing_V"),
        density_floor_m3=_number(value["density_floor_m3"], "model.density_floor_m3"),
        ion_speed_floor_m_s=_number(value["ion_speed_floor_m_s"], "model.ion_speed_floor_m_s"),
    )
    ion_flux = IonFluxParameters(
        mobility_m2_V_s=_positive_number(value["ion_mobility_m2_V_s"], "model.ion_mobility_m2_V_s"),
        regularization_speed_m_s=_positive_number(
            value["ion_flux_regularization_speed_m_s"],
            "model.ion_flux_regularization_speed_m_s",
        ),
    )
    return closure, ion_flux


def _parse_boundaries(value: object) -> tuple[BoundaryCondition, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("boundaries must be a nonempty sequence")
    conditions: list[BoundaryCondition] = []
    groups: set[str] = set()
    for index, raw in enumerate(value):
        label = f"boundaries[{index}]"
        item = _mapping(raw, label)
        _exact_keys(item, {"group", "priority", "potential"}, label)
        group = _name(item["group"], f"{label}.group")
        if group in groups:
            raise ValueError(f"duplicate boundary group: {group}")
        groups.add(group)
        conditions.append(
            BoundaryCondition(
                group,
                _integer(item["priority"], f"{label}.priority"),
                _parse_potential(_mapping(item["potential"], f"{label}.potential"), label),
            )
        )
    return tuple(conditions)


def _parse_potential(value: dict[str, object], boundary_label: str) -> PotentialProfile:
    kind = _text(value.get("kind"), f"{boundary_label}.potential.kind")
    if kind == "constant":
        _exact_keys(value, {"kind", "value_V"}, f"{boundary_label}.potential")
        return ConstantPotential(
            "constant", _number(value["value_V"], f"{boundary_label}.potential.value_V")
        )
    if kind != "radial_exponential":
        raise ValueError(f"{boundary_label}.potential.kind must be constant or radial_exponential")
    required = {
        "kind",
        "inner_value_V",
        "outer_value_V",
        "start_radius_m",
        "transition_length_m",
    }
    _exact_keys(value, required, f"{boundary_label}.potential")
    return RadialExponentialPotential(
        "radial_exponential",
        _number(value["inner_value_V"], f"{boundary_label}.potential.inner_value_V"),
        _number(value["outer_value_V"], f"{boundary_label}.potential.outer_value_V"),
        _nonnegative_number(value["start_radius_m"], f"{boundary_label}.potential.start_radius_m"),
        _positive_number(
            value["transition_length_m"],
            f"{boundary_label}.potential.transition_length_m",
        ),
    )


def _parse_solver(value: dict[str, object]) -> SolverSettings:
    required = {
        "continuation_ramps",
        "max_newton_iterations",
        "relative_residual_tolerance",
        "absolute_residual_tolerance_C",
        "max_linear_iterations",
        "linear_krylov_dimension",
        "linear_relative_tolerance",
        "minimum_line_search_factor",
    }
    _exact_keys(value, required, "solver")
    return SolverSettings(
        continuation_ramps=_number_tuple(value["continuation_ramps"], "solver.continuation_ramps"),
        max_newton_iterations=_integer(
            value["max_newton_iterations"], "solver.max_newton_iterations"
        ),
        relative_residual_tolerance=_number(
            value["relative_residual_tolerance"], "solver.relative_residual_tolerance"
        ),
        absolute_residual_tolerance_C=_number(
            value["absolute_residual_tolerance_C"],
            "solver.absolute_residual_tolerance_C",
        ),
        max_linear_iterations=_integer(
            value["max_linear_iterations"], "solver.max_linear_iterations"
        ),
        linear_krylov_dimension=_integer(
            value["linear_krylov_dimension"], "solver.linear_krylov_dimension"
        ),
        linear_relative_tolerance=_number(
            value["linear_relative_tolerance"], "solver.linear_relative_tolerance"
        ),
        minimum_line_search_factor=_number(
            value["minimum_line_search_factor"], "solver.minimum_line_search_factor"
        ),
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with text keys")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be nonempty text without NUL")
    return value


def _name(value: object, label: str) -> str:
    text = _text(value, label)
    if not text[0].isalpha() or not all(
        character.isalnum() or character == "_" for character in text
    ):
        raise ValueError(f"{label} must be an identifier-like name")
    return text


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a sequence of finite numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


__all__ = [
    "BUILDER_FORMAT_VERSION",
    "BUILDER_REVISION",
    "MODEL_REVISION",
    "BuilderSpecification",
    "build_from_configuration",
    "parse_configuration",
]
