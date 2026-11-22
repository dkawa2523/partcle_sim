"""Strict canonical input/output workflow for the external field preprocessor."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np

from chamber_particles.case_format import (
    DataBundle,
    FieldData,
    Layout,
    RegularLayout,
    read_with_info,
    write,
)
from chamber_particles.fields import (
    RequiredFieldMetadata,
    certify_rz_axis_field_regularity,
    prepare_required_fields,
    rz_axis_accessible,
)
from chamber_particles.yaml_input import parse_document

from .numerics import (
    SOURCE_TIME_DIAGNOSTIC_REVISION,
    CommonPartition,
    ValidationResources,
    layout_summary,
    prepare_partition,
    resample_field,
    source_time_diagnostic,
    validate_pair,
)

PREPROCESSOR_FORMAT_VERSION: Final = 2
PREPROCESSOR_REVISION: Final = "canonical_field_preprocessor_v4"
FIELD_SEMANTICS_REVISION: Final = "weighted_polynomial_space_linear_time_cache_v3"
PRODUCER_VERSION: Final = version("chamber-particles")


@dataclass(frozen=True, slots=True)
class AxisSpecification:
    start_m: float
    stop_m: float
    count: int


@dataclass(frozen=True, slots=True)
class ExistingTarget:
    kind: Literal["existing"]
    layout: str


@dataclass(frozen=True, slots=True)
class RegularTarget:
    kind: Literal["regular"]
    name: str
    axis0: AxisSpecification
    axis1: AxisSpecification


type TargetSpecification = ExistingTarget | RegularTarget


@dataclass(frozen=True, slots=True)
class ErrorLimits:
    value_relative_l2: float
    gradient_relative_l2: float
    boundary_value_relative_l2: float


@dataclass(frozen=True, slots=True)
class FieldSpecification:
    source: str
    output: str
    limits: ErrorLimits


@dataclass(frozen=True, slots=True)
class PreprocessorSpecification:
    data_path: Path
    target: TargetSpecification
    fields: tuple[FieldSpecification, ...]
    validation: ValidationResources


def preprocess_from_configuration(
    configuration_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    source_time_diagnostic_requested: bool = False,
) -> dict[str, object]:
    """Resample selected fields and atomically publish a new canonical bundle."""

    configuration = Path(configuration_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path is not None else None
    if report == output:
        raise ValueError("report_path and output_path must be different")
    if output.exists():
        raise FileExistsError(output)
    if report is not None and report.exists():
        raise FileExistsError(report)
    raw = configuration.read_bytes()
    if type(source_time_diagnostic_requested) is not bool:
        raise ValueError("source_time_diagnostic_requested must be a boolean")
    specification = parse_configuration(raw, base_directory=configuration.parent)
    configuration_sha256 = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    result = _preprocess(
        specification, configuration_sha256, output, source_time_diagnostic_requested
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8", errors="strict") as stream:
            stream.write(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return result


def parse_configuration(raw: bytes, *, base_directory: Path) -> PreprocessorSpecification:
    """Parse the small, exact YAML configuration owned by this tool."""

    document = parse_document(raw)
    root = _mapping(document, "configuration")
    _exact_keys(
        root, {"format_version", "input", "target", "fields", "validation"}, "configuration"
    )
    if _integer(root["format_version"], "format_version") != PREPROCESSOR_FORMAT_VERSION:
        raise ValueError(f"format_version must be {PREPROCESSOR_FORMAT_VERSION}")
    input_mapping = _mapping(root["input"], "input")
    _exact_keys(input_mapping, {"data_path"}, "input")
    data_path = _path(input_mapping["data_path"], "input.data_path", base_directory)
    target = _parse_target(_mapping(root["target"], "target"))
    fields = _parse_fields(root["fields"])
    validation = _parse_validation(_mapping(root["validation"], "validation"))
    return PreprocessorSpecification(data_path, target, fields, validation)


def _parse_validation(value: dict[str, object]) -> ValidationResources:
    _exact_keys(value, {"memory_limit_mb", "workspace_rows", "max_patch_work"}, "validation")
    memory_mb = _finite_float(value["memory_limit_mb"], "validation.memory_limit_mb")
    memory_bytes = memory_mb * 1024.0 * 1024.0
    if not 0.0 < memory_bytes <= np.iinfo(np.intp).max:
        raise ValueError("validation.memory_limit_mb must resolve to positive addressable bytes")
    rows = _integer(value["workspace_rows"], "validation.workspace_rows")
    work = _integer(value["max_patch_work"], "validation.max_patch_work")
    if not 16 <= rows <= np.iinfo(np.intp).max:
        raise ValueError("validation.workspace_rows must be at least 16 and addressable")
    if work <= 0:
        raise ValueError("validation.max_patch_work must be positive")
    return ValidationResources(int(memory_bytes), rows, work)


def _target_counts(target: TargetSpecification, data: DataBundle) -> tuple[int, int]:
    if isinstance(target, RegularTarget):
        return target.axis0.count * target.axis1.count, (target.axis0.count - 1) * (
            target.axis1.count - 1
        )
    layout = next((item for item in data.layouts if item.name == target.layout), None)
    if layout is None:
        return 0, 0  # The target resolver owns the unknown-name error.
    nodes = (
        layout.axis0_m.size * layout.axis1_m.size
        if isinstance(layout, RegularLayout)
        else layout.nodes_m.shape[0]
    )
    return int(nodes), int(layout.cell_support.size)


def _preflight_resources(
    specification: PreprocessorSpecification, data: DataBundle, source_bytes: int
) -> dict[str, int | str]:
    target_nodes, target_cells = _target_counts(specification.target, data)
    fields = {field.name: field for field in data.fields}
    selected_components = selected_snapshots = 0
    for item in specification.fields:
        field = fields.get(item.source)
        if field is not None:
            knots = field.time_s
            count = 1 if knots is None else int(knots.size)
            selected_components += len(field.components) * count
            selected_snapshots += count
    max_components = max((len(field.components) for field in data.fields), default=1)
    source_cells = sum(layout.cell_support.size for layout in data.layouts)
    # Source residency/hash/write transients, retained cache values, layout/index
    # arrays, both sampled workspaces, gradient buffers, and bounded patch scratch.
    planned = (
        3 * source_bytes
        + 8 * target_nodes * selected_components
        + 128 * source_cells
        + 80 * target_cells
        + 80 * target_nodes
        + 4096 * selected_snapshots
        + specification.validation.workspace_rows * (512 + 256 * max_components)
        + 1024 * 1024
    )
    if planned > specification.validation.memory_limit_bytes:
        raise ValueError(
            f"validation memory preflight requires {planned} bytes, exceeding memory_limit_mb"
        )
    return {
        "source_numeric_bytes": source_bytes,
        "planned_owned_array_bytes": int(planned),
        "memory_limit_bytes": specification.validation.memory_limit_bytes,
        "workspace_rows": specification.validation.workspace_rows,
        "max_patch_work": specification.validation.max_patch_work,
        "memory_scope": "owned_numeric_arrays_and_patch_scratch_not_process_rss",
    }


def _preprocess(
    specification: PreprocessorSpecification,
    configuration_sha256: str,
    output: Path,
    source_time_diagnostic_requested: bool,
) -> dict[str, object]:
    data, source_info = read_with_info(
        specification.data_path,
        numeric_array_limit_bytes=specification.validation.memory_limit_bytes,
    )
    resource_report = _preflight_resources(
        specification, data, source_info.footprint.numeric_array_bytes
    )
    layouts = {layout.name: layout for layout in data.layouts}
    fields = {field.name: field for field in data.fields}
    target_layout, generated_target = _resolve_target(specification.target, data, layouts)
    selected = _resolve_fields(specification.fields, fields, target_layout)
    partitions: dict[str, CommonPartition] = {}
    for field, _field_specification in selected:
        validate_pair(field, layouts[field.layout], target_layout)
        if field.layout not in partitions:
            partitions[field.layout] = prepare_partition(
                layouts[field.layout],
                cast(RegularLayout, target_layout),
                specification.validation,
            )
    axis_accessible = rz_axis_accessible(data, target_layout)
    _certify_source_axis_regularity(data, selected, layouts)
    generated_fields: list[FieldData] = []
    field_reports: list[dict[str, object]] = []
    for field, field_specification in selected:
        cached, field_report = resample_field(
            field,
            partitions[field.layout],
            field_specification.output,
            axis_accessible=axis_accessible,
            coordinate_system=data.coordinate_system,
            limits=(
                field_specification.limits.value_relative_l2,
                field_specification.limits.gradient_relative_l2,
                field_specification.limits.boundary_value_relative_l2,
            ),
        )
        _enforce_limits(field_report, field_specification.limits)
        field_report["limits"] = {
            "value_relative_l2": field_specification.limits.value_relative_l2,
            "gradient_relative_l2": field_specification.limits.gradient_relative_l2,
            "boundary_value_relative_l2": (field_specification.limits.boundary_value_relative_l2),
        }
        generated_fields.append(cached)
        field_reports.append(field_report)
    resource_report["patch_work"] = specification.validation.patch_work
    diagnostic_resources = _source_time_diagnostics(
        data,
        selected,
        field_reports,
        partitions,
        resource_report,
        axis_accessible,
        source_time_diagnostic_requested,
    )
    metadata = {
        "preprocessor_revision": PREPROCESSOR_REVISION,
        "configuration_sha256": configuration_sha256,
        "source_provenance": json.loads(data.provenance_json),
        "target_layout": layout_summary(target_layout),
        "resampled_fields": field_reports,
        "validation_resources": resource_report,
        "source_time_diagnostic_resources": diagnostic_resources,
    }
    provenance = json.dumps(
        {
            "producer": "chamber_particles.field_preprocessor",
            "producer_version": PRODUCER_VERSION,
            "source_sha256": source_info.content_hash,
            "field_semantics_revision": FIELD_SEMANTICS_REVISION,
            "producer_metadata": metadata,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    output_bundle = DataBundle(
        coordinate_system=data.coordinate_system,
        provenance_json=provenance,
        geometry=data.geometry,
        layouts=(*data.layouts, target_layout) if generated_target else data.layouts,
        fields=(*data.fields, *generated_fields),
        sources=data.sources,
    )
    _certify_solver_ready_cache(output_bundle, generated_fields)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_info = write(output, output_bundle)
    return {
        "status": "complete",
        "preprocessor_revision": PREPROCESSOR_REVISION,
        "input_path": str(specification.data_path),
        "input_content_hash": source_info.content_hash,
        "configuration_sha256": configuration_sha256,
        "output_path": str(output),
        "output_content_hash": output_info.content_hash,
        "target_layout": layout_summary(target_layout),
        "resampled_fields": field_reports,
        "validation_resources": resource_report,
        "source_time_diagnostic_resources": diagnostic_resources,
    }


def _source_time_diagnostics(
    data: DataBundle,
    selected: tuple[tuple[FieldData, FieldSpecification], ...],
    field_reports: list[dict[str, object]],
    partitions: dict[str, CommonPartition],
    resource_report: Mapping[str, object],
    axis_accessible: bool,
    requested: bool,
) -> dict[str, object]:
    if not requested:
        for report in field_reports:
            report["source_time_diagnostic"] = {
                "status": "NOT_REQUESTED",
                "revision": SOURCE_TIME_DIAGNOSTIC_REVISION,
                "continuous_time_fidelity": "NOT_TESTED",
            }
        return {"requested": False}
    original = next(iter(partitions.values())).resources
    snapshot_count = sum(0 if field.time_s is None else field.time_s.size for field, _ in selected)
    components = max(len(field.components) for field, _ in selected)
    extra_bytes = 4096 * snapshot_count + original.workspace_rows * (256 + 128 * components)
    planned = int(cast(int, resource_report["planned_owned_array_bytes"])) + int(extra_bytes)
    resources = ValidationResources(
        original.memory_limit_bytes,
        original.workspace_rows,
        max(0, original.max_patch_work - original.patch_work),
    )
    for (field, _), report in zip(selected, field_reports, strict=True):
        if (
            planned > original.memory_limit_bytes
            and field.time_s is not None
            and field.time_s.size > 2
        ):
            report["source_time_diagnostic"] = {
                "status": "RESOURCE_LIMITED",
                "revision": SOURCE_TIME_DIAGNOSTIC_REVISION,
                "continuous_time_fidelity": "NOT_TESTED",
                "reason": "diagnostic owned-array memory preflight exceeds memory_limit_mb",
            }
            continue
        report["source_time_diagnostic"] = source_time_diagnostic(
            field,
            replace(partitions[field.layout], resources=resources),
            axis_accessible=axis_accessible,
            coordinate_system=data.coordinate_system,
        )
    return {
        "requested": True,
        "planned_owned_array_bytes": planned,
        "memory_limit_bytes": original.memory_limit_bytes,
        "workspace_rows": resources.workspace_rows,
        "max_patch_work": resources.max_patch_work,
        "attempted_patch_work": resources.patch_work,
        "resource_scope": "remaining_budget_after_all_cache_norm_gates",
    }


def _certify_source_axis_regularity(
    data: DataBundle,
    selected: tuple[tuple[FieldData, FieldSpecification], ...],
    layouts: dict[str, Layout],
) -> None:
    """Reject invalid raw RZ axis values before any sampler can normalize them."""

    by_layout: dict[str, dict[str, FieldData]] = {}
    for field, _specification in selected:
        by_layout.setdefault(field.layout, {})[field.name] = field
    for layout_name, fields in by_layout.items():
        layout = layouts[layout_name]
        accessible = rz_axis_accessible(data, layout)
        certify_rz_axis_field_regularity(
            data.coordinate_system,
            layout,
            fields,
            accessible,
        )


def _certify_solver_ready_cache(data: DataBundle, fields: list[FieldData]) -> None:
    requirements = {
        field.name: RequiredFieldMetadata(
            unit=field.unit,
            components=field.components,
            stored_basis=field.stored_basis,
            positive=False,
        )
        for field in fields
    }
    prepare_required_fields(data, requirements)


def _resolve_target(
    specification: TargetSpecification,
    data: DataBundle,
    layouts: dict[str, Layout],
) -> tuple[Layout, bool]:
    if isinstance(specification, ExistingTarget):
        try:
            return layouts[specification.layout], False
        except KeyError as error:
            raise ValueError(f"unknown target layout {specification.layout!r}") from error
    if specification.name in layouts:
        raise ValueError(f"target layout {specification.name!r} already exists")
    if data.coordinate_system == "axisymmetric_rz" and specification.axis0.start_m < 0.0:
        raise ValueError("axisymmetric target axis0 requires r >= 0")
    axis0 = np.linspace(
        specification.axis0.start_m,
        specification.axis0.stop_m,
        specification.axis0.count,
        dtype="<f8",
    )
    axis1 = np.linspace(
        specification.axis1.start_m,
        specification.axis1.stop_m,
        specification.axis1.count,
        dtype="<f8",
    )
    support = np.ones((axis0.size - 1, axis1.size - 1), dtype="<u1")
    return RegularLayout(specification.name, axis0, axis1, support), True


def _resolve_fields(
    specifications: tuple[FieldSpecification, ...],
    available: dict[str, FieldData],
    target_layout: Layout,
) -> tuple[tuple[FieldData, FieldSpecification], ...]:
    existing_names = set(available)
    output_names: set[str] = set()
    resolved: list[tuple[FieldData, FieldSpecification]] = []
    for specification in specifications:
        try:
            field = available[specification.source]
        except KeyError as error:
            raise ValueError(f"unknown source field {specification.source!r}") from error
        if field.association != "node":
            raise ValueError(
                f"source field {field.name!r} must be node-associated; discontinuous "
                "cell data requires an explicit side-selection model"
            )
        if field.layout == target_layout.name:
            raise ValueError(f"source field {field.name!r} already uses the target layout")
        if specification.output in existing_names or specification.output in output_names:
            raise ValueError(f"output field name is not unique: {specification.output!r}")
        output_names.add(specification.output)
        resolved.append((field, specification))
    return tuple(resolved)


def _enforce_limits(field_report: dict[str, object], limits: ErrorLimits) -> None:
    metrics = cast(dict[str, object], field_report["metrics"])
    checks = (
        ("value", limits.value_relative_l2),
        ("gradient", limits.gradient_relative_l2),
        ("boundary_value", limits.boundary_value_relative_l2),
    )
    for metric_name, limit in checks:
        metric = cast(dict[str, object], metrics[metric_name])
        relative_l2 = metric["relative_l2_upper"]
        if relative_l2 is None:
            raise ValueError(
                f"field {field_report['source_field']!r} {metric_name} relative L2 "
                "is uncertified because the reference lower norm is zero"
            )
        if float(cast(float, relative_l2)) > limit:
            raise ValueError(
                f"field {field_report['source_field']!r} {metric_name} relative L2 "
                f"{float(cast(float, relative_l2)):.17g} exceeds limit {limit:.17g}"
            )


def _parse_target(value: dict[str, object]) -> TargetSpecification:
    kind = _string(value.get("kind"), "target.kind")
    if kind == "existing":
        _exact_keys(value, {"kind", "layout"}, "target")
        return ExistingTarget("existing", _string(value["layout"], "target.layout"))
    if kind == "regular":
        _exact_keys(value, {"kind", "name", "axis0", "axis1"}, "target")
        return RegularTarget(
            "regular",
            _string(value["name"], "target.name"),
            _parse_axis(_mapping(value["axis0"], "target.axis0"), "target.axis0"),
            _parse_axis(_mapping(value["axis1"], "target.axis1"), "target.axis1"),
        )
    raise ValueError("target.kind must be 'existing' or 'regular'")


def _parse_axis(value: dict[str, object], label: str) -> AxisSpecification:
    _exact_keys(value, {"start_m", "stop_m", "count"}, label)
    start = _finite_float(value["start_m"], f"{label}.start_m")
    stop = _finite_float(value["stop_m"], f"{label}.stop_m")
    count = _integer(value["count"], f"{label}.count")
    if stop <= start:
        raise ValueError(f"{label}.stop_m must be greater than start_m")
    if count < 2:
        raise ValueError(f"{label}.count must be at least 2")
    return AxisSpecification(start, stop, count)


def _parse_fields(value: object) -> tuple[FieldSpecification, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("fields must be a nonempty list")
    fields: list[FieldSpecification] = []
    for index, item in enumerate(value):
        label = f"fields[{index}]"
        field = _mapping(item, label)
        _exact_keys(field, {"source", "output", "limits"}, label)
        limits = _mapping(field["limits"], f"{label}.limits")
        _exact_keys(
            limits,
            {
                "value_relative_l2",
                "gradient_relative_l2",
                "boundary_value_relative_l2",
            },
            f"{label}.limits",
        )
        fields.append(
            FieldSpecification(
                _string(field["source"], f"{label}.source"),
                _string(field["output"], f"{label}.output"),
                ErrorLimits(
                    _nonnegative_float(
                        limits["value_relative_l2"], f"{label}.limits.value_relative_l2"
                    ),
                    _nonnegative_float(
                        limits["gradient_relative_l2"],
                        f"{label}.limits.gradient_relative_l2",
                    ),
                    _nonnegative_float(
                        limits["boundary_value_relative_l2"],
                        f"{label}.limits.boundary_value_relative_l2",
                    ),
                ),
            )
        )
    return tuple(fields)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")


def _path(value: object, label: str, base_directory: Path) -> Path:
    text = _string(value, label)
    path = Path(text).expanduser()
    return (base_directory / path).resolve() if not path.is_absolute() else path.resolve()


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _nonnegative_float(value: object, label: str) -> float:
    converted = _finite_float(value, label)
    if converted < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return converted
