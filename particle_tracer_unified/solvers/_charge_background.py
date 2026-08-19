"""Validate and sample plasma backgrounds for particle charging."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.datamodel import QuantitySeriesND, RegularFieldND

from ._charge_model_types import ChargeModelConfig
from .base_field_sampling import sample_regular_time_grid_points_2d
from .compiled_backend_types import (
    CompiledRuntimeBackend,
    RegularRectilinearCompiledBackend,
)
from .plasma_background import PreparedPlasmaBackground


def is_flux_balance_mode(config: ChargeModelConfig) -> bool:
    return str(config.mode) == "oml_linearized_relaxation"


def temperature_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.electron_temperature_quantity:
        return (config.electron_temperature_quantity,)
    return ("Te", "T_e", "electron_temperature_eV", "electron_temperature", "Te_eV")


def density_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.electron_density_quantity:
        return (config.electron_density_quantity,)
    return ("ne", "n_e", "electron_density", "electron_number_density")


def ion_density_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.ion_density_quantity:
        return (config.ion_density_quantity,)
    return ("ni", "n_i", "ion_density", "ion_number_density")


def select_quantity(field: RegularFieldND, names: Sequence[str]):
    for name in names:
        if str(name) in field.quantities:
            return field.quantities[str(name)]
    raise ValueError(
        f"Missing required charge-model field quantity; tried {list(names)}"
    )


def _charge_series(
    field: RegularFieldND,
    names: Sequence[str],
    *,
    expected_unit: str,
    label: str,
) -> QuantitySeriesND:
    series = select_quantity(field, names)
    declared_unit = str(series.unit)
    if declared_unit and declared_unit != expected_unit:
        raise ValueError(
            f"solver.charge_model {label} field {series.name!r} unit must be "
            f"{expected_unit!r}, got {declared_unit!r}"
        )
    return series


def sample_series_2d(
    field: RegularFieldND,
    series,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    return sample_regular_time_grid_points_2d(
        np.asarray(series.data, dtype=np.float64),
        tuple(np.asarray(axis, dtype=np.float64) for axis in field.axes),
        np.asarray(series.times, dtype=np.float64),
        float(t_eval),
        np.asarray(positions, dtype=np.float64),
    )


def sample_temperature_eV(
    config: ChargeModelConfig,
    field: RegularFieldND,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    series = _charge_series(
        field,
        temperature_names(config),
        expected_unit=str(config.electron_temperature_unit),
        label="electron temperature",
    )
    temperature = np.asarray(
        sample_series_2d(field, series, float(t_eval), positions),
        dtype=np.float64,
    )
    source_unit = str(series.unit) or str(config.electron_temperature_unit)
    if source_unit == "K":
        temperature = temperature / 11604.51812155008
    return np.where(
        np.isfinite(temperature) & (temperature > 0.0),
        temperature,
        np.nan,
    )


def sample_positive_series(
    field: RegularFieldND,
    series,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        sample_series_2d(field, series, float(t_eval), positions),
        dtype=np.float64,
    )
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def validate_charge_model_support(
    config: ChargeModelConfig,
    runtime,
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    plasma_background: PreparedPlasmaBackground | None = None,
) -> None:
    if not bool(config.enabled):
        return
    if int(spatial_dim) != 2:
        raise ValueError(
            "solver.charge_model currently supports 2D regular rectilinear fields; "
            "3D is planned separately"
        )
    if not isinstance(compiled, RegularRectilinearCompiledBackend):
        raise ValueError(
            "solver.charge_model requires a regular rectilinear field backend"
        )
    if str(config.background_source) == "plasma_background":
        if plasma_background is None:
            raise ValueError(
                "solver.charge_model.background_source=plasma_background requires "
                "solver.plasma_background.source=saas_constant"
            )
        return
    field_provider = getattr(runtime, "field_provider", None)
    field = getattr(field_provider, "field", None)
    if not isinstance(field, RegularFieldND):
        raise ValueError("solver.charge_model requires a regular field provider")
    _charge_series(
        field,
        temperature_names(config),
        expected_unit=str(config.electron_temperature_unit),
        label="electron temperature",
    )
    if is_flux_balance_mode(config):
        _charge_series(
            field,
            density_names(config),
            expected_unit="1/m^3",
            label="electron density",
        )
        _charge_series(
            field,
            ion_density_names(config),
            expected_unit="1/m^3",
            label="ion density",
        )
        if config.ion_temperature_quantity:
            _charge_series(
                field,
                (config.ion_temperature_quantity,),
                expected_unit="eV",
                label="ion temperature",
            )


@dataclass(frozen=True, slots=True)
class ChargeUpdateBatch:
    indices: np.ndarray
    positions: np.ndarray
    radius: np.ndarray
    old_charge: np.ndarray


@dataclass(frozen=True, slots=True)
class ChargeBackground:
    source: str
    electron_temperature_eV: np.ndarray
    electron_density_m3: np.ndarray | None = None
    ion_density_m3: np.ndarray | None = None
    ion_temperature_eV: np.ndarray | None = None
    debye_length_m: np.ndarray | None = None
    ion_mass_amu: float | None = None
    ion_charge_number: float | None = None


def prepare_charge_update_batch(
    active_mask: np.ndarray,
    x: np.ndarray,
    charge: np.ndarray,
    particle_diameter: np.ndarray,
) -> ChargeUpdateBatch | None:
    indices = np.flatnonzero(np.asarray(active_mask, dtype=bool))
    if indices.size == 0:
        return None
    positions = np.asarray(x[indices, :2], dtype=np.float64)
    radius = 0.5 * np.asarray(particle_diameter[indices], dtype=np.float64)
    if np.any(~np.isfinite(radius) | (radius <= 0.0)):
        raise ValueError(
            "solver.charge_model requires finite positive particle diameter"
        )
    old_charge = np.asarray(charge[indices], dtype=np.float64)
    if np.any(~np.isfinite(old_charge)):
        rows = np.flatnonzero(~np.isfinite(old_charge))[:12].tolist()
        raise ValueError(
            "solver.charge_model requires finite initial charge; "
            f"invalid active rows {rows}"
        )
    return ChargeUpdateBatch(indices, positions, radius, old_charge)


def regular_charge_field(runtime) -> RegularFieldND:
    field_provider = getattr(runtime, "field_provider", None)
    field = getattr(field_provider, "field", None)
    if not isinstance(field, RegularFieldND):
        raise ValueError("solver.charge_model requires a regular field provider")
    return field


def resolve_charge_background(
    *,
    config: ChargeModelConfig,
    runtime,
    batch: ChargeUpdateBatch,
    t_eval: float,
    plasma_background: PreparedPlasmaBackground | None,
    collect: bool,
) -> ChargeBackground:
    source = str(config.background_source)
    oml_mode = str(config.mode) == "oml_linearized_relaxation"
    if source != "plasma_background":
        field = regular_charge_field(runtime)
        return ChargeBackground(
            source=source,
            electron_temperature_eV=sample_temperature_eV(
                config,
                field,
                float(t_eval),
                batch.positions,
            ),
        )
    if plasma_background is None:
        raise ValueError(
            "solver.charge_model.background_source=plasma_background requires "
            "solver.plasma_background.source=saas_constant"
        )
    size = batch.indices.size
    include_plasma = oml_mode or collect
    return ChargeBackground(
        source=source,
        electron_temperature_eV=np.full(
            size,
            float(plasma_background.electron_temperature_eV),
            dtype=np.float64,
        ),
        electron_density_m3=(
            np.full(
                size, float(plasma_background.electron_density_m3), dtype=np.float64
            )
            if include_plasma
            else None
        ),
        ion_density_m3=(
            np.full(size, float(plasma_background.ion_density_m3), dtype=np.float64)
            if include_plasma
            else None
        ),
        ion_temperature_eV=(
            np.full(size, float(plasma_background.ion_temperature_eV), dtype=np.float64)
            if include_plasma
            else None
        ),
        debye_length_m=(
            np.full(size, float(plasma_background.debye_length_m), dtype=np.float64)
            if collect
            else None
        ),
        ion_mass_amu=float(plasma_background.ion_mass_amu) if oml_mode else None,
        ion_charge_number=(
            float(plasma_background.ion_charge_number) if oml_mode else None
        ),
    )


def complete_oml_background(
    *,
    config: ChargeModelConfig,
    runtime,
    batch: ChargeUpdateBatch,
    t_eval: float,
    background: ChargeBackground,
    collect: bool,
    debye_length: Callable[..., np.ndarray],
) -> ChargeBackground:
    if (
        background.electron_density_m3 is not None
        and background.ion_density_m3 is not None
        and background.ion_temperature_eV is not None
    ):
        return background
    field = regular_charge_field(runtime)
    ne = sample_positive_series(
        field,
        _charge_series(
            field,
            density_names(config),
            expected_unit="1/m^3",
            label="electron density",
        ),
        float(t_eval),
        batch.positions,
    )
    ni = sample_positive_series(
        field,
        _charge_series(
            field,
            ion_density_names(config),
            expected_unit="1/m^3",
            label="ion density",
        ),
        float(t_eval),
        batch.positions,
    )
    ti_series = (
        _charge_series(
            field,
            (config.ion_temperature_quantity,),
            expected_unit="eV",
            label="ion temperature",
        )
        if config.ion_temperature_quantity
        else None
    )
    ti = (
        sample_positive_series(field, ti_series, float(t_eval), batch.positions)
        if ti_series is not None
        else np.full(
            batch.positions.shape[0],
            float(config.ion_temperature_eV),
            dtype=np.float64,
        )
    )
    debye = (
        debye_length(
            background.electron_temperature_eV,
            ne,
            ti,
            ni,
            float(config.ion_charge_number),
        )
        if collect
        else None
    )
    return ChargeBackground(
        source=background.source,
        electron_temperature_eV=background.electron_temperature_eV,
        electron_density_m3=ne,
        ion_density_m3=ni,
        ion_temperature_eV=ti,
        debye_length_m=debye,
        ion_mass_amu=background.ion_mass_amu,
        ion_charge_number=background.ion_charge_number,
    )


__all__ = (
    "ChargeBackground",
    "ChargeUpdateBatch",
    "complete_oml_background",
    "density_names",
    "ion_density_names",
    "prepare_charge_update_batch",
    "resolve_charge_background",
    "validate_charge_model_support",
)
