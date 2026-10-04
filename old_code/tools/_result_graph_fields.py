"""Render field and drag-property maps and their tabular data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools._result_graph_common import PYPLOT as plt
from tools._result_graph_maps import (
    as_2d_mask,
    masked_field,
    plot_scalar_map,
)

ScalarMapSpec = tuple[str, str, np.ndarray, str, bool]
OptionalScalarMapSpec = tuple[str, str, np.ndarray | None, str, bool]
DragSourceSpec = tuple[str, str, str, Any, Any, Any]


def _optional_masked_field(
    field: dict[str, np.ndarray], mask: np.ndarray | None, name: str
) -> np.ndarray | None:
    if name not in field:
        return None
    return masked_field(field[name], mask)


def _save_scalar_map_grid(
    *,
    out_dir: Path,
    filename: str,
    specs: list[ScalarMapSpec],
    x: np.ndarray,
    y: np.ndarray,
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None,
    max_columns: int,
    width_per_column: float,
    height_per_row: float,
    figure_title: str | None = None,
) -> bool:
    if not specs:
        return False
    columns = min(max_columns, len(specs))
    rows = int(np.ceil(len(specs) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(width_per_column * columns, height_per_row * rows),
        squeeze=False,
    )
    for ax_plot, (title, label, array, cmap, symmetric) in zip(
        axes.ravel(), specs, strict=False
    ):
        plot_scalar_map(
            fig,
            ax_plot,
            x,
            y,
            array,
            title=title,
            cbar_label=label,
            cmap=cmap,
            symmetric=symmetric,
            edges=edges,
            edge_part_ids=edge_part_ids,
            geometry_payload=geometry,
            medium_summary=medium_summary,
        )
    for empty_axis in axes.ravel()[len(specs) :]:
        empty_axis.axis("off")
    if figure_title is None:
        fig.tight_layout()
    else:
        fig.suptitle(figure_title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / filename, dpi=170)
    plt.close(fig)
    return True


def _field_total_specs(
    ux: np.ndarray | None,
    uy: np.ndarray | None,
    electric_x: np.ndarray | None,
    electric_y: np.ndarray | None,
    viscosity: np.ndarray | None,
) -> list[ScalarMapSpec]:
    specs: list[ScalarMapSpec] = []
    if ux is not None and uy is not None:
        specs.append(
            (
                "Flow speed |u|",
                "|u| [m/s]",
                np.sqrt(ux * ux + uy * uy),
                "viridis",
                False,
            )
        )
    if electric_x is not None and electric_y is not None:
        specs.append(
            (
                "Electric field |E|",
                "|E| [V/m]",
                np.sqrt(electric_x * electric_x + electric_y * electric_y),
                "plasma",
                False,
            )
        )
    if viscosity is not None:
        specs.append(("Dynamic viscosity mu", "mu [Pa s]", viscosity, "cividis", False))
    return specs


def _present_scalar_specs(
    specs: list[OptionalScalarMapSpec],
) -> list[ScalarMapSpec]:
    return [
        (title, label, array, cmap, symmetric)
        for title, label, array, cmap, symmetric in specs
        if array is not None
    ]


def _field_scalar_specs(
    field: dict[str, np.ndarray], mask: np.ndarray | None
) -> list[ScalarMapSpec]:
    descriptions = (
        ("T", "T [K]", "inferno", False),
        ("p", "p", "coolwarm", True),
        ("rho_g", "rho_g [kg/m^3]", "viridis", False),
        ("phi", "phi [V]", "coolwarm", True),
        ("ne", "n_e [1/m^3]", "magma", False),
        ("Te", "T_e [eV]", "plasma", False),
    )
    specs: list[ScalarMapSpec] = []
    for name, label, cmap, symmetric in descriptions:
        array = _optional_masked_field(field, mask, name)
        if array is not None:
            specs.append((name, label, array, cmap, symmetric))
    return specs


def save_field_maps(
    out_dir: Path,
    field: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    if not {"axis_0", "axis_1", "valid_mask"}.issubset(field):
        return []
    x = np.asarray(field["axis_0"], dtype=np.float64)
    y = np.asarray(field["axis_1"], dtype=np.float64)
    mask = np.asarray(field["valid_mask"], dtype=bool)
    ux = _optional_masked_field(field, mask, "ux")
    uy = _optional_masked_field(field, mask, "uy")
    electric_x = _optional_masked_field(field, mask, "E_x")
    electric_y = _optional_masked_field(field, mask, "E_y")
    viscosity = _optional_masked_field(field, mask, "mu")
    plot_groups = (
        (
            "15_mechanics_field_totals.png",
            _field_total_specs(ux, uy, electric_x, electric_y, viscosity),
            2,
            6.6,
            5.4,
        ),
        (
            "16_flow_components_ux_uy.png",
            _present_scalar_specs(
                [
                    ("ux", "u_x [m/s]", ux, "coolwarm", True),
                    ("uy", "u_y [m/s]", uy, "coolwarm", True),
                ]
            ),
            2,
            6.7,
            5.4,
        ),
        (
            "18_electric_field_components_ex_ey.png",
            _present_scalar_specs(
                [
                    ("E_x", "E_x [V/m]", electric_x, "coolwarm", True),
                    ("E_y", "E_y [V/m]", electric_y, "coolwarm", True),
                ]
            ),
            2,
            6.7,
            5.4,
        ),
        (
            "19_scalar_physics_fields.png",
            _field_scalar_specs(field, mask),
            3,
            6.2,
            5.1,
        ),
    )
    saved: list[str] = []
    for filename, specs, max_columns, width, height in plot_groups:
        if _save_scalar_map_grid(
            out_dir=out_dir,
            filename=filename,
            specs=specs,
            x=x,
            y=y,
            geometry=geometry,
            edges=edges,
            edge_part_ids=edge_part_ids,
            medium_summary=medium_summary,
            max_columns=max_columns,
            width_per_column=width,
            height_per_row=height,
        ):
            saved.append(filename)
    return saved


def _drag_gas_report(report: object) -> dict[str, object]:
    if not isinstance(report, dict):
        return {}
    gas_report = report.get("drag_gas_properties", {})
    return gas_report if isinstance(gas_report, dict) else {}


def _drag_source_specs(gas_report: dict[str, object]) -> list[DragSourceSpec]:
    return [
        (
            "rho_g",
            "density",
            "rho_g [kg/m^3]",
            gas_report.get("density_source", "unknown"),
            gas_report.get("fallback_density_kgm3", np.nan),
            gas_report.get("density_used_by_drag_model", 0),
        ),
        (
            "T",
            "temperature",
            "T [K]",
            gas_report.get("temperature_source", "unknown"),
            gas_report.get("fallback_temperature_K", np.nan),
            gas_report.get("temperature_used_by_drag_model", 0),
        ),
        (
            "mu",
            "dynamic_viscosity",
            "mu [Pa s]",
            gas_report.get("dynamic_viscosity_source", "unknown"),
            gas_report.get("fallback_dynamic_viscosity_Pas", np.nan),
            gas_report.get("dynamic_viscosity_used_by_drag_model", 0),
        ),
        (
            "p",
            "pressure_diagnostic",
            "p",
            "diagnostic_only_not_used_by_drag",
            np.nan,
            0,
        ),
    ]


def _drag_summary_rows(
    field: dict[str, np.ndarray],
    mask: np.ndarray | None,
    source_specs: list[DragSourceSpec],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field_name, role, _label, source, fallback, used_by_drag in source_specs:
        row: dict[str, object] = {
            "role": role,
            "field_quantity": field_name,
            "source": str(source),
            "fallback_value": fallback,
            "used_by_drag": int(used_by_drag),
        }
        array = _optional_masked_field(field, mask, field_name)
        if array is not None:
            values = array[np.isfinite(array)]
            if values.size:
                row.update(
                    {
                        "field_min": float(np.nanmin(values)),
                        "field_p50": float(np.nanpercentile(values, 50.0)),
                        "field_p90": float(np.nanpercentile(values, 90.0)),
                        "field_max": float(np.nanmax(values)),
                        "field_mean": float(np.nanmean(values)),
                    }
                )
        rows.append(row)
    return rows


def _drag_property_style(field_name: str, role: str) -> tuple[str, bool]:
    symmetric = role == "pressure_diagnostic"
    if field_name == "T":
        return "inferno", symmetric
    if field_name == "mu":
        return "cividis", symmetric
    return ("coolwarm" if symmetric else "viridis"), symmetric


def _drag_map_specs(
    field: dict[str, np.ndarray],
    mask: np.ndarray | None,
    source_specs: list[DragSourceSpec],
) -> list[ScalarMapSpec]:
    specs: list[ScalarMapSpec] = []
    for field_name, role, label, source, _fallback, _used_by_drag in source_specs:
        array = _optional_masked_field(field, mask, field_name)
        if array is None:
            continue
        cmap, symmetric = _drag_property_style(field_name, role)
        specs.append((f"{field_name}: {source}", label, array, cmap, symmetric))
    return specs


def save_drag_gas_property_maps(
    out_dir: Path,
    field: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    report: dict[str, object],
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    source_specs = _drag_source_specs(_drag_gas_report(report))
    mask = as_2d_mask(field["valid_mask"]) if "valid_mask" in field else None
    sources_filename = "27_drag_gas_property_sources.csv"
    pd.DataFrame(_drag_summary_rows(field, mask, source_specs)).to_csv(
        out_dir / sources_filename, index=False
    )
    saved = [sources_filename]
    if not {"axis_0", "axis_1", "valid_mask"}.issubset(field):
        return saved
    map_filename = "27_drag_gas_properties_used_by_drag.png"
    if _save_scalar_map_grid(
        out_dir=out_dir,
        filename=map_filename,
        specs=_drag_map_specs(field, mask, source_specs),
        x=np.asarray(field["axis_0"], dtype=np.float64),
        y=np.asarray(field["axis_1"], dtype=np.float64),
        geometry=geometry,
        edges=edges,
        edge_part_ids=edge_part_ids,
        medium_summary=medium_summary,
        max_columns=2,
        width_per_column=6.6,
        height_per_row=5.2,
        figure_title="Gas properties for drag; pressure is diagnostic only",
    ):
        saved.append(map_filename)
    return saved
