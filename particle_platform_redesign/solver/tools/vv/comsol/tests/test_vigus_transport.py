from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from tests.verification.microcases import materialize_microcase
from tools.vv.comsol.prepare_vigus_transport import prepare_transport

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import FieldData, RegularLayout, write
from chamber_particles.physics.charge import ELECTRON_MASS_KG
from chamber_particles.physics.forces import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
)
from chamber_particles_tools.inputs import create_case


def _input(tmp_path: Path) -> tuple[Path, Path]:
    path = materialize_microcase("C07", tmp_path / "initial").case_path
    case = load_case(path)
    scalars = {
        "gas_density": (1e-4, "kg/m^3"),
        "gas_temperature": (400, "K"),
        "gas_viscosity": (2e-5, "Pa*s"),
        "gas_pressure": (2, "Pa"),
        "electron_density": (1e13, "1/m^3"),
        "ion_density": (1e13, "1/m^3"),
        "electron_temperature_V": (4, "V"),
        "ion_temperature": (400, "K"),
        "ion_mass": (8.3e-26, "kg"),
        "ion_speed": (2000, "m/s"),
        "rotation_rate": (-100, "1/s"),
        "gas_mean_free_path": (1e-3, "m"),
    }
    fields = [
        FieldData(
            name, "raw", "node", ("value",), "scalar", np.full((4, 1), value, dtype="<f8"), unit
        )
        for name, (value, unit) in scalars.items()
    ]
    fields.extend(
        FieldData(
            name,
            "raw",
            "node",
            ("x", "y"),
            "cartesian_xy",
            np.tile(value, (4, 1)).astype("<f8"),
            unit,
        )
        for name, value, unit in (
            ("gas_velocity", [1, -2], "m/s"),
            ("electric_field", [300, 400], "V/m"),
        )
    )
    layout = RegularLayout(
        "raw",
        np.array([0, 1], dtype="<f8"),
        np.array([0, 1], dtype="<f8"),
        np.ones((1, 1), dtype="<u1"),
    )
    info = write(
        path.parent / "raw.h5", replace(case.data, layouts=(layout,), fields=tuple(fields))
    )
    document = yaml.safe_load(path.read_text())
    document["case"].update(data_path="raw.h5", expected_content_hash=info.content_hash)
    document["time"] = {"start_s": 0, "end_s": 1e-5, "dt_s": 1e-6}
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0, 1e-5]},
    }
    document["physics"].update(
        {
            "noise": {
                "model": "inertial_langevin_fdt",
                "revision": "inertial_langevin_fdt_epstein_linear_midpoint_2d_v2",
                "interval_tree_depth": 1,
                "adaptive_max_depth": 2,
            },
            "drag": {
                "model": "epstein_linear",
                "revision": "epstein_linear_v1",
                "gas_velocity_field": "gas_velocity",
                "gas_density_field": "gas_density",
                "gas_temperature_field": "gas_temperature",
                "gas_mean_free_path_field": "gas_mean_free_path",
                "gas_molecular_mass_kg": 8.3e-26,
                "delta": 1,
                "applicability": "error",
            },
        }
    )
    document["solver"]["integrator"] = "ou_langevin"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    params = {
        "format_version": 2,
        "cmd_slip_coefficients": [1.165, 0.483, 0.997],
        "native_particle_density_kg_m3": 2200,
        "native_capacitance_F": 4 * math.pi * VACUUM_PERMITTIVITY_F_M * 1e-6,
        "maximum_abs_charge_number": 17,
        "native_binding": {"scope": "independent constant primitive verification"},
        "brownian": {"interpretation": "continuous_white_force"},
    }
    parameter_path = tmp_path / "parameters.yaml"
    parameter_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path, parameter_path


def test_transport_producer_reconstructs_independent_source_equations(tmp_path: Path) -> None:
    initial, parameters = _input(tmp_path)
    output = tmp_path / "prepared"
    prepare_transport(initial, parameters, output, interval_tree_depth=1, adaptive_max_depth=2)
    case = load_case(output / "case.yaml")
    f = {field.name: field.values[0] for field in case.data.fields}
    source = case.data.sources[0]
    m, d, a = (
        float(source.mass_kg[0]),
        float(source.drag_diameter_m[0]),
        float(source.electrostatic_radius_m[0]),
    )
    k, e, eps = BOLTZMANN_J_K, ELEMENTARY_CHARGE_C, VACUUM_PERMITTIVITY_F_M
    capacitance = 4 * math.pi * eps * a
    kn = 2e-5 * math.sqrt(math.pi / (2 * 1e-4 * 2)) / d
    beta = m * 18 * 2e-5 / (2200 * d * d * (1 + kn * (1.165 + 0.483 * math.exp(-0.997 / kn))))
    assert f["reference_friction"][0] == pytest.approx(beta, rel=2e-15)
    assert f["reference_force_diffusion"][0] == pytest.approx(6 * math.pi * 2e-5 * d * k * 400)
    ae = math.pi * a * a * 1e13 * math.sqrt(8 * e * 4 / (math.pi * ELECTRON_MASS_KG))
    alpha, bz = e / (capacitance * 4), e * e / (capacitance * k * 400)
    ai = math.pi * a * a * 1e13 * math.sqrt(8 * k * 400 / (math.pi * 8.3e-26))
    plan = yaml.safe_load((output / "case.yaml").read_text())["physics"]["charge"]
    reconstructed = plan["electron_rate_reference_number_s"] * math.exp(
        f["reference_electron_offset"][0]
    )
    assert reconstructed == pytest.approx(ae, rel=1e-14)
    veff = math.sqrt(2000**2 + 8 * k * 400 / (math.pi * 8.3e-26))
    for z in (-30, 0, 3):
        ion_area = math.pi * a * a * (1 - bz * z)
        orbital = (
            math.pi
            * (e * e * z / (2 * math.pi * eps * 8.3e-26 * veff * veff)) ** 2
            * math.log(math.sqrt(eps * 4 / (1e13 * e)) / a)
        )
        native_force = -8.3e-26 * 1e13 * veff * 2000 * (ion_area + orbital) * np.array([0.6, 0.8])
        np.testing.assert_allclose(
            f["reference_ion_f0"] + z * f["reference_ion_f1"] + z * z * f["reference_ion_f2"],
            native_force,
            rtol=3e-15,
        )
        rate = -reconstructed * math.exp(alpha * z) + ai * (1 - bz * z)
        generated = -reconstructed * math.exp(f["reference_alpha"][0] * z) + f[
            "reference_ion_rate"
        ][0] * (1 - f["reference_beta_z"][0] * z)
        assert generated == pytest.approx(rate, rel=2e-14)
    assert f["reference_rotation"][0] == pytest.approx(
        -6.46 * (d / 2) ** 2 * math.sqrt(2e-5 * 1e-4 * 100)
    )
    np.testing.assert_array_equal(
        case.data.sources[0].particle_id, load_case(initial).data.sources[0].particle_id
    )


def test_transport_producer_runs_both_original_and_reference_public_cases(tmp_path: Path) -> None:
    initial, parameters = _input(tmp_path)
    simulate(load_case(initial), tmp_path / "original_result")
    prepared = tmp_path / "prepared"
    receipt = prepare_transport(
        initial, parameters, prepared, interval_tree_depth=1, adaptive_max_depth=2
    )
    simulate(load_case(prepared / "case.yaml"), tmp_path / "reference_result")
    result = open_result(tmp_path / "reference_result")
    failures = result.manifest["failure_reason_counts"]
    assert isinstance(failures, Mapping)
    assert not any(failures.values())
    assert list(result.iter_frames())[-1].time_s == 1e-5
    assert (
        json.loads((prepared / "build_manifest.json").read_text())["content_hash"]
        == receipt["content_hash"]
    )


def test_transport_producer_rejects_undefined_ion_direction(tmp_path: Path) -> None:
    initial, parameters = _input(tmp_path)
    case = load_case(initial)
    fields = tuple(
        replace(f, values=np.zeros_like(f.values)) if f.name == "electric_field" else f
        for f in case.data.fields
    )
    info = write(initial.parent / "zero-e.h5", replace(case.data, fields=fields))
    document = yaml.safe_load(initial.read_text())
    document["case"].update(data_path="zero-e.h5", expected_content_hash=info.content_hash)
    initial.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="undefined at zero electric field"):
        prepare_transport(
            initial, parameters, tmp_path / "must-not-be-written", interval_tree_depth=1
        )
    assert not (tmp_path / "must-not-be-written").exists()


def test_rk4_white_limit_uses_shared_time_covariance_and_preserves_original_case(
    tmp_path: Path,
) -> None:
    initial, parameters = _input(tmp_path)
    original = tmp_path / "continuous"
    prepare_transport(initial, parameters, original, interval_tree_depth=1)
    doc = yaml.safe_load(parameters.read_text())
    doc["brownian"] = {
        "interpretation": "fixed_rk4_white_limit",
        "step_s": 1e-6,
        "force_normalization_step_s": 1e-6,
        "evidence": "independent RK4 quadrature with identical time-keyed noise at repeated times",
    }
    parameters.write_text(yaml.safe_dump(doc), encoding="utf-8")
    mapped = tmp_path / "mapped"
    receipt = prepare_transport(initial, parameters, mapped, interval_tree_depth=1)
    raw = load_case(original / "case.yaml")
    changed = load_case(mapped / "case.yaml")
    # Assemble independent RK4 velocity increments from time-keyed normals.
    # Endpoint values are shared between adjacent intervals, not redrawn.
    n = 1000
    weights = np.zeros(2 * n + 1)
    for j in range(n):
        weights[2 * j : 2 * j + 3] += np.array([1, 4, 1]) / 6
    finite_variance_rate = float(weights @ weights / n)
    asymptotic_rate = finite_variance_rate + 1 / (18 * n)
    old_fields = {f.name: f for f in raw.data.fields}
    for field in changed.data.fields:
        expected = old_fields[field.name].values
        if field.name == "reference_force_diffusion":
            expected = expected * asymptotic_rate
        np.testing.assert_allclose(field.values, expected, rtol=3e-15, atol=0)
    assert receipt["brownian_mapping"]["spectral_density_multiplier"] == pytest.approx(
        asymptotic_rate
    )
    assert load_case(original / "case.yaml").content_hash == raw.content_hash


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"interpretation": "automatic"},
        {
            "interpretation": "fixed_rk4_white_limit",
            "step_s": 5e-7,
            "force_normalization_step_s": 1e-6,
            "evidence": "different normalization",
        },
    ],
)
def test_brownian_interpretation_is_explicit_and_rejects_unmatched_steps(
    tmp_path: Path, invalid: object
) -> None:
    initial, parameters = _input(tmp_path)
    doc = yaml.safe_load(parameters.read_text())
    doc["brownian"] = invalid
    parameters.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=r"brownian|fixed_rk4_white_limit"):
        prepare_transport(initial, parameters, tmp_path / "rejected", interval_tree_depth=1)
    assert not (tmp_path / "rejected").exists()


def test_transport_manifest_failure_leaves_no_case_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, parameters = _input(tmp_path)
    original = initial.read_bytes()
    output = tmp_path / "prepared"
    open_file = Path.open

    def fail_manifest(path: Path, *args, **kwargs):
        if path.name == "build_manifest.json" and args and args[0] == "xb":
            raise OSError("injected manifest write failure")
        return open_file(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_manifest)
        with pytest.raises(OSError, match="injected manifest"):
            prepare_transport(initial, parameters, output, interval_tree_depth=1)
    assert not output.exists()
    assert initial.read_bytes() == original
    receipt = prepare_transport(initial, parameters, output, interval_tree_depth=1)
    case = load_case(output / "case.yaml")
    assert case.content_hash == receipt["content_hash"]
    assert json.loads((output / "build_manifest.json").read_text()) == receipt


@pytest.mark.parametrize(
    "change",
    ["primitive", "layout", "property", "partial", "output", "parameter", "clock", "unit", "basis"],
)
def test_declared_transport_group_cannot_be_republished_stale(tmp_path: Path, change: str) -> None:
    initial, parameters = _input(tmp_path)
    prepared = tmp_path / "prepared"
    receipt = prepare_transport(initial, parameters, prepared, interval_tree_depth=1)
    case = load_case(prepared / "case.yaml")
    document = yaml.safe_load((prepared / "case.yaml").read_text())
    data = case.data
    if change in {"primitive", "output", "clock", "unit", "basis"}:
        target = {"output": "reference_friction", "basis": "gas_velocity"}.get(
            change, "electron_density"
        )
        fields = []
        for item in data.fields:
            if item.name == target:
                if change == "clock":
                    item = replace(
                        item,
                        time_s=np.asarray([0.0, 1e-5]),
                        values=np.stack((item.values, item.values)),
                    )
                elif change == "unit":
                    item = replace(item, unit="1")
                elif change == "basis":
                    item = replace(item, stored_basis="axisymmetric_rz")
                else:
                    item = replace(item, values=item.values * 1.01)
            fields.append(item)
        data = replace(data, fields=tuple(fields))
    elif change == "layout":
        layout = data.layouts[0]
        assert isinstance(layout, RegularLayout)
        data = replace(data, layouts=(replace(layout, axis0_m=layout.axis0_m * 1.01),))
    elif change == "property":
        source = data.sources[0]
        data = replace(data, sources=(replace(source, mass_kg=source.mass_kg * 1.01),))
    elif change == "partial":
        data = replace(data, fields=tuple(f for f in data.fields if f.name != "reference_alpha"))
    else:
        provenance = json.loads(data.provenance_json)
        provenance["derived_field_generations"][0]["parameters"]["native_capacitance_F"] *= 2
        data = replace(data, provenance_json=json.dumps(provenance))
    output = tmp_path / "stale"
    with pytest.raises(ValueError, match=r"stale|incomplete"):
        create_case(data, document, output)
    assert not output.exists()
    assert load_case(prepared / "case.yaml").content_hash == receipt["content_hash"]


def test_declared_groups_keep_independent_prescribed_fields_and_property_authority(
    tmp_path: Path,
) -> None:
    initial, parameters = _input(tmp_path)
    prepared = tmp_path / "prepared"
    prepare_transport(initial, parameters, prepared, interval_tree_depth=1)
    case = load_case(prepared / "case.yaml")
    independent = FieldData(
        "independent_coefficient", "raw", "node", ("value",), "scalar", np.ones((4, 1)), "N"
    )
    data = replace(case.data, fields=(*case.data.fields, independent))
    document = yaml.safe_load((prepared / "case.yaml").read_text())
    # Changing IDs/weights without changing the actual homogeneous authority is
    # legitimate; coefficients depend on mass/dimensions, not sample identity.
    source = data.sources[0]
    data = replace(
        data,
        sources=(
            replace(
                source, particle_id=source.particle_id + 10, model_weight=source.model_weight * 2
            ),
        ),
    )
    published = create_case(data, document, tmp_path / "reused")
    assert "independent_coefficient" in {item.name for item in published.data.fields}
    assert json.loads(published.data.provenance_json)["derived_field_generations"]
    # Independent prescribed data has no implied producer dependencies.
    provenance = json.loads(data.provenance_json)
    del provenance["derived_field_generations"]
    independent_data = replace(
        data,
        provenance_json=json.dumps(provenance),
        fields=tuple(
            replace(item, values=item.values * 2) if item.name == "electron_density" else item
            for item in data.fields
        ),
    )
    create_case(independent_data, document, tmp_path / "independent")
