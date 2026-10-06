from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import BoundaryData, FieldData, RegularLayout, write
from chamber_particles.physics.forces import BOLTZMANN_J_K
from tests.verification.microcases import materialize_microcase

_ARGON_MASS_KG = 6.6335209e-26
_REVISION = "waldmann_gallis_free_molecular_single_species_heat_flux_v1"
_EFFECTIVE_GAS_REVISION = "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"
_PARTICLE_MASS_KG = 4.0e-17
_PARTICLE_DIAMETER_M = 2.0e-6
_GAS_TEMPERATURE_K = 300.0
_HEAT_FLUX_OFFSET_W_M2 = 1.0e-3
_HEAT_FLUX_SLOPE_W_M3 = 5.0e-4


@pytest.mark.parametrize(
    ("integrator", "minimum_order"),
    [("rk4_fixed", 3.5), ("exponential_midpoint", 1.8)],
)
def test_waldmann_affine_heat_flux_time_convergence(
    tmp_path: Path,
    integrator: str,
    minimum_order: float,
) -> None:
    end_time = 1.0
    reference = _analytic_state(end_time)
    position_error: list[float] = []
    velocity_error: list[float] = []
    last_result = None
    for index, dt_s in enumerate((0.125, 0.0625, 0.03125)):
        case_path = _case(
            tmp_path / f"waldmann-{integrator}-{index}",
            integrator=integrator,
            dt_s=dt_s,
            end_time_s=end_time,
        )
        output = tmp_path / f"waldmann-result-{integrator}-{index}"
        simulate(load_case(case_path), output)
        last_result = open_result(output)
        final = last_result.read_final()
        position_error.append(float(np.linalg.norm(final.position_m[0] - reference[:2])))
        velocity_error.append(float(np.linalg.norm(final.velocity_m_s[0] - reference[2:])))

    assert last_result is not None
    for errors in (position_error, velocity_error):
        orders = [math.log2(errors[index] / errors[index + 1]) for index in (0, 1)]
        assert min(orders) >= minimum_order
    assert last_result.manifest["resolved"]["physics_models"]["thermophoresis"] == {
        "model": "waldmann_gallis",
        "revision": _REVISION,
    }
    assert last_result.manifest["compiled_cpu_tile_revision"] == "compiled_cpu_tile_v18"
    assert last_result.manifest["physics_catalog_revision"] == "inertial_langevin_rz_catalog_v17"
    assert (
        last_result.manifest["physics_runtime_revision"]
        == "signed_ion_compiled_physics_runtime_v20"
    )


def test_waldmann_xy_and_rz_use_the_same_stage_physics(tmp_path: Path) -> None:
    outputs = []
    for axisymmetric in (False, True):
        case_path = _case(
            tmp_path / f"waldmann-basis-{axisymmetric}",
            integrator="rk4_fixed",
            dt_s=0.03125,
            end_time_s=1.0,
            axisymmetric_rz=axisymmetric,
        )
        output = tmp_path / f"waldmann-basis-result-{axisymmetric}"
        simulate(load_case(case_path), output)
        outputs.append(open_result(output).read_final())

    shift = np.asarray([1.5, 0.0])
    np.testing.assert_allclose(
        outputs[0].position_m,
        outputs[1].position_m - shift,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        outputs[0].velocity_m_s,
        outputs[1].velocity_m_s,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_effective_gas_heat_flux_composes_with_finite_speed_drag_public_run(
    tmp_path: Path,
) -> None:
    case_path = _case(
        tmp_path / "effective-gas-finite-speed",
        integrator="exponential_midpoint",
        dt_s=0.03125,
        end_time_s=0.5,
        thermophoresis_revision=_EFFECTIVE_GAS_REVISION,
        thermophoresis_maximum_speed_ratio=0.5,
        finite_speed_drag=True,
    )
    output = tmp_path / "effective-gas-finite-speed-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()

    assert bool(np.isfinite(final.position_m).all())
    assert bool(np.isfinite(final.velocity_m_s).all())
    assert result.manifest["resolved"]["physics_models"]["drag"] == {
        "model": "epstein_finite_speed",
        "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
    }
    assert result.manifest["resolved"]["physics_models"]["thermophoresis"] == {
        "model": "waldmann_gallis",
        "revision": _EFFECTIVE_GAS_REVISION,
    }


def _case(
    directory: Path,
    *,
    integrator: str,
    dt_s: float,
    end_time_s: float,
    axisymmetric_rz: bool = False,
    thermophoresis_revision: str = _REVISION,
    thermophoresis_maximum_speed_ratio: float | None = None,
    finite_speed_drag: bool = False,
) -> Path:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    radial_shift = np.asarray([1.5, 0.0]) if axisymmetric_rz else np.zeros(2)
    source = replace(
        case.data.sources[0],
        position_m=np.asarray([[0.25, 0.25]]) + radial_shift,
        velocity_m_s=np.asarray([[0.1, 0.0]]),
        charge_number=np.asarray([0.0]),
        mass_kg=np.asarray([_PARTICLE_MASS_KG]),
        drag_diameter_m=np.asarray([_PARTICLE_DIAMETER_M]),
        electrostatic_radius_m=np.asarray([0.0]),
        displaced_volume_m3=np.asarray([0.0]),
    )
    vector_components = ("r", "z") if axisymmetric_rz else ("x", "y")
    vector_basis = "axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"
    layout = RegularLayout(
        "gas",
        np.asarray([radial_shift[0], radial_shift[0] + 1.0]),
        np.asarray([0.0, 1.0]),
        np.ones((1, 1), dtype=np.uint8),
    )

    def constant_field(
        name: str,
        value: tuple[float, ...],
        components: tuple[str, ...],
        basis: str,
        unit: str,
    ) -> FieldData:
        values = np.repeat(np.asarray([value]), 4, axis=0)
        return FieldData(name, "gas", "node", components, basis, values, unit)

    heat_flux_left = _HEAT_FLUX_OFFSET_W_M2
    heat_flux_right = heat_flux_left + _HEAT_FLUX_SLOPE_W_M3
    heat_flux_values = np.asarray(
        [
            [heat_flux_left, 0.0],
            [heat_flux_left, 0.0],
            [heat_flux_right, 0.0],
            [heat_flux_right, 0.0],
        ]
    )
    fields = [
        constant_field("ug", (0.0, 0.0), vector_components, vector_basis, "m/s"),
        constant_field("tg", (_GAS_TEMPERATURE_K,), ("value",), "scalar", "K"),
        FieldData(
            "qtr",
            "gas",
            "node",
            vector_components,
            vector_basis,
            heat_flux_values,
            "W/m^2",
        ),
        constant_field("mfp", (1.0e-3,), ("value",), "scalar", "m"),
    ]
    if finite_speed_drag:
        fields.append(constant_field("rho", (1.0e-8,), ("value",), "scalar", "kg/m^3"))
    boundaryless = BoundaryData(
        line2=np.empty((0, 2), dtype=np.int64),
        boundary_id=np.empty(0, dtype=np.int32),
        group_id=np.empty(0, dtype=np.int32),
        material_id=np.empty(0, dtype=np.int32),
        owner_cell_type=np.empty(0, dtype=np.uint8),
        owner_cell_local_index=np.empty(0, dtype=np.int64),
        orientation=np.empty(0, dtype=np.int8),
    )
    data_path = paths.case_path.with_name("waldmann-gallis.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system=("axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"),
            geometry=replace(
                case.data.geometry,
                nodes_m=case.data.geometry.nodes_m + radial_shift,
                boundary=boundaryless,
                group_names=(),
            ),
            layouts=(layout,),
            fields=tuple(fields),
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional" if axisymmetric_rz else "cartesian_xy"
    document["time"] = {"start_s": 0.0, "end_s": end_time_s, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    thermophoresis = {
        "model": "waldmann_gallis",
        "revision": thermophoresis_revision,
        "gas_velocity_field": "ug",
        "gas_temperature_field": "tg",
        "gas_translational_heat_flux_field": "qtr",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
        "applicability": "error",
    }
    if thermophoresis_maximum_speed_ratio is not None:
        thermophoresis["maximum_speed_ratio"] = thermophoresis_maximum_speed_ratio
    physics = {
        "charge": {"model": "fixed"},
        "thermophoresis": thermophoresis,
    }
    if finite_speed_drag:
        physics["drag"] = {
            "model": "epstein_finite_speed",
            "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
            "gas_velocity_field": "ug",
            "gas_density_field": "rho",
            "gas_temperature_field": "tg",
            "gas_mean_free_path_field": "mfp",
            "gas_molecular_mass_kg": _ARGON_MASS_KG,
            "diffuse_reflection_fraction": 0.5,
            "maximum_speed_ratio": 1.0,
            "applicability": "error",
        }
    document["physics"] = physics
    document["boundaries"] = []
    document["output"] = {"trajectories": None}
    case_path = paths.case_path.with_name("waldmann-gallis.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _analytic_state(time_s: float) -> np.ndarray:
    radius = 0.5 * _PARTICLE_DIAMETER_M
    mean_speed = math.sqrt(8.0 * BOLTZMANN_J_K * _GAS_TEMPERATURE_K / (math.pi * _ARGON_MASS_KG))
    force_to_acceleration = (32.0 / 15.0) * radius * radius / (_PARTICLE_MASS_KG * mean_speed)
    offset = force_to_acceleration * _HEAT_FLUX_OFFSET_W_M2
    slope = force_to_acceleration * _HEAT_FLUX_SLOPE_W_M3
    rate = math.sqrt(slope)
    equilibrium_shift = offset / slope
    initial_position = 0.25
    initial_velocity = 0.1
    position = (
        (initial_position + equilibrium_shift) * math.cosh(rate * time_s)
        + initial_velocity / rate * math.sinh(rate * time_s)
        - equilibrium_shift
    )
    velocity = (initial_position + equilibrium_shift) * rate * math.sinh(
        rate * time_s
    ) + initial_velocity * math.cosh(rate * time_s)
    return np.asarray([position, 0.25, velocity, 0.0])
