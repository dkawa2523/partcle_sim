from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.force_models import (
    DielectrophoresisForce,
    ForceModelError,
    force_model_to_manifest_inventory,
    force_model_to_native_mapping,
    parse_manifest_force_model,
    parse_native_force_model,
)
from particle_tracer_unified.solvers.force_runtime import (
    ForceBatchState,
    ForceBatchStatic,
    ForcePipeline,
    evaluate_force_pipeline,
)
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    compile_force_runtime_parameters,
    force_runtime_parameters_summary,
)

_AC_PARAMETERS: dict[str, object] = {
    "medium_rel_permittivity": 1.2,
    "particle_rel_permittivity": 3.9,
    "medium_conductivity_Sm": 0.03,
    "particle_conductivity_Sm": 0.2,
    "frequency_Hz": 13.56e6,
}


def _native_ac_model(*, amplitude: str | None = None):
    parameters = dict(_AC_PARAMETERS)
    if amplitude is not None:
        parameters["electric_field_amplitude"] = amplitude
    return parse_native_force_model(
        {"model": "stokes"},
        {
            "dielectrophoresis": {
                "enabled": True,
                "model": "ac_clausius_mossotti",
                "parameters": parameters,
            }
        },
        spatial_dim=2,
    )


def _evaluate_ac_dep(amplitude: str) -> np.ndarray:
    output = np.zeros((1, 2), dtype=np.float64)
    static = ForceBatchStatic(
        particle_diameter=np.asarray([2.0e-6]),
        particle_density=np.asarray([1000.0]),
        particle_mass=np.asarray([3.0e-15]),
        dep_particle_rel_permittivity=np.asarray([3.9]),
        thermophoretic_coeff=np.asarray([np.nan]),
    )
    state = ForceBatchState(velocity=np.zeros((1, 2), dtype=np.float64))
    fields = StageFields(
        points_m=np.zeros((1, 2), dtype=np.float64),
        time_s=0.0,
        values={"electric_magnitude_squared_gradient": np.asarray([[3.0, -4.0]])},
        supported=np.ones(1, dtype=bool),
    )
    params = ForceRuntimeParameters(
        dielectrophoresis_enabled=True,
        dielectrophoresis_model="ac_clausius_mossotti",
        dep_medium_rel_permittivity=1.2,
        dep_medium_conductivity_Sm=0.03,
        dep_particle_conductivity_Sm=0.2,
        dep_frequency_Hz=13.56e6,
        dep_electric_field_amplitude=amplitude,
    )
    return evaluate_force_pipeline(
        output,
        static,
        state,
        fields,
        ForcePipeline(evaluator_names=("dielectrophoresis",), params=params),
    )


def test_legacy_ac_config_defaults_to_explicit_rms_in_every_projection() -> None:
    model = _native_ac_model()
    dep = model.dielectrophoresis

    assert dep.electric_field_amplitude == "rms"
    native_drag, native_forces = force_model_to_native_mapping(model)
    assert native_drag == {"model": "stokes"}
    assert (
        native_forces["dielectrophoresis"]["parameters"]["electric_field_amplitude"]
        == "rms"
    )
    inventory = force_model_to_manifest_inventory(model)
    manifest_dep = next(
        item for item in inventory if item["solver_force"] == "dielectrophoresis"
    )
    assert manifest_dep["parameters"]["electric_field_amplitude"] == "rms"
    assert parse_manifest_force_model(list(inventory), spatial_dim=2) == model

    runtime = compile_force_runtime_parameters(model)
    assert runtime.dep_electric_field_amplitude == "rms"
    summary = force_runtime_parameters_summary(runtime)
    assert summary["dep_electric_field_amplitude"] == "rms"
    equations = summary["implemented_equations"]
    assert isinstance(equations, dict)
    assert str(equations["dielectrophoresis"]).endswith("_rms_electric_field")


def test_peak_amplitude_round_trips_through_native_and_manifest_parsers() -> None:
    model = _native_ac_model(amplitude="peak")
    inventory = force_model_to_manifest_inventory(model)

    assert model.dielectrophoresis.electric_field_amplitude == "peak"
    assert parse_manifest_force_model(list(inventory), spatial_dim=2) == model
    assert compile_force_runtime_parameters(model).dep_electric_field_amplitude == (
        "peak"
    )


@pytest.mark.parametrize(
    ("amplitude", "message"),
    [
        ("instantaneous", "must be one of ['peak', 'rms']"),
        (" rms", "must not contain leading or trailing whitespace"),
    ],
)
def test_amplitude_errors_keep_the_canonical_parameter_path(
    amplitude: str,
    message: str,
) -> None:
    with pytest.raises(ForceModelError) as caught:
        _native_ac_model(amplitude=amplitude)

    assert str(caught.value) == (
        "physics.forces.dielectrophoresis.parameters.electric_field_amplitude: "
        + message
        + (", got 'instantaneous'" if amplitude == "instantaneous" else "")
    )


def test_existing_required_parameter_error_precedes_amplitude_validation() -> None:
    with pytest.raises(ForceModelError) as caught:
        parse_native_force_model(
            {"model": "stokes"},
            {
                "dielectrophoresis": {
                    "enabled": True,
                    "model": "ac_clausius_mossotti",
                    "parameters": {
                        "medium_rel_permittivity": 1.2,
                        "electric_field_amplitude": "invalid",
                    },
                }
            },
            spatial_dim=2,
        )

    assert "AC dielectrophoresis requires explicit frequency_Hz" in str(caught.value)


def test_rms_is_bit_exact_and_peak_is_the_consistent_half_force() -> None:
    rms = _evaluate_ac_dep("rms")
    peak = _evaluate_ac_dep("peak")

    assert [float(value).hex() for value in rms[0]] == [
        "0x1.891ebc4a5fd8fp-45",
        "-0x1.06147d86ea90ap-44",
    ]
    np.testing.assert_array_equal(peak, 0.5 * rms)


def test_invalid_runtime_amplitude_fails_before_force_evaluation() -> None:
    with pytest.raises(
        ValueError,
        match="dielectrophoresis electric_field_amplitude must be 'rms' or 'peak'",
    ):
        _evaluate_ac_dep("unknown")


def test_new_dataclass_parameters_are_appended_with_rms_defaults() -> None:
    semantic = signature(DielectrophoresisForce).parameters
    runtime = signature(ForceRuntimeParameters).parameters

    assert tuple(semantic)[-1] == "electric_field_amplitude"
    assert semantic["electric_field_amplitude"].default == "rms"
    assert tuple(runtime)[-1] == "dep_electric_field_amplitude"
    assert runtime["dep_electric_field_amplitude"].default == "rms"
