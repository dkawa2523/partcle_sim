from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.solvers.force_runtime import (
    ForceBatchState,
    ForceBatchStatic,
    ForcePipeline,
    evaluate_force_pipeline,
)
from particle_tracer_unified.solvers.force_validation import (
    require_batch_quantity,
    require_force_parameter,
)
from particle_tracer_unified.solvers.forces.runtime import ForceRuntimeParameters

_GAS_MOLECULAR_MASS_KG = 4.65e-26


def _valid_inputs(
    evaluator_names: tuple[str, ...],
    *,
    params: ForceRuntimeParameters | None = None,
) -> tuple[np.ndarray, ForceBatchStatic, ForceBatchState, StageFields, ForcePipeline]:
    count = 2
    out = np.zeros((count, 2), dtype=np.float64)
    static = ForceBatchStatic(
        particle_diameter=np.asarray([1.0e-6, 1.5e-6]),
        particle_density=np.asarray([1800.0, 2400.0]),
        particle_mass=np.asarray([1.2e-15, 3.4e-15]),
        dep_particle_rel_permittivity=np.asarray([3.9, 4.2]),
        thermophoretic_coeff=np.asarray([np.nan, 0.75]),
    )
    state = ForceBatchState(
        velocity=np.asarray([[0.2, -0.1], [0.3, 0.4]]),
        charge_over_mass=np.asarray([2.0, -1.0]),
    )
    fields = StageFields(
        points_m=np.asarray([[0.2, 0.3], [0.7, 0.8]]),
        time_s=0.0,
        values={
            "electric_field": np.asarray([[2.0, -1.0], [1.0, 3.0]]),
            "gas_density": np.asarray([1.2, 1.4]),
            "fluid_acceleration": np.asarray([[0.5, -0.2], [0.1, 0.3]]),
            "flow_time_derivative": np.asarray([[0.2, 0.1], [-0.1, 0.3]]),
            "flow_velocity_gradient": np.asarray(
                [
                    [[0.3, 0.1], [-0.2, 0.4]],
                    [[0.1, -0.3], [0.2, 0.5]],
                ]
            ),
            "dynamic_viscosity": np.asarray([1.8e-5, 2.0e-5]),
            "temperature": np.asarray([300.0, 420.0]),
            "temperature_gradient": np.asarray([[20.0, -5.0], [3.0, 8.0]]),
            "electric_magnitude_squared_gradient": np.asarray(
                [[1.0e5, -2.0e5], [3.0e5, 4.0e5]]
            ),
            "flow_velocity": np.asarray([[0.1, 0.0], [-0.2, 0.1]]),
            "vorticity": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        },
        supported=np.ones(count, dtype=bool),
    )
    runtime = params or ForceRuntimeParameters(
        virtual_mass_coefficient=0.5,
        dep_medium_rel_permittivity=1.0,
        dep_particle_rel_permittivity=3.9,
    )
    plan = ForcePipeline(
        evaluator_names=evaluator_names,
        params=runtime,
        gas_molecular_mass_kg=_GAS_MOLECULAR_MASS_KG,
    )
    return out, static, state, fields, plan


def _with_field(fields: StageFields, name: str, value: np.ndarray) -> StageFields:
    return replace(
        fields, values={**fields.values, name: np.asarray(value, dtype=np.float64)}
    )


def test_unknown_force_validation_rules_are_rejected() -> None:
    with pytest.raises(AssertionError, match="unknown force input rule"):
        require_batch_quantity(
            "density",
            np.ones(1),
            (1,),
            rule="unknown",
            forces={"drag"},
        )
    with pytest.raises(AssertionError, match="unknown force parameter rule"):
        require_force_parameter("drag", "density", 1.0, rule="unknown")


def test_all_enabled_force_inputs_are_validated_once_then_evaluated() -> None:
    inputs = _valid_inputs(
        (
            "electric",
            "pressure_gradient",
            "virtual_mass",
            "thermophoresis",
            "dielectrophoresis",
            "lift",
        )
    )

    result = evaluate_force_pipeline(*inputs)

    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))


def test_force_validation_preserves_structural_and_force_error_order() -> None:
    out, static, state, fields, plan = _valid_inputs(("electric", "pressure_gradient"))
    missing_charge = replace(state, charge_over_mass=None)

    with pytest.raises(ValueError, match=r"acceleration.*shape"):
        evaluate_force_pipeline(
            np.zeros(2, dtype=np.float64),
            static,
            missing_charge,
            fields,
            plan,
        )
    with pytest.raises(ValueError, match=r"sample points must match output shape"):
        evaluate_force_pipeline(
            np.zeros((1, 2), dtype=np.float64),
            static,
            missing_charge,
            fields,
            plan,
        )
    with pytest.raises(ValueError, match=r"electric requires.*charge_over_mass"):
        evaluate_force_pipeline(out, static, missing_charge, fields, plan)

    missing_electric = replace(
        fields,
        values={
            name: value
            for name, value in fields.values.items()
            if name != "electric_field"
        },
    )
    with pytest.raises(ValueError, match=r"electric requires field.*electric_field"):
        evaluate_force_pipeline(out, static, state, missing_electric, plan)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_pressure_gradient_rejects_invalid_gas_density_with_particle_row(
    invalid: float,
) -> None:
    out, static, state, fields, plan = _valid_inputs(("pressure_gradient",))
    fields = _with_field(fields, "gas_density", np.asarray([1.2, invalid]))

    with pytest.raises(
        ValueError,
        match=r"pressure_gradient.*gas_density.*invalid particle rows: \[1\]",
    ):
        evaluate_force_pipeline(out, static, state, fields, plan)


def test_pressure_gradient_rejects_nonfinite_vector_instead_of_zeroing_row() -> None:
    out, static, state, fields, plan = _valid_inputs(("pressure_gradient",))
    fields = _with_field(
        fields,
        "fluid_acceleration",
        np.asarray([[0.5, -0.2], [np.inf, 0.3]]),
    )

    with pytest.raises(
        ValueError,
        match=r"pressure_gradient.*fluid_acceleration.*invalid particle rows: \[1\]",
    ):
        evaluate_force_pipeline(out, static, state, fields, plan)


def test_virtual_mass_rejects_wrong_gradient_shape_and_zero_coefficient() -> None:
    out, static, state, fields, plan = _valid_inputs(("virtual_mass",))
    wrong_shape = _with_field(
        fields,
        "flow_velocity_gradient",
        np.zeros((2, 2), dtype=np.float64),
    )
    with pytest.raises(
        ValueError, match=r"virtual_mass.*flow_velocity_gradient.*shape"
    ):
        evaluate_force_pipeline(out, static, state, wrong_shape, plan)

    zero_coefficient = replace(
        plan,
        params=replace(plan.params, virtual_mass_coefficient=0.0),
    )
    with pytest.raises(
        ValueError, match=r"virtual_mass.*coefficient.*strictly positive"
    ):
        evaluate_force_pipeline(out, static, state, fields, zero_coefficient)


def test_thermophoresis_rejects_invalid_particle_and_field_properties() -> None:
    out, static, state, fields, plan = _valid_inputs(("thermophoresis",))
    invalid_diameter = replace(
        static,
        particle_diameter=np.asarray([1.0e-6, 0.0]),
    )
    with pytest.raises(
        ValueError,
        match=r"thermophoresis.*particle_diameter.*invalid particle rows: \[1\]",
    ):
        evaluate_force_pipeline(out, invalid_diameter, state, fields, plan)

    invalid_temperature = _with_field(
        fields, "temperature", np.asarray([np.nan, 420.0])
    )
    with pytest.raises(
        ValueError,
        match=r"thermophoresis.*temperature.*invalid particle rows: \[0\]",
    ):
        evaluate_force_pipeline(out, static, state, invalid_temperature, plan)

    invalid_multiplier = replace(
        static,
        thermophoretic_coeff=np.asarray([np.nan, -0.5]),
    )
    with pytest.raises(
        ValueError,
        match=r"thermophoresis.*thermophoretic_coeff.*invalid particle rows: \[1\]",
    ):
        evaluate_force_pipeline(out, invalid_multiplier, state, fields, plan)


def test_thermophoresis_rejects_nonphysical_parameters_without_floors() -> None:
    out, static, state, fields, plan = _valid_inputs(("thermophoresis",))
    zero_conductivity = replace(
        plan,
        params=replace(plan.params, gas_thermal_conductivity_W_mK=0.0),
    )
    with pytest.raises(
        ValueError,
        match=r"thermophoresis.*gas_thermal_conductivity_W_mK.*strictly positive",
    ):
        evaluate_force_pipeline(out, static, state, fields, zero_conductivity)

    missing_molecular_mass = replace(plan, gas_molecular_mass_kg=np.nan)
    with pytest.raises(
        ValueError,
        match=r"thermophoresis.*gas_molecular_mass_kg.*strictly positive",
    ):
        evaluate_force_pipeline(out, static, state, fields, missing_molecular_mass)


def test_dep_rejects_invalid_medium_permittivity_without_air_fallback() -> None:
    out, static, state, fields, plan = _valid_inputs(("dielectrophoresis",))
    invalid_medium = replace(
        plan,
        params=replace(plan.params, dep_medium_rel_permittivity=0.0),
    )

    with pytest.raises(
        ValueError,
        match=r"dielectrophoresis.*medium_relative_permittivity.*strictly positive",
    ):
        evaluate_force_pipeline(out, static, state, fields, invalid_medium)


def test_dep_rejects_invalid_particle_value_instead_of_using_config_fallback() -> None:
    out, static, state, fields, plan = _valid_inputs(("dielectrophoresis",))
    invalid_particle_value = replace(
        static,
        dep_particle_rel_permittivity=np.asarray([np.nan, 0.0]),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"dielectrophoresis.*particle_relative_permittivity.*"
            r"invalid particle rows: \[1\]"
        ),
    ):
        evaluate_force_pipeline(out, invalid_particle_value, state, fields, plan)

    unspecified = replace(
        static,
        dep_particle_rel_permittivity=np.asarray([np.nan, np.nan]),
    )
    result = evaluate_force_pipeline(out.copy(), unspecified, state, fields, plan)
    assert np.all(np.isfinite(result))


def test_lift_zero_vorticity_is_exact_zero_without_a_numerical_threshold() -> None:
    out, static, state, fields, plan = _valid_inputs(("lift",))
    fields = _with_field(fields, "vorticity", np.zeros((2, 3), dtype=np.float64))

    result = evaluate_force_pipeline(out, static, state, fields, plan)

    np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.float64))


def test_lift_rejects_invalid_viscosity_and_vorticity_shape() -> None:
    out, static, state, fields, plan = _valid_inputs(("lift",))
    invalid_viscosity = _with_field(
        fields, "dynamic_viscosity", np.asarray([1.8e-5, -2.0e-5])
    )
    with pytest.raises(
        ValueError,
        match=r"lift.*dynamic_viscosity.*invalid particle rows: \[1\]",
    ):
        evaluate_force_pipeline(out, static, state, invalid_viscosity, plan)

    wrong_vorticity = _with_field(fields, "vorticity", np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"lift.*vorticity.*shape"):
        evaluate_force_pipeline(out, static, state, wrong_vorticity, plan)


def test_electric_rejects_nonfinite_charge_to_mass_with_particle_row() -> None:
    out, static, state, fields, plan = _valid_inputs(("electric",))
    invalid_state = replace(state, charge_over_mass=np.asarray([2.0, np.nan]))

    with pytest.raises(
        ValueError,
        match=r"electric.*charge_over_mass.*invalid particle rows: \[1\]",
    ):
        evaluate_force_pipeline(out, static, invalid_state, fields, plan)
