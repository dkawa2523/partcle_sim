"""P00 smoke test for the stable package boundary."""

from chamber_particles import (
    CaseError,
    IncompleteResultError,
    SimulationError,
    load_case,
    open_result,
    simulate,
)


def test_public_api_is_importable() -> None:
    operations = (load_case, simulate, open_result)
    exception_types = (CaseError, SimulationError, IncompleteResultError)

    assert all(callable(operation) for operation in operations)
    assert all(issubclass(exception_type, Exception) for exception_type in exception_types)
