"""Empirical refinement and conservation of the bounded first-arrival oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.verification.kramers_reference import KramersResult, integrated_ou_barrier
from tests.verification.ou_reference import quadrature_covariance

OBSERVATION_TIMES = np.asarray([0.4, 0.6, 0.8, 1.0, 1.2])
REFERENCE_ALLOWANCE = 5.0e-3


@pytest.fixture(scope="session")
def kramers_reference() -> tuple[KramersResult, dict[str, KramersResult]]:
    """Fixed grid series, before observing the production particle ensemble.

    Largest state is 512*256 doubles. Refinement differences establish an
    empirical allowance only; they are not a rigorous PDE error enclosure.
    All parameters are dimensionless, with gamma=Theta=1.
    """

    settings = {
        "x128": (128, 256, 0.0005, 4.0, 6.0),
        "x256": (256, 256, 0.0005, 4.0, 6.0),
        "x512": (512, 256, 0.0005, 4.0, 6.0),
        "time_coarse": (256, 256, 0.001, 4.0, 6.0),
        "velocity_fine": (256, 512, 0.0005, 4.0, 6.0),
        "position_domain": (512, 256, 0.0005, 8.0, 6.0),
        "velocity_domain": (256, 384, 0.0005, 4.0, 9.0),
    }
    results = {
        name: integrated_ou_barrier(
            nx=nx, nv=nv, dt=dt, times=OBSERVATION_TIMES, length=length, velocity_limit=limit
        )
        for name, (nx, nv, dt, length, limit) in settings.items()
    }
    reference = results["x512"]
    _check_refinement(reference, results)
    return reference, results


def _check_refinement(
    reference: KramersResult,
    results: dict[str, KramersResult],
    *,
    position_domain: bool = True,
) -> None:
    for result in results.values():
        assert result.minimum_cell_mass >= 0.0
        assert result.maximum_mass_error < 1.0e-10
        assert np.all(np.diff(result.arrival_cdf) > 0.0)
        np.testing.assert_allclose(
            result.arrival_cdf
            + result.surviving_mass
            + result.position_leak
            + result.velocity_leak,
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        )
        assert result.position_leak[-1] < 2.0e-5
        assert result.velocity_leak[-1] < 2.0e-8
    coarse = np.abs(results["x128"].arrival_cdf - results["x256"].arrival_cdf)
    fine = np.abs(results["x256"].arrival_cdf - reference.arrival_cdf)
    assert float(fine.max()) < 0.6 * float(coarse.max())
    # Moment-preserving delta projection narrows with these same dx/dv series.
    # Explicitly vary each other numerical axis at the fixed x256 grid.
    axes = ("time_coarse", "velocity_fine", "velocity_domain")
    if position_domain:
        axes += ("position_domain",)
    allowances = {
        name: float(np.max(np.abs(results[name].arrival_cdf - results["x256"].arrival_cdf)))
        for name in axes
    }
    assert allowances["time_coarse"] < 2.0e-4
    assert allowances["velocity_fine"] < 1.0e-4
    assert allowances.get("position_domain", 0.0) < 1.0e-5
    assert allowances["velocity_domain"] < 1.0e-6
    # A conservative operational empirical budget: two finest spatial
    # differences plus independently measured changes and counted tail losses.
    empirical_indicator = (
        2.0 * float(fine.max())
        + sum(allowances.values())
        + float(reference.position_leak[-1] + reference.velocity_leak[-1])
    )
    assert empirical_indicator < REFERENCE_ALLOWANCE


@pytest.fixture(scope="session")
def radial_kramers_reference() -> tuple[KramersResult, dict[str, KramersResult]]:
    """Folded 2DOF radius: |signed X| hits 0.75 at either physical end.

    Shift the signed interval [-0.75,0.75] to [-1.5,0]; signed X(0)=0.25
    becomes -0.5. Both position ends are physical barriers, so increasing
    position-domain length would change the problem rather than test a tail.
    Refine dx/delta, dv/delta, dt and the artificial velocity domain instead.
    """

    settings = {
        "x128": (128, 256, 0.00025, 6.0),
        "x256": (256, 256, 0.00025, 6.0),
        "x512": (512, 256, 0.00025, 6.0),
        "time_coarse": (256, 256, 0.0005, 6.0),
        "velocity_fine": (256, 512, 0.00025, 6.0),
        "velocity_domain": (256, 384, 0.00025, 9.0),
    }
    results = {
        name: integrated_ou_barrier(
            nx=nx,
            nv=nv,
            dt=dt,
            times=OBSERVATION_TIMES,
            length=1.5,
            velocity_limit=limit,
            left_absorbing=True,
        )
        for name, (nx, nv, dt, limit) in settings.items()
    }
    reference = results["x512"]
    _check_refinement(reference, results, position_domain=False)
    assert np.all(reference.position_leak == 0.0)
    print(
        "radial Kramers refinement="
        + str({name: result.arrival_cdf.tolist() for name, result in results.items()})
    )
    return reference, results


def test_two_absorbing_ou_barriers_contain_both_independent_endpoint_tails(
    radial_kramers_reference: tuple[KramersResult, dict[str, KramersResult]],
) -> None:
    reference, _results = radial_kramers_reference
    endpoint_tail = np.asarray(
        [
            0.5
            * (
                math.erfc(0.5 / math.sqrt(2.0 * quadrature_covariance(time, 1.0, 1.0)[0]))
                + math.erfc(1.0 / math.sqrt(2.0 * quadrature_covariance(time, 1.0, 1.0)[0]))
            )
            for time in OBSERVATION_TIMES
        ]
    )
    assert np.all(reference.arrival_cdf + REFERENCE_ALLOWANCE >= endpoint_tail)


def test_kramers_first_arrival_contains_independent_ou_endpoint_tail(
    kramers_reference: tuple[KramersResult, dict[str, KramersResult]],
) -> None:
    reference, _results = kramers_reference
    # Every continuous trajectory with X(t)>0 has already crossed X=0.
    # The independent no-boundary OU Green integral supplies this necessary
    # lower probability; it does not itself supply a first-arrival solution.
    endpoint_tail = np.asarray(
        [
            0.5 * math.erfc(0.5 / math.sqrt(2.0 * quadrature_covariance(time, 1.0, 1.0)[0]))
            for time in OBSERVATION_TIMES
        ]
    )
    assert np.all(reference.arrival_cdf + REFERENCE_ALLOWANCE >= endpoint_tail)
