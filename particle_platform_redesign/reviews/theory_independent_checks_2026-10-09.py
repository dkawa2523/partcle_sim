"""One-off review calculation. Run from solver with uv run --locked python ../reviews/this_file.py."""

import json
import math
from pathlib import Path

import numpy as np

from chamber_particles.physics.charge import shifted_maxwellian_ion_factors
from chamber_particles.physics.forces import BOLTZMANN_J_K, barnes_collisionless_ion_drag


def gaussian_moments(shift, order):
    """E|w|, E(1/|w|), E(|w| w_x) for w ~ N((shift,0,0),I)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    end = shift + 12.0
    radius = 0.5 * end * (nodes + 1.0)
    radial_weights = 0.5 * end * weights
    density = np.exp(
        -0.5 * (radius[:, None] - shift) ** 2
        - radius[:, None] * shift * (1.0 - nodes[None, :])
    ) / math.sqrt(2.0 * math.pi)
    weighted = radial_weights[:, None] * weights[None, :] * density
    return [
        float(np.sum(weighted * radius[:, None] ** 3)),
        float(np.sum(weighted * radius[:, None])),
        float(np.sum(weighted * radius[:, None] ** 4 * nodes[None, :])),
    ]


thermal = math.sqrt(8.0 / math.pi)
shifts = [0.0, 1.0e-6, 0.5, 2.0, 5.0]
p, h = shifted_maxwellian_ion_factors(np.asarray(shifts))
shift_rows = []
for index, shift in enumerate(shifts):
    coarse = gaussian_moments(shift, 128)
    fine = gaussian_moments(shift, 256)
    p_ref = fine[0] / thermal
    h_ref = math.sqrt(math.pi / 2.0) * fine[1]
    shift_rows.append(
        dict(
            s=shift,
            production_P=float(p[index]),
            quadrature_P=p_ref,
            production_H=float(h[index]),
            quadrature_H=h_ref,
            relative_error_P=float(abs(p[index] / p_ref - 1.0)),
            relative_error_H=float(abs(h[index] / h_ref - 1.0)),
            max_moment_difference_128_256=max(abs(x - y) for x, y in zip(coarse, fine)),
        )
    )

ion_mass = 6.6335209e-26
temperature = 300.0
sigma = math.sqrt(BOLTZMANN_J_K * temperature / ion_mass)
radius = 50.0e-9
particle_mass = 4.0 * math.pi * radius**3 * 2000.0 / 3.0
barnes_rows = []
for drift in [0.001, 0.01, 0.1, 1.0, 3.0]:
    shift = drift * thermal
    exact = gaussian_moments(shift, 256)[2]
    evaluated = barnes_collisionless_ion_drag(
        mass_kg=np.asarray([particle_mass]),
        electrostatic_radius_m=np.asarray([radius]),
        charge_number=np.asarray([0.0]),
        velocity_m_s=np.zeros((1, 2)),
        electron_number_density_m3=np.asarray([1.0e14]),
        positive_ion_number_density_m3=np.asarray([1.0e14]),
        electron_temperature_K=np.asarray([11604.518121550082]),
        positive_ion_temperature_K=np.asarray([temperature]),
        positive_ion_velocity_m_s=np.asarray([[shift * sigma, 0.0]]),
        ion_neutral_mean_free_path_m=np.asarray([1.0]),
        positive_ion_mass_kg=ion_mass,
        maximum_ion_drift_ratio=20.0,
    )
    actual = float(
        evaluated.acceleration_m_s2[0, 0]
        * particle_mass
        / (math.pi * radius**2 * 1.0e14 * ion_mass * sigma**2)
    )
    barnes_rows.append(
        dict(
            U_over_mean_thermal_speed=drift,
            production_dimensionless_momentum=actual,
            independent_Maxwell_momentum=exact,
            production_over_exact=actual / exact,
            applicable=bool(evaluated.applicable[0]),
        )
    )

aggregate = [
    dict(
        minus_phi_over_Vi=value,
        effective_speed_over_stationary_OML=(1.0 + value * math.pi / 4.0) / (1.0 + value),
    )
    for value in [1, 3, 10]
]
talbot_rows = []
for knudsen in [0.01, 0.1, 1.0, 10.0]:
    conductivity_ratio = 0.01

    def correction(kn):
        return 1.17 * (conductivity_ratio + 2.2 * kn) / (
            (1.0 + 3.0 * 1.146 * kn) * (1.0 + 2.0 * conductivity_ratio + 4.4 * kn)
        )

    talbot_rows.append(
        dict(Kn_d=knudsen,conductivity_ratio=conductivity_ratio,
             current_over_radius_convention=correction(knudsen)/correction(2.0*knudsen))
    )
result = dict(
    method="Independent radial/angular Gauss-Legendre integration of shifted 3D Gaussian; no production formula in reference integrand",
    quadrature_orders=[128, 256],
    shifted_OML=shift_rows,
    Barnes_neutral_collection=barnes_rows,
    aggregate_zero_drift_asymptote=dict(
        neglects="1 m/s regularizer and 0.01 V floor",
        attractive_increment_ratio=math.pi / 4.0,
        examples=aggregate,
    ),
    Talbot_convention_comparison=talbot_rows,
)
target = Path(__file__).with_suffix(".json")
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
