from __future__ import annotations

"""Shared source/material/wall catalog schema metadata.

Dataclasses remain explicit in datamodel.py for readability. This module keeps
the operational field names, aliases, and default values in one place so CSV
loaders and source resolution logic do not drift apart.
"""

DEFAULT_SOURCE_LAW_NAME = 'explicit_csv'
DEFAULT_SOURCE_EVENT_TAG = ''

SOURCE_RESOLUTION_SCALAR_DEFAULTS = (
    ('source_speed_scale', 1.0),
    ('source_position_offset_m', 0.0),
    ('source_temperature_K', 300.0),
    ('source_normal_speed_mean_mps', 0.2),
    ('source_normal_speed_std_mps', 0.1),
    ('source_tangent_speed_std_mps', 0.05),
    ('source_resuspension_velocity_scale', 1.0),
    ('source_resuspension_normal_speed_mean_mps', 0.05),
    ('source_resuspension_normal_speed_std_mps', 0.02),
    ('source_resuspension_speed_threshold_mps', 0.0),
    ('source_resuspension_tau_threshold_Pa', 0.0),
    ('source_resuspension_utau_threshold_mps', 0.0),
    ('source_resuspension_shear_length_m', 1e-4),
    ('source_dynamic_viscosity_Pas', 1.8e-5),
    ('source_thermal_accommodation', 1.0),
    ('source_flake_weight', 1.0),
    ('source_reflectivity', 0.0),
    ('source_roughness_rms', 0.0),
    ('source_roughness_corr_length_m', 1e-6),
    ('source_roughness_slope_rms', float('nan')),
    ('source_adhesion_energy_Jm2', 0.0),
    ('source_resuspension_roughness_scale', 0.0),
    ('source_resuspension_adhesion_scale', 0.0),
    ('source_resuspension_tau_roughness_scale', 0.0),
    ('source_resuspension_tau_adhesion_scale', 0.0),
    ('source_resuspension_tau_slope_scale', 0.0),
    ('source_burst_center_s', 0.0),
    ('source_burst_sigma_s', 0.01),
    ('source_burst_amplitude', 0.0),
    ('source_burst_period_s', 0.0),
    ('source_burst_phase_s', 0.0),
    ('source_burst_min_factor', 0.0),
    ('source_burst_max_factor', 10.0),
    ('physics_flow_scale', 1.0),
    ('physics_drag_tau_scale', 1.0),
    ('physics_body_accel_scale', 1.0),
)

SOURCE_RESOLUTION_SCALAR_FIELDS = tuple(name for name, _ in SOURCE_RESOLUTION_SCALAR_DEFAULTS)
SOURCE_RESOLUTION_SCALAR_DEFAULT_MAP = dict(SOURCE_RESOLUTION_SCALAR_DEFAULTS)

SOURCE_CATALOG_TEXT_ALIASES = (
    ('source_default_event_tag', ('source_default_event_tag',)),
    ('wall_law', ('wall_law',)),
)

SOURCE_CATALOG_FLOAT_ALIASES = (
    ('source_speed_scale', ('source_speed_scale',)),
    ('source_position_offset_m', ('source_position_offset_m',)),
    ('source_temperature_K', ('source_temperature_K',)),
    ('source_normal_speed_mean_mps', ('source_normal_speed_mean_mps',)),
    ('source_normal_speed_std_mps', ('source_normal_speed_std_mps',)),
    ('source_tangent_speed_std_mps', ('source_tangent_speed_std_mps',)),
    ('source_resuspension_velocity_scale', ('source_resuspension_velocity_scale',)),
    ('source_resuspension_normal_speed_mean_mps', ('source_resuspension_normal_speed_mean_mps',)),
    ('source_resuspension_normal_speed_std_mps', ('source_resuspension_normal_speed_std_mps',)),
    ('source_resuspension_speed_threshold_mps', ('source_resuspension_speed_threshold_mps',)),
    ('source_resuspension_tau_threshold_Pa', ('source_resuspension_tau_threshold_Pa',)),
    (
        'source_resuspension_utau_threshold_mps',
        ('source_resuspension_utau_threshold_mps', 'source_resuspension_friction_velocity_threshold_mps'),
    ),
    ('source_resuspension_shear_length_m', ('source_resuspension_shear_length_m',)),
    ('source_dynamic_viscosity_Pas', ('source_dynamic_viscosity_Pas',)),
    ('source_thermal_accommodation', ('source_thermal_accommodation',)),
    ('source_flake_weight', ('source_flake_weight',)),
    ('source_reflectivity', ('source_reflectivity',)),
    ('source_roughness_rms', ('source_roughness_rms',)),
    ('source_roughness_corr_length_m', ('source_roughness_corr_length_m',)),
    ('source_roughness_slope_rms', ('source_roughness_slope_rms',)),
    ('source_adhesion_energy_Jm2', ('source_adhesion_energy_Jm2',)),
    ('source_resuspension_roughness_scale', ('source_resuspension_roughness_scale',)),
    ('source_resuspension_adhesion_scale', ('source_resuspension_adhesion_scale',)),
    ('source_resuspension_tau_roughness_scale', ('source_resuspension_tau_roughness_scale',)),
    ('source_resuspension_tau_adhesion_scale', ('source_resuspension_tau_adhesion_scale',)),
    ('source_resuspension_tau_slope_scale', ('source_resuspension_tau_slope_scale',)),
    ('source_burst_center_s', ('source_burst_center_s',)),
    ('source_burst_sigma_s', ('source_burst_sigma_s',)),
    ('source_burst_amplitude', ('source_burst_amplitude',)),
    ('source_burst_period_s', ('source_burst_period_s',)),
    ('source_burst_phase_s', ('source_burst_phase_s',)),
    ('source_burst_min_factor', ('source_burst_min_factor',)),
    ('source_burst_max_factor', ('source_burst_max_factor',)),
    ('wall_stick_probability', ('wall_stick_probability',)),
    ('wall_restitution', ('wall_restitution',)),
    ('wall_diffuse_fraction', ('wall_diffuse_fraction',)),
    (
        'wall_critical_sticking_velocity_mps',
        ('wall_critical_sticking_velocity_mps', 'critical_sticking_velocity_mps'),
    ),
    ('wall_reflectivity', ('wall_reflectivity',)),
    ('wall_roughness_rms', ('wall_roughness_rms',)),
    ('physics_flow_scale', ('physics_flow_scale',)),
    ('physics_drag_tau_scale', ('physics_drag_tau_scale',)),
    ('physics_body_accel_scale', ('physics_body_accel_scale',)),
)

FLAKE_ESCAPE_SOURCE_PARAMETERS = (
    'source_speed_scale',
    'source_position_offset_m',
    'source_normal_speed_mean_mps',
    'source_normal_speed_std_mps',
    'source_tangent_speed_std_mps',
    'source_flake_weight',
)

BURST_ENVELOPE_SOURCE_PARAMETERS = (
    'source_burst_center_s',
    'source_burst_sigma_s',
    'source_burst_amplitude',
    'source_burst_period_s',
    'source_burst_phase_s',
    'source_burst_min_factor',
    'source_burst_max_factor',
)

FLAKE_BURST_SOURCE_PARAMETERS = FLAKE_ESCAPE_SOURCE_PARAMETERS + BURST_ENVELOPE_SOURCE_PARAMETERS

RESUSPENSION_SOURCE_PARAMETERS = (
    'source_speed_scale',
    'source_position_offset_m',
    'source_resuspension_velocity_scale',
    'source_resuspension_normal_speed_mean_mps',
    'source_resuspension_normal_speed_std_mps',
    'source_tangent_speed_std_mps',
)

THERMAL_REEMISSION_SOURCE_PARAMETERS = (
    'source_speed_scale',
    'source_position_offset_m',
    'source_temperature_K',
    'source_thermal_accommodation',
)

