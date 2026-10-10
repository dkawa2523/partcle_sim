"""Read historical field artifacts and compare declared COMSOL/canonical coefficients.

This is an external review calculation, not a solver reader or migration path.
It runs no COMSOL study or trajectory, changes no input, and writes only its JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
SOLVER = REPOSITORY / "particle_platform_redesign/solver"
FIELD_NAMES = (
    "gas_density",
    "gas_dynamic_viscosity",
    "gas_temperature",
    "gas_mean_free_path",
)
GAS_MASS_KG = 1.2753471408396638e-25
BOLTZMANN_J_K = 1.380649e-23
DIAMETER_M = 1.0e-7
ACCOMMODATION = 0.9
INPUTS = (
    (
        "caseP_100nm",
        "_out_m3c1/theory_100nm_30ms_candidate_v3/caseP/candidate_input.h5",
        "c54b4b658213230e82307ca89018538c0240bfcc83e793206d5093f4f08908e9",
    ),
    (
        "caseA_100nm",
        "_out_m3c1/caseA_100nm_exported_p1_v4/candidate_input.h5",
        "14cede2c85e7888368c04da000c5fcaf50cf2135c85be0497633b59d83febea1",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coefficients(values: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    delta = 1.0 + math.pi * ACCOMMODATION / 8.0
    mu_b = (
        values["gas_dynamic_viscosity"]
        * (8.0 + math.pi * ACCOMMODATION)
        * DIAMETER_M
        / (36.0 * values["gas_mean_free_path"])
    )
    beta_b = 3.0 * math.pi * DIAMETER_M * mu_b
    mean_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * values["gas_temperature"] / (math.pi * GAS_MASS_KG)
    )
    beta_c = (
        math.pi * DIAMETER_M**2 / 3.0 * values["gas_density"] * mean_speed * delta
    )
    return mu_b, beta_b, beta_c, beta_b / beta_c


def metrics(values: dict[str, np.ndarray]) -> dict[str, object]:
    _, _, _, ratio = coefficients(values)
    if not np.all(np.isfinite(ratio)):
        raise ValueError("nonfinite coefficient ratio")
    return {
        "rows": int(ratio.size),
        "beta_brownian_over_beta_canonical_min": float(ratio.min()),
        "beta_brownian_over_beta_canonical_max": float(ratio.max()),
        "maximum_absolute_relative_difference": float(np.max(np.abs(ratio - 1.0))),
        "unweighted_rms_relative_difference": float(np.sqrt(np.mean((ratio - 1.0) ** 2))),
        "meaning": "coefficient audit; neither trajectory error nor an area-integrated norm",
    }


def sample_releases(
    nodes: np.ndarray,
    triangles: np.ndarray,
    positions: np.ndarray,
    fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    first = nodes[triangles[:, 0]]
    edge_b = nodes[triangles[:, 1]] - first
    edge_c = nodes[triangles[:, 2]] - first
    determinant = edge_b[:, 0] * edge_c[:, 1] - edge_b[:, 1] * edge_c[:, 0]
    if np.any(determinant == 0.0):
        raise ValueError("degenerate source triangle")
    result = {name: [] for name in FIELD_NAMES}
    for position in positions:
        offset = position - first
        b = (offset[:, 0] * edge_c[:, 1] - offset[:, 1] * edge_c[:, 0]) / determinant
        c = (edge_b[:, 0] * offset[:, 1] - edge_b[:, 1] * offset[:, 0]) / determinant
        matches = np.flatnonzero((b >= -1e-12) & (c >= -1e-12) & (b + c <= 1 + 1e-12))
        if not matches.size:
            raise ValueError("release outside source triangulation")
        cell = int(matches[0])
        weights = np.array([1.0 - b[cell] - c[cell], b[cell], c[cell]])
        for name in FIELD_NAMES:
            result[name].append(float(weights @ fields[name][triangles[cell]]))
    return {name: np.asarray(values) for name, values in result.items()}


def audit_input(case: str, relative: str, expected_sha256: str) -> dict[str, object]:
    path = SOLVER / relative
    actual_hash = sha256(path)
    if actual_hash != expected_sha256:
        raise ValueError(f"historical input hash changed: {case}")
    with h5py.File(path, "r") as artifact:
        layout = artifact["layouts/plasma/unstructured"]
        nodes = np.asarray(layout["nodes_m"])
        triangles = np.asarray(layout["connectivity"])
        supported = np.asarray(layout["cell_support"])
        if not np.all(supported == 1):
            raise ValueError("audit assumes all source cells supported")
        fields = {
            name: np.asarray(artifact[f"fields/{name}/values"]).reshape(-1)
            for name in FIELD_NAMES
        }
        positions = np.asarray(artifact["sources/particles/position_m"])
    centroid_values = {name: field[triangles].mean(axis=1) for name, field in fields.items()}
    mu_b, beta_b, beta_c, ratio = coefficients(centroid_values)
    worst = int(np.argmax(np.abs(ratio - 1.0)))
    return {
        "case": case,
        "path": str(path),
        "sha256": actual_hash,
        "hash_verified": True,
        "nodal": metrics(fields),
        "triangle_centroids": metrics(centroid_values),
        "realized_release_positions": metrics(sample_releases(nodes, triangles, positions, fields)),
        "worst_centroid": {
            "zero_based_cell": worst,
            "zero_based_nodes": triangles[worst].tolist(),
            "position_m": nodes[triangles[worst]].mean(axis=0).tolist(),
            "cell_support": int(supported[worst]),
            "primitives": {name: float(value[worst]) for name, value in centroid_values.items()},
            "comsol_declared_effective_viscosity_Pa_s": float(mu_b[worst]),
            "brownian_equivalent_beta_kg_s": float(beta_b[worst]),
            "canonical_epstein_beta_kg_s": float(beta_c[worst]),
            "force_standard_deviation_ratio": float(np.sqrt(ratio[worst])),
        },
    }


def main() -> None:
    source_settings = (
        "model_dataset/cf4_o2_etch_caseA_nonlinear_sass/cases/"
        "formal_iondrag_theory_consistent"
    )
    report = {
        "tool_revision": "readonly_comsol_coefficient_review_v1",
        "date": "2026-10-09",
        "scope": "historical accepted P1 inputs; declared C2 Brownian coefficient versus canonical Epstein coefficient",
        "comsol_solve_performed": False,
        "trajectory_simulation_performed": False,
        "production_files_modified": False,
        "status": "DECLARED_COEFFICIENTS_DIFFER_INSIDE_P1_CELLS",
        "parameters": {
            "gas_molecular_mass_kg": GAS_MASS_KG,
            "boltzmann_J_K": BOLTZMANN_J_K,
            "drag_diameter_m": DIAMETER_M,
            "accommodation_coefficient": ACCOMMODATION,
        },
        "equations": {
            "mu_B": "mu_P1*(8+pi*sigma)*d/(36*lambda_P1)",
            "beta_B": "3*pi*d*mu_B",
            "beta_canonical": "pi*d^2/3*rho_P1*sqrt(8*kB*T_P1/(pi*m_g))*(1+pi*sigma/8)",
            "ratio": "(mu_P1/lambda_P1)/(rho_P1*sqrt(2*kB*T_P1/(pi*m_g)))",
        },
        "evidence": {
            "comsol_C2_formula": {"path": "particle_platform_redesign/solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java", "lines": [192, 203, 214, 338, 345]},
            "canonical_formula": {"path": "particle_platform_redesign/solver/src/chamber_particles/physics/forces.py", "lines": [1099, 1122, 1133]},
            "caseP_source_axis": {"path": source_settings + "/caseP_100nm/external_reproduction/config/particle_physics_feature_settings.csv", "line": 6, "condition": "Freeze"},
            "caseA_source_axis": {"path": source_settings + "/caseA_100nm/external_reproduction/config/particle_physics_feature_settings.csv", "line": 6, "condition": "Freeze"},
            "C2_contract_axis": {"path": "particle_platform_redesign/solver/tools/vv/comsol/cases/m3c2_caseP_100nm_stochastic_pilot_v1.json", "line": 62, "meaning": "coordinate_crossing_not_material_event"},
            "critical_microcase_axis": {"path": "particle_platform_redesign/solver/tools/vv/comsol/comsol/RunM3CCriticalBoundaries.java", "line": 155, "condition": "Bounce"},
        },
        "official_primary_urls": {
            "brownian_theory": "https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html",
            "rarefied_drag": "https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html",
            "axial_symmetry": "https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.18.html",
            "wall": "https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.04.html",
            "wall_accuracy_and_status": "https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.02.html",
            "time_solver_api": "https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_solver.51.51.html",
            "talbot_documentation": "https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.48.html",
        },
        "inputs": [audit_input(*entry) for entry in INPUTS],
        "interpretation": {
            "proven": "Nonlinear gas closure holds at source nodes but independent P1 interpolation breaks equality of the two declared coefficient expressions inside cells.",
            "not_proven": "Actual COMSOL assembly, trajectory or ensemble effect, and invalidation of a historical population gate.",
            "suggested_binding": "Derive Brownian effective viscosity from the same canonical drag beta: mu_B=beta_canonical/(3*pi*d).",
            "axis_scope": "Source Freeze and candidate coordinate seam have different semantics; a separate Bounce microcase does not certify the chamber source axis.",
        },
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
