"""Bind actual saved t0 rows to explicit collision-regime motivation, not truth."""

import csv
import hashlib
import json
import math
from pathlib import Path


def initial_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            int(float(row["particle_id"])): row
            for row in csv.DictReader(stream)
            if float(row["time_s"]) == 0.0
        }


def main() -> None:
    output = Path(__file__).resolve().parent
    solver = output.parents[2]
    native = solver / "evidence/comsol_speed10000_comparison_2026_10_13/original_comparison/native"
    bindings = []
    rows = []
    for name in ("caseA", "caseP"):
        state_path = native / f"{name}_saved_states_0_to_200us_normalized.csv"
        coefficient_path = native / f"{name}_native_force_coefficients_normalized.csv"
        states = initial_rows(state_path)
        coefficients = initial_rows(coefficient_path)
        assert states.keys() == coefficients.keys()
        for source in (state_path, coefficient_path):
            bindings.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        for particle_id, state in states.items():
            coefficient = coefficients[particle_id]
            temperature = float(coefficient["Brownian_gas_temperature_K"])
            assert temperature == float(state["gas_temperature_K"])
            molecular_mass = float(coefficient["gas_molar_mass_kg_per_mol"]) / 6.02214076e23
            slip = math.hypot(
                float(state["velocity_r_m_per_s"]) - float(state["gas_velocity_r_m_per_s"]),
                float(state["velocity_z_m_per_s"]) - float(state["gas_velocity_z_m_per_s"]),
            )
            rows.append({
                "label": f"{name}_p{particle_id:03}",
                "particle_mass_kg": float(coefficient["mass_kg"]),
                "molecular_mass_kg": molecular_mass,
                "gas_density_kg_m3": float(coefficient["drag_gas_density_kg_per_m3"]),
                "gas_temperature_K": temperature,
                "diameter_m": float(state["diameter_m"]),
                "diffuse_fraction": 0.9,
                "surface_temperature_K": temperature,
                "speed_ratio": slip / math.sqrt(2.0 * 1.380649e-23 * temperature / molecular_mass),
                "gas_mean_free_path_m": float(coefficient["gas_mean_free_path_m"]),
            })
    with (output / "source_regimes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = sorted(rows[:len(rows) // 2], key=lambda row: row["speed_ratio"])
    labels = [selected[index]["label"] for index in (0, len(selected) // 2, len(selected) - 1)]
    receipt = {
        "source_bindings": bindings,
        "rows": len(rows),
        "jump_labels": labels,
        "assumptions": ["One supplied pseudogas, not species-resolved CF4/O2 truth.",
                        "sigma=.9 copied from the selected Maxwell mixed mean law.",
                        "Tw=Tg is a candidate model assumption, not a native measured particle temperature.",
                        "The native source is 2DOF r/z; no phi velocity is supplied.",
                        "Source rows motivate the regime; COMSOL force values are not the oracle."],
    }
    (output / "source_regime_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "jump_labels": labels}))


if __name__ == "__main__":
    main()
