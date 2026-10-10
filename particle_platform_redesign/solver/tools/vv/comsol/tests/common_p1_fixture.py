"""Current-writer inputs for concrete common-P1 adapter tests, never COMSOL truth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from tools.vv.comsol.actual_run_receipt import RECEIPT_NAME, RECEIPT_REVISION
from tools.vv.comsol.meaning_preflight import LAYERS

from chamber_particles.case_format import (
    BoundaryData,
    CaseFileInfo,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    RealizedTableSource,
    write,
)


def write_synthetic_meaning_inventory(
    root: Path, model_sha256: str, field_identity: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Explicit synthetic observed/expected bytes for consumer gates, not physics evidence."""
    meanings = {layer: {"enabled": True, "selector": "synthetic", "value": 1.0} for layer in LAYERS}
    observed = {
        "schema_version": 1,
        "tool_revision": RECEIPT_REVISION,
        "source": {},
        "companion": {"boundary_features": {}, "synthetic_meanings": meanings},
        "terminal_observations": [],
    }
    observed_path = root / RECEIPT_NAME
    observed_path.write_text(json.dumps(observed), encoding="utf-8")
    expected_path = root / "synthetic_expected.json"
    expected_path.write_text(json.dumps(meanings), encoding="utf-8")
    observed_reference = {"path": observed_path.name, "sha256": sha256(observed_path)}
    question = {
        "requested": True,
        "reference_field": {"representation": "canonical", "identity": field_identity},
        "candidate_field": {"representation": "canonical", "identity": field_identity},
        "adapter_lineage": [],
        "outcome": {"status": "NOT_TESTED", "summary": "synthetic consumer test", "evidence": []},
    }
    inventory = {
        "schema_version": 2,
        "inventory_id": "synthetic-consumer-test",
        "source": {
            "kind": "comsol_extracted_model_inventory",
            "model_sha256": model_sha256,
            "comsol_version": "synthetic",
            "component": "comp1",
            "study": "std1",
            "solution": "sol1",
            "dataset": "dset1",
        },
        "layers": {
            layer: [
                {
                    "id": layer,
                    "scope": "required",
                    "source_meaning": layer,
                    "canonical_meaning": layer,
                    "mapping": "direct",
                    "adapter_action": None,
                    "evidence": ["explicit synthetic fixture"],
                    "reason": "consumer gate",
                    "binding": [
                        {
                            "expected": {
                                "path": expected_path.name,
                                "sha256": sha256(expected_path),
                                "pointer": "/" + layer,
                            },
                            "observed": {
                                **observed_reference,
                                "pointer": "/companion/synthetic_meanings/" + layer,
                            },
                        }
                    ],
                }
            ]
            for layer in LAYERS
        },
        "comparison_conditions": {
            "same_canonical_field_solver_parity": question,
            "native_fe_end_to_end_reproduction": {
                **question,
                "requested": False,
                "outcome": {
                    "status": "NOT_APPLICABLE",
                    "summary": "synthetic consumer test",
                    "evidence": [],
                },
            },
        },
    }
    inventory_path = root / "meaning_inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return (
        {"path": inventory_path.name, "sha256": sha256(inventory_path)},
        {**observed_reference, "observation": "EXPORTED"},
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_current_common_p1_template(reference: Path, path: Path, content_hash: str) -> None:
    """Build a current case from reference model parameters and current fixture data."""
    parameters = yaml.safe_load(reference.read_text(encoding="utf-8"))
    document = {
        "format_version": 3,
        "case": {
            "name": "manufactured-common-p1",
            "data_path": "input.h5",
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": "axisymmetric_rz_meridional"},
        "time": parameters["time"],
        "solver": parameters["solver"],
        "resources": {"memory_limit_mb": 512},
        "physics": parameters["physics"],
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": parameters["boundaries"],
        "output": parameters["output"],
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def write_manufactured_common_p1(path: Path, *, negative_ions: bool = False) -> CaseFileInfo:
    """Write seven nodes, five triangles and one explicit 287-row release table."""

    nodes = np.asarray([[0, 0], [0, 1], [0.1, 0], [0.1, 1], [1, 0], [1, 1], [1, 0.5]], dtype="<f8")
    triangles = np.asarray([[0, 2, 3], [0, 3, 1], [2, 4, 6], [2, 6, 5], [2, 5, 3]], dtype="<i8")
    boundary = BoundaryData(
        line2=np.asarray([[0, 2], [2, 4], [4, 6], [6, 5], [5, 3], [3, 1]], dtype="<i8"),
        boundary_id=np.arange(1, 7, dtype="<i4"),
        group_id=np.arange(6, dtype="<i4"),
        material_id=np.zeros(6, dtype="<i4"),
        owner_cell_type=np.ones(6, dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 2, 2, 3, 4, 1], dtype="<i8"),
        orientation=np.ones(6, dtype="<i1"),
    )
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=(
            "wafer",
            "grounded_wall",
            "focus_transition",
            "outer_dielectric",
            "gas_inlet",
            "pump_outlet",
        ),
        tri3=triangles,
        tri3_domain_id=np.ones(5, dtype="<i4"),
    )
    layout = P1TriLayout("plasma", nodes, triangles, np.ones(5, dtype="<u1"))
    primitives = [
        ("gas_density", "kg/m^3", [1.0e-6]),
        ("gas_dynamic_viscosity", "Pa*s", [1.0e-5]),
        ("gas_temperature", "K", [300.0]),
        ("gas_mean_free_path", "m", [1.0e-3]),
        ("electron_number_density", "1/m^3", [1.0e15]),
        ("positive_ion_number_density", "1/m^3", [1.2e15]),
        ("electron_thermal_voltage", "V", [3.0]),
        ("positive_ion_thermal_voltage", "V", [0.05]),
        ("effective_positive_ion_mass", "kg", [6.6335209e-26]),
        ("screening_length", "m", [1.0e-3]),
        ("ion_neutral_mean_free_path", "m", [1.0]),
        ("azimuthal_gas_vorticity", "1/s", [1.0]),
        ("gas_velocity", "m/s", [0.0, 0.0]),
        ("electric_field", "V/m", [0.0, 100.0]),
        ("positive_ion_velocity", "m/s", [0.0, 1000.0]),
        ("gradient_mean_e_squared", "V^2/m^3", [0.0, 10.0]),
        ("gas_translational_heat_flux", "W/m^2", [0.0, 1.0e-8]),
    ]
    if negative_ions:
        primitives.extend(
            [
                ("negative_ion_number_density", "1/m^3", [2.0e14]),
                ("negative_ion_thermal_voltage", "V", [0.03]),
                ("effective_negative_ion_mass", "kg", [6.6335209e-26]),
                ("negative_ion_velocity", "m/s", [0.0, -100.0]),
            ]
        )
    fields = tuple(
        FieldData(
            name=name,
            layout=layout.name,
            association="node",
            components=("value",) if len(values) == 1 else ("r", "z"),
            stored_basis="scalar" if len(values) == 1 else "axisymmetric_rz",
            values=np.broadcast_to(values, (7, len(values))).copy(),
            unit=unit,
        )
        for name, unit, values in primitives
    )
    source = RealizedTableSource(
        name="particles",
        particle_id=np.arange(1, 288, dtype="<i8"),
        release_time_s=np.zeros(287),
        position_m=np.tile([0.3, 0.4], (287, 1)),
        velocity_m_s=np.zeros((287, 2)),
        charge_number=np.full(287, -1.0),
        mass_kg=np.full(287, 4.0e-15),
        drag_diameter_m=np.full(287, 1.0e-7),
        contact_radius_m=np.zeros(287),
        electrostatic_radius_m=np.full(287, 5.0e-8),
        displaced_volume_m3=np.zeros(287),
        model_weight=np.ones(287),
        material_id=np.zeros(287, dtype="<i4"),
    )
    provenance = _provenance("0" * 64, "manufactured_constant_primitives")
    return write(
        path, DataBundle("axisymmetric_rz", provenance, geometry, (layout,), fields, (source,))
    )


def rematerialize_saved_localization_input(reference: Path, path: Path) -> CaseFileInfo:
    """Copy only the registered raw arrays into current DataBundle constructors.

    This fixed external fixture preserves the historical field values and mesh;
    it is not a schema reader, migration path or replacement physics oracle.
    The historical file and its provenance/evidence are never overwritten.
    """

    with h5py.File(reference, "r") as handle:
        boundary = BoundaryData(
            line2=handle["geometry/boundary/line2"][...],
            boundary_id=handle["geometry/boundary/boundary_id"][...],
            group_id=handle["geometry/boundary/group_id"][...],
            material_id=handle["geometry/boundary/material_id"][...],
            owner_cell_type=handle["geometry/boundary/owner_cell_type"][...],
            owner_cell_local_index=handle["geometry/boundary/owner_cell_local_index"][...],
            orientation=handle["geometry/boundary/orientation"][...],
            external_id=handle["geometry/boundary/external_id"][...],
        )
        geometry = GeometryData(
            nodes_m=handle["geometry/nodes_m"][...],
            boundary=boundary,
            group_names=tuple(handle["geometry/groups/names"].asstr()[...]),
            node_external_id=handle["geometry/node_external_id"][...],
            tri3=handle["geometry/cells/tri3"][...],
            tri3_domain_id=handle["geometry/cells/tri3_domain_id"][...],
        )
        layout = P1TriLayout(
            "plasma",
            handle["layouts/plasma/unstructured/nodes_m"][...],
            handle["layouts/plasma/unstructured/connectivity"][...],
            handle["layouts/plasma/unstructured/cell_support"][...],
        )
        fields = tuple(
            FieldData(
                name=name,
                layout=handle[f"fields/{name}/layout"].asstr()[()],
                association=handle[f"fields/{name}/association"].asstr()[()],
                components=tuple(handle[f"fields/{name}/components"].asstr()[...]),
                stored_basis=handle[f"fields/{name}/stored_basis"].asstr()[()],
                values=handle[f"fields/{name}/values"][...],
                unit=handle[f"fields/{name}/unit"].asstr()[()],
            )
            for name in handle["fields"]
        )
        source = RealizedTableSource(
            name="particles",
            particle_id=handle["sources/particles/particle_id"][...],
            release_time_s=handle["sources/particles/release_time_s"][...],
            position_m=handle["sources/particles/position_m"][...],
            velocity_m_s=handle["sources/particles/velocity_m_s"][...],
            charge_number=handle["sources/particles/charge_number"][...],
            mass_kg=handle["sources/particles/mass_kg"][...],
            drag_diameter_m=handle["sources/particles/drag_diameter_m"][...],
            contact_radius_m=np.zeros(287),
            electrostatic_radius_m=handle["sources/particles/electrostatic_radius_m"][...],
            displaced_volume_m3=handle["sources/particles/displaced_volume_m3"][...],
            model_weight=handle["sources/particles/model_weight"][...],
            material_id=handle["sources/particles/material_id"][...],
        )
    provenance = _provenance(sha256(reference), "saved_localization_arrays_point_contact")
    return write(
        path, DataBundle("axisymmetric_rz", provenance, geometry, (layout,), fields, (source,))
    )


def _provenance(source_hash: str, purpose: str) -> str:
    return json.dumps(
        {
            "source_sha256": "sha256:" + source_hash,
            "producer": "common_p1_test_fixture",
            "producer_version": "1",
            "producer_metadata": {"purpose": purpose, "comsol_executed": False},
            "field_semantics_revision": "explicit_common_p1_primitives_v1",
        }
    )
