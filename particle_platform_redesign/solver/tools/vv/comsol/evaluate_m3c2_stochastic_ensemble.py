"""Evaluate M3-C2 R-Z stochastic cohorts without pathwise seed matching.

The input campaign manifest is external V&V data.  It contains two
participants (``comsol`` and ``candidate``), independent seed sets, and one or
more numerical levels per participant.  Every replica points to a normalized
trajectory and terminal-event CSV.  Four-seed pilot campaigns screen numerical
configuration only; final campaigns require a passing pilot report, a
post-pilot authorization receipt, and disjoint registered seeds.

Revision 4 adds a finite-sample full-population R-Z/fate distribution gate for
campaigns whose terminal curves alone are uninformative.  Active particles are
placed in fixed geometry bins and terminal particles in one of three fate
categories, so every source-particle trajectory contributes exactly once at
every output time.  The terminal and distribution families split one
pre-registered familywise error budget.  The four-seed pilot remains a
noninferential numerical-configuration screen.  Continuous summaries are
descriptive.  The tool intentionally makes no pathwise comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from tools.vv.comsol.assemble_m3c2_campaign import _participant_manifest_projection

TOOL_REVISION: Final = "m3c2_stochastic_ensemble_evaluator_v4"
TOOL_REVISION_V5: Final = "m3c2_stochastic_ensemble_evaluator_v5"
PARTICIPANTS: Final = ("comsol", "candidate")
FATES: Final = ("active", "stuck", "held", "escaped")
TERMINAL_CURVES: Final = ("stuck", "held", "escaped", "any_terminal")
TRAJECTORY_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
EVENT_COLUMNS: Final = ("particle_id", "event_time_s", "outcome", "boundary_semantic")
POSITION_FAMILIES: Final = ("mean_position", "covariance", "quantile")
PATH_SENSITIVITY_PURPOSE: Final = "brownian_first_passage_path_depth_sensitivity"
CAMPAIGN_BINDING_KEYS: Final = {
    "contract_sha256",
    "input_sha256",
    "input_content_hash",
}
CASEP_EVIDENCE_STATUS: Final = "PASS_DOCUMENTED_INDEPENDENT_PARTICLE_STREAMS_AND_ONE_WAY_DYNAMICS"
CASEP_CONTRACT_ID: Final = "M3-C2A-caseP-100nm-stochastic-pilot"
CASEP_SOURCE_CASE_ID: Final = "formal_iondrag_theory_consistent/caseP_100nm"
CASEP_EVALUATION_CASE_ID: Final = "M3-C2A_caseP_100nm_common-P1"
CASEP_FEATURE_TAGS: Final = frozenset(
    {
        "auxq",
        "axi1",
        "bf1",
        "depf",
        "df1",
        "dpcon1",
        "ef1",
        "gf1",
        "idf",
        "lf1",
        "liftfm",
        "outin",
        "outpump",
        "pp1",
        "relg1",
        "thpf1",
        "wall1",
    }
)
CASEP_STUDY_PHYSICS_TAGS: Final = frozenset(
    {
        "ptp",
        "spf",
        "ht",
        "mf",
        "pcv",
        "pce2",
        "pcne",
        "pcte",
        "pcni",
        "pcnm",
        "pcgr",
        "pcgz",
        "pcmi",
        "fpt",
        "esass",
        "fptas",
    }
)
CASEP_STUDY_MULTIPHYSICS_TAGS: Final = frozenset({"nipfptp1", "pccptp1", "ehsptp1"})
CASEP_RNG_REVISION: Final = "philox4x32_10_brownian_interval_tree_v1"
CASEP_PROJECTION_REASON: Final = "final前のscope/hash検証補強"
type Purpose = Literal["pilot", "final"]


@dataclass(frozen=True, slots=True)
class _IndependenceEvidenceBinding:
    source_case_id: str
    evaluation_case_id: str
    contract_sha256: str
    input_sha256: str
    input_content_hash: str


@dataclass(frozen=True, slots=True)
class Policy:
    path: Path
    sha256: str
    confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    quantiles: tuple[float, ...]
    radial_bins: int
    axial_bins: int
    pilot_replicas: int
    pilot_levels: int
    max_stabilization_ratio: float
    final_replicas: int
    margins: dict[str, float]
    numerical_fraction: float
    path_sensitivity_fraction: float
    revision: int = 1
    terminal_alpha: float = 0.05
    terminal_margin: float = 0.05
    rz_distribution_alpha: float | None = None
    rz_distribution_margin: float | None = None
    rz_one_sample_radius: float | None = None
    rz_two_sample_radius: float | None = None
    rz_category_count: int | None = None
    rz_participant_union_count: int | None = None
    rz_screening_margin: float | None = None
    expected_case_id: str | None = None
    expected_particle_count: int | None = None
    expected_output_count: int | None = None
    evidence_manifest_path: Path | None = None
    evidence_manifest_sha256: str | None = None
    evidence_binding: _IndependenceEvidenceBinding | None = None
    final_seed_plan: dict[str, tuple[int, ...]] | None = None


@dataclass(frozen=True, slots=True)
class _FinalPolicyRegistration:
    terminal_alpha: float
    terminal_margin: float
    rz_distribution_alpha: float | None = None
    rz_distribution_margin: float | None = None
    rz_one_sample_radius: float | None = None
    rz_two_sample_radius: float | None = None
    rz_category_count: int | None = None
    rz_participant_union_count: int | None = None
    expected_case_id: str | None = None
    expected_particle_count: int | None = None
    expected_output_count: int | None = None
    evidence_manifest_path: Path | None = None
    evidence_manifest_sha256: str | None = None
    evidence_binding: _IndependenceEvidenceBinding | None = None


@dataclass(frozen=True, slots=True)
class Replica:
    seed: int
    trajectory_path: Path
    event_path: Path
    positions_m: np.ndarray
    lifecycle: np.ndarray
    initial_state: np.ndarray
    first_arrival_s: np.ndarray
    terminal_fate: np.ndarray
    performance: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class Level:
    level_id: str
    ordinal: int
    numerical_setting: dict[str, Any]
    replicas: tuple[Replica, ...]


@dataclass(frozen=True, slots=True)
class Campaign:
    path: Path
    sha256: str
    purpose: Purpose
    particle_ids: np.ndarray
    times_s: np.ndarray
    bounds_m: tuple[float, float, float, float]
    geometry_scale_m: float
    participants: dict[str, tuple[Level, ...]]
    case_id: str | None
    canonical_input: dict[str, str] | None
    participant_manifests: dict[str, dict[str, str]] | None
    campaign_binding: dict[str, str] | None
    evaluation_policy_sha256: str | None


@dataclass(frozen=True, slots=True)
class SeedMetrics:
    position_m: np.ndarray
    position_displacement_scaled: np.ndarray
    occupancy: np.ndarray
    fate: np.ndarray
    first_arrival: np.ndarray


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[Any], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error


def _required(mapping: dict[str, Any], key: str, name: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{name}.{key} is required")
    return mapping[key]


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _screening_margins(value: object) -> dict[str, float]:
    margins = {
        str(name): _positive_float(item, f"margin {name}")
        for name, item in _mapping(value, "margins").items()
    }
    expected = {
        "mean_position",
        "covariance",
        "quantile",
        "occupancy_tv",
        "fate_probability",
        "first_arrival_cdf",
    }
    if set(margins) != expected:
        raise ValueError("screening margins are incomplete or unexpected")
    return margins


def _final_seed_plan(value: object, replicas: int) -> dict[str, tuple[int, ...]] | None:
    if value is None:
        return None
    raw = _mapping(value, "final.seed_plan")
    if set(raw) != set(PARTICIPANTS):
        raise ValueError("final seed plan must contain exactly COMSOL and candidate")
    result: dict[str, tuple[int, ...]] = {}
    for participant in PARTICIPANTS:
        seeds = tuple(int(item) for item in _sequence(raw[participant], participant))
        if len(seeds) != replicas or len(set(seeds)) != replicas or any(seed < 0 for seed in seeds):
            raise ValueError(f"final seed plan for {participant} is invalid")
        result[participant] = seeds
    if set(result["comsol"]) & set(result["candidate"]):
        raise ValueError("final participant seed plans must be disjoint")
    return result


def _json_pointer(document: object, pointer: str, name: str) -> object:
    if not pointer.startswith("/"):
        raise ValueError(f"{name} must be an absolute JSON pointer")
    current = document
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"{name} does not resolve")
    return current


def _registered_seed_plan(
    final: dict[str, Any], replicas: int, policy_root: Path, revision: int
) -> dict[str, tuple[int, ...]] | None:
    if revision < 4:
        return _final_seed_plan(final.get("seed_plan"), replicas)
    reference = _mapping(final.get("seed_allocation"), "final.seed_allocation")
    path = _resolve_artifact(reference, policy_root, "final seed allocation")
    allocation = _json(path, "M3-C2 final seed allocation")
    if (
        allocation.get("schema_version"),
        allocation.get("allocation_kind"),
        allocation.get("replicas_per_participant"),
    ) != (1, "m3c2_final_seed_allocation", replicas):
        raise ValueError("final seed allocation identity or cohort size differs")
    pointers = _mapping(reference.get("json_pointers"), "final seed allocation pointers")
    if set(pointers) != set(PARTICIPANTS):
        raise ValueError("final seed allocation pointers must name both participants")
    selected = {
        participant: _json_pointer(
            allocation,
            str(pointers[participant]),
            f"final seed allocation pointer {participant}",
        )
        for participant in PARTICIPANTS
    }
    return _final_seed_plan(selected, replicas)


def _load_registered_final_policy(
    raw: dict[str, Any], final: dict[str, Any], policy_root: Path, revision: int
) -> _FinalPolicyRegistration:
    terminal = _mapping(final.get("terminal_population_gate"), "terminal_population_gate")
    if tuple(_sequence(terminal.get("curves"), "terminal curves")) != TERMINAL_CURVES:
        raise ValueError("terminal population curves must be the registered nonredundant four")
    if terminal.get("method") != "two_sample_union_hoeffding":
        raise ValueError("terminal population gate method is unsupported")
    design = _mapping(final.get("fixed_design"), "final.fixed_design")
    evidence_path, evidence_sha256, evidence_binding = _load_independence_evidence(
        raw.get("independence_evidence"), policy_root, revision
    )
    registration = _FinalPolicyRegistration(
        terminal_alpha=_positive_float(terminal.get("familywise_alpha"), "familywise alpha"),
        terminal_margin=_positive_float(terminal.get("margin"), "terminal margin"),
        expected_case_id=str(design.get("case_id", "")).strip() or None,
        expected_particle_count=int(design.get("particle_count", 0)),
        expected_output_count=int(design.get("output_time_count", 0)),
        evidence_manifest_path=evidence_path,
        evidence_manifest_sha256=evidence_sha256,
        evidence_binding=evidence_binding,
    )
    if revision < 4:
        return registration
    _validate_v4_terminal_registration(final, terminal, registration)
    return _load_v4_distribution_registration(raw, final, registration)


def _validate_v4_terminal_registration(
    final: dict[str, Any],
    terminal: dict[str, Any],
    registration: _FinalPolicyRegistration,
) -> None:
    fixed_design = _mapping(final.get("fixed_design"), "final.fixed_design")
    terminal_observations = int(final.get("replicas_per_participant", 0)) * int(
        fixed_design["particle_count"]
    )
    terminal_endpoints = int(fixed_design["output_time_count"]) * len(TERMINAL_CURVES)
    expected_terminal_radius = math.sqrt(
        math.log(2.0 * terminal_endpoints / registration.terminal_alpha) / terminal_observations
    )
    registered_terminal_radius = _positive_float(
        terminal.get("registered_critical_radius"), "terminal registered critical radius"
    )
    if (
        int(terminal.get("simultaneous_endpoint_count", 0)) != terminal_endpoints
        or int(terminal.get("observations_per_participant", 0)) != terminal_observations
        or not math.isclose(
            registered_terminal_radius,
            expected_terminal_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(terminal.get("maximum_point_difference_that_can_pass", math.nan)),
            registration.terminal_margin - registered_terminal_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("v4 registered terminal radius is inconsistent")


def _v4_familywise_alpha(final: dict[str, Any], terminal_alpha: float, rz_alpha: float) -> None:
    familywise = _mapping(final.get("familywise_error_control"), "familywise error control")
    allocation = _mapping(familywise.get("allocation"), "familywise allocation")
    expected_allocation = {"terminal_population": terminal_alpha, "rz_distribution": rz_alpha}
    observed_allocation = {name: float(value) for name, value in allocation.items()}
    overall = float(familywise.get("overall_alpha", math.nan))
    if familywise.get("decision_rule") != "all_registered_gates_must_pass":
        raise ValueError("v4 familywise decision rule is inconsistent")
    if observed_allocation != expected_allocation or not math.isclose(
        overall, terminal_alpha + rz_alpha, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError("v4 familywise error allocation is inconsistent")


def _v4_distribution_definition(
    raw: dict[str, Any], final: dict[str, Any], rz_gate: dict[str, Any]
) -> tuple[int, int, int, int]:
    fixed_design = _mapping(final.get("fixed_design"), "final.fixed_design")
    occupancy = _mapping(raw.get("occupancy"), "occupancy")
    active_bins = _mapping(rz_gate.get("active_position_bins"), "R-Z active position bins")
    category_count = int(rz_gate.get("category_count", 0))
    radial_bins = int(active_bins.get("radial_bins", 0))
    axial_bins = int(active_bins.get("axial_bins", 0))
    time_count = int(rz_gate.get("output_time_count", 0))
    participant_count = int(rz_gate.get("participant_union_count", 0))
    observed = {
        "method": rz_gate.get("method"),
        "observable": rz_gate.get("observable"),
        "independent_unit": rz_gate.get("independent_unit"),
        "within_unit_dependence": rz_gate.get("within_unit_dependence"),
        "bounds_authority": active_bins.get("bounds_authority"),
        "radial_bins": radial_bins,
        "axial_bins": axial_bins,
        "terminal_categories": tuple(
            _sequence(rz_gate.get("terminal_categories"), "terminal categories")
        ),
        "category_count": category_count,
        "output_time_count": time_count,
        "participant_union_count": participant_count,
    }
    expected = {
        "method": "two_sample_union_multinomial_l1",
        "observable": "full_population_fixed_geometry_rz_fate_partition",
        "independent_unit": "seed_x_fixed_source_particle_trajectory",
        "within_unit_dependence": "time_dependence_retained",
        "bounds_authority": "campaign_scope_geometry_bounds_m_fixed_before_outcomes",
        "radial_bins": int(occupancy.get("radial_bins", 0)),
        "axial_bins": int(occupancy.get("axial_bins", 0)),
        "terminal_categories": FATES[1:],
        "category_count": radial_bins * axial_bins + len(FATES) - 1,
        "output_time_count": int(fixed_design["output_time_count"]),
        "participant_union_count": len(PARTICIPANTS),
    }
    if observed != expected:
        raise ValueError("v4 R-Z distribution gate definition is inconsistent")
    observations = int(rz_gate.get("observations_per_participant_per_time", 0))
    expected_observations = int(final.get("replicas_per_participant", 0)) * int(
        fixed_design["particle_count"]
    )
    if observations != expected_observations:
        raise ValueError("v4 R-Z distribution observation count differs from fixed design")
    return category_count, time_count, participant_count, observations


def _v4_distribution_radii(
    rz_gate: dict[str, Any],
    category_count: int,
    time_count: int,
    participant_count: int,
    observations: int,
) -> tuple[float, float, float]:
    alpha = _positive_float(rz_gate.get("familywise_alpha"), "R-Z familywise alpha")
    expected_one_sample_radius = math.sqrt(
        (category_count * math.log(2.0) + math.log(participant_count * time_count / alpha))
        / (2.0 * observations)
    )
    registered_one_sample_radius = _positive_float(
        rz_gate.get("one_sample_critical_radius"), "R-Z one-sample radius"
    )
    registered_two_sample_radius = _positive_float(
        rz_gate.get("two_sample_critical_radius"), "R-Z two-sample radius"
    )
    margin = _positive_float(rz_gate.get("margin_total_variation"), "R-Z TV margin")
    if (
        rz_gate.get("one_sample_critical_radius_formula") != "sqrt((K*ln(2)+ln(2*T/alpha))/(2*N))"
        or not math.isclose(
            registered_one_sample_radius,
            expected_one_sample_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            registered_two_sample_radius,
            2.0 * registered_one_sample_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(rz_gate.get("maximum_empirical_tv_that_can_pass", math.nan)),
            margin - registered_two_sample_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("v4 registered R-Z distribution radius is inconsistent")
    return registered_one_sample_radius, registered_two_sample_radius, margin


def _load_v4_distribution_registration(
    raw: dict[str, Any],
    final: dict[str, Any],
    registration: _FinalPolicyRegistration,
) -> _FinalPolicyRegistration:
    rz_gate = _mapping(final.get("rz_distribution_gate"), "R-Z distribution gate")
    alpha = _positive_float(rz_gate.get("familywise_alpha"), "R-Z familywise alpha")
    _v4_familywise_alpha(final, registration.terminal_alpha, alpha)
    category_count, time_count, participant_count, observations = _v4_distribution_definition(
        raw, final, rz_gate
    )
    one_sample, two_sample, margin = _v4_distribution_radii(
        rz_gate, category_count, time_count, participant_count, observations
    )
    return _FinalPolicyRegistration(
        terminal_alpha=registration.terminal_alpha,
        terminal_margin=registration.terminal_margin,
        rz_distribution_alpha=alpha,
        rz_distribution_margin=margin,
        rz_one_sample_radius=one_sample,
        rz_two_sample_radius=two_sample,
        rz_category_count=category_count,
        rz_participant_union_count=participant_count,
        expected_case_id=registration.expected_case_id,
        expected_particle_count=registration.expected_particle_count,
        expected_output_count=registration.expected_output_count,
        evidence_manifest_path=registration.evidence_manifest_path,
        evidence_manifest_sha256=registration.evidence_manifest_sha256,
        evidence_binding=registration.evidence_binding,
    )


def _validate_v3_policy(
    pilot: dict[str, Any],
    terminal_margin: float,
    *,
    revision: int,
    rz_distribution_margin: float | None,
    active_position_bins: int,
    rz_category_count: int | None,
) -> float | None:
    expected_selection = (
        "direct_each_macro_level_to_registered_finest_population_summary"
        if revision >= 4
        else "direct_each_macro_level_to_registered_finest_population_terminal_point_summary"
    )
    if pilot.get("selection_method") != expected_selection:
        raise ValueError("pilot selection method is unsupported")
    expected_observables = TERMINAL_CURVES + (
        ("full_population_rz_fate_partition_tv",) if revision >= 4 else ()
    )
    if tuple(_sequence(pilot.get("gating_observables"), "pilot gating observables")) != (
        expected_observables
    ):
        raise ValueError("pilot gating observables differ from the registered policy")
    if pilot.get("registered_finest_reference") != "last_macro_level_by_ordinal":
        raise ValueError("v3 pilot finest-reference rule is unsupported")
    if pilot.get("candidate_path_reference") != "registered_finest_macro_level":
        raise ValueError("v3 candidate path-reference rule is unsupported")
    declared_screening_margin = _positive_float(
        pilot.get("terminal_screening_margin"), "v3 terminal screening margin"
    )
    expected_screening_margin = terminal_margin * float(
        pilot.get("numerical_margin_fraction", math.nan)
    )
    if declared_screening_margin != expected_screening_margin:
        raise ValueError("v3 terminal screening margin differs from final margin fraction")
    if revision < 4:
        return None
    if rz_distribution_margin is None:
        raise ValueError("v4 policy has no R-Z distribution margin")
    partition = _mapping(pilot.get("full_population_partition"), "pilot full-population partition")
    if partition != {
        "active_position_bins": active_position_bins,
        "terminal_categories": list(FATES[1:]),
        "category_count": rz_category_count,
        "conditioning": "none_every_fixed_source_trajectory_has_exactly_one_category_per_time",
    }:
        raise ValueError("v4 pilot full-population partition differs from the final observable")
    rz_screening_margin = _positive_float(
        pilot.get("rz_distribution_screening_margin"), "v4 R-Z screening margin"
    )
    if rz_screening_margin != rz_distribution_margin * float(
        pilot.get("numerical_margin_fraction", math.nan)
    ):
        raise ValueError("v4 R-Z screening margin differs from final margin fraction")
    return rz_screening_margin


def _load_policy(path: Path) -> Policy:
    raw = _json(path, "M3-C2 evaluation policy")
    if (raw.get("schema_version"), raw.get("policy_kind")) != (
        1,
        "m3c2_stochastic_ensemble",
    ):
        raise ValueError("unexpected M3-C2 evaluation-policy identity")
    bootstrap = _mapping(raw.get("bootstrap"), "bootstrap")
    occupancy = _mapping(raw.get("occupancy"), "occupancy")
    pilot = _mapping(raw.get("pilot"), "pilot")
    final = _mapping(raw.get("final"), "final")
    revision = int(raw.get("policy_revision", 1))
    if revision not in {1, 2, 3, 4, 5}:
        raise ValueError("M3-C2 evaluation policy revision must be 1, 2, 3, 4, or 5")
    margins = _screening_margins(
        final.get("equivalence_margins") if revision == 1 else pilot.get("screening_margins")
    )
    confidence = float(bootstrap.get("confidence", math.nan))
    resamples = int(bootstrap.get("resamples", 0))
    if not 0.5 < confidence < 1.0 or resamples < 100:
        raise ValueError("bootstrap confidence/resamples are invalid")
    quantiles = tuple(float(value) for value in _sequence(raw.get("quantiles"), "quantiles"))
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("quantiles must lie strictly inside (0, 1)")
    registration = _FinalPolicyRegistration(
        terminal_alpha=1.0 - confidence,
        terminal_margin=margins["fate_probability"],
    )
    if revision >= 2:
        registration = _load_registered_final_policy(raw, final, path.parent, revision)
    rz_screening_margin = None
    if revision >= 3:
        rz_screening_margin = _validate_v3_policy(
            pilot,
            registration.terminal_margin,
            revision=revision,
            rz_distribution_margin=registration.rz_distribution_margin,
            active_position_bins=int(occupancy.get("radial_bins", 0))
            * int(occupancy.get("axial_bins", 0)),
            rz_category_count=registration.rz_category_count,
        )
    policy = Policy(
        path=path,
        sha256=_sha256(path),
        confidence=confidence,
        bootstrap_resamples=resamples,
        bootstrap_seed=int(_required(bootstrap, "seed", "bootstrap")),
        quantiles=quantiles,
        radial_bins=int(occupancy.get("radial_bins", 0)),
        axial_bins=int(occupancy.get("axial_bins", 0)),
        pilot_replicas=int(pilot.get("replicas_per_participant", 0)),
        pilot_levels=int(pilot.get("minimum_levels", 0)),
        max_stabilization_ratio=float(pilot.get("maximum_fine_to_coarse_ratio", math.nan)),
        final_replicas=int(final.get("replicas_per_participant", 0)),
        margins=margins,
        numerical_fraction=float(pilot.get("numerical_margin_fraction", math.nan)),
        path_sensitivity_fraction=float(pilot.get("path_sensitivity_margin_fraction", math.nan)),
        revision=revision,
        terminal_alpha=registration.terminal_alpha,
        terminal_margin=registration.terminal_margin,
        rz_distribution_alpha=registration.rz_distribution_alpha,
        rz_distribution_margin=registration.rz_distribution_margin,
        rz_one_sample_radius=registration.rz_one_sample_radius,
        rz_two_sample_radius=registration.rz_two_sample_radius,
        rz_category_count=registration.rz_category_count,
        rz_participant_union_count=registration.rz_participant_union_count,
        rz_screening_margin=rz_screening_margin,
        expected_case_id=registration.expected_case_id,
        expected_particle_count=registration.expected_particle_count,
        expected_output_count=registration.expected_output_count,
        evidence_manifest_path=registration.evidence_manifest_path,
        evidence_manifest_sha256=registration.evidence_manifest_sha256,
        evidence_binding=registration.evidence_binding,
        final_seed_plan=_registered_seed_plan(
            final,
            int(final.get("replicas_per_participant", 0)),
            path.parent,
            revision,
        ),
    )
    _validate_policy_values(policy)
    return policy


def _validate_policy_values(policy: Policy) -> None:
    if policy.radial_bins < 2 or policy.axial_bins < 2:
        raise ValueError("occupancy requires at least two bins per axis")
    if policy.pilot_replicas < 2 or policy.pilot_levels < 3 or policy.final_replicas < 2:
        raise ValueError("pilot/final cohort sizes are too small")
    if not 0.0 < policy.numerical_fraction < 1.0:
        raise ValueError("numerical_margin_fraction must lie inside (0, 1)")
    if not 0.0 < policy.path_sensitivity_fraction < 1.0:
        raise ValueError("path_sensitivity_margin_fraction must lie inside (0, 1)")
    if not 0.0 < policy.max_stabilization_ratio <= 1.0:
        raise ValueError("maximum_fine_to_coarse_ratio must lie inside (0, 1]")
    if policy.revision >= 2:
        _validate_registered_policy_values(policy)


def _validate_registered_policy_values(policy: Policy) -> None:
    if not 0.0 < policy.terminal_alpha < 0.5:
        raise ValueError("familywise alpha must lie inside (0, 0.5)")
    if not 0.0 < policy.terminal_margin <= 1.0:
        raise ValueError("terminal population margin must lie inside (0, 1]")
    fixed_design_is_valid = all(
        (
            bool(policy.expected_case_id),
            (policy.expected_particle_count or 0) > 0,
            (policy.expected_output_count or 0) >= 2,
        )
    )
    if not fixed_design_is_valid:
        raise ValueError("v2 fixed-design particle/time counts are invalid")
    if policy.final_seed_plan is None:
        raise ValueError("v2 final policy must register disjoint participant seed plans")
    if policy.revision >= 4:
        if not all(
            (
                policy.rz_distribution_alpha is not None,
                policy.rz_distribution_margin is not None,
                policy.rz_one_sample_radius is not None,
                policy.rz_two_sample_radius is not None,
                policy.rz_category_count == policy.radial_bins * policy.axial_bins + 3,
                policy.rz_participant_union_count == len(PARTICIPANTS),
                policy.rz_screening_margin is not None,
            )
        ):
            raise ValueError("v4 R-Z distribution policy is incomplete")


def _resolve_artifact(record: object, root: Path, name: str) -> Path:
    artifact = _mapping(record, name)
    path = (root / str(artifact.get("path"))).resolve()
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    expected = str(artifact.get("sha256", "")).lower()
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"{name} SHA-256 differs")
    return path


def _evidence_artifact(record: object, root: Path, name: str) -> Path:
    artifact = _mapping(record, name)
    if set(artifact) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} must contain exactly path, bytes, and sha256")
    relative = Path(str(artifact["path"]))
    path = relative if relative.is_absolute() else root / relative
    expected = str(artifact["sha256"]).lower()
    if (
        not path.is_file()
        or path.stat().st_size != int(artifact["bytes"])
        or len(expected) != 64
        or _sha256(path) != expected
    ):
        raise ValueError(f"{name} identity differs: {path}")
    return path


def _casep_evidence_artifacts(raw: dict[str, Any], root: Path) -> dict[str, Path]:
    records = _mapping(raw.get("locked_artifacts"), "locked artifacts")
    expected = {
        "contract",
        "source_model",
        "common_p1_input",
        "java_runner",
        "powershell_launcher",
        "feature_exporter",
        "feature_inventory",
        "study_inventory",
        "candidate_rng",
        "candidate_rng_test",
    }
    if set(records) != expected:
        raise ValueError("Case-P RNG evidence locked-artifact inventory differs")
    artifacts = {
        name: _evidence_artifact(records[name], root, f"Case-P evidence {name}")
        for name in sorted(expected)
    }
    documentation = _sequence(raw.get("comsol_rng_documentation"), "COMSOL RNG documentation")
    if len(documentation) != 4:
        raise ValueError("Case-P evidence must lock the four installed COMSOL RNG documents")
    for index, value in enumerate(documentation):
        item = _mapping(value, f"COMSOL RNG documentation {index}")
        if set(item) != {"artifact", "anchors", "claim"}:
            raise ValueError("COMSOL RNG documentation record has unexpected keys")
        _evidence_artifact(item["artifact"], root, f"COMSOL RNG documentation {index}")
        if not _sequence(item["anchors"], f"COMSOL RNG documentation {index} anchors"):
            raise ValueError("COMSOL RNG documentation anchors are empty")
    return artifacts


def _casep_scope(raw: dict[str, Any]) -> dict[str, Any]:
    scope = _mapping(raw.get("casep_scope"), "Case-P evidence scope")
    expected = {
        "contract_id": CASEP_CONTRACT_ID,
        "source_case_id": CASEP_SOURCE_CASE_ID,
        "evaluation_case_id": CASEP_EVALUATION_CASE_ID,
        "workflow": "caseP",
        "physics_tag": "fpt",
        "particle_count": 287,
        "output_count": 121,
    }
    if scope != expected:
        raise ValueError("Case-P RNG evidence scope differs from the registered design")
    return scope


def _casep_binding(raw: dict[str, Any]) -> _IndependenceEvidenceBinding:
    binding = _mapping(raw.get("campaign_binding"), "Case-P evidence campaign binding")
    expected_keys = {
        "contract_sha256",
        "input_sha256",
        "input_content_hash",
        "source_model_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError("Case-P evidence campaign binding has unexpected keys")
    for key in ("contract_sha256", "input_sha256", "source_model_sha256"):
        _validate_sha256_digest(str(binding[key]), f"Case-P evidence {key}")
    _validate_content_hash(str(binding["input_content_hash"]), "Case-P evidence input hash")
    return _IndependenceEvidenceBinding(
        source_case_id=CASEP_SOURCE_CASE_ID,
        evaluation_case_id=CASEP_EVALUATION_CASE_ID,
        contract_sha256=str(binding["contract_sha256"]),
        input_sha256=str(binding["input_sha256"]),
        input_content_hash=str(binding["input_content_hash"]),
    )


def _casep_contract_semantics(
    contract_path: Path,
    artifacts: dict[str, Path],
    binding: _IndependenceEvidenceBinding,
) -> None:
    contract = _json(contract_path, "Case-P stochastic contract")
    campaign = _mapping(contract.get("campaign"), "Case-P contract campaign")
    scope = _mapping(contract.get("scope"), "Case-P contract scope")
    source = _mapping(contract.get("source_model"), "Case-P contract source model")
    common = _mapping(contract.get("common_p1_input"), "Case-P contract common input")
    comsol = _mapping(
        _mapping(contract.get("stochastic_physics"), "Case-P stochastic physics").get("comsol"),
        "Case-P COMSOL stochastic physics",
    )
    candidate = _mapping(
        _mapping(contract.get("stochastic_physics"), "Case-P stochastic physics").get("candidate"),
        "Case-P candidate stochastic physics",
    )
    checks = (
        contract.get("contract_id") == CASEP_CONTRACT_ID,
        campaign.get("case_id") == binding.source_case_id,
        campaign.get("evaluation_case_id") == binding.evaluation_case_id,
        scope.get("workflow") == "caseP",
        scope.get("particle_count") == 287,
        scope.get("output_count") == 121,
        source.get("sha256") == _sha256(artifacts["source_model"]),
        common.get("file_sha256") == binding.input_sha256,
        common.get("content_hash") == binding.input_content_hash,
        _sha256(artifacts["common_p1_input"]) == binding.input_sha256,
        comsol.get("physics_tag") == "fpt",
        comsol.get("feature_tag") == "bf1",
        comsol.get("random_number_args") == "UserDefined",
        comsol.get("seed_parameter") == "P_brownian_seed",
        comsol.get("sole_seed_authority") == "fpt.bf1.i",
        comsol.get("out_of_plane_degrees_of_freedom") is False,
        candidate.get("seed_authority") == "case_random_seed",
    )
    if not all(checks):
        raise ValueError("Case-P stochastic contract differs from the RNG evidence binding")


def _require_source_tokens(path: Path, tokens: tuple[str, ...], name: str) -> None:
    text = path.read_text(encoding="utf-8")
    if any(token not in text for token in tokens):
        raise ValueError(f"{name} no longer contains its registered semantic checks")


def _casep_source_semantics(artifacts: dict[str, Path]) -> None:
    _require_source_tokens(
        artifacts["java_runner"],
        (
            'physics.prop("RandomNumberArgs").set("RandomNumberArgs", "UserDefined")',
            'brownian.set("i", RunM3C2StochasticRequest.seedParameter())',
            'time.setSolveFor("/physics/" + PHYSICS, true)',
            'step.setSolveFor("/multiphysics/" + tag, false)',
            "ModelUtil.loadCopy(",
            'String thermoTag = "m3c2HeatFlux"',
        ),
        "Case-P Java runner",
    )
    _require_source_tokens(
        artifacts["powershell_launcher"],
        (
            "Get-FileHash -Algorithm SHA256",
            '"-nosave"',
            '"-error", "on"',
            '"-np", "1"',
            "RunM3C2StochasticCampaign.java",
        ),
        "Case-P PowerShell launcher",
    )
    _require_source_tokens(
        artifacts["feature_exporter"],
        (
            "for(String ft:ph.feature().tags())",
            "for(String ft:m.study(st).feature().tags())",
            "String[]pr=f.properties()",
            "Arrays.sort(pr)",
        ),
        "Case-P exhaustive inventory exporter",
    )
    _require_source_tokens(
        artifacts["candidate_rng"],
        (
            f'BROWNIAN_RNG_REVISION = "{CASEP_RNG_REVISION}"',
            'particles = _nonnegative_uint64_vector(particle_id, "particle IDs")',
            'macro_low, macro_high = _split_uint64(macro_interval, "macro interval")',
            'root_low, root_high = _split_uint64(root_interval, "Brownian root interval")',
            "stream ^ component_word",
            "particle & mask32",
            "particle >> np.uint64(32)",
        ),
        "candidate RNG implementation",
    )
    _require_source_tokens(
        artifacts["candidate_rng_test"],
        (
            "test_brownian_normals_have_stable_interval_tree_identity",
            "particles[::-1]",
            "assert all(not np.array_equal(root, variant) for variant in variants)",
        ),
        "candidate RNG verification",
    )


def _csv_rows(path: Path, expected_header: tuple[str, ...], name: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError(f"{name} header differs")
        return [dict(row) for row in reader]


def _csv_row_count_with_required_columns(
    path: Path, required_columns: tuple[str, ...], name: str
) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if len(fieldnames) != len(set(fieldnames)) or set(required_columns).difference(fieldnames):
            raise ValueError(f"{name} columns are incomplete or duplicated")
        return sum(1 for _row in reader)


def _study_pairs(value: str, name: str) -> dict[str, str]:
    stripped = value.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError(f"{name} is not a bracketed solve-for map")
    values = [item.strip() for item in stripped[1:-1].split(",")]
    if not values or len(values) % 2:
        raise ValueError(f"{name} is not a tag/state sequence")
    result = dict(zip(values[::2], values[1::2], strict=True))
    if len(result) * 2 != len(values) or set(result.values()) - {"on", "off"}:
        raise ValueError(f"{name} contains duplicate tags or invalid states")
    return result


def _casep_study_authority(study_rows: list[dict[str, str]]) -> dict[str, str]:
    selected = {
        row["property"]: row["value"]
        for row in study_rows
        if row["study_tag"] == "stdP100"
        and row["feature_tag"] == "time"
        and row["property"] in {"activate", "activateCoupling"}
    }
    if set(selected) != {"activate", "activateCoupling"}:
        raise ValueError("Case-P study inventory lacks solve-for authority rows")
    return selected


def _validate_casep_study_isolation(selected: dict[str, str]) -> None:
    physics = _study_pairs(selected["activate"], "Case-P study physics activation")
    couplings = _study_pairs(selected["activateCoupling"], "Case-P study multiphysics activation")
    active_casep_physics = {tag for tag in CASEP_STUDY_PHYSICS_TAGS if physics.get(tag) == "on"}
    if (
        not CASEP_STUDY_PHYSICS_TAGS.issubset(physics)
        or active_casep_physics != {"fpt"}
        or set(couplings) != CASEP_STUDY_MULTIPHYSICS_TAGS
        or set(couplings.values()) != {"off"}
    ):
        raise ValueError("Case-P study inventory is not particle-only")


def _casep_inventory_semantics(artifacts: dict[str, Path]) -> None:
    feature_rows = _csv_rows(
        artifacts["feature_inventory"],
        (
            "physics_tag",
            "physics_label",
            "feature_tag",
            "feature_label",
            "selected_entities",
            "property",
            "value",
        ),
        "Case-P particle feature inventory",
    )
    if (
        not feature_rows
        or {row["physics_tag"] for row in feature_rows} != {"fpt"}
        or {row["feature_tag"] for row in feature_rows} != CASEP_FEATURE_TAGS
    ):
        raise ValueError("Case-P particle feature inventory allowlist differs")
    study_rows = _csv_rows(
        artifacts["study_inventory"],
        ("study_tag", "study_label", "feature_tag", "feature_label", "property", "value"),
        "Case-P study inventory",
    )
    _validate_casep_study_isolation(_casep_study_authority(study_rows))


def _validate_casep_evidence_manifest(
    raw: dict[str, Any], manifest_path: Path
) -> _IndependenceEvidenceBinding:
    if (
        raw.get("schema_version"),
        raw.get("evidence_revision"),
        raw.get("manifest_kind"),
        raw.get("status"),
    ) != (2, 2, "m3c2_rng_noninteraction_evidence", CASEP_EVIDENCE_STATUS):
        raise ValueError("Case-P RNG/noninteraction evidence v2 is not accepted")
    _casep_scope(raw)
    binding = _casep_binding(raw)
    artifacts = _casep_evidence_artifacts(raw, _repository_root())
    if _sha256(artifacts["contract"]) != binding.contract_sha256:
        raise ValueError("Case-P evidence contract hash differs from its campaign binding")
    campaign_binding = _mapping(raw["campaign_binding"], "Case-P campaign binding")
    if _sha256(artifacts["source_model"]) != campaign_binding["source_model_sha256"]:
        raise ValueError("Case-P evidence source-model hash differs from its binding")
    _casep_contract_semantics(artifacts["contract"], artifacts, binding)
    _casep_source_semantics(artifacts)
    _casep_inventory_semantics(artifacts)
    assumptions = _mapping(raw.get("statistical_assumptions"), "statistical assumptions")
    expected_assumptions = {
        "independent_unit": "seed_x_fixed_source_particle_trajectory",
        "fixed_source_design": True,
        "particle_particle_interaction": False,
        "two_way_background_coupling": False,
        "within_trajectory_time_dependence": "retained_and_covered_by_simultaneous_union_bound",
        "cross_solver_pathwise_rng_equality": False,
    }
    if any(assumptions.get(key) != value for key, value in expected_assumptions.items()):
        raise ValueError("Case-P RNG evidence statistical assumptions differ")
    if manifest_path.parent.name != "rng_noninteraction_v2":
        raise ValueError("Case-P RNG evidence v2 is outside its versioned directory")
    return binding


def _load_independence_evidence(
    record: object, root: Path, revision: int
) -> tuple[Path, str, _IndependenceEvidenceBinding | None]:
    path = _resolve_artifact(record, root, "independence evidence")
    raw = _json(path, "M3-C2 RNG/noninteraction evidence")
    if revision >= 5:
        binding = _validate_casep_evidence_manifest(raw, path)
        return path, _sha256(path), binding
    if (
        raw.get("schema_version"),
        raw.get("manifest_kind"),
        raw.get("status"),
    ) != (
        1,
        "m3c2_rng_noninteraction_evidence",
        "PASS_DOCUMENTED_INDEPENDENT_PARTICLE_STREAMS_AND_ONE_WAY_DYNAMICS",
    ):
        raise ValueError("M3-C2 RNG/noninteraction evidence is not accepted")
    assumptions = _mapping(raw.get("statistical_assumptions"), "statistical assumptions")
    if assumptions.get("independent_unit") != "seed_x_fixed_source_particle_trajectory":
        raise ValueError("M3-C2 evidence does not support the registered independent unit")
    return path, _sha256(path), None


def _scope_artifact(
    record: object,
    root: Path,
    name: str,
    *,
    require_content_hash: bool = False,
) -> dict[str, str]:
    raw = _mapping(record, name)
    _resolve_artifact(raw, root, name)
    result = {
        "path": Path(str(raw.get("path"))).as_posix(),
        "sha256": str(raw.get("sha256")).lower(),
    }
    if require_content_hash:
        content_hash = str(raw.get("content_hash", ""))
        digest = content_hash.removeprefix("sha256:")
        if (
            not content_hash.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"{name}.content_hash is invalid")
        result["content_hash"] = content_hash
    return result


def _optional_float(value: str, source: Path, line: int, column: str) -> float:
    stripped = value.strip()
    if not stripped:
        return math.nan
    try:
        number = float(stripped)
    except ValueError as error:
        raise ValueError(f"{source}:{line}: invalid {column}") from error
    if math.isnan(number):
        return math.nan
    if not math.isfinite(number):
        raise ValueError(f"{source}:{line}: nonfinite {column}")
    return number


def _particle_index(particle_ids: np.ndarray) -> dict[int, int]:
    return {int(particle_id): index for index, particle_id in enumerate(particle_ids)}


def _match_time(value: str, times_s: np.ndarray, source: Path, line: int) -> int:
    try:
        time_s = float(value)
    except ValueError as error:
        raise ValueError(f"{source}:{line}: invalid time_s") from error
    insertion = int(np.searchsorted(times_s, time_s))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times_s)]
    closest = min(candidates, key=lambda index: abs(float(times_s[index]) - time_s))
    tolerance = max(2.0e-14, 64.0 * abs(float(np.spacing(times_s[closest]))))
    if abs(float(times_s[closest]) - time_s) > tolerance:
        raise ValueError(f"{source}:{line}: unexpected output time")
    return closest


def _read_trajectory(
    path: Path, particle_ids: np.ndarray, times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(times_s), len(particle_ids))
    positions = np.full((*shape, 2), np.nan, dtype=np.float64)
    lifecycle = np.full(shape, "", dtype="<U8")
    initial_state = np.full((len(particle_ids), 5), np.nan, dtype=np.float64)
    particles = _particle_index(particle_ids)
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(TRAJECTORY_COLUMNS).difference(reader.fieldnames or ()):
            raise ValueError(f"{path}: trajectory columns are incomplete")
        for line, row in enumerate(reader, start=2):
            try:
                particle = particles[int(row["particle_id"])]
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line}: unexpected particle/time key") from error
            time = _match_time(row["time_s"], times_s, path, line)
            key = (time, particle)
            if key in seen:
                raise ValueError(f"{path}:{line}: duplicate particle/time key")
            seen.add(key)
            state = row["lifecycle"].strip()
            if state not in FATES:
                raise ValueError(f"{path}:{line}: unsupported lifecycle {state!r}")
            values = [
                _optional_float(row[column], path, line, column)
                for column in TRAJECTORY_COLUMNS[2:7]
            ]
            observed = np.isfinite(values)
            if state == "escaped":
                if np.any(observed):
                    raise ValueError(f"{path}:{line}: escaped suffix must remain unobserved")
            elif not np.all(observed):
                raise ValueError(f"{path}:{line}: observed state contains missing values")
            positions[time, particle] = values[:2]
            lifecycle[time, particle] = state
            if time == 0:
                initial_state[particle] = values
    if np.any(lifecycle[0] != "active"):
        raise ValueError(f"{path}: every particle must start active")
    if not np.isfinite(initial_state).all():
        raise ValueError(f"{path}: every particle must have one finite initial source state")
    return positions, lifecycle, initial_state


def _validate_lifecycle(path: Path, lifecycle: np.ndarray) -> None:
    for particle in range(lifecycle.shape[1]):
        terminal = ""
        for state in lifecycle[:, particle]:
            current = str(state)
            if not terminal and current == "active":
                continue
            if not terminal:
                terminal = current
            elif current != terminal:
                raise ValueError(f"{path}: lifecycle changes after terminal state")


def _read_events(
    path: Path,
    particle_ids: np.ndarray,
    times_s: np.ndarray,
    positions: np.ndarray,
    lifecycle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arrivals = np.full(len(particle_ids), np.inf, dtype=np.float64)
    fates = np.full(len(particle_ids), "active", dtype="<U8")
    particles = _particle_index(particle_ids)
    seen: set[int] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(EVENT_COLUMNS).difference(reader.fieldnames or ()):
            raise ValueError(f"{path}: event columns are incomplete")
        for line, row in enumerate(reader, start=2):
            outcome = row["outcome"].strip()
            if outcome not in FATES[1:]:
                continue
            try:
                particle = particles[int(row["particle_id"])]
                event_time = float(row["event_time_s"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line}: invalid terminal event") from error
            if particle in seen:
                raise ValueError(f"{path}:{line}: duplicate terminal event")
            if not math.isfinite(event_time) or not 0.0 <= event_time <= float(times_s[-1]):
                raise ValueError(f"{path}:{line}: terminal-event time is outside the run")
            if not row["boundary_semantic"].strip():
                raise ValueError(f"{path}:{line}: boundary_semantic is empty")
            arrivals[particle] = event_time
            fates[particle] = outcome
            seen.add(particle)
    _reconcile_event_lifecycle(path, times_s, positions, lifecycle, arrivals, fates)
    return arrivals, fates


def _reconcile_event_lifecycle(
    path: Path,
    times_s: np.ndarray,
    positions: np.ndarray,
    lifecycle: np.ndarray,
    arrivals: np.ndarray,
    fates: np.ndarray,
) -> None:
    for particle in range(len(arrivals)):
        for time_index, time_s in enumerate(times_s):
            expected = fates[particle] if time_s >= arrivals[particle] else "active"
            observed = str(lifecycle[time_index, particle])
            if not observed:
                if expected != "escaped":
                    raise ValueError(f"{path}: only an escaped suffix may be omitted")
                lifecycle[time_index, particle] = "escaped"
            elif observed != expected:
                raise ValueError(f"{path}: terminal event and scheduled lifecycle disagree")
            if expected == "escaped" and np.isfinite(positions[time_index, particle]).any():
                raise ValueError(f"{path}: escaped suffix position must remain unobserved")
    _validate_lifecycle(path, lifecycle)


def _load_performance(record: object, root: Path, name: str) -> dict[str, Any] | None:
    if record is None:
        return None
    path = _resolve_artifact(record, root, name)
    raw = _json(path, name)
    required = ("wall_time_s", "peak_rss_bytes", "output_bytes", "particle_count", "output_frames")
    measurement_status = raw.get("measurement_status")
    if measurement_status in {
        "NOT_MEASURED_OR_NOT_RECOVERABLE",
        "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP",
    }:
        memory = _mapping(raw.get("memory_plan"), f"{name}.memory_plan")
        reason = (
            str(raw.get("non_authoritative_reason"))
            if measurement_status == "NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP"
            else "runner_receipt_explicitly_marks_timing_and_peak_rss_unrecoverable"
        )
        return {
            "measurement_status": "NOT_MEASURED",
            "reason": reason,
            "particle_count": int(raw.get("particle_count", 0)),
            "output_frames": int(raw.get("output_frames", 0)),
            "output_bytes": int(raw.get("output_bytes", -1)),
            "solver_planned_bytes": int(memory.get("planned_bytes", 0)),
            "peak_rss_bytes": None,
            "wall_time_s": None,
            "stage_times_s": {},
        }
    if any(key not in raw for key in required):
        if raw.get("timing_status") != "NOT_RECOVERABLE_AFTER_RUN":
            raise ValueError(f"{name} is missing a required performance value")
        counts = _mapping(raw.get("counts"), f"{name}.counts")
        memory = _mapping(raw.get("memory_plan"), f"{name}.memory_plan")
        return {
            "measurement_status": "NOT_MEASURED",
            "reason": "runner_receipt_explicitly_marks_elapsed_time_unrecoverable",
            "particle_count": int(counts.get("particles", 0)),
            "output_frames": int(counts.get("frames", 0)),
            "output_bytes": int(raw.get("artifact_bytes", -1)),
            "solver_planned_bytes": int(memory.get("planned_bytes", 0)),
            "peak_rss_bytes": None,
            "wall_time_s": None,
            "stage_times_s": {},
        }
    wall = _positive_float(raw["wall_time_s"], f"{name}.wall_time_s")
    if int(raw["peak_rss_bytes"]) <= 0 or int(raw["output_bytes"]) < 0:
        raise ValueError(f"{name} has invalid memory/output bytes")
    stages = _mapping(raw.get("stage_times_s") or {}, f"{name}.stage_times_s")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in stages.values()):
        raise ValueError(f"{name} has invalid stage time")
    if math.fsum(float(value) for value in stages.values()) > wall * 1.05:
        raise ValueError(f"{name} stage times exceed wall time")
    return {**raw, "stage_times_s": stages, "measurement_status": "MEASURED"}


def _load_replica(
    raw: object,
    root: Path,
    particle_ids: np.ndarray,
    times_s: np.ndarray,
    name: str,
) -> Replica:
    record = _mapping(raw, name)
    trajectory = _resolve_artifact(record.get("trajectory"), root, f"{name}.trajectory")
    events = _resolve_artifact(record.get("events"), root, f"{name}.events")
    positions, lifecycle, initial_state = _read_trajectory(trajectory, particle_ids, times_s)
    arrivals, fates = _read_events(events, particle_ids, times_s, positions, lifecycle)
    return Replica(
        seed=int(_required(record, "seed", name)),
        trajectory_path=trajectory,
        event_path=events,
        positions_m=positions,
        lifecycle=lifecycle,
        initial_state=initial_state,
        first_arrival_s=arrivals,
        terminal_fate=fates,
        performance=_load_performance(record.get("performance"), root, f"{name}.performance"),
    )


def _load_levels(
    raw: object,
    root: Path,
    particle_ids: np.ndarray,
    times_s: np.ndarray,
    participant: str,
) -> tuple[Level, ...]:
    levels: list[Level] = []
    for index, value in enumerate(_sequence(raw, f"{participant}.levels")):
        record = _mapping(value, f"{participant}.levels[{index}]")
        replicas = tuple(
            sorted(
                (
                    _load_replica(
                        replica,
                        root,
                        particle_ids,
                        times_s,
                        f"{participant}.{record.get('level_id')}.replicas[{replica_index}]",
                    )
                    for replica_index, replica in enumerate(
                        _sequence(record.get("replicas"), f"{participant}.replicas")
                    )
                ),
                key=lambda replica: replica.seed,
            )
        )
        seeds = [replica.seed for replica in replicas]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{participant} level contains duplicate seeds")
        levels.append(
            Level(
                level_id=str(record.get("level_id")),
                ordinal=int(_required(record, "ordinal", f"{participant}.level")),
                numerical_setting=_mapping(record.get("numerical_setting"), "numerical_setting"),
                replicas=replicas,
            )
        )
    levels.sort(key=lambda level: level.ordinal)
    if len({level.ordinal for level in levels}) != len(levels):
        raise ValueError(f"{participant} level ordinals are not unique")
    if any(not level.level_id for level in levels) or len(
        {level.level_id for level in levels}
    ) != len(levels):
        raise ValueError(f"{participant} level IDs must be nonempty and unique")
    if levels and [level.ordinal for level in levels] != list(range(len(levels))):
        raise ValueError(f"{participant} levels must be ordered coarse-to-fine from zero")
    return tuple(levels)


def _validate_sha256_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} is not a SHA-256 digest")


def _validate_content_hash(value: str, name: str) -> None:
    digest = value.removeprefix("sha256:")
    if (
        not value.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} is invalid")


def _campaign_binding(
    raw: dict[str, Any],
    canonical_input: dict[str, str] | None,
    policy: Policy,
) -> dict[str, str] | None:
    value = raw.get("campaign_binding")
    if value is None:
        if policy.revision >= 4:
            raise ValueError("v4 campaign is missing its campaign binding")
        return None
    record = _mapping(value, "campaign binding")
    if set(record) != CAMPAIGN_BINDING_KEYS or any(
        not isinstance(record.get(key), str) for key in CAMPAIGN_BINDING_KEYS
    ):
        raise ValueError("campaign binding must contain exactly the registered string fields")
    binding = {key: cast(str, record[key]).lower() for key in CAMPAIGN_BINDING_KEYS}
    for key in ("contract_sha256", "input_sha256"):
        _validate_sha256_digest(binding[key], f"campaign binding {key}")
    _validate_content_hash(binding["input_content_hash"], "campaign binding input_content_hash")
    if canonical_input is None or (
        binding["input_sha256"] != canonical_input["sha256"]
        or binding["input_content_hash"] != canonical_input["content_hash"]
    ):
        raise ValueError("campaign binding differs from the canonical input")
    evidence = policy.evidence_binding
    if evidence is not None and binding != {
        "contract_sha256": evidence.contract_sha256,
        "input_sha256": evidence.input_sha256,
        "input_content_hash": evidence.input_content_hash,
    }:
        raise ValueError("campaign binding differs from the Case-P RNG evidence")
    return binding


def _study_isolation_record(value: object, name: str) -> None:
    record = _mapping(value, name)
    physics = _mapping(record.get("physics_solve_for"), f"{name}.physics_solve_for")
    multiphysics = _mapping(record.get("multiphysics_solve_for"), f"{name}.multiphysics_solve_for")
    if (
        record.get("status") != "PASS"
        or record.get("enabled_physics") != ["fpt"]
        or record.get("enabled_multiphysics") != []
        or set(physics) != CASEP_STUDY_PHYSICS_TAGS
        or {tag for tag, enabled in physics.items() if enabled is True} != {"fpt"}
        or any(not isinstance(enabled, bool) for enabled in physics.values())
        or set(multiphysics) != CASEP_STUDY_MULTIPHYSICS_TAGS
        or any(enabled is not False for enabled in multiphysics.values())
    ):
        raise ValueError(f"{name} is not an exact particle-only study")


def _casep_comsol_participant(raw: dict[str, Any], purpose: Purpose) -> None:
    if (
        raw.get("schema_version"),
        raw.get("manifest_kind"),
        raw.get("participant"),
        raw.get("purpose"),
        raw.get("case_id"),
    ) != (1, "m3c2_participant", "comsol", purpose, CASEP_EVALUATION_CASE_ID):
        raise ValueError("Case-P COMSOL participant manifest identity differs")
    summary = _mapping(raw.get("study_isolation"), "COMSOL study-isolation summary")
    if summary != {
        "required_enabled_multiphysics": [],
        "required_enabled_physics": ["fpt"],
        "status": "PASS",
        "validated_for_every_replica": True,
    }:
        raise ValueError("Case-P COMSOL study-isolation summary differs")
    replica_count = 0
    for level_index, level_value in enumerate(_sequence(raw.get("levels"), "COMSOL levels")):
        level = _mapping(level_value, f"COMSOL level {level_index}")
        for replica_index, replica_value in enumerate(
            _sequence(level.get("replicas"), f"COMSOL level {level_index} replicas")
        ):
            replica = _mapping(replica_value, f"COMSOL replica {replica_index}")
            if replica.get("status") != "COMPLETE":
                raise ValueError("Case-P COMSOL participant contains an incomplete replica")
            _study_isolation_record(
                replica.get("study_isolation"),
                f"COMSOL level {level_index} replica {replica_index} study isolation",
            )
            replica_count += 1
    if replica_count == 0:
        raise ValueError("Case-P COMSOL participant contains no replicas")


def _validate_casep_candidate_identity(raw: dict[str, Any], purpose: Purpose) -> None:
    identity = _mapping(raw.get("campaign_identity"), "candidate campaign identity")
    if (
        raw.get("tool_revision"),
        raw.get("status"),
        raw.get("participant"),
        raw.get("purpose"),
        identity.get("case_id"),
        identity.get("evaluation_case_id"),
    ) != (
        "m3c2_candidate_campaign_runner_v4",
        "COMPLETE",
        "candidate",
        purpose,
        CASEP_SOURCE_CASE_ID,
        CASEP_EVALUATION_CASE_ID,
    ):
        raise ValueError("Case-P candidate participant manifest identity differs")


def _validate_casep_candidate_binding(raw: dict[str, Any]) -> None:
    binding = _mapping(raw.get("campaign_binding"), "candidate campaign binding")
    if set(binding) != CAMPAIGN_BINDING_KEYS or not all(
        isinstance(binding[key], str) and binding[key] for key in CAMPAIGN_BINDING_KEYS
    ):
        raise ValueError("Case-P candidate participant campaign binding differs")
    policy_sha256 = raw.get("evaluation_policy_sha256")
    if not isinstance(policy_sha256, str) or policy_sha256 != policy_sha256.lower():
        raise ValueError("Case-P candidate participant policy hash is invalid")
    _validate_sha256_digest(policy_sha256, "Case-P candidate participant policy hash")


def _casep_candidate_participant(raw: dict[str, Any], purpose: Purpose) -> None:
    _validate_casep_candidate_identity(raw, purpose)
    _validate_casep_candidate_binding(raw)
    levels = _mapping(raw.get("levels"), "candidate participant levels")
    replica_count = 0
    for level_name, level_value in levels.items():
        level = _mapping(level_value, f"candidate level {level_name}")
        for replica_index, replica_value in enumerate(
            _sequence(level.get("replicas"), f"candidate level {level_name} replicas")
        ):
            replica = _mapping(replica_value, f"candidate replica {replica_index}")
            noise = _mapping(
                _mapping(replica.get("resolved_physics_models"), "resolved physics").get("noise"),
                "resolved Brownian model",
            )
            if (
                replica.get("status") != "COMPLETE"
                or replica.get("brownian_rng_revision") != CASEP_RNG_REVISION
                or noise.get("revision")
                != "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1"
            ):
                raise ValueError("Case-P candidate RNG authority differs")
            replica_count += 1
    if replica_count == 0:
        raise ValueError("Case-P candidate participant contains no replicas")


def _validate_casep_participant_manifests(
    records: dict[str, Any], root: Path, purpose: Purpose
) -> None:
    paths = {
        participant: _resolve_artifact(
            records[participant], root, f"{participant} participant manifest"
        )
        for participant in PARTICIPANTS
    }
    _casep_comsol_participant(_json(paths["comsol"], "COMSOL participant"), purpose)
    _casep_candidate_participant(_json(paths["candidate"], "candidate participant"), purpose)


def _embedded_candidate_path_reference(
    participants: dict[str, Any], purpose: Purpose
) -> str | None:
    if purpose == "final":
        return None
    candidate = _mapping(participants.get("candidate"), "candidate campaign participant")
    references = [
        _mapping(level.get("numerical_setting"), "candidate numerical setting").get(
            "reference_level_id"
        )
        for level in (
            _mapping(value, "candidate campaign level")
            for value in _sequence(candidate.get("levels"), "candidate campaign levels")
        )
        if _mapping(level.get("numerical_setting"), "candidate numerical setting").get("purpose")
        == PATH_SENSITIVITY_PURPOSE
    ]
    if len(references) != 1 or not isinstance(references[0], str) or not references[0]:
        raise ValueError("candidate campaign path reference is invalid")
    return references[0]


def _validate_participant_manifest_projection(
    records: dict[str, Any],
    root: Path,
    purpose: Purpose,
    campaign: dict[str, Any],
    participants: dict[str, Any],
) -> None:
    paths = {
        participant: _resolve_artifact(
            records[participant], root, f"{participant} participant manifest"
        )
        for participant in PARTICIPANTS
    }
    projection = _participant_manifest_projection(
        _json(paths["comsol"], "COMSOL participant"),
        _json(paths["candidate"], "candidate participant"),
        paths["comsol"].parent,
        paths["candidate"].parent,
        purpose,
        root,
        _embedded_candidate_path_reference(participants, purpose),
    )
    expected = {
        "case_id": campaign.get("case_id"),
        "campaign_binding": campaign.get("campaign_binding"),
        "evaluation_policy_sha256": _participant_projection_policy_sha256(campaign, root),
        "pilot_authorization": campaign.get("pilot_authorization"),
        "participants": participants,
    }
    if projection != expected:
        raise ValueError("campaign participants differ from their hash-locked manifests")


def _projection_artifact(record: object, root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    path = _resolve_artifact(record, root, name)
    return path, _json(path, name)


def _participant_projection_policy_sha256(campaign: dict[str, Any], root: Path) -> object:
    raw_projection = campaign.get("policy_projection")
    if raw_projection is None:
        return campaign.get("evaluation_policy_sha256")
    projection = _mapping(raw_projection, "Case-P v5 campaign projection")
    expected_keys = {
        "schema_version",
        "projection_kind",
        "tool_revision",
        "reason",
        "source_campaign",
        "source_policy",
        "target_policy",
        "changed_fields",
        "unchanged_payload_sha256",
        "raw_solver_outputs_reused_without_rerun",
    }
    if set(projection) != expected_keys or (
        projection.get("schema_version"),
        projection.get("projection_kind"),
        projection.get("tool_revision"),
        projection.get("reason"),
        projection.get("changed_fields"),
        projection.get("raw_solver_outputs_reused_without_rerun"),
    ) != (
        1,
        "m3c2_caseP_policy_only_campaign_projection",
        "m3c2_caseP_pilot_v5_projection_v1",
        CASEP_PROJECTION_REASON,
        ["evaluation_policy_sha256"],
        True,
    ):
        raise ValueError("Case-P v5 campaign projection declaration differs")
    _, source = _projection_artifact(
        projection.get("source_campaign"), root, "Case-P v4 source campaign"
    )
    source_policy_path, source_policy = _projection_artifact(
        projection.get("source_policy"), root, "Case-P v4 source policy"
    )
    target_policy_path, target_policy = _projection_artifact(
        projection.get("target_policy"), root, "Case-P v5 target policy"
    )
    source_policy_sha256 = _sha256(source_policy_path)
    target_policy_sha256 = _sha256(target_policy_path)
    if (
        source_policy.get("policy_revision"),
        target_policy.get("policy_revision"),
        source.get("evaluation_policy_sha256"),
        campaign.get("evaluation_policy_sha256"),
    ) != (4, 5, source_policy_sha256, target_policy_sha256):
        raise ValueError("Case-P v5 projection policy transition differs")
    reconstructed = dict(campaign)
    reconstructed.pop("policy_projection")
    reconstructed["evaluation_policy_sha256"] = source_policy_sha256
    source_without_policy = dict(source)
    source_without_policy.pop("evaluation_policy_sha256", None)
    if reconstructed != source or projection.get("unchanged_payload_sha256") != _object_sha256(
        source_without_policy
    ):
        raise ValueError("Case-P v5 campaign projection changes raw campaign content")
    return source_policy_sha256


def _campaign_evaluation_policy_sha256(raw: dict[str, Any], policy: Policy) -> str | None:
    value = raw.get("evaluation_policy_sha256")
    if value is None:
        if policy.revision >= 4:
            raise ValueError("v4 campaign is missing its evaluation policy SHA-256")
        return None
    if not isinstance(value, str) or value != value.lower():
        raise ValueError("campaign evaluation policy SHA-256 must be lowercase hexadecimal")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("campaign evaluation policy SHA-256 is invalid")
    if value != policy.sha256:
        raise ValueError("campaign was produced for another evaluation policy")
    return value


def _campaign_identity(
    raw: dict[str, Any],
    scope: dict[str, Any],
    root: Path,
    policy: Policy,
    purpose: Purpose,
    particle_count: int,
    output_count: int,
) -> tuple[
    str | None,
    dict[str, str] | None,
    dict[str, dict[str, str]] | None,
    dict[str, str] | None,
    str | None,
]:
    case_id = str(raw.get("case_id", "")).strip() or None
    participant_manifests = None
    if raw.get("participant_manifests") is not None:
        manifest_records = _mapping(raw.get("participant_manifests"), "participant manifests")
        if set(manifest_records) != set(PARTICIPANTS):
            raise ValueError("participant manifests must contain exactly COMSOL and candidate")
        if policy.revision >= 5:
            _validate_casep_participant_manifests(manifest_records, root, purpose)
        participant_manifests = {
            participant: _scope_artifact(
                manifest_records[participant], root, f"{participant} participant manifest"
            )
            for participant in PARTICIPANTS
        }
    canonical_input = None
    if scope.get("canonical_input") is not None:
        canonical_input = _scope_artifact(
            scope.get("canonical_input"),
            root,
            "canonical input",
            require_content_hash=True,
        )
    required_identity_is_present = all(
        (case_id is not None, canonical_input is not None, participant_manifests is not None)
    )
    if policy.revision >= 2 and not required_identity_is_present:
        raise ValueError("v2 campaign is missing required scope identity artifacts")
    fixed_design_matches = all(
        (
            case_id == policy.expected_case_id,
            particle_count == policy.expected_particle_count,
            output_count == policy.expected_output_count,
        )
    )
    if policy.revision >= 2 and not fixed_design_matches:
        raise ValueError("v2 campaign differs from the registered fixed source/time design")
    if (
        policy.evidence_binding is not None
        and case_id != policy.evidence_binding.evaluation_case_id
    ):
        raise ValueError("campaign case ID differs from the Case-P RNG evidence")
    return (
        case_id,
        canonical_input,
        participant_manifests,
        _campaign_binding(raw, canonical_input, policy),
        _campaign_evaluation_policy_sha256(raw, policy),
    )


def _load_campaign(path: Path, policy: Policy) -> Campaign:
    raw = _json(path, "M3-C2 campaign manifest")
    if (raw.get("schema_version"), raw.get("manifest_kind")) != (1, "m3c2_campaign"):
        raise ValueError("unexpected M3-C2 campaign-manifest identity")
    purpose = str(raw.get("purpose"))
    if purpose not in {"pilot", "final"}:
        raise ValueError("campaign purpose must be pilot or final")
    scope = _mapping(raw.get("scope"), "campaign scope")
    particle_ids, times_s, bounds, scale = _campaign_scope(scope)
    root = path.parent
    (
        case_id,
        canonical_input,
        participant_manifests,
        campaign_binding,
        evaluation_policy_sha256,
    ) = _campaign_identity(
        raw,
        scope,
        root,
        policy,
        cast(Purpose, purpose),
        len(particle_ids),
        len(times_s),
    )
    participant_records = _mapping(raw.get("participants"), "participants")
    if set(participant_records) != set(PARTICIPANTS):
        raise ValueError("campaign must contain exactly COMSOL and candidate")
    if case_id == CASEP_EVALUATION_CASE_ID:
        _validate_participant_manifest_projection(
            _mapping(raw.get("participant_manifests"), "participant manifests"),
            root,
            cast(Purpose, purpose),
            raw,
            participant_records,
        )
    participants = {
        participant: _load_levels(
            _mapping(participant_records[participant], participant).get("levels"),
            root,
            particle_ids,
            times_s,
            participant,
        )
        for participant in PARTICIPANTS
    }
    _validate_campaign_shape(cast(Purpose, purpose), participants, policy)
    return Campaign(
        path=path,
        sha256=_sha256(path),
        purpose=cast(Purpose, purpose),
        particle_ids=particle_ids,
        times_s=times_s,
        bounds_m=bounds,
        geometry_scale_m=scale,
        participants=participants,
        case_id=case_id,
        canonical_input=canonical_input,
        participant_manifests=participant_manifests,
        campaign_binding=campaign_binding,
        evaluation_policy_sha256=evaluation_policy_sha256,
    )


def _campaign_scope(
    scope: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]:
    particle_ids = np.asarray(
        [int(value) for value in _sequence(scope.get("particle_ids"), "particle_ids")],
        dtype=np.int64,
    )
    times_s = np.asarray(
        [float(value) for value in _sequence(scope.get("output_times_s"), "output_times_s")],
        dtype=np.float64,
    )
    if (
        particle_ids.size == 0
        or len({int(value) for value in particle_ids}) != particle_ids.size
        or times_s.size < 2
        or not np.isfinite(times_s).all()
        or np.any(np.diff(times_s) <= 0.0)
        or times_s[0] != 0.0
    ):
        raise ValueError("campaign particle IDs or output times are invalid")
    bounds_values = [float(value) for value in _sequence(scope.get("geometry_bounds_m"), "bounds")]
    if len(bounds_values) != 4:
        raise ValueError("geometry_bounds_m must be [r_min, r_max, z_min, z_max]")
    r_min, r_max, z_min, z_max = bounds_values
    scale = math.hypot(r_max - r_min, z_max - z_min)
    if not all(math.isfinite(value) for value in bounds_values) or scale <= 0.0:
        raise ValueError("geometry bounds are invalid")
    return particle_ids, times_s, (r_min, r_max, z_min, z_max), scale


def _validate_campaign_shape(
    purpose: Purpose, participants: dict[str, tuple[Level, ...]], policy: Policy
) -> None:
    expected_replicas = policy.pilot_replicas if purpose == "pilot" else policy.final_replicas
    seed_sets = {
        participant: _participant_seed_set(participant, levels, purpose, expected_replicas)
        for participant, levels in participants.items()
    }
    if seed_sets["comsol"] & seed_sets["candidate"]:
        raise ValueError("participant seed sets must be disjoint")
    if purpose == "final" and policy.final_seed_plan is not None:
        for participant in PARTICIPANTS:
            if seed_sets[participant] != set(policy.final_seed_plan[participant]):
                raise ValueError(f"{participant} final seeds differ from the registered plan")
    if purpose == "pilot":
        _validate_pilot_level_purposes(participants, policy)


def _participant_seed_set(
    participant: str,
    levels: tuple[Level, ...],
    purpose: Purpose,
    expected_replicas: int,
) -> set[int]:
    if purpose == "final" and len(levels) != 1:
        raise ValueError(f"{participant} has the wrong number of numerical levels")
    seeds_by_level = [{replica.seed for replica in level.replicas} for level in levels]
    if any(len(level.replicas) != expected_replicas for level in levels):
        raise ValueError(f"{participant} has the wrong replica count")
    if any(seeds != seeds_by_level[0] for seeds in seeds_by_level[1:]):
        raise ValueError(f"{participant} numerical levels do not reuse one pilot seed set")
    return seeds_by_level[0]


def _validate_pilot_level_purposes(
    participants: dict[str, tuple[Level, ...]], policy: Policy
) -> None:
    for participant, levels in participants.items():
        _validate_participant_pilot_levels(participant, levels, policy)


def _validate_participant_pilot_levels(
    participant: str, levels: tuple[Level, ...], policy: Policy
) -> None:
    macro = [
        level
        for level in levels
        if level.numerical_setting.get("purpose") == "macro_step_convergence"
    ]
    path = [
        level
        for level in levels
        if level.numerical_setting.get("purpose") == PATH_SENSITIVITY_PURPOSE
    ]
    if len(macro) != policy.pilot_levels:
        raise ValueError(f"{participant} must contain exactly three macro convergence levels")
    if participant == "comsol":
        if path or len(levels) != len(macro):
            raise ValueError("COMSOL pilot must contain only three macro convergence levels")
        return
    if len(path) != 1 or len(levels) != len(macro) + 1:
        raise ValueError("candidate pilot must contain one separate path sensitivity level")
    reference_id = path[0].numerical_setting.get("reference_level_id")
    if reference_id not in {level.level_id for level in macro}:
        raise ValueError("candidate path sensitivity must reference one macro level")
    if policy.revision >= 3 and reference_id != macro[-1].level_id:
        raise ValueError("v3 candidate path sensitivity must reference the finest macro level")


def _seed_metrics(replica: Replica, campaign: Campaign, policy: Policy) -> SeedMetrics:
    time_count = len(campaign.times_s)
    bins = policy.radial_bins * policy.axial_bins
    occupancy = np.zeros((time_count, bins), dtype=np.float64)
    fate = np.zeros((time_count, len(replica.first_arrival_s), len(FATES)), dtype=np.float64)
    first_arrival = np.zeros((time_count, len(replica.first_arrival_s)), dtype=np.float64)
    r_min, r_max, z_min, z_max = campaign.bounds_m
    for time_index, time_s in enumerate(campaign.times_s):
        active = replica.lifecycle[time_index] == "active"
        points = replica.positions_m[time_index, active]
        if len(points):
            if np.any(points[:, 0] < r_min) or np.any(points[:, 0] > r_max):
                raise ValueError("active radial coordinate lies outside geometry bounds")
            if np.any(points[:, 1] < z_min) or np.any(points[:, 1] > z_max):
                raise ValueError("active axial coordinate lies outside geometry bounds")
            histogram, _, _ = np.histogram2d(
                points[:, 0],
                points[:, 1],
                bins=(policy.radial_bins, policy.axial_bins),
                range=((r_min, r_max), (z_min, z_max)),
            )
            occupancy[time_index] = histogram.ravel() / len(points)
        arrived = replica.first_arrival_s <= time_s
        fate[time_index, :, 0] = ~arrived
        for fate_index, fate_name in enumerate(FATES[1:], start=1):
            fate[time_index, :, fate_index] = arrived & (replica.terminal_fate == fate_name)
        first_arrival[time_index] = arrived
    initial = replica.positions_m[0:1]
    displacement = (replica.positions_m - initial) / campaign.geometry_scale_m
    displacement[replica.lifecycle != "active"] = np.nan
    active_positions = replica.positions_m.copy()
    active_positions[replica.lifecycle != "active"] = np.nan
    return SeedMetrics(active_positions, displacement, occupancy, fate, first_arrival)


def _level_metrics(level: Level, campaign: Campaign, policy: Policy) -> dict[str, np.ndarray]:
    metrics = [_seed_metrics(replica, campaign, policy) for replica in level.replicas]
    return {
        "position_m": np.stack([item.position_m for item in metrics]),
        "position_displacement": np.stack([item.position_displacement_scaled for item in metrics]),
        "occupancy": np.stack([item.occupancy for item in metrics]),
        "fate": np.stack([item.fate for item in metrics]),
        "first_arrival": np.stack([item.first_arrival for item in metrics]),
    }


def _population_point_summary(
    metrics: dict[str, np.ndarray], campaign: Campaign, policy: Policy
) -> dict[str, np.ndarray]:
    """Aggregate weak observables over seed by fixed-source particle units."""

    displacements = metrics["position_displacement"]
    time_count = displacements.shape[1]
    mean_position = np.full((time_count, 2), np.nan, dtype=np.float64)
    covariance = np.full((time_count, 3), np.nan, dtype=np.float64)
    quantile = np.full((time_count, len(policy.quantiles), 2), np.nan, dtype=np.float64)
    occupancy = np.zeros((time_count, policy.radial_bins * policy.axial_bins), dtype=np.float64)
    active_count = np.zeros(time_count, dtype=np.int64)
    r_min, r_max, z_min, z_max = campaign.bounds_m
    for time_index in range(time_count):
        values = displacements[:, time_index].reshape(-1, 2)
        active = np.isfinite(values).all(axis=1)
        active_values = values[active]
        active_count[time_index] = len(active_values)
        if len(active_values):
            mean = np.mean(active_values, axis=0)
            centered = active_values - mean
            mean_position[time_index] = mean
            covariance[time_index] = (
                np.mean(centered[:, 0] * centered[:, 0]),
                np.mean(centered[:, 0] * centered[:, 1]),
                np.mean(centered[:, 1] * centered[:, 1]),
            )
            quantile[time_index] = np.quantile(active_values, policy.quantiles, axis=0)
        positions = metrics["position_m"][:, time_index].reshape(-1, 2)
        present = np.isfinite(positions).all(axis=1)
        if np.any(present):
            histogram, _, _ = np.histogram2d(
                positions[present, 0],
                positions[present, 1],
                bins=(policy.radial_bins, policy.axial_bins),
                range=((r_min, r_max), (z_min, z_max)),
            )
            occupancy[time_index] = histogram.ravel() / np.count_nonzero(present)
    terminal = np.mean(_terminal_population_values(metrics), axis=(0, 2))
    return {
        "mean_position": mean_position,
        "covariance": covariance,
        "quantile": quantile,
        "terminal_population": terminal,
        "occupancy": occupancy,
        "rz_fate_partition": _full_population_rz_fate_partition(metrics, campaign, policy),
        "active_count": active_count,
    }


def _full_population_rz_fate_partition(
    metrics: dict[str, np.ndarray], campaign: Campaign, policy: Policy
) -> np.ndarray:
    """Return one fixed categorical distribution per output time.

    Each seed-by-source-particle unit is counted once: active units occupy one
    fixed geometry bin, while terminal units occupy stuck, held, or escaped.
    """

    positions = metrics["position_m"]
    fate = metrics["fate"]
    time_count = positions.shape[1]
    active_bin_count = policy.radial_bins * policy.axial_bins
    partition = np.zeros((time_count, active_bin_count + len(FATES) - 1), dtype=np.float64)
    observation_count = int(positions.shape[0] * positions.shape[2])
    r_min, r_max, z_min, z_max = campaign.bounds_m
    for time_index in range(time_count):
        active = fate[:, time_index, :, 0].astype(np.bool_)
        active_positions = positions[:, time_index][active]
        if len(active_positions):
            if not np.isfinite(active_positions).all():
                raise ValueError("active R-Z/fate partition contains nonfinite positions")
            histogram, _, _ = np.histogram2d(
                active_positions[:, 0],
                active_positions[:, 1],
                bins=(policy.radial_bins, policy.axial_bins),
                range=((r_min, r_max), (z_min, z_max)),
            )
            if int(np.sum(histogram)) != len(active_positions):
                raise ValueError("active R-Z/fate partition lies outside fixed geometry bounds")
            partition[time_index, :active_bin_count] = histogram.ravel()
        partition[time_index, active_bin_count:] = np.sum(fate[:, time_index, :, 1:], axis=(0, 1))
    partition /= observation_count
    if not np.allclose(np.sum(partition, axis=1), 1.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("full-population R-Z/fate partition is not exhaustive")
    return partition


def _population_continuous_point_report(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    defined = np.isfinite(left) & np.isfinite(right)
    if not np.any(defined):
        return {
            "status": "NOT_APPLICABLE_NO_COMMON_ACTIVE_POPULATION",
            "defined_scalar_endpoints": 0,
            "evidence_role": "descriptive_non_gating_point_summary",
        }
    difference = np.abs(left - right)
    return {
        "status": "EVALUATED",
        "aggregation": "seed_x_fixed_source_particle_population_by_time",
        "defined_scalar_endpoints": int(np.count_nonzero(defined)),
        "maximum_absolute_difference": float(np.max(difference[defined])),
        "evidence_role": "descriptive_non_gating_point_summary",
        "bootstrap_used": False,
    }


def _population_terminal_point_report(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = np.abs(left - right)
    return {
        "status": "EVALUATED",
        "aggregation": "seed_x_fixed_source_particle_population_by_time",
        "curves": list(TERMINAL_CURVES),
        "active_curve_role": "derived_as_one_minus_any_terminal_not_counted",
        "first_arrival_role": "represented_once_as_any_terminal",
        "maximum_absolute_difference": float(np.max(difference)),
        "maximum_absolute_difference_by_curve": {
            curve: float(np.max(difference[:, index]))
            for index, curve in enumerate(TERMINAL_CURVES)
        },
        "maximum_absolute_difference_by_time": [
            float(value) for value in np.max(difference, axis=1)
        ],
        "bootstrap_used": False,
    }


def _population_occupancy_point_report(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    by_time = 0.5 * np.sum(np.abs(left - right), axis=1)
    return {
        "status": "EVALUATED",
        "aggregation": "all_active_seed_x_fixed_source_particle_units_by_time",
        "maximum_total_variation": float(np.max(by_time)),
        "total_variation_by_time": [float(value) for value in by_time],
        "evidence_role": "auxiliary_descriptive_non_gating_point_summary",
        "bootstrap_used": False,
    }


def _population_rz_fate_point_report(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    by_time = 0.5 * np.sum(np.abs(left - right), axis=1)
    return {
        "status": "EVALUATED",
        "aggregation": "all_seed_x_fixed_source_particle_units_by_time",
        "partition": "fixed_active_rz_bins_plus_stuck_held_escaped",
        "maximum_total_variation": float(np.max(by_time)),
        "total_variation_by_time": [float(value) for value in by_time],
        "bootstrap_used": False,
    }


def _compare_population_point_summaries(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    return {
        family: _population_continuous_point_report(left[family], right[family])
        for family in POSITION_FAMILIES
    } | {
        "terminal_population": _population_terminal_point_report(
            left["terminal_population"], right["terminal_population"]
        ),
        "occupancy": _population_occupancy_point_report(left["occupancy"], right["occupancy"]),
        "rz_fate_partition": _population_rz_fate_point_report(
            left["rz_fate_partition"], right["rz_fate_partition"]
        ),
    }


def _object_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scope_fingerprint(campaign: Campaign, policy: Policy) -> dict[str, object]:
    design = {
        "case_id": campaign.case_id,
        "canonical_input": campaign.canonical_input,
        **(
            {"campaign_binding": campaign.campaign_binding}
            if campaign.campaign_binding is not None
            else {}
        ),
        "particle_ids": [int(value) for value in campaign.particle_ids],
        "particle_count": len(campaign.particle_ids),
        "output_times_s": [float(value) for value in campaign.times_s],
        "output_time_count": len(campaign.times_s),
        "geometry_bounds_m": [float(value) for value in campaign.bounds_m],
        "policy_sha256": policy.sha256,
    }
    payload = {
        "purpose": campaign.purpose,
        **design,
        "participant_manifests": campaign.participant_manifests,
        "numerical_settings": {
            participant: [
                {
                    "level_id": level.level_id,
                    "ordinal": level.ordinal,
                    "numerical_setting": level.numerical_setting,
                }
                for level in levels
            ]
            for participant, levels in campaign.participants.items()
        },
    }
    return {
        "algorithm": "sha256_canonical_json_v1",
        "sha256": _object_sha256(payload),
        "common_design_sha256": _object_sha256(design),
        "payload": payload,
    }


def _source_identity(campaign: Campaign) -> dict[str, object]:
    entries = [
        (participant, level.level_id, replica.seed, replica.initial_state)
        for participant, levels in campaign.participants.items()
        for level in levels
        for replica in level.replicas
    ]
    baseline = entries[0][3]
    dynamic_names = ("velocity_r_m_per_s", "velocity_z_m_per_s", "charge_number_e")
    epsilon_factor = 64.0
    mismatches: list[dict[str, object]] = []
    maximum_dynamic = np.zeros(3, dtype=np.float64)
    maximum_budget_fraction = 0.0
    for participant, level_id, seed, state in entries[1:]:
        position_equal = np.array_equal(state[:, :2], baseline[:, :2])
        difference = np.abs(state[:, 2:] - baseline[:, 2:])
        scale = np.maximum(1.0, np.maximum(np.abs(state[:, 2:]), np.abs(baseline[:, 2:])))
        budget = epsilon_factor * np.finfo(np.float64).eps * scale
        within_budget = difference <= budget
        maximum_dynamic = np.maximum(maximum_dynamic, np.max(difference, axis=0))
        maximum_budget_fraction = max(maximum_budget_fraction, float(np.max(difference / budget)))
        if not position_equal or not bool(np.all(within_budget)):
            mismatches.append(
                {
                    "participant": participant,
                    "level_id": level_id,
                    "seed": seed,
                    "position_exact": position_equal,
                    "dynamic_components_outside_roundoff_budget": int(
                        np.count_nonzero(~within_budget)
                    ),
                }
            )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "comparison": (
            "canonical_source_hash_and_fixed_particle_ids_plus_t0_position_exact_and_"
            "velocity_charge_roundoff_bounded"
        ),
        "canonical_input": campaign.canonical_input,
        "particle_count": int(baseline.shape[0]),
        "position_rule": "exact_float64_by_particle_id_for_r_m_and_z_m",
        "velocity_charge_rule": "absolute_difference_le_64_eps_times_max_1_abs_values",
        "velocity_charge_epsilon_factor": epsilon_factor,
        "maximum_absolute_dynamic_difference_by_component": {
            name: float(value) for name, value in zip(dynamic_names, maximum_dynamic, strict=True)
        },
        "maximum_roundoff_budget_fraction": maximum_budget_fraction,
        "replica_level_states_compared": len(entries),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _common_active_coverage(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], campaign: Campaign
) -> dict[str, object]:
    left_active = left["fate"][..., 0].astype(np.bool_)
    right_active = right["fate"][..., 0].astype(np.bool_)
    common = np.all(left_active, axis=0) & np.all(right_active, axis=0)
    counts = np.count_nonzero(common, axis=1)
    particle_count = len(campaign.particle_ids)
    return {
        "conditioning": "particle_id_and_time_active_in_every_seed_on_both_participants",
        "defined_particle_time_pairs": int(np.count_nonzero(common)),
        "possible_particle_time_pairs": int(common.size),
        "fraction_defined": float(np.count_nonzero(common) / common.size),
        "by_time": [
            {
                "time_s": float(time_s),
                "common_active_particle_ids": int(count),
                "fraction": float(count / particle_count),
            }
            for time_s, count in zip(campaign.times_s, counts, strict=True)
        ],
    }


def _terminal_population_values(metrics: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (metrics["fate"][..., 1:], metrics["first_arrival"][..., None]),
        axis=-1,
    )


def _terminal_population_gate(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], policy: Policy
) -> dict[str, object]:
    left_values = _terminal_population_values(left)
    right_values = _terminal_population_values(right)
    if left_values.shape != right_values.shape:
        raise ValueError("terminal population arrays do not align")
    sample_count = int(left_values.shape[0] * left_values.shape[2])
    endpoint_count = int(left_values.shape[1] * left_values.shape[3])
    point = np.mean(left_values, axis=(0, 2)) - np.mean(right_values, axis=(0, 2))
    maximum = float(np.max(np.abs(point)))
    radius = math.sqrt(math.log(2.0 * endpoint_count / policy.terminal_alpha) / sample_count)
    bound = maximum + radius
    return {
        "status": "PASS" if bound <= policy.terminal_margin else "FAIL",
        "role": (
            "confirmatory_terminal_population_gate"
            if policy.revision >= 4
            else "sole_confirmatory_stochastic_equivalence_gate"
        ),
        "method": "two_sample_union_hoeffding",
        "independent_unit": "seed_x_fixed_source_particle_trajectory",
        "within_unit_dependence": "time_and_terminal_curve_dependence_retained",
        "curves": list(TERMINAL_CURVES),
        "active_curve_role": "derived_as_one_minus_any_terminal_not_double_counted",
        "first_arrival_role": "represented_once_as_any_terminal",
        "observations_per_participant": sample_count,
        "simultaneous_endpoint_count": endpoint_count,
        "familywise_alpha": policy.terminal_alpha,
        "confidence": 1.0 - policy.terminal_alpha,
        "critical_radius_formula": "sqrt(log(2*K/alpha)/N)",
        "maximum_absolute_difference": maximum,
        "critical_radius": radius,
        "simultaneous_confidence_bound": bound,
        "margin": policy.terminal_margin,
        "maximum_point_difference_that_can_pass": policy.terminal_margin - radius,
        "maximum_absolute_difference_by_curve": {
            curve: float(np.max(np.abs(point[:, index])))
            for index, curve in enumerate(TERMINAL_CURVES)
        },
    }


def _rz_distribution_gate(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    campaign: Campaign,
    policy: Policy,
) -> dict[str, object]:
    if any(
        value is None
        for value in (
            policy.rz_distribution_alpha,
            policy.rz_distribution_margin,
            policy.rz_one_sample_radius,
            policy.rz_two_sample_radius,
            policy.rz_category_count,
            policy.rz_participant_union_count,
        )
    ):
        raise ValueError("R-Z distribution gate requires a complete revision-4 policy")
    left_partition = _full_population_rz_fate_partition(left, campaign, policy)
    right_partition = _full_population_rz_fate_partition(right, campaign, policy)
    point_by_time = 0.5 * np.sum(np.abs(left_partition - right_partition), axis=1)
    maximum = float(np.max(point_by_time))
    radius = float(cast(float, policy.rz_two_sample_radius))
    margin = float(cast(float, policy.rz_distribution_margin))
    bound = maximum + radius
    observation_count = int(left["position_m"].shape[0] * left["position_m"].shape[2])
    expected_observation_count = policy.final_replicas * int(policy.expected_particle_count or 0)
    if observation_count != expected_observation_count:
        raise ValueError("R-Z distribution gate observation count differs from fixed design")
    if len(campaign.times_s) != policy.expected_output_count:
        raise ValueError("R-Z distribution gate output-time count differs from fixed design")
    return {
        "status": "PASS" if bound <= margin else "FAIL",
        "role": "confirmatory_full_population_distribution_gate",
        "method": "two_sample_union_multinomial_l1",
        "observable": "full_population_fixed_geometry_rz_fate_partition",
        "independent_unit": "seed_x_fixed_source_particle_trajectory",
        "within_unit_dependence": "time_dependence_retained",
        "active_position_bins": {
            "radial_bins": policy.radial_bins,
            "axial_bins": policy.axial_bins,
            "bounds_m": [float(value) for value in campaign.bounds_m],
            "bounds_authority": "campaign_scope_geometry_bounds_m_fixed_before_outcomes",
        },
        "terminal_categories": list(FATES[1:]),
        "category_count": policy.rz_category_count,
        "output_time_count": len(campaign.times_s),
        "participant_union_count": policy.rz_participant_union_count,
        "observations_per_participant_per_time": observation_count,
        "familywise_alpha": policy.rz_distribution_alpha,
        "confidence": 1.0 - float(cast(float, policy.rz_distribution_alpha)),
        "one_sample_critical_radius_formula": ("sqrt((K*ln(2)+ln(2*T/alpha))/(2*N))"),
        "one_sample_critical_radius": policy.rz_one_sample_radius,
        "two_sample_critical_radius": radius,
        "maximum_empirical_total_variation": maximum,
        "maximum_empirical_total_variation_by_time": [float(value) for value in point_by_time],
        "simultaneous_confidence_bound": bound,
        "margin_total_variation": margin,
        "maximum_empirical_tv_that_can_pass": margin - radius,
    }


def _defined_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    return np.isfinite(left_flat).all(axis=0) & np.isfinite(right_flat).all(axis=0)


def _position_statistic(values: np.ndarray, family: str, policy: Policy) -> np.ndarray:
    if family == "mean_position":
        return np.mean(values, axis=0).ravel()
    if family == "quantile":
        return np.quantile(values, policy.quantiles, axis=0).ravel()
    centered = values - np.mean(values, axis=0, keepdims=True)
    denominator = max(len(values) - 1, 1)
    rr = np.sum(centered[:, :, 0] * centered[:, :, 0], axis=0) / denominator
    rz = np.sum(centered[:, :, 0] * centered[:, :, 1], axis=0) / denominator
    zz = np.sum(centered[:, :, 1] * centered[:, :, 1], axis=0) / denominator
    return np.stack((rr, rz, zz), axis=1).ravel()


def _uniform_position_bound(
    left: np.ndarray,
    right: np.ndarray,
    family: str,
    *,
    paired: bool,
    policy: Policy,
    rng: np.random.Generator,
) -> dict[str, object]:
    common = np.isfinite(left).all(axis=(0, 3)) & np.isfinite(right).all(axis=(0, 3))
    if not np.any(common):
        return {
            "status": "NOT_APPLICABLE_NO_COMMON_ACTIVE_DISTRIBUTION",
            "defined_particle_time_pairs": 0,
        }
    left_values = left[:, common, :]
    right_values = right[:, common, :]
    point = _position_statistic(left_values, family, policy) - _position_statistic(
        right_values, family, policy
    )
    deviations = np.empty(policy.bootstrap_resamples, dtype=np.float64)
    for sample in range(policy.bootstrap_resamples):
        left_rows = rng.integers(0, len(left_values), len(left_values))
        right_rows = left_rows if paired else rng.integers(0, len(right_values), len(right_values))
        boot = _position_statistic(left_values[left_rows], family, policy) - _position_statistic(
            right_values[right_rows], family, policy
        )
        deviations[sample] = float(np.max(np.abs(boot - point)))
    critical = float(np.quantile(deviations, policy.confidence))
    return {
        "status": "EVALUATED",
        "conditioning": "particle_id_and_time_with_all_seeds_active_on_both_sides",
        "defined_particle_time_pairs": int(np.count_nonzero(common)),
        "maximum_absolute_difference": float(np.max(np.abs(point))),
        "simultaneous_confidence_bound": float(np.max(np.abs(point)) + critical),
        "bootstrap_critical_radius": critical,
    }


def _uniform_mean_bound(
    left: np.ndarray,
    right: np.ndarray,
    *,
    paired: bool,
    policy: Policy,
    rng: np.random.Generator,
) -> dict[str, object]:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    defined = _defined_columns(left, right)
    if not np.any(defined):
        return {"status": "NOT_APPLICABLE_NO_COMMON_ACTIVE_DISTRIBUTION", "defined": 0}
    left_values = left_flat[:, defined]
    right_values = right_flat[:, defined]
    point = np.mean(left_values, axis=0) - np.mean(right_values, axis=0)
    deviations = np.empty(policy.bootstrap_resamples, dtype=np.float64)
    for sample in range(policy.bootstrap_resamples):
        left_rows = rng.integers(0, len(left_values), len(left_values))
        right_rows = left_rows if paired else rng.integers(0, len(right_values), len(right_values))
        boot = np.mean(left_values[left_rows], axis=0) - np.mean(right_values[right_rows], axis=0)
        deviations[sample] = float(np.max(np.abs(boot - point)))
    critical = float(np.quantile(deviations, policy.confidence))
    return {
        "status": "EVALUATED",
        "defined": int(np.count_nonzero(defined)),
        "maximum_absolute_difference": float(np.max(np.abs(point))),
        "simultaneous_confidence_bound": float(np.max(np.abs(point)) + critical),
        "bootstrap_critical_radius": critical,
    }


def _uniform_tv_bound(
    left: np.ndarray,
    right: np.ndarray,
    *,
    paired: bool,
    policy: Policy,
    rng: np.random.Generator,
) -> dict[str, object]:
    point_by_time = 0.5 * np.sum(np.abs(np.mean(left, axis=0) - np.mean(right, axis=0)), axis=1)
    bootstrap = np.empty(policy.bootstrap_resamples, dtype=np.float64)
    for sample in range(policy.bootstrap_resamples):
        left_rows = rng.integers(0, len(left), len(left))
        right_rows = left_rows if paired else rng.integers(0, len(right), len(right))
        difference = np.mean(left[left_rows], axis=0) - np.mean(right[right_rows], axis=0)
        bootstrap[sample] = float(np.max(0.5 * np.sum(np.abs(difference), axis=1)))
    return {
        "status": "EVALUATED",
        "maximum_total_variation": float(np.max(point_by_time)),
        "simultaneous_confidence_bound": float(np.quantile(bootstrap, policy.confidence)),
    }


def _compare_metric_sets(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    policy: Policy,
    *,
    paired: bool,
    seed_offset: int,
) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for index, family in enumerate(POSITION_FAMILIES):
        rng = np.random.default_rng(policy.bootstrap_seed + seed_offset + index)
        reports[family] = _uniform_position_bound(
            left["position_displacement"],
            right["position_displacement"],
            family,
            paired=paired,
            policy=policy,
            rng=rng,
        )
    for index, family in enumerate(("fate", "first_arrival"), start=10):
        reports[family] = _uniform_mean_bound(
            left[family],
            right[family],
            paired=paired,
            policy=policy,
            rng=np.random.default_rng(policy.bootstrap_seed + seed_offset + index),
        )
    reports["occupancy"] = _uniform_tv_bound(
        left["occupancy"],
        right["occupancy"],
        paired=paired,
        policy=policy,
        rng=np.random.default_rng(policy.bootstrap_seed + seed_offset + 19),
    )
    for report in reports.values():
        report["evidence_role"] = (
            "four_seed_configuration_screening_not_95_percent_convergence_or_accuracy"
        )
        report["bootstrap_bound_interpretation"] = (
            "heuristic_screening_summary_not_a_finite_sample_confidence_guarantee"
        )
    return reports


def _compare_descriptive_metric_sets(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    policy: Policy,
    *,
    seed_offset: int,
) -> dict[str, dict[str, object]]:
    """Report continuous and occupancy diagnostics without making final gates."""

    reports: dict[str, dict[str, object]] = {}
    for index, family in enumerate(POSITION_FAMILIES):
        report = _uniform_position_bound(
            left["position_displacement"],
            right["position_displacement"],
            family,
            paired=False,
            policy=policy,
            rng=np.random.default_rng(policy.bootstrap_seed + seed_offset + index),
        )
        reports[family] = {
            **report,
            "evidence_role": "descriptive_not_in_confirmatory_final_decision",
        }
    reports["occupancy"] = {
        **_uniform_tv_bound(
            left["occupancy"],
            right["occupancy"],
            paired=False,
            policy=policy,
            rng=np.random.default_rng(policy.bootstrap_seed + seed_offset + 19),
        ),
        "evidence_role": "auxiliary_not_in_confirmatory_final_decision",
    }
    return reports


def _compare_v3_descriptive_point_summaries(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    campaign: Campaign,
    policy: Policy,
) -> dict[str, dict[str, object]]:
    left_summary = _population_point_summary(left, campaign, policy)
    right_summary = _population_point_summary(right, campaign, policy)
    reports = _compare_population_point_summaries(left_summary, right_summary)
    return {family: reports[family] for family in (*POSITION_FAMILIES, "occupancy")}


def _bound(report: dict[str, object]) -> float | None:
    if report.get("status") != "EVALUATED":
        return None
    return float(cast(Any, report["simultaneous_confidence_bound"]))


def _gate_reports(
    reports: dict[str, dict[str, object]],
    margins: dict[str, float],
    *,
    screening: bool = False,
) -> tuple[str, list[dict[str, object]]]:
    gates: list[dict[str, object]] = []
    mapping = {
        "mean_position": "mean_position",
        "covariance": "covariance",
        "quantile": "quantile",
        "occupancy": "occupancy_tv",
        "fate": "fate_probability",
        "first_arrival": "first_arrival_cdf",
    }
    for family, margin_name in mapping.items():
        bound = _bound(reports[family])
        if family == "occupancy":
            status = (
                "AUXILIARY_NOT_APPLICABLE"
                if bound is None
                else "AUXILIARY_PASS"
                if bound <= margins[margin_name]
                else "AUXILIARY_FAIL"
            )
        elif bound is None:
            status = "NOT_APPLICABLE" if family in POSITION_FAMILIES else "BLOCKED"
        else:
            status = "PASS" if bound <= margins[margin_name] else "FAIL"
        gates.append(
            {
                "family": family,
                "status": status,
                "simultaneous_confidence_bound": bound,
                "margin": margins[margin_name],
                "role": (
                    "auxiliary_configuration_screening_not_in_overall_status"
                    if family == "occupancy"
                    else "configuration_screening_not_95_percent_convergence_or_accuracy"
                    if screening
                    else "scientific_accuracy_gate"
                ),
            }
        )
    failed = any(row["status"] == "FAIL" for row in gates)
    blocked = any(row["status"] == "BLOCKED" for row in gates)
    return ("FAIL" if failed else "BLOCKED" if blocked else "PASS"), gates


def _v3_terminal_screening_gate(
    report: dict[str, object], *, base_margin: float, fraction: float
) -> dict[str, object]:
    point_difference = float(cast(Any, report["maximum_absolute_difference"]))
    margin = base_margin * fraction
    return {
        "status": "PASS" if point_difference <= margin else "FAIL",
        "role": "noninferential_configuration_selection_gate",
        "observable_family": "terminal_population_curves",
        "maximum_absolute_point_difference": point_difference,
        "base_final_margin": base_margin,
        "pilot_margin_fraction": fraction,
        "screening_margin": margin,
        "bootstrap_used": False,
        "pathwise_comparison": False,
    }


def _v4_rz_screening_gate(
    report: dict[str, object], *, screening_margin: float
) -> dict[str, object]:
    point_difference = float(cast(Any, report["maximum_total_variation"]))
    return {
        "status": "PASS" if point_difference <= screening_margin else "FAIL",
        "role": "noninferential_configuration_selection_gate",
        "observable_family": "full_population_rz_fate_partition_tv",
        "maximum_empirical_total_variation": point_difference,
        "screening_margin": screening_margin,
        "bootstrap_used": False,
        "pathwise_comparison": False,
    }


def _registered_population_screening_gates(
    reports: dict[str, dict[str, object]], policy: Policy, *, margin_fraction: float
) -> tuple[dict[str, object], dict[str, object] | None, bool]:
    terminal_gate = _v3_terminal_screening_gate(
        reports["terminal_population"],
        base_margin=policy.terminal_margin,
        fraction=margin_fraction,
    )
    distribution_gate = None
    if policy.revision >= 4:
        if policy.rz_distribution_margin is None:
            raise ValueError("v4 policy has no R-Z distribution margin")
        distribution_gate = _v4_rz_screening_gate(
            reports["rz_fate_partition"],
            screening_margin=policy.rz_distribution_margin * margin_fraction,
        )
    passed = terminal_gate["status"] == "PASS" and (
        distribution_gate is None or distribution_gate["status"] == "PASS"
    )
    return terminal_gate, distribution_gate, passed


def _v3_adjacent_terminal_diagnostics(
    summaries: list[dict[str, np.ndarray]], levels: tuple[Level, ...], policy: Policy
) -> dict[str, object]:
    differences = [
        float(
            cast(
                Any,
                _population_terminal_point_report(
                    summaries[index]["terminal_population"],
                    summaries[index + 1]["terminal_population"],
                )["maximum_absolute_difference"],
            )
        )
        for index in range(len(summaries) - 1)
    ]
    coarse, fine = differences[-2:]
    ratio = fine / coarse if coarse > 0.0 else 0.0 if fine == 0.0 else None
    return {
        "role": "descriptive_only_not_used_for_v3_selection",
        "adjacent_pairs": [
            {
                "left_level": levels[index].level_id,
                "right_level": levels[index + 1].level_id,
                "maximum_terminal_point_difference": value,
            }
            for index, value in enumerate(differences)
        ],
        "fine_to_coarse_ratio": ratio,
        "historical_v2_registered_maximum_ratio": policy.max_stabilization_ratio,
        "reason_not_gated": (
            "ratio_of_two_noisy_four_seed_point_differences_is_unstable_and_does_not_"
            "measure_numerical_convergence"
        ),
    }


def _v3_candidate_path_screening(
    participant: str,
    levels: tuple[Level, ...],
    macro_levels: tuple[Level, ...],
    reference_summary: dict[str, np.ndarray],
    campaign: Campaign,
    policy: Policy,
) -> dict[str, object] | None:
    if participant != "candidate":
        return None
    path_level = next(
        level
        for level in levels
        if level.numerical_setting.get("purpose") == PATH_SENSITIVITY_PURPOSE
    )
    reference_level = macro_levels[-1]
    if path_level.numerical_setting.get("reference_level_id") != reference_level.level_id:
        raise ValueError("v3 path sensitivity does not reference the finest macro level")
    path_summary = _population_point_summary(
        _level_metrics(path_level, campaign, policy), campaign, policy
    )
    reports = _compare_population_point_summaries(path_summary, reference_summary)
    gate, distribution_gate, passed = _registered_population_screening_gates(
        reports, policy, margin_fraction=policy.path_sensitivity_fraction
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "evidence_role": "noninferential_registered_population_configuration_screening",
        "reference_level": reference_level.level_id,
        "sensitivity_level": path_level.level_id,
        "comparison": "direct_population_point_summary_to_finest_macro_level",
        "reports": reports,
        "terminal_gate": gate,
        "rz_distribution_gate": distribution_gate,
        "continuous_trajectory_equivalence_established": False,
    }


def _v3_pilot_participant(
    participant: str,
    levels: tuple[Level, ...],
    campaign: Campaign,
    policy: Policy,
) -> dict[str, object]:
    macro_levels = tuple(
        level
        for level in levels
        if level.numerical_setting.get("purpose") == "macro_step_convergence"
    )
    summaries = [
        _population_point_summary(_level_metrics(level, campaign, policy), campaign, policy)
        for level in macro_levels
    ]
    reference_level = macro_levels[-1]
    reference_summary = summaries[-1]
    comparisons: list[dict[str, object]] = []
    selected_level: str | None = None
    for level, summary in zip(macro_levels, summaries, strict=True):
        reports = _compare_population_point_summaries(summary, reference_summary)
        gate, distribution_gate, gates_pass = _registered_population_screening_gates(
            reports, policy, margin_fraction=policy.numerical_fraction
        )
        if selected_level is None and gates_pass:
            selected_level = level.level_id
        comparisons.append(
            {
                "level_id": level.level_id,
                "reference_level_id": reference_level.level_id,
                "numerical_setting": level.numerical_setting,
                "reports": reports,
                "terminal_gate": gate,
                "rz_distribution_gate": distribution_gate,
                "active_population_units_by_time": [
                    int(value) for value in summary["active_count"]
                ],
                "reference_active_population_units_by_time": [
                    int(value) for value in reference_summary["active_count"]
                ],
            }
        )
    if selected_level is None:
        raise ValueError("pilot finest reference did not pass its identity comparison")
    path = _v3_candidate_path_screening(
        participant,
        levels,
        macro_levels,
        reference_summary,
        campaign,
        policy,
    )
    status = "PASS" if path is None else str(path["status"])
    return {
        "participant": participant,
        "status": status,
        "evidence_role": "four_seed_noninferential_registered_population_configuration_screening",
        "selection_method": "direct_each_macro_level_to_registered_finest_reference",
        "gating_observables": list(TERMINAL_CURVES)
        + (["full_population_rz_fate_partition_tv"] if policy.revision >= 4 else []),
        "descriptive_non_gating_observables": [
            "active_population_mean_position",
            "active_population_covariance",
            "active_population_registered_quantiles",
            "active_population_occupancy_total_variation",
        ],
        "comparisons_to_finest": comparisons,
        "adjacent_terminal_diagnostics": _v3_adjacent_terminal_diagnostics(
            summaries, macro_levels, policy
        ),
        "path_sensitivity": path,
        "reference_level": reference_level.level_id,
        "largest_screened_macro_level": selected_level,
        "final_level_selection": "NOT_AUTHORIZED_REQUIRES_POST_PILOT_RECEIPT",
        "continuous_weak_convergence_established": False,
        "continuous_trajectory_equivalence_established": False,
        "v2_method_correction": (
            "per_id_simultaneous_bootstrap_bounds_and_adjacent_difference_ratios_are_not_"
            "used_for_v3_selection"
        ),
    }


def _pilot_participant(
    participant: str,
    levels: tuple[Level, ...],
    campaign: Campaign,
    policy: Policy,
) -> dict[str, object]:
    macro_levels = tuple(
        level
        for level in levels
        if level.numerical_setting.get("purpose") == "macro_step_convergence"
    )
    metrics = [_level_metrics(level, campaign, policy) for level in macro_levels]
    pair_reports = [
        _compare_metric_sets(
            metrics[index], metrics[index + 1], policy, paired=True, seed_offset=100 * index
        )
        for index in range(len(metrics) - 1)
    ]
    numerical_margins = {
        name: value * policy.numerical_fraction for name, value in policy.margins.items()
    }
    coarse_status, coarse_gates = _gate_reports(pair_reports[-2], numerical_margins, screening=True)
    fine_status, gates = _gate_reports(pair_reports[-1], numerical_margins, screening=True)
    stabilization = _stabilization_rows(pair_reports, numerical_margins, policy)
    macro_status = _convergence_status(fine_status, stabilization)
    status = macro_status
    path_sensitivity = _candidate_path_sensitivity(
        participant, levels, macro_levels, metrics, campaign, policy
    )
    if path_sensitivity is not None:
        status = _combined_status(status, str(path_sensitivity["status"]))
    largest_qualified = None
    if macro_status == "PASS":
        largest_qualified = (
            macro_levels[0].level_id if coarse_status == "PASS" else macro_levels[1].level_id
        )
    return {
        "participant": participant,
        "status": status,
        "evidence_role": (
            "four_seed_configuration_screening_not_95_percent_convergence_or_accuracy"
        ),
        "macro_configuration_screening_status": macro_status,
        "level_comparison": (
            "four_seed_paired_whole_seed_common_random_number_configuration_screening_"
            "for_weak_observables; "
            "same seed labels give valid levelwise marginals but do not establish the "
            "same continuous Brownian path, 95% convergence, accuracy, or strong/pathwise "
            "convergence"
        ),
        "levels": [
            {"level_id": level.level_id, "numerical_setting": level.numerical_setting}
            for level in levels
        ],
        "coarse_pair": pair_reports[-2],
        "coarse_pair_gates": coarse_gates,
        "fine_pair": pair_reports[-1],
        "fine_pair_gates": gates,
        "stabilization": stabilization,
        "path_sensitivity": path_sensitivity,
        "reference_level": macro_levels[-1].level_id,
        "largest_qualified_macro_level": largest_qualified,
        "pragmatic_step_screening": (
            "largest level satisfying the preregistered heuristic without changing any margin"
        ),
        "final_level_selection": ("NOT_SELECTED_REQUIRES_POST_PILOT_AUTHORIZATION_RECEIPT"),
    }


def _stabilization_rows(
    pair_reports: list[dict[str, dict[str, object]]],
    margins: dict[str, float],
    policy: Policy,
) -> list[dict[str, object]]:
    stabilization: list[dict[str, object]] = []
    for family in (*POSITION_FAMILIES, "occupancy", "fate", "first_arrival"):
        coarse = _bound(pair_reports[-2][family])
        fine = _bound(pair_reports[-1][family])
        if coarse is None or fine is None:
            status = "NOT_APPLICABLE" if family in POSITION_FAMILIES else "BLOCKED"
            ratio = None
        else:
            margin = margins[_margin_name(family)]
            ratio = fine / coarse if coarse > 0.0 else 0.0
            if fine > margin:
                status = "FAIL"
            elif coarse <= margin:
                status = "PASS_TOLERANCE_PLATEAU"
            else:
                status = "PASS" if ratio <= policy.max_stabilization_ratio else "FAIL"
        if family == "occupancy":
            status = f"AUXILIARY_{status}"
        stabilization.append(
            {
                "family": family,
                "status": status,
                "fine_to_coarse_ratio": ratio,
                "evidence_role": (
                    "four_seed_configuration_screening_not_95_percent_convergence_or_accuracy"
                ),
            }
        )
    return stabilization


def _margin_name(family: str) -> str:
    return {
        "occupancy": "occupancy_tv",
        "fate": "fate_probability",
        "first_arrival": "first_arrival_cdf",
    }.get(family, family)


def _convergence_status(status: str, stabilization: list[dict[str, object]]) -> str:
    if any(row["status"] == "FAIL" for row in stabilization):
        return "FAIL"
    if any(row["status"] == "BLOCKED" for row in stabilization):
        return "BLOCKED"
    return status


def _candidate_path_sensitivity(
    participant: str,
    levels: tuple[Level, ...],
    macro_levels: tuple[Level, ...],
    macro_metrics: list[dict[str, np.ndarray]],
    campaign: Campaign,
    policy: Policy,
) -> dict[str, object] | None:
    if participant != "candidate":
        return None
    path_level = next(
        level
        for level in levels
        if level.numerical_setting.get("purpose") == PATH_SENSITIVITY_PURPOSE
    )
    reference_id = str(path_level.numerical_setting["reference_level_id"])
    reference_index = next(
        index for index, level in enumerate(macro_levels) if level.level_id == reference_id
    )
    reference_level = macro_levels[reference_index]
    reports = _compare_metric_sets(
        macro_metrics[reference_index],
        _level_metrics(path_level, campaign, policy),
        policy,
        paired=True,
        seed_offset=700,
    )
    margins = {
        name: value * policy.path_sensitivity_fraction for name, value in policy.margins.items()
    }
    status, gates = _gate_reports(reports, margins, screening=True)
    return {
        "status": status,
        "evidence_role": (
            "four_seed_configuration_screening_not_95_percent_convergence_or_accuracy"
        ),
        "comparison": (
            "four_seed_paired_whole_seed_configuration_screening_valid_only_for_the_"
            "registered_nested_tree_sensitivity_at_one_macro_step"
        ),
        "reference_level": reference_level.level_id,
        "sensitivity_level": path_level.level_id,
        "reports": reports,
        "gates": gates,
    }


def _combined_status(left: str, right: str) -> str:
    if "FAIL" in {left, right}:
        return "FAIL"
    if "BLOCKED" in {left, right}:
        return "BLOCKED"
    return "PASS"


def _cross_observable_rows(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], campaign: Campaign
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, time_s in enumerate(campaign.times_s):
        left_mean = _nanmean_positions(left["position_m"][:, index])
        right_mean = _nanmean_positions(right["position_m"][:, index])
        shared = np.isfinite(left_mean).all(axis=1) & np.isfinite(right_mean).all(axis=1)
        differences = (left_mean[shared] - right_mean[shared]) / campaign.geometry_scale_m
        norms = np.linalg.norm(differences, axis=1)
        left_fate = np.mean(left["fate"][:, index], axis=(0, 1))
        right_fate = np.mean(right["fate"][:, index], axis=(0, 1))
        occupancy_tv = 0.5 * np.sum(
            np.abs(
                np.mean(left["occupancy"][:, index], axis=0)
                - np.mean(right["occupancy"][:, index], axis=0)
            )
        )
        rows.append(
            {
                "time_s": float(time_s),
                "shared_particle_ids_for_mean": int(np.count_nonzero(shared)),
                "mean_position_difference_scaled": (
                    float(np.sqrt(np.mean(norms * norms))) if len(norms) else math.nan
                ),
                "maximum_position_difference_scaled": (
                    float(np.max(norms)) if len(norms) else math.nan
                ),
                "occupancy_total_variation": float(occupancy_tv),
                **{
                    f"{participant}_{fate}": float(values[fate_index])
                    for participant, values in (("comsol", left_fate), ("candidate", right_fate))
                    for fate_index, fate in enumerate(FATES)
                },
                "comsol_first_arrival_cdf": float(np.mean(left["first_arrival"][:, index, :])),
                "candidate_first_arrival_cdf": float(np.mean(right["first_arrival"][:, index, :])),
            }
        )
    return rows


def _nanmean_positions(values: np.ndarray) -> np.ndarray:
    counts = np.count_nonzero(np.isfinite(values), axis=0)
    result = np.full(values.shape[1:], np.nan, dtype=np.float64)
    np.divide(np.nansum(values, axis=0), counts, out=result, where=counts > 0)
    return result


def _performance_summary(campaign: Campaign) -> dict[str, object]:
    participants = {
        participant: [_performance_level(participant, level) for level in levels]
        for participant, levels in campaign.participants.items()
    }
    return {
        "policy": (
            "A stage is only a candidate for a bounded optimization when it is at least "
            "25% of end-to-end wall time in a >=10000-particle accepted-accuracy workload."
        ),
        "participants": participants,
    }


def _performance_level(participant: str, level: Level) -> dict[str, object]:
    records = [replica.performance for replica in level.replicas]
    if all(record is None for record in records):
        return {"level_id": level.level_id, "status": "NOT_REPORTED"}
    if any(record is None for record in records):
        raise ValueError(f"{participant}/{level.level_id} performance reporting is partial")
    complete = cast(list[dict[str, Any]], records)
    workloads = {(int(row["particle_count"]), int(row["output_frames"])) for row in complete}
    if len(workloads) != 1:
        raise ValueError(f"{participant}/{level.level_id} performance workload differs")
    measurement = {str(row.get("measurement_status")) for row in complete}
    if measurement == {"NOT_MEASURED"}:
        return _unmeasured_performance(level, complete)
    if measurement != {"MEASURED"}:
        raise ValueError(f"{participant}/{level.level_id} performance timing is partial")
    return _measured_performance(level, complete, next(iter(workloads))[0])


def _unmeasured_performance(level: Level, records: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "level_id": level.level_id,
        "status": "NOT_MEASURED",
        "reason": "runner receipts contain no recoverable elapsed time or peak RSS",
        "maximum_solver_planned_bytes": max(int(row["solver_planned_bytes"]) for row in records),
        "median_output_bytes": int(np.median([int(row["output_bytes"]) for row in records])),
        "workload_classification": "PILOT_ONLY_NOT_OPTIMIZATION_AUTHORITY",
    }


def _measured_performance(
    level: Level, records: list[dict[str, Any]], particle_count: int
) -> dict[str, object]:
    stages = sorted(
        set().union(*(_mapping(row.get("stage_times_s", {}), "stage times") for row in records))
    )
    stage_medians = {
        stage: float(
            np.median(
                [
                    float(_mapping(row.get("stage_times_s", {}), "stage times").get(stage, 0.0))
                    for row in records
                ]
            )
        )
        for stage in stages
    }
    wall = float(np.median([float(row["wall_time_s"]) for row in records]))
    return {
        "level_id": level.level_id,
        "status": "CHARACTERIZED",
        "median_wall_time_s": wall,
        "maximum_peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in records),
        "median_output_bytes": int(np.median([int(row["output_bytes"]) for row in records])),
        "median_stage_times_s": stage_medians,
        "stages_at_or_above_25_percent": [
            stage for stage, value in stage_medians.items() if value / wall >= 0.25
        ],
        "workload_classification": (
            "REPRESENTATIVE_PROFILE"
            if particle_count >= 10_000
            else "PILOT_ONLY_NOT_OPTIMIZATION_AUTHORITY"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _polyline(
    values_x: np.ndarray,
    values_y: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> str:
    x, y, width, height = box
    finite = np.isfinite(values_x) & np.isfinite(values_y)
    values_x = values_x[finite]
    values_y = values_y[finite]
    if not len(values_x):
        return ""
    x_min, x_max = x_range or (float(np.min(values_x)), float(np.max(values_x)))
    y_min, y_max = y_range or (float(np.min(values_y)), float(np.max(values_y)))
    x_span = max(x_max - x_min, np.finfo(float).eps)
    y_span = max(y_max - y_min, np.finfo(float).eps)
    return " ".join(
        f"{x + width * (float(x_value) - x_min) / x_span:.2f},"
        f"{y + height - height * (float(y_value) - y_min) / y_span:.2f}"
        for x_value, y_value in zip(values_x, values_y, strict=True)
    )


def _write_rz_svg(
    path: Path,
    campaign: Campaign,
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> None:
    left_mean = _nanmean_positions(left["position_m"]) * 1.0e3
    right_mean = _nanmean_positions(right["position_m"]) * 1.0e3
    all_r = np.concatenate((left_mean[:, :, 0].ravel(), right_mean[:, :, 0].ravel()))
    all_z = np.concatenate((left_mean[:, :, 1].ravel(), right_mean[:, :, 1].ravel()))
    all_r = all_r[np.isfinite(all_r)]
    all_z = all_z[np.isfinite(all_z)]
    r_range = (float(np.min(all_r)), float(np.max(all_r)))
    z_range = (float(np.min(all_z)), float(np.max(all_z)))
    box = (90.0, 80.0, 820.0, 620.0)
    items = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780" viewBox="0 0 1000 780">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="38" font-size="24" font-family="Arial">M3-C2 ensemble mean R-Z trajectories</text>',
        '<rect x="90" y="80" width="820" height="620" fill="none" stroke="#64748b"/>',
    ]
    for values, color in ((left_mean, "#d97706"), (right_mean, "#2563eb")):
        for particle in range(values.shape[1]):
            points = _polyline(
                values[:, particle, 0],
                values[:, particle, 1],
                box,
                x_range=r_range,
                y_range=z_range,
            )
            items.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="1" opacity="0.42"/>'
            )
    items.extend(
        (
            f'<text x="90" y="730" font-size="13" font-family="Arial">r range {float(np.nanmin(all_r)):.4g} to {float(np.nanmax(all_r)):.4g} mm</text>',
            f'<text x="600" y="730" font-size="13" font-family="Arial">z range {float(np.nanmin(all_z)):.4g} to {float(np.nanmax(all_z)):.4g} mm</text>',
            '<text x="90" y="755" fill="#d97706" font-size="13" font-family="Arial">COMSOL: per-ID seed mean</text>',
            '<text x="300" y="755" fill="#2563eb" font-size="13" font-family="Arial">candidate: per-ID seed mean</text>',
            "</svg>",
            "",
        )
    )
    path.write_text("\n".join(items), encoding="utf-8")


def _write_observable_svg(path: Path, rows: list[dict[str, object]]) -> None:
    times = np.asarray([float(cast(Any, row["time_s"])) * 1.0e3 for row in rows])
    series = (
        ("mean_position_difference_scaled", "#7c3aed", "mean position / geometry scale"),
        ("occupancy_total_variation", "#059669", "R-Z occupancy total variation"),
    )
    items = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="38" font-size="24" font-family="Arial">M3-C2 ensemble observable differences</text>',
    ]
    for panel, (column, color, label) in enumerate(series):
        box = (90.0, 80.0 + panel * 325.0, 920.0, 250.0)
        values = np.asarray([float(cast(Any, row[column])) for row in rows])
        points = _polyline(times, values, box)
        items.extend(
            (
                f'<rect x="{box[0]}" y="{box[1]}" width="{box[2]}" height="{box[3]}" fill="none" stroke="#94a3b8"/>',
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<text x="{box[0]}" y="{box[1] - 12}" font-size="14" font-family="Arial">{html.escape(label)}</text>',
                f'<text x="{box[0]}" y="{box[1] + box[3] + 22}" font-size="12" font-family="Arial">time 0 to {times[-1]:.4g} ms; max {float(np.max(values)):.4g}</text>',
            )
        )
    items.extend(("</svg>", ""))
    path.write_text("\n".join(items), encoding="utf-8")


def _write_outputs(
    output: Path,
    manifest: dict[str, object],
    rows: list[dict[str, object]],
    performance: dict[str, object],
    campaign: Campaign,
    metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "ensemble_metrics.csv", rows)
    (output / "performance.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_rz_svg(
        output / "rz_ensemble_trajectory.svg", campaign, metrics["comsol"], metrics["candidate"]
    )
    _write_observable_svg(output / "ensemble_observable_differences.svg", rows)
    (output / "README.md").write_text(
        "# M3-C2 stochastic ensemble evaluation\n\n"
        f"Decision: `{manifest['status']}`.\n\n"
        "Pathwise equality is not evaluated. Revision-2/3 final accuracy authority is the "
        "finite-sample terminal-population gate; revision 4 additionally requires the "
        "finite-sample full-population R-Z/fate distribution gate. "
        "Per-ID position moments, covariance, quantiles, and fixed-bin R-Z occupancy are "
        "descriptive or auxiliary and cannot change the final decision. Pilot calculations "
        "are four-seed configuration screening, not 95% convergence or accuracy evidence. "
        "SVG figures are explanatory and are not gate authority.\n",
        encoding="utf-8",
    )


def _campaign_seed_plan(campaign: Campaign) -> dict[str, list[int]]:
    return {
        participant: sorted(replica.seed for replica in levels[0].replicas)
        for participant, levels in campaign.participants.items()
    }


def _selected_final_levels(campaign: Campaign) -> dict[str, dict[str, object]]:
    if campaign.purpose != "final":
        raise ValueError("selected final levels are defined only for a final campaign")
    return {
        participant: {
            "level_id": levels[0].level_id,
            "ordinal": levels[0].ordinal,
            "numerical_setting": levels[0].numerical_setting,
        }
        for participant, levels in campaign.participants.items()
    }


def _casep_final_authorization_artifacts(
    receipt: dict[str, Any], receipt_root: Path
) -> dict[str, Path]:
    return {
        name: _resolve_artifact(receipt.get(name), receipt_root, f"Case-P final {name}")
        for name in (
            "seed_allocation",
            "rng_noninteraction_evidence",
            "geometry_tolerance_qualification",
        )
    }


def _candidate_final_boundary_event_rows(campaign: Campaign) -> int:
    return sum(
        _csv_row_count_with_required_columns(
            replica.event_path, EVENT_COLUMNS, "candidate final boundary events"
        )
        for replica in campaign.participants["candidate"][0].replicas
    )


def _validate_casep_final_qualification(
    qualification: dict[str, Any], selected: dict[str, dict[str, object]], campaign: Campaign
) -> int:
    candidate_setting = _mapping(
        selected["candidate"].get("numerical_setting"), "selected candidate setting"
    )
    qualified_setting = _mapping(qualification.get("selected_setting"), "Case-P qualified setting")
    reference = _mapping(qualification.get("reference"), "Case-P tolerance reference")
    candidate = _mapping(qualification.get("candidate"), "Case-P tolerance candidate")
    if (
        qualification.get("schema_version"),
        qualification.get("qualification_kind"),
        qualification.get("tool_revision"),
        qualification.get("status"),
        qualification.get("classification"),
        qualification.get("validity_condition"),
        qualification.get("seed"),
    ) != (
        1,
        "m3c2_caseP_geometry_rtol_pre_final",
        "m3c2_caseP_event_tolerance_pre_final_evaluator_v2",
        "PASS",
        "PASS_NO_EVENT_OPERATIONAL_BRIDGE",
        "zero_candidate_boundary_events",
        919008,
    ):
        raise ValueError("Case-P final geometry-tolerance qualification differs")
    if (
        qualified_setting.get("dt_s") != candidate_setting.get("dt_s")
        or qualified_setting.get("brownian_interval_tree_depth")
        != candidate_setting.get("brownian_interval_tree_depth")
        or reference.get("geometry_rtol") != candidate_setting.get("geometry_rtol")
        or candidate.get("geometry_rtol") != 1.0e-9
    ):
        raise ValueError("Case-P final setting differs from its geometry-tolerance bridge")
    event_rows = _candidate_final_boundary_event_rows(campaign)
    if event_rows:
        raise ValueError(
            "Case-P final evaluation is rejected because its zero-event geometry-tolerance "
            "qualification does not cover candidate boundary events"
        )
    return event_rows


def _validate_casep_final_authorization_evidence(
    receipt: dict[str, Any],
    receipt_root: Path,
    campaign: Campaign,
    policy: Policy,
    selected: dict[str, dict[str, object]],
    actual_seed_plan: dict[str, list[int]],
) -> dict[str, object]:
    artifacts = _casep_final_authorization_artifacts(receipt, receipt_root)
    if (
        policy.evidence_manifest_path is None
        or policy.evidence_manifest_sha256 is None
        or artifacts["rng_noninteraction_evidence"] != policy.evidence_manifest_path
        or _sha256(artifacts["rng_noninteraction_evidence"]) != policy.evidence_manifest_sha256
    ):
        raise ValueError("Case-P final authorization uses another RNG/noninteraction evidence")
    allocation = _json(artifacts["seed_allocation"], "Case-P final seed allocation")
    allocation_seeds = _mapping(
        allocation.get("participant_seed_sets"), "Case-P allocated final seeds"
    )
    normalized_allocation = {
        participant: sorted(
            int(seed)
            for seed in _sequence(allocation_seeds.get(participant), f"{participant} seeds")
        )
        for participant in PARTICIPANTS
    }
    if normalized_allocation != actual_seed_plan:
        raise ValueError("Case-P final authorization seed allocation differs")
    physical = _mapping(receipt.get("physical_applicability"), "physical applicability")
    if physical != {
        "status": "NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED",
        "caseP_contains_negative_ions": True,
        "locked_charge_model_includes_negative_ion_current": False,
        "permitted_interpretation": "same_form_numerical_sensitivity_comparison_only",
        "forbidden_interpretation": "physical_validation_of_caseP_charging_or_trajectory_truth",
    }:
        raise ValueError("Case-P physical-applicability limitation differs")
    event_gate = _mapping(receipt.get("candidate_final_event_gate"), "candidate final event gate")
    if event_gate != {
        "validity_condition": "zero_candidate_boundary_events",
        "decision": "REJECT_FINAL_EVALUATION_IF_ANY_CANDIDATE_FINAL_BOUNDARY_EVENT_IS_PRESENT",
        "enforcement_owner": "m3c2_stochastic_ensemble_evaluator_v5",
    }:
        raise ValueError("Case-P final boundary-event rejection rule differs")
    if receipt.get("execution_status") != "NOT_RUN":
        raise ValueError("Case-P final selection receipt was not issued before execution")
    qualification = _json(
        artifacts["geometry_tolerance_qualification"], "Case-P tolerance qualification"
    )
    event_rows = _validate_casep_final_qualification(qualification, selected, campaign)
    return {
        "rng_noninteraction_evidence_sha256": _sha256(artifacts["rng_noninteraction_evidence"]),
        "geometry_tolerance_qualification_sha256": _sha256(
            artifacts["geometry_tolerance_qualification"]
        ),
        "candidate_final_boundary_event_rows": event_rows,
        "physical_applicability": physical["status"],
    }


def _verify_final_authorization(
    path: Path | None,
    *,
    campaign: Campaign,
    policy: Policy,
    pilot_report_path: Path,
    pilot_report: dict[str, Any],
    final_scope: dict[str, object],
) -> dict[str, object]:
    if path is None:
        raise ValueError("v2 final comparison requires --authorization-receipt")
    resolved = path.resolve()
    receipt = _json(resolved, "M3-C2 post-pilot final authorization receipt")
    if (
        receipt.get("schema_version"),
        receipt.get("receipt_kind"),
        receipt.get("status"),
    ) != (
        1,
        "m3c2_post_pilot_final_authorization",
        "AUTHORIZED_FOR_CONFIRMATORY_FINAL",
    ):
        raise ValueError("final authorization receipt is not an accepted authorization")
    pilot_scope = _mapping(pilot_report.get("scope_fingerprint"), "pilot scope fingerprint")
    if campaign.campaign_binding is not None:
        pilot_payload = _mapping(pilot_scope.get("payload"), "pilot scope fingerprint payload")
        if pilot_payload.get("campaign_binding") != campaign.campaign_binding:
            raise ValueError("pilot and final campaign bindings differ")
    actual_seed_plan = _campaign_seed_plan(campaign)
    expected = {
        "policy_sha256": policy.sha256,
        "pilot_report_sha256": _sha256(pilot_report_path),
        "pilot_scope_sha256": pilot_scope.get("sha256"),
        "common_design_sha256": final_scope.get("common_design_sha256"),
        "seed_plan_sha256": _object_sha256(actual_seed_plan),
    }
    if pilot_scope.get("common_design_sha256") != final_scope.get("common_design_sha256"):
        raise ValueError("pilot and final common fixed designs differ")
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise ValueError(f"final authorization receipt {field} differs")
    selected = _selected_final_levels(campaign)
    if receipt.get("selected_final_levels") != selected:
        raise ValueError("final authorization receipt selected levels differ")
    if policy.final_seed_plan is None:
        raise ValueError("v2 final policy has no registered seed plan")
    registered_seed_plan = {
        participant: sorted(policy.final_seed_plan[participant]) for participant in PARTICIPANTS
    }
    if actual_seed_plan != registered_seed_plan:
        raise ValueError("final campaign seed plan differs from the registered policy")
    casep_evidence = None
    if policy.revision >= 5 and campaign.case_id == CASEP_EVALUATION_CASE_ID:
        casep_evidence = _validate_casep_final_authorization_evidence(
            receipt,
            resolved.parent,
            campaign,
            policy,
            selected,
            actual_seed_plan,
        )
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "status": receipt["status"],
        "pilot_report_sha256": expected["pilot_report_sha256"],
        "selected_final_levels": selected,
        "seed_plan": actual_seed_plan,
        "seed_plan_sha256": expected["seed_plan_sha256"],
        **({"casep_pre_final_evidence": casep_evidence} if casep_evidence is not None else {}),
    }


def _evaluation_manifest_metadata(
    policy: Policy,
    campaign: Campaign,
    scope_fingerprint: dict[str, object],
    source_identity: dict[str, object],
) -> dict[str, object]:
    confirmatory_final = campaign.purpose == "final" and policy.revision >= 2
    if policy.revision >= 3:
        bootstrap_execution = f"NOT_RUN_POLICY_REVISION_{policy.revision}"
        bootstrap_interpretation = f"not_used_by_policy_revision_{policy.revision}"
    elif campaign.purpose == "pilot":
        bootstrap_execution = "LEGACY_EXECUTED_FOR_SCREENING_OR_DESCRIPTION"
        bootstrap_interpretation = "four_seed_configuration_screening_not_a_confidence_guarantee"
    else:
        bootstrap_execution = "LEGACY_EXECUTED_FOR_SCREENING_OR_DESCRIPTION"
        bootstrap_interpretation = "descriptive_only_not_in_confirmatory_final_decision"
    independence_evidence = None
    if policy.evidence_manifest_path is not None:
        independence_evidence = {
            "path": str(policy.evidence_manifest_path),
            "sha256": policy.evidence_manifest_sha256,
        }
    return {
        "resampling_unit": (
            "seed_x_fixed_source_particle_trajectory"
            if confirmatory_final
            else "whole_seed_cluster_configuration_screening"
        ),
        "pathwise_equality_evaluated": False,
        "scope_fingerprint": scope_fingerprint,
        "source_identity": source_identity,
        "geometry_scale_m": campaign.geometry_scale_m,
        "confidence": 1.0 - policy.terminal_alpha if confirmatory_final else policy.confidence,
        "bootstrap_confidence": policy.confidence,
        "bootstrap_resamples": policy.bootstrap_resamples,
        "bootstrap_execution": bootstrap_execution,
        "bootstrap_interpretation": bootstrap_interpretation,
        "independence_evidence": independence_evidence,
        "performance_classification": "SEPARATE_FROM_ACCURACY_DECISION",
    }


def _tool_revision(policy: Policy) -> str:
    return TOOL_REVISION_V5 if policy.revision >= 5 else TOOL_REVISION


def evaluate(
    policy_path: Path,
    campaign_path: Path,
    output: Path,
    *,
    pilot_report_path: Path | None = None,
    authorization_receipt_path: Path | None = None,
) -> dict[str, object]:
    policy = _load_policy(policy_path.resolve())
    campaign = _load_campaign(campaign_path.resolve(), policy)
    if campaign.purpose == "final" and policy.revision < 2:
        raise ValueError("policy revision 1 cannot authorize a confirmatory final evaluation")
    scope_fingerprint = _scope_fingerprint(campaign, policy)
    source_identity = _source_identity(campaign)
    metrics = {
        participant: _level_metrics(_reported_level(campaign, levels), campaign, policy)
        for participant, levels in campaign.participants.items()
    }
    rows = _cross_observable_rows(metrics["comsol"], metrics["candidate"], campaign)
    performance = _performance_summary(campaign)
    if campaign.purpose == "pilot":
        status, decision = _pilot_decision(campaign, policy)
    else:
        status, decision = _final_decision(
            campaign,
            policy,
            metrics,
            pilot_report_path,
            authorization_receipt_path,
            scope_fingerprint,
        )
    if source_identity["status"] != "PASS":
        status = "FAIL"
    seed_sets = {
        participant: [replica.seed for replica in levels[-1].replicas]
        for participant, levels in campaign.participants.items()
    }
    metadata = _evaluation_manifest_metadata(policy, campaign, scope_fingerprint, source_identity)
    manifest: dict[str, object] = {
        "schema_version": policy.revision,
        "tool_revision": _tool_revision(policy),
        "phase": campaign.purpose,
        "status": status,
        "policy": {
            "path": str(policy.path),
            "sha256": policy.sha256,
            "revision": policy.revision,
        },
        "campaign": {"path": str(campaign.path), "sha256": campaign.sha256},
        "seed_sets": seed_sets,
        **metadata,
        **decision,
    }
    _write_outputs(output.resolve(), manifest, rows, performance, campaign, metrics)
    return manifest


def _reported_level(campaign: Campaign, levels: tuple[Level, ...]) -> Level:
    if campaign.purpose == "final":
        return levels[0]
    return next(
        level
        for level in reversed(levels)
        if level.numerical_setting.get("purpose") == "macro_step_convergence"
    )


def _pilot_decision(campaign: Campaign, policy: Policy) -> tuple[str, dict[str, object]]:
    pilot = (
        {
            participant: _v3_pilot_participant(
                participant, campaign.participants[participant], campaign, policy
            )
            for participant in PARTICIPANTS
        }
        if policy.revision >= 3
        else {
            participant: _pilot_participant(
                participant, campaign.participants[participant], campaign, policy
            )
            for participant in PARTICIPANTS
        }
    )
    status = "PASS"
    for report in pilot.values():
        status = _combined_status(status, str(report["status"]))
    key = "pilot_configuration_screening" if policy.revision >= 2 else "pilot_convergence"
    return status, {
        key: pilot,
        "pilot_evidence_role": (
            "four_seed_configuration_screening_not_95_percent_convergence_or_accuracy"
        ),
        "screening_margins_immutable_after_observation": True,
        "cross_participant_result": "CHARACTERIZED_NOT_GATED_IN_PILOT",
    }


def _confirmatory_population_gates(
    campaign: Campaign,
    policy: Policy,
    metrics: dict[str, dict[str, np.ndarray]],
) -> tuple[str, dict[str, object], dict[str, object] | None, list[dict[str, object]]]:
    terminal = _terminal_population_gate(metrics["comsol"], metrics["candidate"], policy)
    distribution = (
        _rz_distribution_gate(metrics["comsol"], metrics["candidate"], campaign, policy)
        if policy.revision >= 4
        else None
    )
    gates = [terminal]
    status = str(terminal["status"])
    if distribution is not None:
        gates.append(distribution)
        status = _combined_status(status, str(distribution["status"]))
    return status, terminal, distribution, gates


def _final_descriptive_reports(
    campaign: Campaign,
    policy: Policy,
    metrics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, object]]:
    if policy.revision >= 3:
        return _compare_v3_descriptive_point_summaries(
            metrics["comsol"], metrics["candidate"], campaign, policy
        )
    return _compare_descriptive_metric_sets(
        metrics["comsol"], metrics["candidate"], policy, seed_offset=1000
    )


def _final_decision(
    campaign: Campaign,
    policy: Policy,
    metrics: dict[str, dict[str, np.ndarray]],
    pilot_report_path: Path | None,
    authorization_receipt_path: Path | None,
    scope_fingerprint: dict[str, object],
) -> tuple[str, dict[str, object]]:
    if policy.revision < 2:
        raise ValueError("policy revision 1 cannot authorize a confirmatory final evaluation")
    if pilot_report_path is None:
        raise ValueError("final comparison requires --pilot-report")
    resolved_pilot = pilot_report_path.resolve()
    pilot_report = _json(resolved_pilot, "M3-C2 pilot report")
    if (
        pilot_report.get("schema_version") != policy.revision
        or pilot_report.get("tool_revision") != _tool_revision(policy)
        or pilot_report.get("phase") != "pilot"
        or pilot_report.get("status") != "PASS"
    ):
        raise ValueError("final comparison requires a passing pilot report")
    pilot_policy = _mapping(pilot_report.get("policy"), "pilot policy")
    if (
        pilot_policy.get("sha256") != policy.sha256
        or pilot_policy.get("revision") != policy.revision
    ):
        raise ValueError("pilot report uses another evaluation policy")
    pilot_seeds = {
        int(seed)
        for values in _mapping(pilot_report.get("seed_sets"), "pilot seed sets").values()
        for seed in _sequence(values, "pilot participant seeds")
    }
    final_seeds = {
        replica.seed for levels in campaign.participants.values() for replica in levels[0].replicas
    }
    if pilot_seeds & final_seeds:
        raise ValueError("pilot and final seed sets overlap")
    authorization = _verify_final_authorization(
        authorization_receipt_path,
        campaign=campaign,
        policy=policy,
        pilot_report_path=resolved_pilot,
        pilot_report=pilot_report,
        final_scope=scope_fingerprint,
    )
    status, terminal_gate, rz_distribution_gate, gates = _confirmatory_population_gates(
        campaign, policy, metrics
    )
    return status, {
        "authorization_receipt": authorization,
        "terminal_population_gate": terminal_gate,
        "rz_distribution_gate": rz_distribution_gate,
        "gates": gates,
        "descriptive_continuous_metrics": _final_descriptive_reports(campaign, policy, metrics),
        "common_active_coverage": _common_active_coverage(
            metrics["comsol"], metrics["candidate"], campaign
        ),
        "claim_scope": (
            "approximately_the_same_registered_population_observables_as_COMSOL_for_the_"
            "locked_case_design_without_pathwise_RNG_equality"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    arguments = parser.parse_args()
    report = evaluate(
        arguments.policy,
        arguments.campaign,
        arguments.output,
        pilot_report_path=arguments.pilot_report,
        authorization_receipt_path=arguments.authorization_receipt,
    )
    print(json.dumps({"phase": report["phase"], "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
