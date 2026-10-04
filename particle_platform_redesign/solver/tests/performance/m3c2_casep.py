"""Profile the accepted 287-particle M3-C2 Case-P workload.

The manual ``profile`` suite reads the registered Case-P final selection and
the completed candidate campaign, then runs three locked seeds in isolated
children.  Each child performs one public-API warm-up, one unprofiled
measurement, and one separate cProfile observation.  ``smoke`` uses a caller-
supplied small case to exercise the same plumbing without making a Case-P
performance claim.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
import math
import os
import pstats
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from chamber_particles import load_case, open_result, simulate
from tests.performance.p09_memory import (
    _directory_bytes,
    _machine_metadata,
    _process_memory_bytes,
)
from tests.performance.p14_matrix import _scientific_payload_digest

type Suite = Literal["smoke", "profile"]

_EXPECTED_PROFILE_SEEDS = (319032, 319047, 319063)
_EXPECTED_LEVEL = "macro_coarse"
_EXPECTED_PARTICLES = 287
_EXPECTED_OUTPUT_COUNT = 121
_EXPECTED_END_S = 0.03
_EXPECTED_SETTING = {
    "dt_s": 2.0e-5,
    "brownian_interval_tree_depth": 3,
    "geometry_rtol": 1.0e-8,
    "purpose": "accepted_final",
}
_OWNER_SHARE_THRESHOLD = 0.25
_MINIMUM_OPTIMIZATION_PARTICLES = 10_000
_EXPECTED_FULL_DIGESTS = {
    319032: "sha256:9c0679b19a4362db7f86160376a0b6ccdb4957d5aaeeea57ee786ab3e0ad5fcb",
    319047: "sha256:ba6caf7b711f76a8129d39b8c4fe3a5b4016389d468b44067166797410a86a76",
    319063: "sha256:e665fc77cb6a9882d56a0733ff4fc3497ce9708ec6544775e0b9ae55f48e7eb5",
}
_BOUNDED_PROFILE_OWNERS = frozenset(
    {
        "allocation_staging",
        "boundary_response",
        "event_broad_localize",
        "field_locate_sample",
        "ou_rng_charge",
        "physics_coefficients",
        "writer",
    }
)
_SELECTION_RELATIVE = Path("evidence/m3c2/caseP_100nm_final_campaign_v1/selection_receipt.json")
_REGISTRATION_RELATIVE = Path("evidence/m3c2/caseP_100nm_final_campaign_v1/final_registration.json")
_FINAL_RESULT_RELATIVE = Path(
    "evidence/m3c2/caseP_100nm_final_campaign_v1/final_result_receipt.json"
)
_DEFAULT_CAMPAIGN_RELATIVE = Path("_out_m3c2/caseP_100nm_candidate_final_v1")


@dataclass(frozen=True, slots=True)
class CasePAuthority:
    """Registered numerical setting and selected performance seeds."""

    level: str
    seeds: tuple[int, ...]
    setting: dict[str, object]
    selection_path: Path
    registration_path: Path
    final_result_path: Path
    allocation_path: Path
    candidate_manifest_path: Path
    performance_policy_path: Path
    hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class _CaseObservation:
    seed: int
    case_path: Path
    expected_particles: int | None
    fixture: bool
    baseline_result_path: Path | None
    baseline_manifest_sha256: str | None
    expected_scientific_payload_sha256: str | None


@dataclass(frozen=True, slots=True)
class _DecisionFacts:
    seeds: tuple[int, ...]
    owners: tuple[str, ...]
    shares: tuple[float, ...]
    wall_overhead_ratios: tuple[float, ...]
    process_overhead_ratios: tuple[float | None, ...]
    complete_seed_set: bool
    exact_pairs: bool
    baseline_identity: bool
    same_owner: bool
    bounded_owner: bool
    threshold_met: bool
    cross_seed_work_equal: bool
    profile_overhead_recorded: bool


def main(argv: list[str] | None = None) -> None:
    """Run the human-facing driver or one isolated measurement child."""

    arguments = _arguments(argv)
    if arguments.worker_spec is not None:
        _worker(arguments.worker_spec, arguments.worker_output)
        return
    report = _driver(arguments)
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if arguments.json_path is None:
        print(rendered, end="")
        return
    arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_path.write_text(rendered, encoding="utf-8")


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    solver_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "profile"), default="smoke")
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=solver_root / _DEFAULT_CAMPAIGN_RELATIVE,
        help="prepared final candidate campaign; used only by --suite profile",
    )
    parser.add_argument(
        "--fixture-case",
        type=Path,
        help="small public-API case required by --suite smoke",
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    result = parser.parse_args(argv)
    if result.worker_spec is not None:
        if result.worker_output is None:
            parser.error("--worker-spec requires --worker-output")
        return result
    if result.suite == "smoke" and result.fixture_case is None:
        parser.error("--suite smoke requires --fixture-case")
    if result.suite == "profile" and result.fixture_case is not None:
        parser.error("--fixture-case is valid only with --suite smoke")
    return result


def load_casep_authority(solver_root: Path | None = None) -> CasePAuthority:
    """Load and cross-check the committed Case-P final selection authorities."""

    root = Path(__file__).resolve().parents[2] if solver_root is None else solver_root.resolve()
    selection_path = root / _SELECTION_RELATIVE
    registration_path = root / _REGISTRATION_RELATIVE
    final_result_path = root / _FINAL_RESULT_RELATIVE
    selection = _load_json(selection_path, "Case-P selection receipt")
    registration = _load_json(registration_path, "Case-P final registration")
    final_result = _load_json(final_result_path, "Case-P final result receipt")
    if selection.get("status") != "AUTHORIZED_FOR_CONFIRMATORY_FINAL":
        raise ValueError("Case-P selection does not authorize the confirmatory final")
    execution = _mapping(registration.get("execution_authorization"), "execution authorization")
    if execution.get("status") != "AUTHORIZED":
        raise ValueError("Case-P final registration is not authorized")
    if final_result.get("status") != "PASS":
        raise ValueError("Case-P final result is not an accepted accuracy anchor")
    level, setting = _selected_setting(selection)
    allocation_path, profile_seeds = _selected_profile_seeds(
        selection_path,
        registration_path,
        selection,
        registration,
        execution,
    )
    candidate_manifest_path, performance_policy_path = _accepted_final_artifacts(
        final_result_path,
        final_result,
        selection_path,
        registration_path,
        setting,
    )
    return CasePAuthority(
        level=level,
        seeds=profile_seeds,
        setting=setting,
        selection_path=selection_path,
        registration_path=registration_path,
        final_result_path=final_result_path,
        allocation_path=allocation_path,
        candidate_manifest_path=candidate_manifest_path,
        performance_policy_path=performance_policy_path,
        hashes={
            "selection_receipt_sha256": _sha256(selection_path),
            "final_registration_sha256": _sha256(registration_path),
            "final_result_receipt_sha256": _sha256(final_result_path),
            "seed_allocation_sha256": _sha256(allocation_path),
            "candidate_campaign_manifest_sha256": _sha256(candidate_manifest_path),
            "performance_policy_sha256": _sha256(performance_policy_path),
        },
    )


def _selected_setting(selection: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    selected_levels = _mapping(selection.get("selected_final_levels"), "selected final levels")
    candidate = _mapping(selected_levels.get("candidate"), "selected candidate level")
    level = str(candidate.get("level_id"))
    setting = dict(_mapping(candidate.get("numerical_setting"), "candidate numerical setting"))
    if level != _EXPECTED_LEVEL or setting != _EXPECTED_SETTING:
        raise ValueError("Case-P selected candidate setting differs from the accepted setting")
    return level, setting


def _selected_profile_seeds(
    selection_path: Path,
    registration_path: Path,
    selection: Mapping[str, object],
    registration: Mapping[str, object],
    execution: Mapping[str, object],
) -> tuple[Path, tuple[int, int, int]]:
    selection_allocation = _mapping(selection.get("seed_allocation"), "selection seed allocation")
    registration_allocation = _mapping(
        registration.get("participant_seed_source"), "registration seed allocation"
    )
    allocation_path = _verified_reference(selection_path, selection_allocation)
    registration_allocation_path = _verified_reference(registration_path, registration_allocation)
    if allocation_path != registration_allocation_path:
        raise ValueError("Case-P selection and registration name different seed authorities")
    registration_selection = _mapping(
        execution.get("selection_receipt"), "registration selection receipt"
    )
    if _verified_reference(registration_path, registration_selection) != selection_path.resolve():
        raise ValueError("Case-P registration names a different selection receipt")
    allocation = _load_json(allocation_path, "Case-P final seed allocation")
    seed_sets = _mapping(allocation.get("participant_seed_sets"), "participant seed sets")
    candidate_seeds = tuple(int(value) for value in _sequence(seed_sets.get("candidate"), "seeds"))
    if len(candidate_seeds) < 3:
        raise ValueError("Case-P final candidate allocation has fewer than three seeds")
    profile_seeds = (
        candidate_seeds[0],
        candidate_seeds[(len(candidate_seeds) - 1) // 2],
        candidate_seeds[-1],
    )
    if profile_seeds != _EXPECTED_PROFILE_SEEDS:
        raise ValueError("Case-P first/middle/last performance seeds differ from the plan")
    return allocation_path, profile_seeds


def _accepted_final_artifacts(
    final_result_path: Path,
    final_result: Mapping[str, object],
    selection_path: Path,
    registration_path: Path,
    setting: Mapping[str, object],
) -> tuple[Path, Path]:
    design = _mapping(final_result.get("design"), "final result design")
    candidate_design = _mapping(design.get("candidate_setting"), "final candidate setting")
    final_design_checks = (
        design.get("particle_count") == _EXPECTED_PARTICLES,
        design.get("output_time_count") == _EXPECTED_OUTPUT_COUNT,
        design.get("time_end_s") == _EXPECTED_END_S,
        candidate_design.get("dt_s") == setting["dt_s"],
        candidate_design.get("brownian_interval_tree_depth")
        == setting["brownian_interval_tree_depth"],
        candidate_design.get("geometry_rtol") == setting["geometry_rtol"],
    )
    if not all(final_design_checks):
        raise ValueError("Case-P final result design differs from the selected setting")
    artifacts = _mapping(final_result.get("artifacts"), "final result artifacts")
    final_selection_path = _artifact_path(
        final_result_path, artifacts, "selection_receipt", "final selection artifact"
    )
    final_registration_path = _artifact_path(
        final_result_path, artifacts, "final_registration", "final registration artifact"
    )
    if (
        final_selection_path != selection_path.resolve()
        or final_registration_path != registration_path.resolve()
    ):
        raise ValueError("Case-P final result names different selection authorities")
    candidate_manifest_path = _artifact_path(
        final_result_path,
        artifacts,
        "candidate_campaign_manifest",
        "candidate campaign artifact",
    )
    performance_policy_path = _artifact_path(
        final_result_path, artifacts, "performance_record", "performance policy artifact"
    )
    performance_policy = _load_json(performance_policy_path, "Case-P performance policy")
    expected_policy = (
        "A stage is only a candidate for a bounded optimization when it is at least 25% "
        "of end-to-end wall time in a >=10000-particle accepted-accuracy workload."
    )
    if performance_policy.get("policy") != expected_policy:
        raise ValueError("Case-P performance policy differs from the registered 10k/25% gate")
    return candidate_manifest_path, performance_policy_path


def _artifact_path(
    owner: Path,
    artifacts: Mapping[str, object],
    key: str,
    label: str,
) -> Path:
    return _verified_reference(owner, _mapping(artifacts.get(key), label))


def _verified_reference(owner: Path, reference: Mapping[str, object]) -> Path:
    path = (owner.parent / str(reference.get("path"))).resolve()
    if not path.is_file() or _sha256(path) != reference.get("sha256"):
        raise ValueError(f"referenced authority differs: {path}")
    return path


def _driver(arguments: argparse.Namespace) -> dict[str, object]:
    suite: Suite = arguments.suite
    solver_root = Path(__file__).resolve().parents[2]
    authority = load_casep_authority(solver_root)
    cases = (
        _smoke_case(arguments.fixture_case, authority)
        if suite == "smoke"
        else _profile_cases(arguments.campaign_root.resolve(), authority)
    )
    with tempfile.TemporaryDirectory(prefix="chamber-particles-m3c2-casep-") as temporary:
        root = Path(temporary)
        observations = [
            _launch_worker(
                item,
                root / "runs" / f"seed-{item.seed}",
                root / "numba-cache" / f"seed-{item.seed}",
            )
            for item in cases
        ]
    return {
        "benchmark": "m3c2_casep_owner_profile_v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "suite": suite,
        "non_gating_seconds": True,
        "authority": _authority_record(authority, solver_root),
        "conditions": {
            "public_api_path": ["load_case", "simulate", "open_result"],
            "child_process_per_seed": True,
            "same_process_order": ["warmup", "unprofiled_measurement", "separate_cprofile"],
            "private_numba_cache_per_seed": True,
            "numba_num_threads": 1,
            "profile_scope": "load_case + simulate + open_result in a separate cProfile run",
            "owner_share_threshold": _OWNER_SHARE_THRESHOLD,
            "bounded_profile_owners": sorted(_BOUNDED_PROFILE_OWNERS),
            "minimum_particle_count_for_optimization": _MINIMUM_OPTIMIZATION_PARTICLES,
            "profile_particle_count": _EXPECTED_PARTICLES,
            "fixture_is_performance_evidence": False,
        },
        "machine": _machine_metadata(),
        "observations": observations,
        "decision": profile_decision(observations),
        "claim_limits": [
            "Absolute timing, process time, RSS, and profile overhead are machine-local.",
            "The smoke suite validates harness plumbing only.",
            "A consistent 287-particle owner is discovery only; the accepted policy still requires a 10k confirmation before optimization.",
            "cProfile self-time is an entry-point hotspot hypothesis; unattributed native time cannot authorize an owner.",
            "Case-P physical applicability remains NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED.",
            "No outcome authorizes a scientific or schema change.",
            "The harness neither changes nor times COMSOL.",
        ],
        "driver_sha256": _sha256(Path(__file__)),
    }


def _smoke_case(path: Path, authority: CasePAuthority) -> tuple[_CaseObservation, ...]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"smoke fixture case is missing: {resolved}")
    return (
        _CaseObservation(
            authority.seeds[0],
            resolved,
            None,
            True,
            None,
            None,
            None,
        ),
    )


def _profile_cases(campaign_root: Path, authority: CasePAuthority) -> tuple[_CaseObservation, ...]:
    manifest_path = campaign_root / "candidate_campaign_manifest.json"
    if _sha256(manifest_path) != authority.hashes["candidate_campaign_manifest_sha256"]:
        raise ValueError("Case-P prepared campaign differs from the accepted campaign manifest")
    manifest = _load_json(manifest_path, "Case-P candidate campaign manifest")
    expected_header = {
        "status": "COMPLETE",
        "purpose": "final",
        "particle_count": _EXPECTED_PARTICLES,
        "output_count": _EXPECTED_OUTPUT_COUNT,
        "time_end_s": _EXPECTED_END_S,
    }
    if any(manifest.get(key) != value for key, value in expected_header.items()):
        raise ValueError("Case-P candidate campaign does not match the accepted final shape")
    levels = _mapping(manifest.get("levels"), "candidate campaign levels")
    level = _mapping(levels.get(authority.level), "accepted candidate campaign level")
    if any(level.get(key) != value for key, value in authority.setting.items()):
        raise ValueError("Case-P campaign level differs from the registered setting")
    replicas = _sequence(level.get("replicas"), "candidate replicas")
    by_seed = {int(_mapping(value, "candidate replica")["seed"]): value for value in replicas}
    observations = []
    for seed in authority.seeds:
        replica = _mapping(by_seed.get(seed), f"candidate seed {seed}")
        _validate_replica_setting(replica, authority, seed)
        case_path = campaign_root / str(replica.get("case"))
        if not case_path.is_file() or _sha256(case_path) != replica.get("case_sha256"):
            raise ValueError(f"Case-P case identity differs for seed {seed}")
        _validate_case_document(case_path, authority, seed)
        baseline_result_path = campaign_root / str(replica.get("result"))
        baseline_manifest_path = baseline_result_path / "run.json"
        baseline_manifest_sha256 = str(replica.get("result_manifest_sha256"))
        if (
            not baseline_result_path.is_dir()
            or not baseline_manifest_path.is_file()
            or _sha256(baseline_manifest_path) != baseline_manifest_sha256
        ):
            raise ValueError(f"Case-P baseline result identity differs for seed {seed}")
        observations.append(
            _CaseObservation(
                seed,
                case_path,
                _EXPECTED_PARTICLES,
                False,
                baseline_result_path,
                baseline_manifest_sha256,
                _EXPECTED_FULL_DIGESTS[seed],
            )
        )
    return tuple(observations)


def _validate_replica_setting(
    replica: Mapping[str, object], authority: CasePAuthority, seed: int
) -> None:
    expected = {
        "status": "COMPLETE",
        "seed": seed,
        "level": authority.level,
        "dt_s": authority.setting["dt_s"],
        "brownian_interval_tree_depth": authority.setting["brownian_interval_tree_depth"],
        "geometry_rtol": authority.setting["geometry_rtol"],
    }
    if any(replica.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Case-P candidate replica setting differs for seed {seed}")


def _validate_case_document(case_path: Path, authority: CasePAuthority, seed: int) -> None:
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Case-P case is not a mapping: {case_path}")
    time_block = _mapping(document.get("time"), "case time")
    solver = _mapping(document.get("solver"), "case solver")
    event = _mapping(solver.get("event"), "case event")
    physics = _mapping(document.get("physics"), "case physics")
    noise = _mapping(physics.get("noise"), "case noise")
    output = _mapping(document.get("output"), "case output")
    trajectories = _mapping(output.get("trajectories"), "case trajectories")
    schedule = _mapping(trajectories.get("schedule"), "case output schedule")
    times = _sequence(schedule.get("explicit_times_s"), "case output times")
    checks = (
        solver.get("integrator") == "ou_langevin",
        solver.get("seed") == seed,
        time_block.get("end_s") == _EXPECTED_END_S,
        time_block.get("dt_s") == authority.setting["dt_s"],
        event.get("geometry_rtol") == authority.setting["geometry_rtol"],
        noise.get("interval_tree_depth") == authority.setting["brownian_interval_tree_depth"],
        trajectories.get("selection") == "all",
        len(times) == _EXPECTED_OUTPUT_COUNT,
    )
    if not all(checks):
        raise ValueError(f"Case-P case document differs from the accepted final: {case_path}")


def _launch_worker(
    item: _CaseObservation, output_root: Path, cache_directory: Path
) -> dict[str, object]:
    output_root.mkdir(parents=True)
    cache_directory.mkdir(parents=True)
    cache_initially_empty = not any(cache_directory.iterdir())
    if not cache_initially_empty:
        raise RuntimeError("Case-P profile worker cache was not initially empty")
    specification = output_root / "worker-spec.json"
    specification.write_text(
        json.dumps(
            {
                "seed": item.seed,
                "case_path": str(item.case_path),
                "expected_particles": item.expected_particles,
                "fixture": item.fixture,
                "baseline_result_path": (
                    None if item.baseline_result_path is None else str(item.baseline_result_path)
                ),
                "baseline_manifest_sha256": item.baseline_manifest_sha256,
                "expected_scientific_payload_sha256": (item.expected_scientific_payload_sha256),
                "numba_cache_directory": str(cache_directory),
                "numba_cache_initially_empty": cache_initially_empty,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "tests.performance.m3c2_casep",
        "--worker-spec",
        str(specification),
        "--worker-output",
        str(output_root / "outputs"),
    ]
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(cache_directory)
    environment["NUMBA_DISABLE_JIT"] = "0"
    environment["NUMBA_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Case-P profile worker for seed {item.seed} failed with exit "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        observation = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Case-P profile worker for seed {item.seed} returned invalid JSON"
        ) from error
    if not isinstance(observation, dict):
        raise RuntimeError("Case-P profile worker returned a non-object")
    return observation


def _worker(specification_path: Path, output_root: Path | None) -> None:
    if output_root is None:
        raise SystemExit("worker mode requires --worker-output")
    specification = _load_json(specification_path, "Case-P worker specification")
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        raise RuntimeError("Case-P profile workers require NUMBA_NUM_THREADS=1")
    cache_directory = Path(str(os.environ.get("NUMBA_CACHE_DIR"))).resolve()
    expected_cache_directory = Path(str(specification.get("numba_cache_directory"))).resolve()
    cache_matches_specification = cache_directory == expected_cache_directory
    cache_initially_empty = bool(specification.get("numba_cache_initially_empty"))
    if not cache_matches_specification or not cache_initially_empty:
        raise RuntimeError("Case-P profile worker requires its own initially empty Numba cache")
    seed = int(specification["seed"])
    case_path = Path(str(specification["case_path"]))
    expected = specification.get("expected_particles")
    expected_particles = None if expected is None else int(expected)
    fixture = bool(specification.get("fixture"))
    output_root.mkdir(parents=True, exist_ok=False)

    baseline = _baseline_identity(specification, expected_particles)

    warmup, _ = _public_run(
        case_path,
        output_root / "warmup",
        expected_particles,
        fixture=fixture,
        profile=False,
    )
    gc.collect()
    measured, _ = _public_run(
        case_path,
        output_root / "measured",
        expected_particles,
        fixture=fixture,
        profile=False,
    )
    gc.collect()
    profiled, profiler = _public_run(
        case_path,
        output_root / "profiled",
        expected_particles,
        fixture=fixture,
        profile=True,
    )
    if profiler is None:
        raise RuntimeError("Case-P profile worker did not create a profile")
    identity = _paired_identity(warmup, measured, profiled, baseline)
    if not all(
        identity[key]
        for key in (
            "exact_scientific_payload",
            "exact_work_counts",
            "exact_case_identity",
            "exact_algorithm_revisions",
        )
    ):
        raise RuntimeError("profiling changed Case-P science, work, case, or revisions")
    if not fixture and (
        not identity["exact_registered_baseline_science"]
        or not identity["exact_registered_baseline_work_counts"]
        or not identity["exact_registered_baseline_case_identity"]
        or not identity["exact_registered_baseline_algorithm_revisions"]
    ):
        raise RuntimeError("Case-P rerun differs from the accepted baseline")
    owner_self_seconds = _profile_owners(profiler)
    owner_shares = _owner_shares(owner_self_seconds)
    dominant_owner = next(iter(owner_shares))
    measured_timing = _mapping(measured["timing_s"], "measured timing")
    profile_timing = _mapping(profiled["timing_s"], "profile timing")
    simulate_process_ratio = _optional_ratio(
        float(profile_timing["simulate_process_s"]),
        float(measured_timing["simulate_process_s"]),
    )
    end_to_end_process_ratio = _optional_ratio(
        float(profile_timing["public_end_to_end_process_s"]),
        float(measured_timing["public_end_to_end_process_s"]),
    )
    observation = {
        "seed": seed,
        "fixture": fixture,
        "case_path": str(case_path),
        "case_sha256": _sha256(case_path),
        "environment": {
            "process_id": os.getpid(),
            "numba_cache_dir_private": cache_matches_specification,
            "numba_cache_dir_initially_empty": cache_initially_empty,
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
        },
        "warmup": warmup,
        "measured": measured,
        "accepted_baseline": baseline,
        "profile": {
            **profiled,
            "owner_self_seconds": owner_self_seconds,
            "owner_shares": owner_shares,
            "owner_share_basis": (
                "owner cProfile self seconds / all self seconds across the profiled "
                "load_case + simulate + open_result workflow"
            ),
            "dominant_owner": dominant_owner,
            "dominant_owner_share": owner_shares[dominant_owner],
            "top_functions": _profile_top_functions(profiler),
            "source_functions": _profile_source_functions(profiler),
        },
        "profile_overhead": {
            "simulate_wall_ratio": _ratio(
                float(profile_timing["simulate_wall_s"]),
                float(measured_timing["simulate_wall_s"]),
            ),
            "simulate_process_ratio": simulate_process_ratio,
            "public_end_to_end_wall_ratio": _ratio(
                float(profile_timing["public_end_to_end_wall_s"]),
                float(measured_timing["public_end_to_end_wall_s"]),
            ),
            "public_end_to_end_process_ratio": end_to_end_process_ratio,
            "public_end_to_end_wall_fraction": (
                _ratio(
                    float(profile_timing["public_end_to_end_wall_s"]),
                    float(measured_timing["public_end_to_end_wall_s"]),
                )
                - 1.0
            ),
            "public_end_to_end_process_fraction": (
                None if end_to_end_process_ratio is None else end_to_end_process_ratio - 1.0
            ),
            "process_ratio_available": end_to_end_process_ratio is not None,
            "excluded_from_baseline_timing": True,
        },
        "identity": identity,
    }
    print(json.dumps(observation, allow_nan=False, sort_keys=True))


def _baseline_identity(
    specification: Mapping[str, object], expected_particles: int | None
) -> dict[str, object] | None:
    raw_path = specification.get("baseline_result_path")
    if raw_path is None:
        return None
    baseline_path = Path(str(raw_path))
    manifest_path = baseline_path / "run.json"
    expected_manifest_sha256 = str(specification.get("baseline_manifest_sha256"))
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Case-P accepted baseline manifest changed before worker execution")
    result = open_result(baseline_path)
    manifest = _mapping(result.manifest, "accepted baseline manifest")
    counts = _mapping(manifest.get("counts"), "accepted baseline counts")
    if (
        manifest.get("status") != "complete"
        or int(counts.get("failure_events", -1)) != 0
        or expected_particles is None
        or int(counts.get("particles", -1)) != expected_particles
    ):
        raise RuntimeError("Case-P accepted baseline is not the complete 287-particle run")
    digest = _scientific_payload_digest(result)
    expected_digest = str(specification.get("expected_scientific_payload_sha256"))
    if digest != expected_digest:
        raise RuntimeError("Case-P accepted baseline full scientific digest changed")
    return {
        "result_path": str(baseline_path),
        "result_manifest_sha256": expected_manifest_sha256,
        "scientific_payload_sha256": digest,
        "expected_scientific_payload_sha256": expected_digest,
        "work_counts": _work_counts(manifest, require_diagnostics=True),
        "algorithm_revisions": {
            key: value for key, value in manifest.items() if key.endswith("_revision")
        },
        "result_artifact_bytes": _directory_bytes(baseline_path),
        "case_identity": {
            "case_file_hash": manifest.get("case_file_hash"),
            "data_content_hash": manifest.get("data_content_hash"),
        },
    }


def _public_run(
    case_path: Path,
    output: Path,
    expected_particles: int | None,
    *,
    fixture: bool,
    profile: bool,
) -> tuple[dict[str, object], cProfile.Profile | None]:
    peak_before, current_before, rss_source = _process_memory_bytes()
    profiler = cProfile.Profile() if profile else None
    if profiler is not None:
        profiler.enable()
    total_wall = time.perf_counter()
    total_process = time.process_time()
    case, load_wall, load_process = _timed_call(lambda: load_case(case_path))
    summary, simulate_wall, simulate_process = _timed_call(lambda: simulate(case, output))
    result, open_wall, open_process = _timed_call(lambda: open_result(output))
    total_wall_elapsed = time.perf_counter() - total_wall
    total_process_elapsed = time.process_time() - total_process
    if profiler is not None:
        profiler.disable()
    peak_after, current_after, _ = _process_memory_bytes()
    manifest = _mapping(result.manifest, "result manifest")
    counts = _mapping(manifest.get("counts"), "result counts")
    if manifest.get("status") != "complete" or int(counts.get("failure_events", -1)) != 0:
        raise RuntimeError("Case-P performance observation did not complete without failures")
    if expected_particles is not None and summary.particle_count != expected_particles:
        raise RuntimeError("Case-P performance particle count differs from the campaign")
    artifact_bytes = _directory_bytes(output)
    work_counts = _work_counts(manifest, require_diagnostics=not fixture)
    return (
        {
            "timing_s": {
                "load_case_wall_s": load_wall,
                "load_case_process_s": load_process,
                "simulate_wall_s": simulate_wall,
                "simulate_process_s": simulate_process,
                "open_result_wall_s": open_wall,
                "open_result_process_s": open_process,
                "public_end_to_end_wall_s": total_wall_elapsed,
                "public_end_to_end_process_s": total_process_elapsed,
            },
            "memory": {
                "rss_source": rss_source,
                "peak_rss_before_bytes": peak_before,
                "peak_rss_after_bytes": peak_after,
                "additional_process_high_water_bytes": max(0, peak_after - peak_before),
                "current_rss_before_bytes": current_before,
                "current_rss_after_bytes": current_after,
                "solver_memory_plan": manifest.get("memory_plan"),
            },
            "particle_count": summary.particle_count,
            "result_artifact_bytes": artifact_bytes,
            "result_artifact_bytes_per_simulate_wall_s": _ratio(
                float(artifact_bytes), simulate_wall
            ),
            "scientific_payload_sha256": _scientific_payload_digest(result),
            "scientific_payload_digest_revision": "p14_full_payload_label_order_v1",
            "work_counts": work_counts,
            "algorithm_revisions": {
                key: value for key, value in manifest.items() if key.endswith("_revision")
            },
            "case_identity": {
                "case_file_hash": case.case_file_hash,
                "data_content_hash": case.content_hash,
            },
        },
        profiler,
    )


def _timed_call(function: Any) -> tuple[Any, float, float]:
    wall_started = time.perf_counter()
    process_started = time.process_time()
    value = function()
    return value, time.perf_counter() - wall_started, time.process_time() - process_started


def _work_counts(manifest: Mapping[str, object], *, require_diagnostics: bool) -> dict[str, object]:
    counts = _mapping(manifest.get("counts"), "result counts")
    event_value = manifest.get("event_refinement")
    boundary_value = manifest.get("boundary_interactions")
    if require_diagnostics:
        event = _mapping(event_value, "event refinement")
        boundary = _mapping(boundary_value, "boundary interactions")
    else:
        event = event_value if isinstance(event_value, Mapping) else {}
        boundary = boundary_value if isinstance(boundary_value, Mapping) else {}
    particle_count = int(counts.get("particles", 0))
    macro_steps = int(counts.get("macro_steps", 0))
    return {
        "particles": particle_count,
        "macro_steps": macro_steps,
        "particle_macro_roots": particle_count * macro_steps,
        "brownian_interval_tree_depth": manifest.get("brownian_interval_tree_depth"),
        "ou_leaf_pieces": event.get("accepted_particle_pieces"),
        "accepted_particle_pieces": event.get("accepted_particle_pieces"),
        "candidate_queries": event.get("candidate_queries"),
        "refinements": event.get("refinements"),
        "maximum_refinement_depth": event.get("maximum_refinement_depth"),
        "wall_events": boundary.get("wall_events"),
        "axis_crossings": boundary.get("axis_crossings"),
        "residual_splits": boundary.get("residual_splits"),
        "boundary_events": counts.get("boundary_events"),
        "failure_events": counts.get("failure_events"),
        "frames": counts.get("frames"),
        "frame_rows": counts.get("frame_rows"),
    }


def _paired_identity(
    warmup: Mapping[str, object],
    measured: Mapping[str, object],
    profiled: Mapping[str, object],
    baseline: Mapping[str, object] | None,
) -> dict[str, object]:
    runs = (warmup, measured, profiled)
    digests = {str(value["scientific_payload_sha256"]) for value in runs}
    work = _serialized_run_values(runs, "work_counts")
    cases = _serialized_run_values(runs, "case_identity")
    revisions = _serialized_run_values(runs, "algorithm_revisions")
    identity: dict[str, object] = {
        "exact_scientific_payload": len(digests) == 1,
        "exact_work_counts": len(work) == 1,
        "exact_case_identity": len(cases) == 1,
        "exact_algorithm_revisions": len(revisions) == 1,
    }
    identity.update(_registered_baseline_comparison(baseline, digests, work, cases, revisions))
    return identity


def _serialized_run_values(runs: tuple[Mapping[str, object], ...], key: str) -> set[str]:
    return {_serialized_identity(value[key]) for value in runs}


def _registered_baseline_comparison(
    baseline: Mapping[str, object] | None,
    digests: set[str],
    work: set[str],
    cases: set[str],
    revisions: set[str],
) -> dict[str, object]:
    if baseline is None:
        return {
            "registered_baseline_available": False,
            "exact_registered_baseline_science": None,
            "exact_registered_baseline_work_counts": None,
            "exact_registered_baseline_case_identity": None,
            "exact_registered_baseline_algorithm_revisions": None,
        }
    return {
        "registered_baseline_available": True,
        "exact_registered_baseline_science": (
            str(baseline["scientific_payload_sha256"]) in digests and len(digests) == 1
        ),
        "exact_registered_baseline_work_counts": (
            _serialized_identity(baseline["work_counts"]) in work and len(work) == 1
        ),
        "exact_registered_baseline_case_identity": (
            _serialized_identity(baseline["case_identity"]) in cases and len(cases) == 1
        ),
        "exact_registered_baseline_algorithm_revisions": (
            _serialized_identity(baseline["algorithm_revisions"]) in revisions
            and len(revisions) == 1
        ),
    }


def _serialized_identity(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _profile_owners(profile: cProfile.Profile) -> dict[str, float]:
    owners: dict[str, float] = {}
    statistics_rows = cast(Any, pstats.Stats(profile)).stats
    for (filename, _line, name), statistics_row in statistics_rows.items():
        owner = _profile_owner(filename, name)
        owners[owner] = owners.get(owner, 0.0) + float(statistics_row[2])
    return dict(sorted(owners.items(), key=lambda item: item[1], reverse=True))


def _owner_shares(owner_seconds: Mapping[str, float]) -> dict[str, float]:
    total = math.fsum(owner_seconds.values())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("Case-P cProfile produced no finite owner self time")
    return {
        owner: seconds / total
        for owner, seconds in sorted(owner_seconds.items(), key=lambda item: item[1], reverse=True)
    }


def _profile_top_functions(profile: cProfile.Profile, limit: int = 20) -> dict[str, object]:
    rows: list[dict[str, Any]] = []
    statistics_rows = cast(Any, pstats.Stats(profile)).stats
    for (filename, line, name), statistics_row in statistics_rows.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = statistics_row
        rows.append(
            {
                "source": _profile_source(filename),
                "line": line,
                "function": name,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": float(self_seconds),
                "cumulative_seconds": float(cumulative_seconds),
            }
        )
    return {
        "top_by_self_seconds": sorted(
            rows,
            key=lambda item: (-float(item["self_seconds"]), str(item["source"]), int(item["line"])),
        )[:limit],
        "top_by_cumulative_seconds": sorted(
            rows,
            key=lambda item: (
                -float(item["cumulative_seconds"]),
                str(item["source"]),
                int(item["line"]),
            ),
        )[:limit],
    }


def _profile_source_functions(profile: cProfile.Profile) -> list[dict[str, object]]:
    """Keep source-local raw rows so later audits can reclassify owners."""

    rows: list[dict[str, Any]] = []
    statistics_rows = cast(Any, pstats.Stats(profile)).stats
    for (filename, line, name), statistics_row in statistics_rows.items():
        source = _profile_source(filename)
        if not source.startswith("chamber_particles/"):
            continue
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = statistics_row
        rows.append(
            {
                "source": source,
                "line": line,
                "function": name,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": float(self_seconds),
                "cumulative_seconds": float(cumulative_seconds),
            }
        )
    return sorted(
        rows, key=lambda item: (str(item["source"]), int(item["line"]), str(item["function"]))
    )


def _profile_source(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    for marker in ("/chamber_particles/", "/tests/performance/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    if normalized.startswith("{") or normalized.startswith("~"):
        return normalized
    return f"runtime_or_dependency/{normalized.rsplit('/', 1)[-1]}"


def _profile_owner(filename: str, function_name: str) -> str:
    normalized = filename.replace("\\", "/")
    if normalized.startswith(("{", "~")):
        return "native_unattributed"
    marker = "/chamber_particles/"
    if marker not in normalized:
        return "runtime_or_dependency"
    relative = normalized.split(marker, 1)[1]
    module = relative.split("/", 1)[0].removesuffix(".py")
    name = function_name.lower()
    if module in {"cpu", "engine"}:
        return _runtime_profile_owner(module, name)
    return _module_profile_owner(relative, module, name)


def _module_profile_owner(relative: str, module: str, name: str) -> str:
    if relative.startswith("physics/"):
        return "physics_coefficients"
    if module == "output":
        return _output_profile_owner(name)
    if module == "geometry":
        if _contains_profile_token(name, ("candidate", "aabb")):
            return "event_broad_localize"
        return "geometry_containment_or_prepare"
    if module == "fields":
        if _contains_profile_token(
            name,
            (
                "sample",
                "locate",
                "local_component",
                "local_regular",
                "local_unstructured",
                "count_local",
                "fill_local",
            ),
        ):
            return "field_locate_sample"
        return "field_prepare_or_bounds"
    return {
        "boundaries": "boundary_response",
        "events": "event_broad_localize",
        "rng": "ou_rng_charge",
        "stochastic": "ou_rng_charge",
    }.get(module, module)


def _output_profile_owner(name: str) -> str:
    if _contains_profile_token(
        name,
        (
            "write",
            "commit",
            "finalize",
            "begin_epoch",
            "append",
            "resizable",
            "atomic",
            "sync_file",
            "temporary_path",
            "create_partial",
            "require_running",
            "require_open_segment",
        ),
    ):
        return "writer"
    if _contains_profile_token(
        name,
        (
            "read",
            "iter",
            "open",
            "validate",
            "summary",
            "join",
            "reorder",
            "counts_from",
            "committed_segment",
            "checkpoint_counts",
            "one_dimensional",
            "has_layout",
            "candidate_ranges",
        ),
    ):
        return "result_read"
    return "output_shared_unattributed"


def _runtime_profile_owner(module: str, name: str) -> str:
    if module == "engine" and _contains_profile_token(name, ("write_", "flush_")):
        return "writer"
    if _contains_profile_token(
        name,
        (
            "event",
            "candidate",
            "refine",
            "chord",
            "localize",
            "locate_curved",
            "locate_exact",
            "build_curved_rows",
        ),
    ):
        return "event_broad_localize"
    if _contains_profile_token(name, ("field", "sample")):
        return "field_locate_sample"
    if module == "engine" and _contains_profile_token(
        name, ("brownian", "joint_ou", "langevin", "random", "rng", "charge")
    ):
        return "ou_rng_charge"
    if _contains_profile_token(name, ("alloc", "buffer", "pack", "scatter", "workspace")):
        return "allocation_staging"
    return "engine_orchestration" if module == "engine" else "compiled_tile"


def _contains_profile_token(name: str, tokens: tuple[str, ...]) -> bool:
    return any(token in name for token in tokens)


def profile_decision(observations: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Apply the 287-particle discovery gate without authorizing optimization."""

    facts = _decision_facts(observations)
    owner_discovery_consistent = all(
        (
            facts.complete_seed_set,
            facts.exact_pairs,
            facts.baseline_identity,
            facts.same_owner,
            facts.bounded_owner,
            facts.threshold_met,
            facts.cross_seed_work_equal,
            facts.profile_overhead_recorded,
        )
    )
    reasons = _decision_reasons(facts, owner_discovery_consistent)
    return {
        "required_seeds": list(_EXPECTED_PROFILE_SEEDS),
        "observed_seeds": list(facts.seeds),
        "complete_seed_set": facts.complete_seed_set,
        "paired_science_and_work_identity": facts.exact_pairs,
        "accepted_baseline_identity": facts.baseline_identity,
        "dominant_owner_consistent": facts.same_owner,
        "dominant_owner": facts.owners[0] if facts.same_owner else None,
        "dominant_owner_is_bounded": facts.bounded_owner,
        "dominant_owner_shares": list(facts.shares),
        "minimum_dominant_owner_share": min(facts.shares) if facts.shares else None,
        "owner_share_threshold": _OWNER_SHARE_THRESHOLD,
        "owner_share_threshold_met_in_every_seed": facts.threshold_met,
        "cross_seed_work_counts_equal": facts.cross_seed_work_equal,
        "profile_overhead": {
            "recorded_separately": facts.profile_overhead_recorded,
            "public_end_to_end_wall_ratios": list(facts.wall_overhead_ratios),
            "public_end_to_end_process_ratios": list(facts.process_overhead_ratios),
            "excluded_from_baseline_timing": facts.profile_overhead_recorded,
        },
        "owner_discovery_consistent": owner_discovery_consistent,
        "minimum_optimization_particle_count": _MINIMUM_OPTIMIZATION_PARTICLES,
        "10k_confirmation_required": owner_discovery_consistent,
        "optimization_authorized": False,
        "reasons": reasons,
    }


def _decision_facts(observations: Sequence[Mapping[str, Any]]) -> _DecisionFacts:
    seeds = tuple(sorted(int(item["seed"]) for item in observations))
    profiles = [_mapping(item.get("profile"), "profile observation") for item in observations]
    owners = tuple(str(item.get("dominant_owner")) for item in profiles)
    shares = tuple(float(item.get("dominant_owner_share", 0.0)) for item in profiles)
    identities = [_mapping(item.get("identity"), "paired identity") for item in observations]
    exact_pairs = _paired_decision_identity(identities)
    baseline_identity = _accepted_baseline_identity(observations, identities)
    cross_seed_work_equal = _cross_seed_work_equal(observations)
    wall_overhead_ratios, process_overhead_ratios, profile_overhead_recorded = (
        _profile_overhead_facts(observations)
    )
    same_owner = len(set(owners)) == 1 and bool(owners)
    bounded_owner = same_owner and owners[0] in _BOUNDED_PROFILE_OWNERS
    threshold_met = len(shares) == len(_EXPECTED_PROFILE_SEEDS) and all(
        share >= _OWNER_SHARE_THRESHOLD for share in shares
    )
    return _DecisionFacts(
        seeds=seeds,
        owners=owners,
        shares=shares,
        wall_overhead_ratios=wall_overhead_ratios,
        process_overhead_ratios=process_overhead_ratios,
        complete_seed_set=seeds == _EXPECTED_PROFILE_SEEDS,
        exact_pairs=exact_pairs,
        baseline_identity=baseline_identity,
        same_owner=same_owner,
        bounded_owner=bounded_owner,
        threshold_met=threshold_met,
        cross_seed_work_equal=cross_seed_work_equal,
        profile_overhead_recorded=profile_overhead_recorded,
    )


def _paired_decision_identity(identities: Sequence[Mapping[str, object]]) -> bool:
    return all(
        bool(item.get("exact_scientific_payload"))
        and bool(item.get("exact_work_counts"))
        and bool(item.get("exact_case_identity"))
        and bool(item.get("exact_algorithm_revisions"))
        for item in identities
    )


def _accepted_baseline_identity(
    observations: Sequence[Mapping[str, Any]], identities: Sequence[Mapping[str, object]]
) -> bool:
    return bool(observations) and all(
        not bool(item.get("fixture"))
        and bool(identity.get("registered_baseline_available"))
        and bool(identity.get("exact_registered_baseline_science"))
        and bool(identity.get("exact_registered_baseline_work_counts"))
        and bool(identity.get("exact_registered_baseline_case_identity"))
        and bool(identity.get("exact_registered_baseline_algorithm_revisions"))
        for item, identity in zip(observations, identities, strict=True)
    )


def _cross_seed_work_equal(observations: Sequence[Mapping[str, Any]]) -> bool:
    work_rows = {
        _serialized_identity(
            _mapping(
                _mapping(item.get("measured"), "measured observation").get("work_counts"),
                "work counts",
            )
        )
        for item in observations
    }
    return len(work_rows) == 1 and bool(work_rows)


def _profile_overhead_facts(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], tuple[float | None, ...], bool]:
    overheads = [
        _mapping(item.get("profile_overhead"), "profile overhead") for item in observations
    ]
    wall_overhead_ratios = tuple(
        float(item.get("public_end_to_end_wall_ratio", 0.0)) for item in overheads
    )
    process_overhead_ratios = tuple(
        _optional_float(item.get("public_end_to_end_process_ratio")) for item in overheads
    )
    profile_overhead_recorded = len(overheads) == len(observations) and all(
        _profile_overhead_row_recorded(item, observation, wall_ratio, process_ratio)
        for item, observation, wall_ratio, process_ratio in zip(
            overheads,
            observations,
            wall_overhead_ratios,
            process_overhead_ratios,
            strict=True,
        )
    )
    return wall_overhead_ratios, process_overhead_ratios, profile_overhead_recorded


def _profile_overhead_row_recorded(
    overhead: Mapping[str, object],
    observation: Mapping[str, Any],
    wall_ratio: float,
    process_ratio: float | None,
) -> bool:
    wall_recorded = (
        bool(overhead.get("excluded_from_baseline_timing"))
        and math.isfinite(wall_ratio)
        and wall_ratio > 0.0
    )
    if not wall_recorded:
        return False
    if bool(observation.get("fixture")):
        return process_ratio is None or (math.isfinite(process_ratio) and process_ratio > 0.0)
    return process_ratio is not None and math.isfinite(process_ratio) and process_ratio > 0.0


def _decision_reasons(facts: _DecisionFacts, owner_discovery_consistent: bool) -> list[str]:
    reasons: list[str] = []
    if not facts.complete_seed_set:
        reasons.append("three_registered_seeds_not_measured")
    if not facts.exact_pairs:
        reasons.append("profile_changed_science_or_work")
    if not facts.baseline_identity:
        reasons.append("accepted_baseline_identity_not_confirmed")
    if not facts.same_owner:
        reasons.append("dominant_owner_not_consistent_across_seeds")
    elif not facts.bounded_owner:
        reasons.append("dominant_owner_is_not_a_preregistered_bounded_change")
    if not facts.threshold_met:
        reasons.append("owner_share_below_25_percent_in_at_least_one_seed")
    if not facts.cross_seed_work_equal:
        reasons.append("work_counts_differ_across_seeds")
    if not facts.profile_overhead_recorded:
        reasons.append("profile_overhead_not_recorded_separately")
    if owner_discovery_consistent:
        reasons.extend(
            [
                "287_particle_owner_discovery_only",
                "10000_particle_accepted_accuracy_confirmation_required",
            ]
        )
    return reasons


def _authority_record(authority: CasePAuthority, solver_root: Path) -> dict[str, object]:
    return {
        "level": authority.level,
        "seeds": list(authority.seeds),
        "numerical_setting": authority.setting,
        "selection_receipt": str(authority.selection_path.relative_to(solver_root)).replace(
            "\\", "/"
        ),
        "final_registration": str(authority.registration_path.relative_to(solver_root)).replace(
            "\\", "/"
        ),
        "final_result_receipt": str(authority.final_result_path.relative_to(solver_root)).replace(
            "\\", "/"
        ),
        "seed_allocation": str(authority.allocation_path.relative_to(solver_root)).replace(
            "\\", "/"
        ),
        "candidate_campaign_manifest": str(
            authority.candidate_manifest_path.relative_to(solver_root)
        ).replace("\\", "/"),
        "performance_policy": str(
            authority.performance_policy_path.relative_to(solver_root)
        ).replace("\\", "/"),
        "minimum_optimization_particle_count": _MINIMUM_OPTIMIZATION_PARTICLES,
        **authority.hashes,
    }


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("Case-P timing ratio is not finite and positive")
    return numerator / denominator


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        return None
    return numerator / denominator


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
