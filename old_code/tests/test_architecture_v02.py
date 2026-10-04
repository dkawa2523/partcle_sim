from __future__ import annotations

import ast
import importlib
import re
from dataclasses import fields
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Any, get_args, get_origin, get_type_hints

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "particle_tracer_unified"
_MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _relative_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        if node.module:
            imports.add(node.module)
        else:
            imports.update(alias.name for alias in node.names)
    return imports


def _absolute_import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def _has_main_guard(path: Path) -> bool:
    for node in _tree(path).body:
        if not isinstance(node, ast.If):
            continue
        names = {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
        values = {
            item.value
            for item in ast.walk(node.test)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        if "__name__" in names and "__main__" in values:
            return True
    return False


def _assigns_process_argv(path: Path) -> bool:
    for node in ast.walk(_tree(path)):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            raw_targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            targets.extend(raw_targets)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "argv"
                and isinstance(target.value, ast.Name)
                and target.value.id == "sys"
            ):
                return True
    return False


def _assert_direct_exports(facade: ModuleType, owners: dict[str, ModuleType]) -> None:
    for name, owner in owners.items():
        assert getattr(facade, name) is getattr(owner, name)


def _parser_commands(parser: Any) -> set[str]:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            return {str(name) for name in choices}
    return set()


def _markdown_local_targets(path: Path) -> list[Path]:
    targets: list[Path] = []
    for raw_target in _MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
        target = raw_target.partition(" ")[0].strip("<>")
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        local_path = target.partition("#")[0]
        if local_path:
            targets.append((path.parent / local_path).resolve())
    return targets


def test_public_api_imports_only_the_lightweight_preflight_result_type() -> None:
    import particle_tracer_unified as package
    from particle_tracer_unified import application, preflight_types

    assert package.ValidationReport is preflight_types.ValidationReport
    assert (
        get_type_hints(application.validate_case)["return"]
        is preflight_types.ValidationReport
    )
    package_imports = _relative_imports(PACKAGE / "__init__.py")
    assert "preflight_types" in package_imports
    assert "preflight" not in package_imports


def test_application_facade_reexports_owned_public_value_types() -> None:
    from particle_tracer_unified import _application_types as types
    from particle_tracer_unified import application

    _assert_direct_exports(
        application,
        dict.fromkeys(
            (
                "ArtifactManifest",
                "ArtifactRecord",
                "RunStats",
                "SimulationCase",
                "SimulationPlan",
                "SimulationResult",
                "SimulationState",
            ),
            types,
        ),
    )


def test_distribution_has_one_console_script_and_no_root_runner() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {
        "particle-tracer": "particle_tracer_unified.cli:main"
    }
    assert not (ROOT / "run_from_yaml.py").exists()


def test_cli_routes_subcommands_without_mutating_process_argv() -> None:
    assert not _assigns_process_argv(PACKAGE / "cli.py")
    for module_name in (
        "particle_tracer_unified.compare.field_compare",
        "particle_tracer_unified.compare.acceleration_compare",
        "particle_tracer_unified.compare.trajectory_compare",
        "particle_tracer_unified.compare.boundary_compare",
        "tools.export_visualizations",
        "particle_tracer_unified.comsol_case.cli",
    ):
        entrypoint = importlib.import_module(module_name).main
        assert callable(entrypoint)
        assert "argv" in signature(entrypoint).parameters


def test_visualization_workers_are_libraries_not_duplicate_cli_entrypoints() -> None:
    for name in (
        "export_result_graphs.py",
        "export_trajectory_animation.py",
        "export_mechanics_visuals.py",
        "export_boundary_diagnostics_visuals.py",
    ):
        path = ROOT / "tools" / name
        assert "main" not in _top_level_definitions(path)
        assert not _has_main_guard(path)


def test_comsol_case_builder_has_one_way_responsibility_modules() -> None:
    from particle_tracer_unified import cli

    package = PACKAGE / "comsol_case"
    expected = {
        "builder.py",
        "cli.py",
        "contracts.py",
        "fields.py",
        "mesh.py",
        "profiles.py",
        "reporting.py",
    }
    assert expected <= {path.name for path in package.glob("*.py")}
    assert cli._COMSOL_COMMANDS["build-case"] == (
        "particle_tracer_unified.comsol_case.cli"
    )
    assert "argparse" not in _absolute_import_roots(package / "builder.py")
    for name in ("mesh.py", "fields.py", "contracts.py"):
        assert "builder" not in _relative_imports(package / name)


def test_visualize_cli_forwards_help_to_the_owned_parser(monkeypatch) -> None:
    from particle_tracer_unified import cli

    calls: list[tuple[str, tuple[str, ...]]] = []

    def routed(module_name: str, arguments) -> int:
        calls.append((module_name, tuple(arguments)))
        return 17

    monkeypatch.setattr(cli, "_module_main", routed)

    assert cli.main(["visualize", "--help"]) == 17
    assert calls == [("tools.export_visualizations", ("--help",))]


def test_root_package_exposes_only_the_four_application_operations() -> None:
    import particle_tracer_unified as package

    public_operations = {
        name
        for name in package.__all__
        if callable(getattr(package, name))
        and not isinstance(getattr(package, name), type)
    }
    assert public_operations == {
        "load_case",
        "simulate",
        "validate_case",
        "write_result",
    }
    assert {
        "load_run_config",
        "parse_run_config",
        "sample_one",
        "validate_artifacts",
    }.isdisjoint(package.__all__)


def test_solver_layer_does_not_import_io_comsol_or_visualization() -> None:
    forbidden_external_roots = {
        "json",
        "matplotlib",
        "pandas",
        "pathlib",
        "tools",
        "yaml",
    }
    offenders = {
        str(path.relative_to(ROOT)): sorted(
            forbidden_external_roots & _absolute_import_roots(path)
        )
        for path in sorted((PACKAGE / "solvers").rglob("*.py"))
        if forbidden_external_roots & _absolute_import_roots(path)
    }
    assert offenders == {}


def test_solver_boundary_dependency_uses_the_domain_protocol() -> None:
    from particle_tracer_unified.domain import BoundaryQuery
    from particle_tracer_unified.solvers.collision_detection import (
        classify_trial_collisions_2d,
        classify_trial_collisions_3d,
    )

    for operation in (classify_trial_collisions_2d, classify_trial_collisions_3d):
        annotation = get_type_hints(operation)["boundary_service"]
        assert get_origin(annotation) is BoundaryQuery


def test_removed_source_generation_and_json_schema_stay_removed() -> None:
    from particle_tracer_unified import cli

    commands = _parser_commands(cli._build_parser())
    assert {"run", "check", "migrate", "compare", "artifacts"} <= commands
    assert {"source", "generate-source", "schema"}.isdisjoint(commands)


def test_solver_context_has_no_legacy_runtime_wrapper_or_raw_config() -> None:
    from particle_tracer_unified.application import SimulationResult
    from particle_tracer_unified.configuration import RunConfig
    from particle_tracer_unified.core.datamodel import SolverContext

    assert "config" not in SolverContext.__dataclass_fields__
    assert not hasattr(RunConfig, "to_runtime_mapping")
    assert not hasattr(RunConfig, "_to_adapter_mapping")
    assert "case" not in SimulationResult.__dataclass_fields__


def test_solver_context_is_core_owned_with_typed_solver_binding() -> None:
    from particle_tracer_unified.core.datamodel import SolverContext
    from particle_tracer_unified.solvers.runtime_context import RuntimeSolverContext
    from particle_tracer_unified.solvers.runtime_plan import SolverPlan
    from particle_tracer_unified.solvers.runtime_setup import RuntimeOptions

    assert tuple(field.name for field in fields(SolverContext)) == (
        "spatial_dim",
        "coordinate_system",
        "particles",
        "geometry_provider",
        "field_provider",
        "gas",
        "wall_catalog",
        "force_catalog",
        "plan",
        "options",
    )
    assert get_origin(RuntimeSolverContext) is SolverContext
    assert get_args(RuntimeSolverContext) == (SolverPlan, RuntimeOptions)

    opaque: Any = object()
    plan: Any = object()
    options: Any = object()
    context = RuntimeSolverContext(
        spatial_dim=3,
        coordinate_system="cartesian_3d",
        particles=opaque,
        geometry_provider=opaque,
        field_provider=opaque,
        gas=opaque,
        wall_catalog=opaque,
        force_catalog=opaque,
        plan=plan,
        options=options,
    )
    assert type(context) is SolverContext
    assert context.plan is plan
    assert context.options is options


def test_boundary_protocol_keeps_required_geometry_argument_names() -> None:
    from particle_tracer_unified.domain import BoundaryQuery

    assert tuple(signature(BoundaryQuery.polyline_hit).parameters) == (
        "self",
        "start_m",
        "stage_points_m",
    )
    assert tuple(signature(BoundaryQuery.nearest_projection).parameters) == (
        "self",
        "point_m",
        "inside_reference_m",
    )


def test_production_context_builder_consumes_typed_physics_directly() -> None:
    from particle_tracer_unified.configuration import RunConfig
    from particle_tracer_unified.core.datamodel import SolverContext
    from particle_tracer_unified.io import runtime_builder

    annotations = get_type_hints(runtime_builder.build_solver_context)
    assert annotations["config"] is RunConfig
    assert annotations["return"] is SolverContext
    assert runtime_builder.__all__ == ("build_solver_context",)


def test_solver_modules_do_not_reintroduce_raw_config_parsers_or_dead_updates() -> None:
    from particle_tracer_unified.solvers import _charge_model_types, charge_model
    from particle_tracer_unified.solvers.runtime_plan import SolverPlan
    from particle_tracer_unified.solvers.runtime_setup import RuntimeOptions

    assert charge_model.ChargeModelConfig is _charge_model_types.ChargeModelConfig
    assert SolverPlan.__module__.endswith("runtime_plan")
    assert RuntimeOptions.__module__.endswith("runtime_setup")


def test_brownian_saved_path_has_one_composition_owner() -> None:
    from particle_tracer_unified.solvers import (
        _stochastic_composition as composition,
    )
    from particle_tracer_unified.solvers import stochastic_motion
    from particle_tracer_unified.solvers.segment_motion import SegmentMotionTrace

    _assert_direct_exports(
        stochastic_motion,
        dict.fromkeys(
            (
                "compose_piecewise_langevin_paths",
                "compose_piecewise_langevin_state",
                "compose_piecewise_langevin_trace",
                "resolve_piecewise_valid_mask_prefix",
            ),
            composition,
        ),
    )
    assert "stochastic_motion" not in _relative_imports(
        PACKAGE / "solvers" / "_stochastic_composition.py"
    )
    assert {
        "tau_start_s",
        "tau_mid_s",
    } <= SegmentMotionTrace.__dataclass_fields__.keys()


def test_force_registry_exposes_only_the_typed_constructor() -> None:
    from particle_tracer_unified.solvers import forces
    from particle_tracer_unified.solvers.forces import registry, runtime

    assert forces.resolve_force_catalog is registry.resolve_force_catalog
    assert (
        forces.compile_force_runtime_parameters
        is runtime.compile_force_runtime_parameters
    )
    assert set(registry.__all__) == {
        "ForceBinding",
        "ForceCatalog",
        "force_catalog_summary",
        "resolve_force_catalog",
    }


def test_force_semantics_have_one_domain_owner_and_no_mapping_handoff_types() -> None:
    from particle_tracer_unified import (
        _force_model_parsing as parsing,
    )
    from particle_tracer_unified import (
        _force_model_serialization as serialization,
    )
    from particle_tracer_unified import (
        _force_model_types as types,
    )
    from particle_tracer_unified import (
        force_models,
    )

    _assert_direct_exports(
        force_models,
        {
            "ForceModel": types,
            "ForceModelError": types,
            "parse_manifest_force_model": parsing,
            "parse_native_force_model": parsing,
            "force_model_to_native_mapping": serialization,
        },
    )


def test_old_contract_and_solver_writer_modules_are_deleted() -> None:
    from particle_tracer_unified import _application_types, application, domain, writer

    assert application.SimulationResult is _application_types.SimulationResult
    assert domain.BoundaryHit.__module__ == domain.__name__
    assert writer.write_result.__module__ == writer.__name__


def test_runtime_state_execution_and_contact_have_one_way_ownership() -> None:
    from particle_tracer_unified.solvers import (
        _contact_dynamics,
        _runtime_execution_context,
        _runtime_outcome,
        _runtime_preparation,
        contact_sliding,
        runtime_execution,
    )

    _assert_direct_exports(
        runtime_execution,
        {
            "RunExecutionContext": _runtime_execution_context,
            "StepLoopResult": _runtime_execution_context,
            "append_snapshot": _runtime_outcome,
            "finalize_runtime_execution": _runtime_outcome,
            "initialize_debug_buffers": _runtime_outcome,
            "prepare_runtime_execution": _runtime_preparation,
        },
    )
    _assert_direct_exports(
        contact_sliding,
        {
            "ContactDynamicsBatch": _contact_dynamics,
            "advance_contact_relaxation": _contact_dynamics,
            "displaced_fluid_factors": _contact_dynamics,
        },
    )
    for name in (
        "_contact_dynamics.py",
        "_contact_sliding_2d.py",
        "_contact_sliding_3d.py",
        "_runtime_execution_context.py",
        "_runtime_outcome.py",
        "_runtime_preparation.py",
    ):
        imports = _relative_imports(PACKAGE / "solvers" / name)
        assert "high_fidelity_runtime" not in imports
        assert "contact_sliding" not in imports


def test_boundary_facade_has_no_obsolete_bisection_or_boolean_mask_api() -> None:
    from particle_tracer_unified.core import (
        _boundary_contact_2d,
        _boundary_hits_2d,
        _boundary_hits_3d,
        boundary_hits,
    )

    _assert_direct_exports(
        boundary_hits,
        {
            "nearest_hit_on_boundary_edges": _boundary_contact_2d,
            "nearest_boundary_edge_features_2d": _boundary_hits_2d,
            "segment_hit_from_boundary_edges": _boundary_hits_2d,
            "segment_hit_from_boundary_triangles": _boundary_hits_3d,
        },
    )


def test_collision_solver_has_explicit_responsibility_modules_without_facades() -> None:
    from particle_tracer_unified.solvers import (
        _collision_detection_trace,
        _collision_detection_types,
        _collision_types,
        collision_detection,
        high_fidelity_collision,
    )

    _assert_direct_exports(
        high_fidelity_collision,
        {
            "CollidingParticleAdvanceResult": _collision_types,
            "CollisionSegmentInputs": _collision_types,
        },
    )
    _assert_direct_exports(
        collision_detection,
        {
            "TrialCollisionBatch": _collision_detection_types,
            "promote_stage_trace_collisions": _collision_detection_trace,
        },
    )
    for name in (
        "collision_detection.py",
        "collision_hit_localization.py",
        "wall_response.py",
        "_collision_types.py",
        "_collision_resolution.py",
        "_collision_wall_events.py",
        "_collision_detection_trace.py",
        "_collision_detection_types.py",
        "_collision_detection_2d.py",
        "_collision_detection_3d.py",
        "_collision_detection_candidates.py",
    ):
        assert "high_fidelity_collision" not in _relative_imports(
            PACKAGE / "solvers" / name
        )


def test_compiled_field_backend_has_one_way_responsibility_modules() -> None:
    from particle_tracer_unified.solvers import (
        base_field_sampling,
        compiled_backend_types,
        field_compilation,
        force_field_assembly,
    )

    backend_types = set(get_args(compiled_backend_types.CompiledRuntimeBackend))
    assert backend_types == {
        compiled_backend_types.RegularRectilinearCompiledBackend,
        compiled_backend_types.TriangleMesh2DCompiledBackend,
    }
    assert field_compilation.compile_runtime_backend.__module__ == (
        field_compilation.__name__
    )
    assert base_field_sampling.sample_regular_time_grid_points_2d.__module__ == (
        base_field_sampling.__name__
    )
    assert force_field_assembly.sample_compiled_stage_fields.__module__ == (
        force_field_assembly.__name__
    )

    imports = {
        name: _relative_imports(PACKAGE / "solvers" / f"{name}.py")
        for name in (
            "compiled_backend_types",
            "field_compilation",
            "base_field_sampling",
        )
    }
    assert all("force_field_assembly" not in values for values in imports.values())
    assert "base_field_sampling" not in imports["field_compilation"]
    assert "field_compilation" not in imports["base_field_sampling"]


def test_motion_progression_has_one_numerical_owner() -> None:
    from particle_tracer_unified.solvers import (
        _segment_motion_contracts as contracts,
    )
    from particle_tracer_unified.solvers import (
        _segment_motion_scalar as scalar,
    )
    from particle_tracer_unified.solvers import (
        segment_motion,
    )
    from particle_tracer_unified.solvers import (
        segment_motion_batch as batch,
    )

    _assert_direct_exports(
        segment_motion,
        {
            "SegmentMotionBatchDestination": contracts,
            "SegmentMotionBatchRequest": contracts,
            "SegmentMotionRequest": contracts,
            "SegmentMotionTrace": scalar,
            "trace_motion_batch": batch,
            "trace_motion_segment": scalar,
        },
    )
    assert "segment_motion_batch" not in _relative_imports(
        PACKAGE / "solvers" / "_segment_motion_scalar.py"
    )
    assert "segment_motion" not in _relative_imports(
        PACKAGE / "solvers" / "segment_motion_batch.py"
    )
    assert _top_level_definitions(PACKAGE / "solvers" / "segment_motion.py") == set()


def test_documentation_is_consolidated_into_the_five_product_guides() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "comsol_vv.md",
        ROOT / "docs" / "input_artifacts.md",
        ROOT / "docs" / "physics_numerics.md",
    )
    assert all(path.is_file() for path in documents)
    broken_links = {
        str(path.relative_to(ROOT)): [
            str(target.relative_to(ROOT))
            if target.is_relative_to(ROOT)
            else str(target)
            for target in _markdown_local_targets(path)
            if not target.exists()
        ]
        for path in documents
    }
    assert not {path: targets for path, targets in broken_links.items() if targets}
