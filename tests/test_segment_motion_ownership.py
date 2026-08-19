from __future__ import annotations

import ast
import inspect
from pathlib import Path

import particle_tracer_unified.solvers.segment_motion as public_motion
import particle_tracer_unified.solvers.segment_motion_batch as batch_motion
from particle_tracer_unified.solvers import _segment_motion_contracts as contracts
from particle_tracer_unified.solvers import _segment_motion_scalar as scalar_motion

SOLVERS = Path(public_motion.__file__).resolve().parent


def _local_imports(module_name: str) -> set[str]:
    source = (SOLVERS / f"{module_name}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }


def test_public_motion_api_is_a_direct_view_of_its_owners() -> None:
    expected_owners = {
        "SegmentMotionBatchDestination": contracts,
        "SegmentMotionBatchRequest": contracts,
        "SegmentMotionBatchTrace": batch_motion,
        "SegmentMotionRequest": contracts,
        "SegmentMotionTrace": scalar_motion,
        "ValidMaskPrefixResolution": contracts,
        "resolve_valid_mask_prefix": scalar_motion,
        "trace_motion_batch": batch_motion,
        "trace_motion_segment": scalar_motion,
    }

    assert set(public_motion.__all__) == set(expected_owners)
    for name, owner in expected_owners.items():
        assert getattr(public_motion, name) is getattr(owner, name)


def test_motion_owner_dependencies_are_one_way() -> None:
    public_source = (SOLVERS / "segment_motion.py").read_text(encoding="utf-8")
    public_tree = ast.parse(public_source)

    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for node in public_tree.body
    )
    assert _local_imports("_segment_motion_contracts").isdisjoint(
        {
            "segment_motion",
            "segment_motion_batch",
            "_segment_motion_scalar",
            "_segment_stage_dynamics",
        }
    )
    assert "segment_motion_batch" not in _local_imports("_segment_motion_scalar")
    assert "segment_motion" not in _local_imports("segment_motion_batch")
    assert "segment_motion_batch" not in _local_imports("_segment_stage_dynamics")


def test_public_motion_call_signatures_remain_stable() -> None:
    assert str(inspect.signature(public_motion.trace_motion_segment)) == (
        "(request: 'SegmentMotionRequest') -> 'SegmentMotionTrace'"
    )
    assert str(inspect.signature(public_motion.resolve_valid_mask_prefix)) == (
        "(request: 'SegmentMotionRequest', *, max_halving_count: 'int', "
        "require_clean_prefix: 'bool' = False) -> 'ValidMaskPrefixResolution'"
    )
    assert str(inspect.signature(public_motion.trace_motion_batch)) == (
        "(request: 'SegmentMotionBatchRequest', destination: "
        "'SegmentMotionBatchDestination | None' = None) -> 'SegmentMotionBatchTrace'"
    )
