from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def resolve_path(repo_root: Path, path_value: Path) -> Path:
    return (
        path_value if path_value.is_absolute() else (repo_root / path_value).resolve()
    )


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _absolutize_config_paths(config: dict[str, Any], source_config: Path) -> None:
    base_dir = source_config.parent
    inputs = config.get("inputs", {})
    if not isinstance(inputs, dict):
        return
    for key in ("particles", "boundaries", "comsol_manifest"):
        value = inputs.get(key)
        if value is not None and str(value).strip():
            path = Path(str(value))
            inputs[key] = str(
                path if path.is_absolute() else (base_dir / path).resolve()
            )
    for section in ("geometry", "field"):
        provider = inputs.get(section, {})
        if isinstance(provider, dict) and provider.get("path") is not None:
            path = Path(str(provider["path"]))
            provider["path"] = str(
                path if path.is_absolute() else (base_dir / path).resolve()
            )


def write_config_variant(
    *,
    source_config: Path,
    output_config: Path,
    override_t_end: float | None,
    artifact_mode: str | None,
) -> Path:
    config = load_yaml_mapping(source_config)
    _absolutize_config_paths(config, source_config)
    if override_t_end is not None:
        config.setdefault("time", {})["t_end"] = float(override_t_end)
    if artifact_mode is not None:
        output = config.setdefault("output", {})
        output["mode"] = str(artifact_mode)
        if artifact_mode == "debug":
            output.setdefault("trajectory_interval_steps", 1)
        else:
            output.pop("trajectory_interval_steps", None)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_config


def parse_named_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = str(value).partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"Expected NAME=path, got: {value}")
    return str(name).strip(), Path(raw_path.strip())


_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RESERVED_RUN_NAMES = frozenset({"reference", "configs", "comparison_summary.json"})


def validate_run_specs(
    run_specs: Iterable[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    validated: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw_name, path in run_specs:
        name = str(raw_name)
        if _RUN_NAME_PATTERN.fullmatch(name) is None or name in _RESERVED_RUN_NAMES:
            raise ValueError(
                f"invalid run name {name!r}; use letters, digits, dot, underscore, "
                "or hyphen "
                "and avoid reserved comparison artifact names"
            )
        if name in seen:
            raise ValueError(f"duplicate comparison run name: {name}")
        seen.add(name)
        validated.append((name, Path(path)))
    return validated


def resolve_comparison_inputs(
    args: argparse.Namespace, repo_root: Path
) -> tuple[Path, list[tuple[str, Path]], Path]:
    reference_config = resolve_path(repo_root, args.reference_config)
    run_specs = validate_run_specs(
        (name, resolve_path(repo_root, path)) for name, path in args.run
    )
    output_root = resolve_path(repo_root, args.output_root)
    if not reference_config.exists():
        raise FileNotFoundError(f"reference config not found: {reference_config}")
    for run_name, run_config in run_specs:
        if not run_config.exists():
            raise FileNotFoundError(
                f"run config not found for {run_name}: {run_config}"
            )
    return reference_config, run_specs, output_root


def execution_configs(
    args: argparse.Namespace,
    *,
    reference_config: Path,
    run_specs: list[tuple[str, Path]],
    staging_dir: Path,
    comparison_dir: Path,
) -> tuple[Path, Path, list[tuple[str, Path, Path]]]:
    if args.override_t_end is None and args.artifact_mode is None:
        return (
            reference_config,
            reference_config,
            [(name, path, path) for name, path in run_specs],
        )
    staging_config_dir = staging_dir / "configs"
    published_config_dir = comparison_dir / "configs"
    execution_reference = write_config_variant(
        source_config=reference_config,
        output_config=staging_config_dir / f"{reference_config.stem}_reference.yaml",
        override_t_end=args.override_t_end,
        artifact_mode=args.artifact_mode,
    )
    execution_runs: list[tuple[str, Path, Path]] = []
    for run_name, run_config in run_specs:
        execution_config = write_config_variant(
            source_config=run_config,
            output_config=staging_config_dir / f"{run_name}.yaml",
            override_t_end=args.override_t_end,
            artifact_mode=args.artifact_mode,
        )
        execution_runs.append(
            (
                run_name,
                execution_config,
                published_config_dir / execution_config.name,
            )
        )
    return (
        execution_reference,
        published_config_dir / execution_reference.name,
        execution_runs,
    )
