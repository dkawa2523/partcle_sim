"""Top-level run configuration assembly and YAML persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ._configuration_core import (
    ConfigurationError,
    enum,
    error,
    finite_number,
    integer,
    mapping,
    reject_unknown,
    required,
)
from ._configuration_inputs import CaseConfig, InputsConfig
from ._configuration_physics import PhysicsConfig

SCHEMA_VERSION = 2


DEFAULT_MAX_SUBSTEP_SPLITS = 4
MAX_SUBSTEP_SPLITS_LIMIT = 12


@dataclass(frozen=True)
class TimeConfig:
    """Nominal macro step, end time, and adaptive refinement budget.

    ``max_substep_splits`` is the number of times one nominal step may be
    halved, so the substep budget is ``2 ** max_substep_splits``.  It is an
    explicit input because the value that resolves a smooth free flight is not
    the value that resolves a sheath transit or a near-wall approach, and a
    trace that exhausts the budget without a safety proof is stopped rather
    than accepted.
    """

    dt: float
    t_end: float
    max_substep_splits: int = DEFAULT_MAX_SUBSTEP_SPLITS

    @classmethod
    def from_mapping(cls, value: Any, path: str = "time") -> TimeConfig:
        data = mapping(value, path)
        reject_unknown(data, {"dt", "t_end", "max_substep_splits"}, path)
        dt = finite_number(
            required(data, "dt", path),
            f"{path}.dt",
            minimum=0.0,
            exclusive_minimum=True,
        )
        t_end = finite_number(
            required(data, "t_end", path), f"{path}.t_end", minimum=0.0
        )
        splits = (
            DEFAULT_MAX_SUBSTEP_SPLITS
            if data.get("max_substep_splits") is None
            else integer(
                data["max_substep_splits"],
                f"{path}.max_substep_splits",
                minimum=0,
            )
        )
        if splits > MAX_SUBSTEP_SPLITS_LIMIT:
            raise error(
                f"{path}.max_substep_splits",
                f"must be at most {MAX_SUBSTEP_SPLITS_LIMIT}",
            )
        return cls(dt=dt, t_end=t_end, max_substep_splits=int(splits))

    def to_mapping(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "dt": float(self.dt),
            "t_end": float(self.t_end),
        }
        if int(self.max_substep_splits) != DEFAULT_MAX_SUBSTEP_SPLITS:
            result["max_substep_splits"] = int(self.max_substep_splits)
        return result


@dataclass(frozen=True)
class OutputConfig:
    mode: str
    trajectory_interval_steps: int | None = None

    @classmethod
    def from_mapping(cls, value: Any, path: str = "output") -> OutputConfig:
        data = mapping(value, path)
        reject_unknown(data, {"mode", "trajectory_interval_steps"}, path)
        mode = enum(required(data, "mode", path), {"standard", "debug"}, f"{path}.mode")
        interval = (
            None
            if data.get("trajectory_interval_steps") is None
            else integer(
                data["trajectory_interval_steps"],
                f"{path}.trajectory_interval_steps",
                minimum=1,
            )
        )
        if mode == "standard" and interval is not None:
            raise error(
                f"{path}.trajectory_interval_steps",
                "is only valid when output.mode is debug",
            )
        if mode == "debug" and interval is None:
            raise error(
                f"{path}.trajectory_interval_steps",
                "is required when output.mode is debug",
            )
        return cls(mode=mode, trajectory_interval_steps=interval)

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.trajectory_interval_steps is not None:
            result["trajectory_interval_steps"] = int(self.trajectory_interval_steps)
        return result


@dataclass(frozen=True)
class RunConfig:
    case: CaseConfig
    inputs: InputsConfig
    physics: PhysicsConfig
    time: TimeConfig
    output: OutputConfig
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Any) -> RunConfig:
        data = mapping(value, "config")
        reject_unknown(
            data,
            {"schema_version", "case", "inputs", "physics", "time", "output"},
            "config",
        )
        version = integer(required(data, "schema_version", "config"), "schema_version")
        if version != SCHEMA_VERSION:
            raise error(
                "schema_version",
                f"must be {SCHEMA_VERSION}; use particle-tracer migrate "
                "for legacy inputs",
            )
        case_cfg = CaseConfig.from_mapping(required(data, "case", "config"))
        inputs = InputsConfig.from_mapping(
            required(data, "inputs", "config"),
            adapter=case_cfg.adapter,
            spatial_dim=case_cfg.spatial_dim,
        )
        physics = PhysicsConfig.from_mapping(
            required(data, "physics", "config"),
            adapter=case_cfg.adapter,
            spatial_dim=case_cfg.spatial_dim,
            coordinate_system=case_cfg.coordinate_system,
        )
        time = TimeConfig.from_mapping(required(data, "time", "config"))
        output = OutputConfig.from_mapping(required(data, "output", "config"))
        return cls(
            case=case_cfg, inputs=inputs, physics=physics, time=time, output=output
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case": self.case.to_mapping(),
            "inputs": self.inputs.to_mapping(),
            "physics": self.physics.to_mapping(),
            "time": self.time.to_mapping(),
            "output": self.output.to_mapping(),
        }


def parse_run_config(value: Mapping[str, Any]) -> RunConfig:
    """Parse an in-memory canonical mapping with strict unknown-key checks."""

    return RunConfig.from_mapping(value)


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    if value is None:
        value = {}
    return parse_run_config(value)


def dump_run_config(config: RunConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.to_mapping(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination
