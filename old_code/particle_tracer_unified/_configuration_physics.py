"""Gas, stochastic, and force-model configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._configuration_charge import ChargeConfig
from ._configuration_core import (
    ConfigurationError,
    enum,
    error,
    finite_number,
    integer,
    mapping,
    reject_unknown,
    required,
    strict_bool,
)
from .force_models import (
    ForceModel,
    ForceModelError,
    force_model_to_native_mapping,
    parse_native_force_model,
)


@dataclass(frozen=True)
class GasConfig:
    temperature_K: float | None = None
    dynamic_viscosity_Pas: float | None = None
    density_kgm3: float | None = None
    molecular_mass_amu: float | None = None

    @classmethod
    def from_mapping(cls, value: Any, path: str = "physics.gas") -> GasConfig:
        data = mapping(value, path)
        allowed = {
            "temperature_K",
            "dynamic_viscosity_Pas",
            "density_kgm3",
            "molecular_mass_amu",
        }
        reject_unknown(data, allowed, path)
        parsed: dict[str, float | None] = {}
        for name in sorted(allowed):
            parsed[name] = (
                None
                if data.get(name) is None
                else finite_number(
                    data[name],
                    f"{path}.{name}",
                    minimum=0.0,
                    exclusive_minimum=True,
                )
            )
        return cls(**parsed)

    def require_for_drag(self, model: str) -> None:
        required_fields = {
            "none": (),
            "stokes": ("dynamic_viscosity_Pas",),
            "stokes_cunningham": (
                "temperature_K",
                "dynamic_viscosity_Pas",
                "density_kgm3",
                "molecular_mass_amu",
            ),
            "schiller_naumann": ("dynamic_viscosity_Pas", "density_kgm3"),
            "epstein": ("temperature_K", "density_kgm3", "molecular_mass_amu"),
        }[model]
        missing = [name for name in required_fields if getattr(self, name) is None]
        if missing:
            raise error(
                "physics.gas", f"drag model {model!r} requires {', '.join(missing)}"
            )

    def to_mapping(self) -> dict[str, float]:
        return {
            name: float(value)
            for name in (
                "temperature_K",
                "dynamic_viscosity_Pas",
                "density_kgm3",
                "molecular_mass_amu",
            )
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True)
class StochasticConfig:
    enabled: bool
    model: str = "underdamped_langevin"
    temperature_source: str = "field_T_then_gas"
    seed: int | None = None

    @classmethod
    def from_mapping(
        cls, value: Any, path: str = "physics.stochastic"
    ) -> StochasticConfig:
        data = mapping(value, path)
        reject_unknown(data, {"enabled", "model", "temperature_source", "seed"}, path)
        enabled = strict_bool(required(data, "enabled", path), f"{path}.enabled")
        model = enum(
            data.get("model", "underdamped_langevin"),
            {"underdamped_langevin"},
            f"{path}.model",
        )
        source = enum(
            data.get("temperature_source", "field_T_then_gas"),
            {"field_T_then_gas", "gas"},
            f"{path}.temperature_source",
        )
        seed = (
            None
            if data.get("seed") is None
            else integer(data["seed"], f"{path}.seed", minimum=0)
        )
        return cls(enabled=enabled, model=model, temperature_source=source, seed=seed)

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": bool(self.enabled),
            "model": self.model,
            "temperature_source": self.temperature_source,
        }
        if self.seed is not None:
            result["seed"] = int(self.seed)
        return result


def _validate_drag_declaration(adapter: str, drag_payload: Any, path: str) -> None:
    if adapter == "native" and drag_payload is None:
        raise error(
            path,
            "native adapter requires an explicit drag model "
            "(use model: none for ballistic motion)",
        )
    if adapter == "comsol" and drag_payload is not None:
        raise error(
            f"{path}.drag",
            "COMSOL drag is declared by the manifest force inventory",
        )


def _validate_force_declaration(
    adapter: str,
    raw_forces: Mapping[str, Any],
    path: str,
) -> None:
    if adapter == "comsol" and raw_forces:
        raise error(
            f"{path}.forces",
            "COMSOL force enablement and laws are declared only by the manifest",
        )


def _build_native_force_model(
    *,
    adapter: str,
    drag_payload: Any,
    raw_forces: Mapping[str, Any],
    spatial_dim: int,
    gas: GasConfig,
    path: str,
) -> ForceModel | None:
    if adapter != "native":
        return None
    try:
        force_model = parse_native_force_model(
            drag_payload,
            raw_forces,
            spatial_dim=spatial_dim,
            path=path,
        )
    except ForceModelError as exc:
        raise ConfigurationError(str(exc)) from exc
    gas.require_for_drag(force_model.drag.model)
    return force_model


def _validate_stochastic_drag(
    *,
    adapter: str,
    stochastic: StochasticConfig | None,
    force_model: ForceModel | None,
    path: str,
) -> None:
    if (
        adapter == "native"
        and stochastic is not None
        and stochastic.enabled
        and force_model is not None
        and force_model.drag.model == "none"
    ):
        raise error(f"{path}.stochastic", "requires a dissipative drag model")


def _validate_axisymmetric_physics(
    *,
    coordinate_system: str,
    stochastic: StochasticConfig | None,
    force_model: ForceModel | None,
    path: str,
) -> None:
    if coordinate_system != "axisymmetric_rz":
        return
    if stochastic is not None and stochastic.enabled:
        raise error(f"{path}.stochastic", "is not supported for axisymmetric_rz")
    if force_model is not None and force_model.lift.enabled:
        raise error(f"{path}.forces.lift", "is not supported for axisymmetric_rz")


DEFAULT_MAX_WALL_HITS_PER_STEP = 5
MAX_WALL_HITS_PER_STEP_LIMIT = 64


@dataclass(frozen=True)
class WallInteractionConfig:
    """How repeated contact with the same wall is resolved.

    ``contact_sliding`` is a numerical device, not a wall law.  When a particle
    reflects off one boundary part twice inside a single macro step it is
    pinned to that wall and advanced along its tangent, which bounds the Zeno
    behaviour of a particle pressed into a surface.  COMSOL's particle tracing
    has no contact model for point particles: it keeps resolving individual
    bounces.  A case built to reproduce COMSOL therefore disables sliding,
    while a case that needs a settled particle to stay put enables it.

    ``max_hits_per_step`` bounds how many wall events one macro step resolves.
    Reaching it with time left is a numerical stop, so a case that expects many
    bounces per step must raise it rather than accept a truncated step.
    """

    contact_sliding: bool = True
    max_hits_per_step: int = DEFAULT_MAX_WALL_HITS_PER_STEP

    @classmethod
    def from_mapping(
        cls, value: Any, path: str = "physics.wall_interaction"
    ) -> WallInteractionConfig:
        data = mapping(value, path)
        reject_unknown(data, {"contact_sliding", "max_hits_per_step"}, path)
        contact_sliding = (
            True
            if data.get("contact_sliding") is None
            else strict_bool(data["contact_sliding"], f"{path}.contact_sliding")
        )
        max_hits = (
            DEFAULT_MAX_WALL_HITS_PER_STEP
            if data.get("max_hits_per_step") is None
            else integer(
                data["max_hits_per_step"],
                f"{path}.max_hits_per_step",
                minimum=1,
            )
        )
        if max_hits > MAX_WALL_HITS_PER_STEP_LIMIT:
            raise error(
                f"{path}.max_hits_per_step",
                f"must be at most {MAX_WALL_HITS_PER_STEP_LIMIT}",
            )
        return cls(
            contact_sliding=bool(contact_sliding),
            max_hits_per_step=int(max_hits),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if not self.contact_sliding:
            result["contact_sliding"] = False
        if int(self.max_hits_per_step) != DEFAULT_MAX_WALL_HITS_PER_STEP:
            result["max_hits_per_step"] = int(self.max_hits_per_step)
        return result


@dataclass(frozen=True)
class PhysicsConfig:
    force_model: ForceModel | None
    gas: GasConfig
    charge: ChargeConfig | None = None
    stochastic: StochasticConfig | None = None
    seed: int = 12345
    wall_interaction: WallInteractionConfig = field(
        default_factory=WallInteractionConfig
    )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        adapter: str,
        spatial_dim: int,
        coordinate_system: str,
        path: str = "physics",
    ) -> PhysicsConfig:
        data = mapping(value, path)
        reject_unknown(
            data,
            {
                "drag",
                "gas",
                "forces",
                "charge",
                "stochastic",
                "seed",
                "wall_interaction",
            },
            path,
        )
        drag_payload = data.get("drag")
        _validate_drag_declaration(adapter, drag_payload, path)
        gas = GasConfig.from_mapping(data.get("gas", {}), f"{path}.gas")
        raw_forces = mapping(data.get("forces", {}), f"{path}.forces")
        _validate_force_declaration(adapter, raw_forces, path)
        force_model = _build_native_force_model(
            adapter=adapter,
            drag_payload=drag_payload,
            raw_forces=raw_forces,
            spatial_dim=spatial_dim,
            gas=gas,
            path=path,
        )
        charge = (
            None
            if data.get("charge") is None
            else ChargeConfig.from_mapping(data["charge"], f"{path}.charge")
        )
        stochastic = (
            None
            if data.get("stochastic") is None
            else StochasticConfig.from_mapping(data["stochastic"], f"{path}.stochastic")
        )
        seed = integer(data.get("seed", 12345), f"{path}.seed", minimum=0)
        _validate_stochastic_drag(
            adapter=adapter,
            stochastic=stochastic,
            force_model=force_model,
            path=path,
        )
        _validate_axisymmetric_physics(
            coordinate_system=coordinate_system,
            stochastic=stochastic,
            force_model=force_model,
            path=path,
        )
        return cls(
            force_model=force_model,
            gas=gas,
            charge=charge,
            stochastic=stochastic,
            seed=seed,
            wall_interaction=WallInteractionConfig.from_mapping(
                data.get("wall_interaction", {}),
                f"{path}.wall_interaction",
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "gas": self.gas.to_mapping(),
            "seed": int(self.seed),
        }
        if self.force_model is not None:
            drag, forces = force_model_to_native_mapping(self.force_model)
            result["drag"] = drag
            result["forces"] = forces
        else:
            result["forces"] = {}
        if self.charge is not None:
            result["charge"] = self.charge.to_mapping()
        if self.stochastic is not None:
            result["stochastic"] = self.stochastic.to_mapping()
        wall_interaction = self.wall_interaction.to_mapping()
        if wall_interaction:
            result["wall_interaction"] = wall_interaction
        return result
