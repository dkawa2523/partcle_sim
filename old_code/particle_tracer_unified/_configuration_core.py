"""Strict scalar and mapping parsers shared by configuration sections."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


class ConfigurationError(ValueError):
    """A precise error in a canonical v0.2 configuration."""


def error(path: str, message: str) -> ConfigurationError:
    return ConfigurationError(f"{path}: {message}")


def mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise error(path, "must be a mapping")
    return {str(key): item for key, item in value.items()}


def reject_unknown(
    data: Mapping[str, Any], allowed: set[str] | frozenset[str], path: str
) -> None:
    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        raise error(path, f"unknown key(s): {', '.join(unknown)}")


def required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise error(path, f"missing required key {key!r}")
    return data[key]


def string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise error(path, "must be a string")
    if value != value.strip():
        raise error(path, "must not contain leading or trailing whitespace")
    if not value and not allow_empty:
        raise error(path, "must not be empty")
    return value


def strict_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise error(path, "must be a YAML boolean (true or false)")
    return bool(value)


def integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(path, "must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise error(path, f"must be >= {minimum}")
    return result


def finite_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise error(path, "must be finite")
    if minimum is not None:
        invalid = result <= minimum if exclusive_minimum else result < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            raise error(path, f"must be {operator} {minimum:g}")
    return result


def enum(value: Any, allowed: set[str] | frozenset[str], path: str) -> str:
    text = string(value, path)
    if text not in allowed:
        raise error(path, f"must be one of {', '.join(sorted(allowed))}")
    return text


def parameters(value: Any, path: str) -> dict[str, Any]:
    data = mapping(value, path)

    def reject_legacy_bool_strings(item: Any, item_path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject_legacy_bool_strings(child, f"{item_path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                reject_legacy_bool_strings(child, f"{item_path}[{index}]")
        elif isinstance(item, str) and item.strip().lower() in {
            "true",
            "false",
            "yes",
            "no",
            "on",
            "off",
        }:
            raise error(
                item_path, "boolean-like strings are not allowed; use a YAML boolean"
            )

    reject_legacy_bool_strings(data, path)
    return deepcopy(data)


def optional_string(value: Any, path: str) -> str | None:
    return None if value is None else string(value, path)
