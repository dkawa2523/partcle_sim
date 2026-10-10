"""Safe YAML document syntax shared by configuration consumers."""

from __future__ import annotations

import yaml


class _DocumentLoader(yaml.SafeLoader):
    """Construct mappings without duplicate or implicit merge keys."""


def _construct_mapping(
    loader: _DocumentLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("YAML merge keys are not supported; write explicit keys")
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable scalar values") from error
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DocumentLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_document(raw: bytes) -> object:
    """Decode one safe UTF-8 document; domain consumers validate its meaning."""
    try:
        return yaml.load(raw.decode("utf-8"), Loader=_DocumentLoader)
    except UnicodeDecodeError as error:
        raise ValueError("YAML document is not valid UTF-8") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML document: {error}") from error
