from __future__ import annotations

from pathlib import Path

import yaml

from particle_tracer_unified.integrity import sha256_file

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "assets.yaml"


def test_large_binary_asset_ledger_matches_repository_files() -> None:
    payload = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["artifact_type"] == "large_binary_asset_ledger"

    assets = payload["assets"]
    assert assets
    logical_names = [str(asset["logical_name"]) for asset in assets]
    repository_paths = [str(asset["repository_path"]) for asset in assets]
    assert len(logical_names) == len(set(logical_names))
    assert len(repository_paths) == len(set(repository_paths))

    for asset in assets:
        assert set(asset) == {
            "logical_name",
            "repository_path",
            "size_bytes",
            "sha256",
            "external_uri",
            "status",
        }
        path = (ROOT / str(asset["repository_path"])).resolve()
        assert path.is_relative_to((ROOT / "data").resolve())
        assert path.is_file()
        assert path.stat().st_size == int(asset["size_bytes"])
        assert sha256_file(path) == str(asset["sha256"])
        assert asset["status"] in {
            "tracked_pending_external_uri",
            "externally_verified",
        }
        if asset["status"] == "externally_verified":
            assert isinstance(asset["external_uri"], str)
            assert asset["external_uri"]
