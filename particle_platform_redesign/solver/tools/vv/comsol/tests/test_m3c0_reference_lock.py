from __future__ import annotations

from pathlib import Path

from tools.vv.comsol.lock_m3c0_reference import (
    _allowed_variant_difference,
    _normalized_text_sha256,
    _seed_rows,
    load_config,
)


def test_comsol_export_date_does_not_change_normalized_payload_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text('% Version,6.4\n% Date,"one"\n1,2\n', encoding="utf-8")
    second.write_text('% Version,6.4\n% Date,"two"\n1,2\n', encoding="utf-8")

    assert _normalized_text_sha256(first) == _normalized_text_sha256(second)


def test_only_ion_drag_force_expression_is_an_allowed_pair_difference() -> None:
    assert _allowed_variant_difference("particle_physics_feature_settings.csv:fpt/idf/[3]/F")
    assert not _allowed_variant_difference("particle_physics_feature_settings.csv:fpt/liftfm/[3]/F")
    assert not _allowed_variant_difference(
        "particle_physics_feature_settings.csv:fpt/idf/[3]/ParticlesToAffect"
    )


def test_reference_config_preregisters_unique_seed_per_package() -> None:
    config_path = Path(__file__).parents[1] / "cases/m3c0_reference_lock.yaml"
    config = load_config(config_path)
    package_rows = [
        {"case_id": f"{variant}/{case}"} for variant in config.variants for case in config.cases
    ]

    rows = _seed_rows(config, package_rows)

    assert len(package_rows) == 12
    assert len(rows) == 12 * 32
    assert len({row["seed"] for row in rows}) == len(rows)
