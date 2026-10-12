from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import write
from tests.scenarios.test_brownian_run import _brownian_first_passage_case


@pytest.mark.parametrize(("law", "terminal_code"), (("stick", 2), ("hold", 5), ("escape", 3)))
def test_terminal_cohort_preserves_continuing_brownian_particles_and_output(
    tmp_path: Path, law: str, terminal_code: int
) -> None:
    path = _brownian_first_passage_case(
        tmp_path / "base", particle_count=23, tree_depth=6, adaptive_max_depth=8, seed=82017
    )
    case = load_case(path)
    source = case.data.sources[0]
    continuing = np.arange(23) % 5 == 0
    velocity = source.velocity_m_s.copy()
    velocity[~continuing, 0] = 0.5
    source = replace(source, velocity_m_s=velocity)
    times = [index / 64 for index in range(65)]
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["boundaries"][0]["law"] = law
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": times},
    }
    results = []
    for label, selected in (("mixed", np.ones(23, dtype=np.bool_)), ("continuing", continuing)):
        cohort = replace(
            source,
            **{
                item.name: value[selected]
                for item in fields(source)
                if isinstance(value := getattr(source, item.name), np.ndarray)
            },
        )
        data_path = tmp_path / f"{label}.h5"
        info = write(data_path, replace(case.data, sources=(cohort,)))
        variant = {
            **document,
            "case": {
                **document["case"],
                "data_path": data_path.name,
                "expected_content_hash": info.content_hash,
            },
            "output": {
                **document["output"],
                "probes": {
                    "particle_ids": cohort.particle_id.tolist(),
                    "schedule": {"explicit_times_s": times},
                },
            },
        }
        variant_path = tmp_path / f"{label}.yaml"
        variant_path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"{label}-result"
        simulate(load_case(variant_path), output)
        results.append(open_result(output))

    mixed, isolated = results
    mixed_final = mixed.read_final()
    isolated_final = isolated.read_final()
    np.testing.assert_array_equal(mixed_final.lifecycle[~continuing], terminal_code)
    assert mixed.read_failure_events().particle_id.size == 0
    events = mixed.read_boundary_events()
    terminal_ids = source.particle_id[~continuing]
    terminal_events = np.flatnonzero(np.isin(events.particle_id, terminal_ids))
    terminal_events = terminal_events[np.argsort(events.particle_id[terminal_events])]
    np.testing.assert_array_equal(events.particle_id[terminal_events], terminal_ids)
    for final_name, event_name in (
        ("position_m", "position_post_m"),
        ("velocity_m_s", "velocity_post_m_s"),
        ("charge_number", "charge_number_post"),
    ):
        assert (
            getattr(mixed_final, final_name)[~continuing].tobytes()
            == getattr(events, event_name)[terminal_events].tobytes()
        )
    if law == "hold":
        assert (mixed_final.velocity_m_s[~continuing, 0] > 0.0).all()
    for item in fields(isolated_final):
        expected = getattr(isolated_final, item.name)
        actual = getattr(mixed_final, item.name)[continuing]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()

    for mixed_frame, isolated_frame in zip(
        mixed.iter_frames(), isolated.iter_frames(), strict=True
    ):
        assert mixed_frame.time_s == isolated_frame.time_s
        selected = np.isin(mixed_frame.particle_id, isolated_final.particle_id)
        for name in ("particle_id", "position_m", "velocity_m_s", "charge_number", "lifecycle"):
            assert (
                getattr(mixed_frame, name)[selected].tobytes()
                == getattr(isolated_frame, name).tobytes()
            )
        if mixed_frame.time_s >= times[1]:
            retired = np.isin(mixed_frame.particle_id, source.particle_id[~continuing])
            if law == "escape":
                assert not retired.any()
            else:
                np.testing.assert_array_equal(mixed_frame.particle_id[retired], terminal_ids)
                np.testing.assert_array_equal(mixed_frame.lifecycle[retired], terminal_code)
                expected_rows = np.searchsorted(
                    mixed_final.particle_id, mixed_frame.particle_id[retired]
                )
                for name in ("position_m", "velocity_m_s", "charge_number"):
                    assert (
                        getattr(mixed_frame, name)[retired].tobytes()
                        == getattr(mixed_final, name)[expected_rows].tobytes()
                    )

    for mixed_probe, isolated_probe in zip(
        mixed.iter_probes(), isolated.iter_probes(), strict=True
    ):
        assert mixed_probe.time_s == isolated_probe.time_s
        selected = np.isin(mixed_probe.particle_id, isolated_final.particle_id)
        for name in ("particle_id", "position_m", "velocity_m_s", "charge_number", "lifecycle"):
            assert (
                getattr(mixed_probe, name)[selected].tobytes()
                == getattr(isolated_probe, name).tobytes()
            )
