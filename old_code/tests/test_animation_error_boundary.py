from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.artist import Artist

import tools.export_trajectory_animation as animation


def _empty_update(_frame: int) -> list[Artist]:
    return []


def _write_animation_input(
    output_dir: Path,
    *,
    positions: np.ndarray | None = None,
    times: list[float] | None = None,
    particle_count: int = 1,
    final_state: str = "active_free_flight",
) -> None:
    output_dir.mkdir()
    trajectory = (
        positions
        if positions is not None
        else np.asarray([[[0.0, 0.0]], [[1.0, 1.0]]], dtype=np.float64)
    )
    np.save(output_dir / "trajectory.npy", trajectory)
    pd.DataFrame({"time_s": times or [0.0, 1.0]}).to_csv(
        output_dir / "trajectory_frames.csv", index=False
    )
    pd.DataFrame(
        {
            "particle_id": np.arange(particle_count),
            "final_state": [final_state] * particle_count,
        }
    ).to_csv(output_dir / "final_particles.csv", index=False)


def test_missing_animation_input_is_recoverable(tmp_path: Path) -> None:
    with pytest.raises(
        animation.AnimationInputNotFoundError,
        match=r"trajectory\.npy not found",
    ) as caught:
        animation.export_trajectory_animations(tmp_path)

    assert isinstance(caught.value, animation.AnimationExportError)
    assert isinstance(caught.value, FileNotFoundError)


def test_downsample_selection_is_bounded_and_deterministic() -> None:
    assert animation._select_frame_indices(0, 3).size == 0
    assert animation._select_frame_indices(5, 1).tolist() == [0]
    assert animation._select_frame_indices(5, 3).tolist() == [0, 2, 4]
    assert animation._select_particle_indices(0, 2, "uniform").size == 0

    first = animation._select_particle_indices(8, 3, "random")
    second = animation._select_particle_indices(8, 3, "random")
    np.testing.assert_array_equal(first, second)
    with pytest.raises(animation.AnimationInputError, match="unsupported"):
        animation._select_particle_indices(8, 3, "first")


@pytest.mark.parametrize(
    ("positions", "spatial_dim", "message"),
    [
        (np.zeros((1, 2)), 2, "shaped"),
        (np.zeros((1, 1, 3)), 2, "dimensionality mismatch"),
        (np.zeros((0, 1, 2)), 2, "at least one frame"),
        (np.full((1, 1, 2), np.inf), 2, "finite values"),
    ],
)
def test_animation_position_validation_reports_invalid_input(
    positions: np.ndarray, spatial_dim: int, message: str
) -> None:
    with pytest.raises(animation.AnimationInputError, match=message):
        animation._validate_positions(positions, spatial_dim)


def test_animation_input_preserves_validation_order(tmp_path: Path) -> None:
    frame_mismatch = tmp_path / "frame_mismatch"
    _write_animation_input(frame_mismatch, times=[0.0])
    with pytest.raises(animation.AnimationInputError, match="time frame count"):
        animation.export_trajectory_animations(frame_mismatch)

    particle_mismatch = tmp_path / "particle_mismatch"
    _write_animation_input(particle_mismatch, particle_count=2)
    with pytest.raises(animation.AnimationInputError, match="particle count"):
        animation.export_trajectory_animations(particle_mismatch)

    invalid_state = tmp_path / "invalid_state"
    _write_animation_input(invalid_state, final_state="unknown")
    with pytest.raises(animation.AnimationInputError, match="unsupported final_state"):
        animation.export_trajectory_animations(invalid_state)


def test_animation_wall_event_read_failure_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_to_read(_output_dir: Path) -> pd.DataFrame:
        raise OSError("unreadable wall events")

    monkeypatch.setattr(animation, "load_wall_events", fail_to_read)
    with pytest.raises(animation.AnimationInputError, match="unreadable wall events"):
        animation._load_animation_wall_events(tmp_path, enabled=True)


def test_gif_writer_failure_closes_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenAnimation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def save(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(
        "tools.export_trajectory_animation.FuncAnimation", BrokenAnimation
    )
    fig = plt.figure()
    figure_number = fig.number

    with pytest.raises(animation.AnimationWriteError, match="disk full"):
        animation._write_gif(
            fig,
            _empty_update,
            frame_count=1,
            out_path=tmp_path / "animation.gif",
            fps=1,
        )

    assert not plt.fignum_exists(figure_number)


def test_gif_writer_does_not_hide_programming_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenAnimation:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def save(self, *_args: object, **_kwargs: object) -> None:
            raise AttributeError("unexpected bug")

    monkeypatch.setattr(
        "tools.export_trajectory_animation.FuncAnimation", BrokenAnimation
    )
    fig = plt.figure()
    figure_number = fig.number

    with pytest.raises(AttributeError, match="unexpected bug"):
        animation._write_gif(
            fig,
            _empty_update,
            frame_count=1,
            out_path=tmp_path / "animation.gif",
            fps=1,
        )

    assert not plt.fignum_exists(figure_number)
