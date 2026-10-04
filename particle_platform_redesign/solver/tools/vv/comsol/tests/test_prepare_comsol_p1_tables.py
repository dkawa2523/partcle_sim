from __future__ import annotations

from pathlib import Path

import numpy as np
from tools.vv.comsol.prepare_comsol_p1_tables import write_sectionwise_p1


def test_sectionwise_writer_preserves_exact_p1_connectivity(tmp_path: Path) -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    connectivity = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    path = tmp_path / "field.txt"

    write_sectionwise_p1(path, nodes, connectivity, "field", values)

    sections: dict[str, list[str]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%"):
            current = line
            sections[current] = []
        else:
            sections[current].append(line)
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Coordinates"]),
        nodes,
    )
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Elements"], dtype=np.int64),
        connectivity + 1,
    )
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Data (field)"]),
        values,
    )
