from pathlib import Path
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(cfg: Path, out_dir: Path):
    cmd = [sys.executable, str(ROOT / 'run_from_yaml.py'), str(cfg), '--output-dir', str(out_dir)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    assert (out_dir / 'final_particles.csv').exists()
    assert (out_dir / 'prepared_runtime_summary.json').exists()
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    assert float(report['timing_s']['solver_core_s']) >= 0.0
    assert int(report['memory_estimate_bytes']['estimated_numpy_bytes']) > 0


@pytest.mark.parametrize(
    ('case_name', 'config_relpath'),
    [
        ('minimal_2d', Path('examples/minimal_2d/run_config.yaml')),
        ('minimal_3d', Path('examples/minimal_3d/run_config.yaml')),
    ],
)
def test_minimal_cli_runs(case_name: str, config_relpath: Path, tmp_path: Path):
    _run(ROOT / config_relpath, tmp_path / case_name)
