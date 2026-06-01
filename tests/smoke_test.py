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


def test_minimal_surface_release_production_example_runs_standard(tmp_path: Path):
    out_dir = tmp_path / 'minimal_surface_release_production'
    check_dir = tmp_path / 'minimal_surface_release_production_check'
    cfg = ROOT / 'examples' / 'minimal_surface_release_production' / 'run_config.yaml'

    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'run_from_yaml.py'),
            str(cfg),
            '--check-input',
            '--output-dir',
            str(check_dir),
        ],
        cwd=str(ROOT),
        check=True,
    )
    assert (check_dir / 'provider_contract_report.json').exists()
    assert (check_dir / 'input_contract_report.json').exists()
    assert (check_dir / 'source_particle_diagnostics.csv').exists()

    _run(cfg, out_dir)

    assert (out_dir / 'wall_summary.json').exists()
    assert (out_dir / 'coating_summary_by_part.csv').exists()
    assert not (out_dir / 'positions_2d.npy').exists()
    assert not (out_dir / 'wall_events.csv').exists()
    assert not (out_dir / 'runtime_step_summary.csv').exists()
    assert not (out_dir / 'source_particle_diagnostics.csv').exists()
    assert not (out_dir / 'collision_diagnostics.json').exists()
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    prepared = json.loads((out_dir / 'prepared_runtime_summary.json').read_text(encoding='utf-8'))
    wall_summary = json.loads((out_dir / 'wall_summary.json').read_text(encoding='utf-8'))
    assert report['output_mode'] == 'standard'
    assert wall_summary['by_part']['101']['stuck'] == 1
    source_summary = prepared['source_model_summary']
    assert int(source_summary['boundary_release_applied_count']) == 2
    assert float(source_summary['boundary_release_capture_tolerance_m']) == pytest.approx(5.0e-4)
    assert float(source_summary['boundary_release_inward_offset_m']) == pytest.approx(2.0e-6)
