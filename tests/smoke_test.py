from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def _run(cfg: Path, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [sys.executable, str(ROOT / 'run_from_yaml.py'), str(cfg), '--output-dir', str(out_dir)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    assert (out_dir / 'final_particles.csv').exists()
    assert (out_dir / 'prepared_runtime_summary.json').exists()


def test_minimal_2d():
    _run(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', ROOT / 'tests' / '_out_2d')


def test_minimal_3d():
    _run(ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', ROOT / 'tests' / '_out_3d')
