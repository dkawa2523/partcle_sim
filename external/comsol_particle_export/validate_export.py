from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from comsol_particle_export.validate_export import main


if __name__ == "__main__":
    raise SystemExit(main())
