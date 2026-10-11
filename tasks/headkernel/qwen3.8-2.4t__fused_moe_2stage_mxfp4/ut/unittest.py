#!/usr/bin/env python3
"""Compatibility entrypoint for the protected complete two-case correctness run."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    task = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable, str(task / 'scripts/task_runner.py'), 'correctness']))
