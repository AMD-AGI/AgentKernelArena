#!/usr/bin/env python3
"""Run the protected generated-input correctness controller."""
from pathlib import Path
import subprocess
import sys
if __name__ == '__main__':
    task = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.run([sys.executable, str(task / 'scripts/generated_task_runner.py'), 'correctness'], cwd=task).returncode)
