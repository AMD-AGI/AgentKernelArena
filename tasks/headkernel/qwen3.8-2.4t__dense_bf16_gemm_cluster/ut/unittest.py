#!/usr/bin/env python3
"""Compatibility entrypoint for the complete protected dense correctness phase.

Fresh results are under build; historical selection/negative evidence is immutable.
"""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    task = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.run([sys.executable, str(task / "scripts/task_runner.py"),
                                   "correctness", *sys.argv[1:]], check=False).returncode)
