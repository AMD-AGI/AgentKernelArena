#!/usr/bin/env python3
"""Public seven-action task runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_runtime import run
if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
