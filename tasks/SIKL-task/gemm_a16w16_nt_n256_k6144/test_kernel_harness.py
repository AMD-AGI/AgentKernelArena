#!/usr/bin/env python3
"""Alternative CLI entry for the public seven-action task runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
