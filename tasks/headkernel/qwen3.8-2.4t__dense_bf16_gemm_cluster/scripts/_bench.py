#!/usr/bin/env python3
"""Protected entrypoint for complete native-routing graph evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _native_route_bench import main

if __name__ == "__main__":
    main()
