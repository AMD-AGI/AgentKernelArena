"""Synthetic GPU probes used to attest sanitizer installations."""

from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parent

__all__ = ["PROBE_ROOT"]
