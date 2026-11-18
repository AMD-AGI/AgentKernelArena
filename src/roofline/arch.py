"""Peak bandwidth and throughput of a target GPU."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import yaml

ARCH_DIR = Path(__file__).resolve().parent / "arch"
# ``matrix_bf16x3``: products emulated with 3 products at the ``matrix_bf16`` rate.
EMULATED_KEY = re.compile(r"(matrix_[a-z0-9]+)x([1-9][0-9]*)")


@dataclass(frozen=True)
class Peaks:
    name: str
    source: str
    mem_bw_gbps: float
    tflops: dict[str, float]
    unconfirmed: tuple[str, ...] = ()

    @property
    def bytes_per_s(self) -> float:
        return self.mem_bw_gbps * 1e9

    def matrix_flops_per_s(self, key: str) -> float:
        """The peak of a datasheet key, or of ``matrix_bf16xN``: one FLOP emulated with N bf16 products."""
        emulated = EMULATED_KEY.fullmatch(key)
        if emulated:
            return self.tflops[emulated[1]] * 1e12 / int(emulated[2])
        return self.tflops[key] * 1e12

    @property
    def vector_flops_per_s(self) -> float:
        """The fastest vector rate of any precision: a vector FLOP count is never charged at a slower one."""
        rates = [value for key, value in self.tflops.items() if key.startswith("vector_")]
        return max(rates) * 1e12

    def describe(self) -> dict[str, Any]:
        return {"source": self.source, "mem_bw_gbps": self.mem_bw_gbps}


def load_peaks(name_or_path: str) -> Peaks:
    """Load a bundled peak file by name (``mi355x_datasheet``) or any peak YAML/JSON by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = ARCH_DIR / f"{name_or_path}.yaml"
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    tflops = {str(k): float(v) for k, v in raw["max_achievable_tflops"].items()}
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    name = str(raw.get("name", path.stem))
    return Peaks(name=name, source=f"{name}@sha256:{digest}", mem_bw_gbps=float(raw["mem_bw_gbps"]),
                 tflops=tflops, unconfirmed=tuple(raw.get("unconfirmed", ())))
