"""The arithmetic a float32 matrix product is charged at.

Operator models charge a product of float32 operands at the ``matrix_fp32`` peak, as TraceLens does.
A kernel does not have to multiply in float32: it can split each float32 operand into 16-bit parts
and multiply those on the bf16/fp16 matrix cores with float32 accumulation, and it is correct
whenever the result still passes the task's tolerance. Each kept part product runs at the 16-bit
rate, so ``N`` products are charged at the 16-bit peak divided by ``N`` (key ``matrix_bf16xN`` or
``matrix_fp16xN``):

| Arithmetic | Products | What a kernel does | MI355X datasheet rate |
|---|---|---|---|
| ``bf16``   | 1 | rounds both operands to bf16                            | 2516.6 TFLOPS |
| ``bf16x3`` | 3 | hi*hi + hi*lo + lo*hi of a two-part bf16 split           | 838.9 |
| ``fp16x3`` | 3 | the same with fp16 parts (11-bit significands)           | 838.9 |
| ``bf16x6`` | 6 | three-part bf16 split, all products of order <= 2        | 419.4 |
| ``bf16x9`` | 9 | three-part bf16 split, every product (float32-exact)     | 279.6 |
| ``fp32``   | - | float32 matrix instructions                              | 157.3 |

Which of these passes depends on the task's tolerance and inputs, so it is measured, not derived: a
precision probe runs each performance case through the evaluator's own correctness check with every
matrix product replaced by the emulated arithmetic. Its results are measurements and stay outside
the repository, like measured peaks. ``floors --precision-probe <file>`` reads them::

    {"source": "<probe name, GPU, date>",
     "tasks": {"<task_id>": {"package_sha256": "<task_package_sha256 the probe ran on>",
                             "arithmetic": "<cheapest passing arithmetic of most cases>",
                             "cases": {"<case_id>": "<arithmetic, where it differs>"}}}}

Charging. Without a probe file every float32 product is charged at the float32 matrix peak. With
one, a case is charged at the cheapest arithmetic the probe saw pass for it. A task the probe did
not cover, or whose package (``task_package_sha256``, as task materialization computes it) changed
since the probe, stays at the float32 peak, and its note says so. Every float32 product keeps the
``precision_substitution_sensitive`` note: a kernel may find a cheaper arithmetic than the floor
charges, which a measurement beating the floor would reveal.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
# Products of 16-bit matrix instructions per emulated product, and the peak they run at.
PRODUCTS = {"bf16": ("matrix_bf16", 1), "bf16x3": ("matrix_bf16", 3), "fp16x3": ("matrix_fp16", 3),
            "bf16x6": ("matrix_bf16", 6), "bf16x9": ("matrix_bf16", 9)}
FP32 = "fp32"
ARITHMETIC = (*PRODUCTS, FP32)
UNPROBED = FP32


def load_probe(path: str | Path) -> dict:
    """Read a precision-probe file (format in the module docstring) and check its arithmetic names."""
    probe = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    for task_id, entry in probe["tasks"].items():
        if len(entry["package_sha256"]) != 64:
            raise ValueError(f"{task_id}: package_sha256 is not a SHA-256 digest")
        for arithmetic in (entry["arithmetic"], *entry.get("cases", {}).values()):
            if arithmetic not in ARITHMETIC:
                raise ValueError(f"{task_id}: unknown arithmetic {arithmetic!r}")
    return probe


@lru_cache(maxsize=None)
def package_sha256(task_dir: Path) -> str:
    """The digest task materialization records as ``task_package_sha256``."""
    from ..task_materialization import _STALE_OUTPUTS, _Deadline, _tree_manifest, package_digest

    return package_digest(_tree_manifest(Path(task_dir), _Deadline(600), excludes=tuple(_STALE_OUTPUTS)))


def probed_arithmetic(probe: dict | None, task_id: str, case_id: str) -> str | None:
    """The cheapest arithmetic the probe saw pass for one case, or None when it has no result."""
    entry = (probe or {}).get("tasks", {}).get(task_id)
    if entry is None:
        return None
    return entry.get("cases", {}).get(case_id, entry["arithmetic"])


def fp32_arithmetic(task_id: str, case_id: str, task_dir: Path | None = None,
                    probe: dict | None = None) -> tuple[str, str]:
    """The arithmetic one case's float32 products are charged at, and the note that explains it."""
    probed = probed_arithmetic(probe, task_id, case_id)
    if probed is None:
        return UNPROBED, f"fp32_arithmetic:{UNPROBED} (no probe result; float32 matrix peak)"
    task_dir = Path(task_dir) if task_dir is not None else TASKS_DIR / task_id
    if not task_dir.is_dir() or package_sha256(task_dir) != probe["tasks"][task_id]["package_sha256"]:
        return UNPROBED, (f"fp32_arithmetic:{UNPROBED} (task package changed since the probe, which saw "
                          f"{probed} pass; float32 matrix peak until it is probed again)")
    return probed, (f"fp32_arithmetic:{probed} (cheapest arithmetic that passed the evaluator's check on both "
                    f"seeds; {probe['source']})")


def charge_key(arithmetic: str) -> str:
    """The peak key a float32 product is charged at under ``arithmetic``."""
    if arithmetic == FP32:
        return "matrix_fp32"
    peak, products = PRODUCTS[arithmetic]
    return peak if products == 1 else f"{peak}x{products}"


def charge_fp32_products(task_id: str, case_id: str, flops_matrix: dict[str, int],
                         task_dir: Path | None = None, probe: dict | None = None) -> tuple[dict[str, int], list[str]]:
    """Move the ``matrix_fp32`` FLOPs of a case to the key of its arithmetic; the FLOP count is unchanged."""
    if "matrix_fp32" not in flops_matrix:
        return flops_matrix, []
    arithmetic, note = fp32_arithmetic(task_id, case_id, task_dir, probe)
    charged = {key: flops for key, flops in flops_matrix.items() if key != "matrix_fp32"}
    key = charge_key(arithmetic)
    charged[key] = charged.get(key, 0) + flops_matrix["matrix_fp32"]
    return charged, [note]
