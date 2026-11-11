"""TileLens control: a Triton copy, optionally off by one element.

Usage: tilelens_probe.py [safe|oob|bounds_safe|bounds_oob]

``safe``/``oob`` use the same programmatic API as the schema-v3 lane: a
Sanitizer client with abort_on_error off, whose records are read after the
launch. ``bounds_safe``/``bounds_oob`` run the copy concretely in Triton's
interpreter (``TRITON_INTERPRET=1``) under the lane's own bounds check, which
must confirm a TileLens finding before it counts.
"""

from __future__ import annotations

import json
import sys

import torch
import triton
import triton.language as tl

MODE = sys.argv[1] if len(sys.argv) > 1 else "safe"
if not MODE.startswith("bounds"):
    # Imported before the kernel is defined, as in the lane.
    import tilelens
    from tilelens.clients import Sanitizer


@triton.jit
def copy_kernel(source, destination, shift: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK) + shift
    tl.store(destination + offsets, tl.load(source + offsets))


def main() -> int:
    mode = MODE
    shift = 1 if mode in ("oob", "bounds_oob") else 0
    source = torch.arange(1024, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    if mode.startswith("bounds"):
        from src.tools.arena_eval.sanitize import _bounds_patch

        state = {"violations": [], "violation_count": 0, "launches": 0, "kernel": None, "starts": None,
                 "ends": None}
        unpatch = _bounds_patch(state)
        try:
            copy_kernel[(1,)](source, destination, shift, BLOCK=1024)
        finally:
            unpatch()
        print("BOUNDS_RESULT " + json.dumps({"shift": shift, "launches": state["launches"],
                                             "violations": state["violation_count"]}), flush=True)
        return 0
    client = Sanitizer(abort_on_error=False)
    kernel = tilelens.trace(client=client)(copy_kernel)
    kernel[(1,)](source, destination, shift, BLOCK=1024)
    print("TILELENS_RESULT " + json.dumps({"shift": shift, "records": len(client.records)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
