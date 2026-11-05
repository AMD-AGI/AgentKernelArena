"""Verify this task's upstream Python source and qualified runtime requirements."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime():
    import torch
    import aiter

    if not torch.cuda.is_available():
        raise RuntimeError("This task requires an AMD gfx950 GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"Expected gfx950, observed {arch}")
    identity = json.loads((ROOT / "source_identity.json").read_text())
    installed = Path(aiter.__file__).resolve().parent
    for relative, digest in identity["aiter_files"].items():
        for path in (ROOT / "upstream" / relative, installed / Path(relative).relative_to("aiter")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"AITER source differs from the task pin: {relative}")
    # Use the normal pinned table, not a caller-selected replacement table.
    overrides = [name for name in os.environ if name.startswith("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE")]
    if overrides:
        raise RuntimeError("Undeclared GEMM dispatch overrides: " + ", ".join(sorted(overrides)))
    flydsl = importlib.metadata.version("flydsl")
    if flydsl != "0.4.0":
        raise RuntimeError(f"Expected FlyDSL 0.4.0, observed {flydsl}")
    return {"arch": arch, "torch": torch.__version__, "hip": torch.version.hip,
            "flydsl": flydsl, "aiter_revision": identity["revision"],
            "aiter_python_sources_verified": True}


if __name__ == "__main__":
    print(json.dumps(verify_runtime()))
