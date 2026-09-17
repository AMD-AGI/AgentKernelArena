"""Record and compare the actual runtime used for a frozen v2 baseline."""
import importlib.metadata
import json
import os
import sys
from src.task_spec import required_gpu_arches


def _runtime_identity() -> dict:
    import torch

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("V2 GPU tasks require an available ROCm GPU in the scoring runtime")
    properties = torch.cuda.get_device_properties(0)
    identity = {"python": sys.version.split()[0], "torch": str(torch.__version__), "hip": torch.version.hip,
                "gpu_name": properties.name, "gpu_arch": properties.gcnArchName.split(":", 1)[0]}
    for name in ("triton", "flydsl", "aiter"):
        try:
            identity[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            identity[name] = None
    for name in ("AKA_SCORING_IMAGE_RUNTIME_REF", "AKA_SCORING_IMAGE_REFERENCE",
                 "AKA_EVAL_TOOLS_SCORING_IMAGE_ID", "AKA_SCORING_IMAGE_ID", "AKA_DOCKER_IMAGE"):
        if name in os.environ:
            identity[name] = os.environ[name]
    return identity


def bind_session_runtime(session) -> dict:
    runtime = _runtime_identity()
    required_arch = required_gpu_arches(session.spec.to_mapping().get("platform_support"))
    if required_arch and runtime["gpu_arch"] not in required_arch:
        raise RuntimeError(f"Task requires {required_arch}; actual GPU is {runtime['gpu_arch']}")
    path = session.state_directory / "runtime_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != runtime:
            raise RuntimeError("Scoring runtime changed since baseline capture; start a separate run")
    else:
        with path.open("x") as handle:
            json.dump(runtime, handle, indent=2, sort_keys=True, allow_nan=False)
    return runtime
