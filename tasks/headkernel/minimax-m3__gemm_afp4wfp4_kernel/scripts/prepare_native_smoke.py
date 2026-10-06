"""Prepare a reviewable exact-image smoke command; never launch a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, common, binding_helper, expectation):
    output, common, binding_helper, expectation = [Path(p).resolve() for p in (output, common, binding_helper, expectation)]
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("smoke output must be outside the task snapshot")
    for path in (ROOT, output, common, binding_helper, expectation):
        if "," in str(path):
            raise ValueError("Docker bind paths must not contain commas")
    pins = json.loads((ROOT / "SOURCE-PROVENANCE.json").read_text())
    shared_sha = json.loads((ROOT / "capture/INTEGRATION.json").read_text())["shared_recorder"]["runtime_capture_sha256"]
    if hashlib.sha256((common / "runtime_capture.py").read_bytes()).hexdigest() != shared_sha:
        raise ValueError("shared capture implementation differs from the pinned adapter")
    gpu = json.loads(expectation.read_text())
    if not re.fullmatch(r"/dev/dri/renderD[0-9]+", gpu.get("render_device", "")) or not re.fullmatch(r"GPU-[0-9a-fA-F]{16}", gpu.get("rocr_uuid", "")):
        raise ValueError("trusted physical GPU expectation is required")
    output.mkdir(parents=True, exist_ok=False)
    task_snapshot = output / "task"
    shutil.copytree(ROOT, task_snapshot, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "build"))
    common_snapshot = output / "common"
    common_snapshot.mkdir()
    (common_snapshot / "runtime_capture.py").write_bytes((common / "runtime_capture.py").read_bytes())
    (output / "GPU-EXPECTED.json").write_text(json.dumps(gpu, indent=2) + "\n")
    (output / "gpu_binding.py").write_bytes(binding_helper.read_bytes())
    (output / "results").mkdir()
    name = "aka-fp4-native-smoke-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--pull=never", "--name", name, "--network=none",
               "--device", "/dev/kfd", "--device", gpu["render_device"], "--shm-size=2g", "--workdir", "/task"]
    for source, target, readonly in ((task_snapshot, "/task", True), (common_snapshot, "/capture-common", True),
                                      (output / "gpu_binding.py", "/gpu_binding.py", True),
                                      (output / "GPU-EXPECTED.json", "/gpu-expectation.json", True),
                                      (output / "results", "/results", False)):
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for setting in ("ROCR_VISIBLE_DEVICES=" + gpu["rocr_uuid"], "HIP_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=0",
                    "PYTHONDONTWRITEBYTECODE=1", "TRITON_CACHE_DIR=/results/triton-cache"):
        command += ["--env", setting]
    command += ["--entrypoint", "python3", pins["runtime_image"], "/gpu_binding.py", "--expected", "/gpu-expectation.json",
                "--proof", "/results/GPU-PREFLIGHT.json", "--", "/task/scripts/native_smoke.py",
                "--common", "/capture-common", "--output", "/results/smoke"]
    plan = {"schema": "minimax-fp4-native-smoke-plan-v1", "status": "PREPARED_NOT_EXECUTED", "command": command,
            "cleanup_command": ["docker", "rm", "-f", name], "timeout_seconds": 1800,
            "image": pins["runtime_image"], "gpu": gpu, "GPU_actions": False, "scheduler_actions": False,
            "scoreable": False, "performance_samples": 0,
            "common_sha256": hashlib.sha256((common_snapshot / "runtime_capture.py").read_bytes()).hexdigest(),
            "binding_helper_sha256": hashlib.sha256((output / "gpu_binding.py").read_bytes()).hexdigest(),
            "source_sha256": {str(p.relative_to(task_snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in task_snapshot.rglob("*") if p.is_file()}}
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(shlex.join(command))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "common", "binding-helper", "expectation"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output, args.common, args.binding_helper, args.expectation)


if __name__ == "__main__":
    main()
