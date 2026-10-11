"""Host admission for a fresh whole-case grant before the normal Docker run."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import socket
import subprocess
import sys

import yaml

from src.case_parallelism import enabled, verify_helpers
from src.tools import case_parallel as cp


def scheduler_state(job_id):
    result = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)], check=True,
                            capture_output=True, text=True, timeout=20)
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result.stdout))


def validate_scheduler(plan, state, node):
    cp.require(state.get("JobId") == str(plan["job_id"]) and state.get("JobState") == "RUNNING",
               "allocation is not the requested running job")
    cp.require(state.get("Restarts") == str(plan["restart_generation"]), "allocation restart generation changed")
    cp.require(state.get("NodeList") == plan["node"] and node.split(".")[0] == plan["node"].split(".")[0],
               "allocation/host node differs")
    tres = dict(item.split("=", 1) for item in state.get("AllocTRES", "").split(",") if "=" in item)
    memory = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", tres.get("mem", ""))
    cp.require(memory is not None and tres.get("cpu", "").isdigit()
               and tres.get("gres/gpu", "").isdigit(), "scheduler lacks explicit allocated CPU/memory/GPU capacity")
    memory_gib = float(memory[1]) * {"": 1 / 1024, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}[memory[2]]
    requested = plan["allocation_capacity"]
    cp.require(requested["cpus"] <= int(tres["cpu"]) and requested["memory_gib"] <= memory_gib
               and requested["gpus"] <= int(tres["gres/gpu"]), "plan capacity exceeds current scheduler allocation")


def admit_plan(repo, path, sha256, image, config_path, *, read_scheduler=scheduler_state,
               select_gpu=cp.gpu_binding.select_gpu, hostname=socket.gethostname):
    repo = Path(repo).resolve(strict=True)
    raw = cp._read(path)
    task_relative = raw.get("task_path", "")
    cp.require(isinstance(task_relative, str) and task_relative.startswith("tasks/")
               and ".." not in Path(task_relative).parts, "invalid host task path")
    task = repo / task_relative
    config = yaml.safe_load(cp._relative(task, "config.yaml").read_text())
    cp.require(enabled(config), "task does not opt into whole-case workers")
    verify_helpers(task, config)
    manifest = cp._read(cp._relative(task, config["trusted_evaluation"].get("case_manifest", "cases.json")))
    plan = cp.load_plan(task, manifest, {"manifest_sha256": cp.fingerprint(manifest)}, path=path, sha256=sha256)
    cp.require(plan["runtime_image"] == image, "selected Docker image differs from plan")
    actual_commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                   capture_output=True, text=True, timeout=10).stdout.strip()
    cp.require(actual_commit == plan["baseline_commit"], "execution commit differs from group plan")
    run_path = Path(config_path)
    if not run_path.is_absolute():
        run_path = repo / run_path
    run_config = yaml.safe_load(cp._regular(run_path).read_text())
    cp.require(run_config.get("agent", {}).get("template") == "task_validator"
               and run_config.get("tasks") == [task_relative.removeprefix("tasks/")],
               "GPU group requires one original full-task validator selection")
    timeouts = {phase: config.get(phase + "_timeout") for phase in ("compile", "correctness", "performance")}
    cp.require(all(type(value) is int and value > 0 for value in timeouts.values())
               and plan.get("phase_timeouts") == timeouts, "plan phase caps differ from original task config")
    validate_scheduler(plan, read_scheduler(plan["job_id"]), hostname())
    for binding in plan["bindings"]:
        physical = select_gpu(binding["render_device"])
        cp.require(physical == {key: value for key, value in binding.items() if key != "worker_id"},
                   "granted render node physical identity changed")
    cp.require("," not in plan["_plan_path"], "Docker plan mount path contains a comma")
    resources = plan["resources"]
    count = len(plan["bindings"])
    return ["--mount", f"type=bind,src={plan['_plan_path']},dst=/run/aka-case-gpu-plan.json,readonly",
            "--env", cp.PLAN_ENV + "=/run/aka-case-gpu-plan.json", "--env", cp.HASH_ENV + "=" + sha256,
            "--env", "ROCR_VISIBLE_DEVICES=" + ",".join(binding["rocr_uuid"] for binding in plan["bindings"]),
            "--env", "HIP_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(count)),
            "--env", "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(count)),
            "--env", "GPU_DEVICE_ORDINAL=" + ",".join(str(i) for i in range(count)),
            "--cpus", str(count * resources["cpus_per_worker"] + resources["coordinator_cpus"]),
            "--memory", str(count * resources["memory_gib_per_worker"] + resources["coordinator_memory_gib"]) + "g",
            "--device=/dev/kfd", *("--device=" + binding["render_device"] for binding in plan["bindings"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "plan", "sha256", "image", "config"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    argv = admit_plan(args.repo, args.plan, args.sha256, args.image, args.config)
    sys.stdout.buffer.write(b"\0".join(value.encode() for value in argv) + b"\0")


if __name__ == "__main__":
    main()
