"""Host admission for a fresh whole-case grant before the normal Docker run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import yaml

from src.case_parallelism import enabled, verify_helpers
from src.tools import case_parallel as cp


def scheduler_state(job_id):
    result = subprocess.run(["scontrol", "show", "job", "-o", str(job_id)], check=True,
                            capture_output=True, text=True, timeout=20, env={**os.environ, "TZ": "UTC"})
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result.stdout))


SPUR = "/usr/local/bin/spur"


def spur_fields(stdout):
    """Read provider fields, retaining slash keys and rejecting ambiguity.

    Command and SubmitLine contain user-authored shell text, not scheduler
    fields. Their arguments must not override controller identity fields.
    """
    fields = {}
    for line in stdout.splitlines():
        if line.lstrip().startswith(("Command=", "SubmitLine=")):
            continue
        for key, value in re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=(\S*)", line):
            cp.require(key not in fields, "duplicate Spur controller field: " + key)
            fields[key] = value
    return fields


def spur_scheduler_state(job_id):
    cp.require(re.fullmatch(r"[0-9]+", str(job_id)), "invalid Spur job ID")
    argv = [SPUR, "show", "job", str(job_id)]
    started = time.time()
    result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=20,
                            env={**os.environ, "TZ": "UTC", "TOKIO_WORKER_THREADS": "1"})
    return {**spur_fields(result.stdout), "_provider": "spur", "_receipt": {
        "argv": argv, "returncode": result.returncode, "stdout": result.stdout,
        "stderr": result.stderr, "started_unix": started, "observed_unix": time.time()}}


def plan_scheduler_state(plan):
    provider = plan.get("scheduler_provider", "slurm")
    cp.require(provider in {"slurm", "spur"}, "unsupported scheduler provider")
    return spur_scheduler_state(plan["job_id"]) if provider == "spur" else scheduler_state(plan["job_id"])


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    return (parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)).timestamp()


def _memory_bytes(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", str(value))
    cp.require(match is not None, "invalid recorded Spur memory quantity")
    size = Decimal(match[1]) * {"": 1024**2, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match[2]]
    cp.require(size > 0 and size == int(size), "non-integral recorded Spur memory size")
    return int(size)


def _receipt(evidence, name):
    raw = evidence.get(name + "_json")
    cp.require(isinstance(raw, str) and hashlib.sha256(raw.encode()).hexdigest() == evidence.get(name + "_sha256"),
               "missing or changed Spur " + name + " receipt")
    value = cp.strict_json(raw)
    cp.require(isinstance(value, dict), "Spur receipt must be an object")
    return value


def validate_spur_reservation(plan, state):
    """Admit an explicit recorded root reservation; never manufacture AllocTRES.

    Spur's display reports requests. The additional root-owned readiness and
    mount-namespace receipts supply the recorded reservation and actual GPU
    scope. This deliberately supports only one node and one task, with the
    complete recorded GPU pool. Missing evidence does not select a fallback.
    """
    evidence = plan.get("spur_allocation")
    cp.require(state.get("_provider") == "spur" and isinstance(evidence, dict)
               and evidence.get("schema") == "spur-recorded-reservation-v1",
               "Spur requires explicit recorded reservation evidence")
    ready = _receipt(evidence, "ready")
    assignment = _receipt(evidence, "assignment")
    binding = _receipt(evidence, "binding")
    renders = _receipt(evidence, "renders")
    cp.require(plan["allocation_receipt_sha256"] == evidence["ready_sha256"], "plan allocation receipt differs")
    job, node = str(plan["job_id"]), plan["node"]
    cp.require(all(str(row.get("job_id")) == job and row.get("node") == node for row in (ready, assignment, binding)),
               "recorded Spur allocation belongs to another job/node")
    cp.require(ready.get("state") == "RUNNING" and ready.get("holder_ready") is True
               and ready.get("restarts") == plan["restart_generation"]
               and ready.get("start_unix") == _timestamp(state["StartTime"]),
               "recorded Spur reservation generation/start differs")
    recorded = ready.get("ready", {})
    cp.require(str(recorded.get("job_id")) == job and recorded.get("node") == node
               and ready.get("assignment") == assignment,
               "Spur readiness does not bind the allocation assignment")
    profile = recorded.get("resource_profile", {})
    cp.require(all(type(profile.get(key)) is int and profile[key] > 0 for key in ("cpus", "memory_MiB", "gpus")),
               "recorded Spur reservation has no explicit CPU/memory/GPU profile")
    cp.require(state.get("NumNodes") == "1" and state.get("NumTasks") == "1",
               "Spur recorded reservations support one node and one task only")
    requested = {}
    for item in state.get("ReqTRES", "").split(","):
        cp.require(item.count("=") == 1, "malformed Spur requested resource entry")
        key, value = item.split("=", 1)
        cp.require(key not in requested, "duplicate Spur requested resource entry")
        requested[key] = value
    cp.require(requested.get("cpu") == state.get("CPUs/Task") == state.get("MinCPUsNode") == str(profile["cpus"])
               and requested.get("node") == "1" and requested.get("gres/gpu") == state.get("ReqGPUs") == str(profile["gpus"]),
               "Spur recorded reservation disagrees with current controller request")
    memory = profile["memory_MiB"] * 1024**2
    cp.require(_memory_bytes(requested.get("mem")) == _memory_bytes(state.get("MinMemoryNode")) == memory,
               "Spur recorded memory reservation differs from controller request")
    cp.require(re.fullmatch(r"gpu:[A-Za-z0-9_.-]+:" + str(profile["gpus"]) + r"/node", state.get("TresPerNode", "")),
               "Spur requested GPU type/count is ambiguous")
    selected = assignment.get("assignment_env", {}).get("SPUR_JOB_GPUS", "").split(",")
    cp.require(all(re.fullmatch(r"[0-9]+", value) for value in selected)
               and len(set(selected)) == len(selected) == profile["gpus"], "Spur GPU assignment is incomplete")
    selected = [int(value) for value in selected]
    cp.require(binding.get("original_assignment_sha256") == evidence["assignment_sha256"],
               "physical Spur pool does not bind the provider assignment")
    cp.require(binding.get("worker_visibility_changed") is False and binding.get("math_performed") is False
               and renders.get("source") == "Scheduler job mount namespace /dev/dri, matched by device major/minor to PCI sysfs",
               "Spur device scope was not recorded in the unchanged allocation namespace")
    rows = binding.get("binding")
    cp.require(isinstance(rows, list) and len(rows) == len(plan["bindings"]) == len(selected),
               "Spur whole group must use the complete recorded GPU pool")
    cp.require(all(row.get("scheduler_registry_ids_for_allocation") == selected for row in rows),
               "physical pool belongs to a different scheduler assignment")
    # Registry IDs are not assumed equal to AMD-SMI local/global ordinals.
    expected = {(b["render_device"], b["pci_bus_id"], b["rocr_uuid"]) for b in plan["bindings"]}
    observed = {(b.get("render_device"), b.get("bdf"), b.get("ROCR_UUID")) for b in rows}
    cp.require(observed == expected and len(observed) == len(rows), "Spur assigned physical GPU pool differs")
    actual_renders = renders.get("renders")
    cp.require(isinstance(actual_renders, list) and len(actual_renders) == len(rows), "missing scheduler mount-namespace GPU scope")
    cp.require({(row.get("render_device"), row.get("bdf")) for row in actual_renders}
               == {(row[0], row[1]) for row in expected}, "Spur mount-namespace device pool differs")
    for row in actual_renders:
        minor = int(row["render_device"].removeprefix("/dev/dri/renderD"))
        cp.require(row.get("major_minor") == row.get("sysfs_dev") == "226:" + str(minor),
                   "Spur assigned render node does not match its sysfs device")
    for row in (ready, assignment, binding, renders):
        observed_unix = row.get("observed_unix")
        cp.require(type(observed_unix) in (int, float) and ready["start_unix"] <= observed_unix <= time.time() + 5,
                   "Spur allocation evidence is from a different start interval")
    capacity = plan["allocation_capacity"]
    cp.require(capacity["cpus"] <= profile["cpus"] and capacity["memory_gib"] * 1024**3 <= memory
               and capacity["gpus"] == len(selected), "plan exceeds the explicitly recorded Spur reservation")
    return {"provider": "spur", "capacity_basis": "root_recorded_single_node_single_task_reservation",
            "recorded_profile": profile, "controller_request_fields_preserved": True,
            "synthetic_AllocTRES_created": False, "individual_registry_to_ordinal_mapping_assumed": False,
            "assigned_gpu_ids": selected, "allocation_receipt_sha256": evidence["ready_sha256"]}


def validate_scheduler(plan, state, node):
    cp.require(state.get("JobId") == str(plan["job_id"]) and state.get("JobState") == "RUNNING",
               "allocation is not the requested running job")
    cp.require(state.get("Restarts") == str(plan["restart_generation"]), "allocation restart generation changed")
    cp.require(state.get("NodeList") == plan["node"] and node.split(".")[0] == plan["node"].split(".")[0],
               "allocation/host node differs")
    try:
        ending = datetime.fromisoformat(state["EndTime"])
        end = (ending if ending.tzinfo is not None else ending.replace(tzinfo=timezone.utc)).timestamp()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Whole-case parallelism: scheduler lease end is missing or invalid") from exc
    cp.require(plan["work_stop_unix"] <= end, "plan work cutoff exceeds the scheduler lease")
    provider = plan.get("scheduler_provider", "slurm")
    cp.require(provider in {"slurm", "spur"}, "unsupported scheduler provider")
    if provider == "spur":
        return validate_spur_reservation(plan, state)
    tres = dict(item.split("=", 1) for item in state.get("AllocTRES", "").split(",") if "=" in item)
    memory = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", tres.get("mem", ""))
    cp.require(memory is not None and tres.get("cpu", "").isdigit()
               and tres.get("gres/gpu", "").isdigit(), "scheduler lacks explicit allocated CPU/memory/GPU capacity")
    memory_gib = float(memory[1]) * {"": 1 / 1024, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}[memory[2]]
    requested = plan["allocation_capacity"]
    cp.require(requested["cpus"] <= int(tres["cpu"]) and requested["memory_gib"] <= memory_gib
               and requested["gpus"] <= int(tres["gres/gpu"]), "plan capacity exceeds current scheduler allocation")


def admit_plan(repo, path, sha256, image, config_path, *, read_scheduler=None,
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
    validate_scheduler(plan, read_scheduler(plan["job_id"]) if read_scheduler is not None else plan_scheduler_state(plan), hostname())
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
