"""Whole original cases on an explicitly granted, fixed group of physical GPUs.

This file and gpu_binding.py are copied unchanged into opted-in tasks' ut/.
Only the parent may finalize a report. Workers emit non-scoreable receipts;
the ordinary full-manifest numerical and source validators remain mandatory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

try:
    from . import gpu_binding
except ImportError:  # Self-contained task copy.
    import gpu_binding


PLAN_ENV = "AKA_CASE_GPU_PLAN"
HASH_ENV = "AKA_CASE_GPU_PLAN_SHA256"
_SHA = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,128}")
_PHASES = {"compile", "correctness", "performance"}
_LEGS = {"framework", "reference", "candidate"}


def require(value, message):
    if not value:
        raise ValueError("Whole-case parallelism: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    def bad(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def _regular(path):
    path = Path(path).absolute()
    require(all(not p.is_symlink() for p in (path, *path.parents)), "symlinked input path")
    require(path.is_file(), "missing regular input: " + str(path))
    return path


def _relative(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.parts and ".." not in relative.parts,
            "path must stay in the task")
    return _regular(Path(root) / relative)


def _read(path):
    return strict_json(_regular(path).read_bytes())


def _write(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "refusing stale output: " + str(path))
    temporary = path.with_name(path.name + "." + secrets.token_hex(12) + ".tmp")
    with temporary.open("x") as stream:
        stream.write(canonical(value) + "\n")
    temporary.replace(path)


def input_tree_identity(root):
    """Independent digest; never reinterpret a task's package_sha256 producer.

    Runtime outputs live in build/. Validator output/audit paths are excluded;
    all other files (including candidate, fixtures and protected UT) are bound.
    """
    root = Path(root).resolve(strict=True)
    ignored = {"build", ".git", ".validator_audit", ".validator_torch_extensions",
               "task_result.yaml", "validation_report.yaml"}
    entries = {}
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        names[:] = sorted(name for name in names if name != "__pycache__"
                          and not (base == root and name in ignored))
        for name in sorted(names + files):
            path = base / name
            relative = path.relative_to(root)
            if relative.parts[0] in ignored or path.suffix == ".pyc":
                continue
            require(not path.is_symlink(), "input tree contains a symlink: " + str(relative))
            if path.is_file():
                entries[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint(entries)


def _sources(root, request):
    sources = request.get("source_sha256")
    require(isinstance(sources, dict) and sources, "missing source identity")
    require(all(isinstance(value, str) and _SHA.fullmatch(value) for value in sources.values()),
            "invalid source digest")
    actual = {name: hashlib.sha256(_relative(root, name).read_bytes()).hexdigest() for name in sources}
    require(actual == sources, "source identity changed")
    return actual


def _manifest_ids(manifest):
    cases = manifest.get("cases")
    require(isinstance(cases, list) and cases, "empty manifest")
    ids = [case.get("case_id") for case in cases]
    require(all(isinstance(name, str) and name for name in ids) and len(set(ids)) == len(ids),
            "invalid original case IDs")
    return ids


def validate_capacity(plan):
    """Check a fresh grant, including explicit coordinator headroom."""
    resources, capacity = plan.get("resources", {}), plan.get("allocation_capacity", {})
    keys = ("cpus_per_worker", "memory_gib_per_worker", "coordinator_cpus", "coordinator_memory_gib")
    require(all(type(resources.get(key)) is int and resources[key] > 0 for key in keys),
            "explicit positive worker resources and coordinator headroom required")
    require(all(type(capacity.get(key)) is int and capacity[key] > 0 for key in ("cpus", "memory_gib", "gpus")),
            "explicit allocation capacity required")
    count = len(plan["bindings"])
    cpu = count * resources["cpus_per_worker"] + resources["coordinator_cpus"]
    memory = count * resources["memory_gib_per_worker"] + resources["coordinator_memory_gib"]
    require(cpu <= capacity["cpus"] and memory <= capacity["memory_gib"] and count <= capacity["gpus"],
            "worker resources plus coordinator exceed granted capacity")
    for key, expected in (("total_cpus", cpu), ("total_memory_gib", memory)):
        require(key not in resources or resources[key] == expected, "resource total excludes coordinator or differs")


def validate_plan(plan, manifest, *, now=None):
    require(isinstance(plan, dict) and plan.get("schema") == "case-gpu-plan-v1"
            and plan.get("authorized") is True, "an authorized group plan is required")
    require(isinstance(plan.get("plan_id"), str) and _TOKEN.fullmatch(plan["plan_id"]), "invalid plan ID")
    require(re.fullmatch(r"[0-9]+", str(plan.get("job_id", ""))) and isinstance(plan.get("node"), str)
            and bool(plan["node"]) and type(plan.get("restart_generation")) is int
            and plan["restart_generation"] >= 0, "missing allocation identity/generation")
    require(isinstance(plan.get("allocation_receipt_sha256"), str)
            and _SHA.fullmatch(plan["allocation_receipt_sha256"]), "missing allocation receipt digest")
    require(re.fullmatch(r"[0-9a-f]{40}", str(plan.get("baseline_commit", ""))), "full baseline commit required")
    stop = plan.get("work_stop_unix")
    require(type(stop) in (int, float) and math.isfinite(stop) and stop > (time.time() if now is None else now),
            "expired or invalid work cutoff")
    task = PurePosixPath(plan.get("task_path", ""))
    require(len(task.parts) >= 3 and task.parts[0] == "tasks" and ".." not in task.parts,
            "invalid task path")
    require(plan.get("manifest_sha256") == fingerprint(manifest), "plan does not bind the full original manifest")
    require(plan.get("runtime_image") == manifest.get("runtime_image")
            and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", str(plan.get("runtime_image", ""))),
            "plan runtime image differs")
    ids = _manifest_ids(manifest)
    bindings, assignments = plan.get("bindings"), plan.get("assignments")
    require(isinstance(bindings, list) and 1 < len(bindings) <= len(ids), "invalid group size")
    require(isinstance(assignments, list) and len(assignments) == len(bindings), "assignment count differs")
    for index, binding in enumerate(bindings):
        require(isinstance(binding, dict) and type(binding.get("worker_id")) is int
                and binding["worker_id"] == index, "bindings must have consecutive ordered worker IDs")
        render = re.fullmatch(r"/dev/dri/renderD([0-9]+)", str(binding.get("render_device", "")))
        require(render and type(binding.get("drm_minor")) is int and binding["drm_minor"] == int(render[1]),
                "invalid render binding")
        require(re.fullmatch(r"GPU-[0-9a-f]{16}", str(binding.get("rocr_uuid", ""))), "stable ROCr UUID required")
        require(gpu_binding.pci_address(binding.get("pci_bus_id", "")) == binding["pci_bus_id"],
                "canonical PCI identity required")
        require(re.fullmatch(r"[0-9]+", str(binding.get("kfd_node", ""))), "KFD identity required")
    for key in ("render_device", "drm_minor", "kfd_node", "rocr_uuid", "pci_bus_id"):
        require(len({row[key] for row in bindings}) == len(bindings), "duplicate physical GPU: " + key)
    union = []
    for index, assignment in enumerate(assignments):
        require(isinstance(assignment, dict) and set(assignment) == {"worker_id", "case_ids"}
                and type(assignment["worker_id"]) is int and assignment["worker_id"] == index,
                "invalid assignment worker")
        selected = assignment["case_ids"]
        require(isinstance(selected, list) and selected and all(isinstance(name, str) for name in selected),
                "empty or malformed assignment")
        require(selected == [name for name in ids if name in set(selected)], "assignment is not in original manifest order")
        union.extend(selected)
    require(len(union) == len(ids) and set(union) == set(ids), "assignments are not an exact disjoint full case union")
    require(plan.get("assignment_sha256") == fingerprint(assignments), "assignment digest differs")
    validate_capacity(plan)
    return plan


def load_plan(root, manifest, request, *, path=None, sha256=None):
    path = os.environ.get(PLAN_ENV) if path is None else path
    sha256 = os.environ.get(HASH_ENV) if sha256 is None else sha256
    if not path and not sha256:
        require(not request.get("case_parallelism") and not request.get("gpu_group"), "group request has no plan")
        return None
    require(path and isinstance(sha256, str) and _SHA.fullmatch(sha256), "both plan path and SHA256 are required")
    file = _regular(path)
    require(not (file.stat().st_mode & 0o222) or bool(os.statvfs(file).f_flag & os.ST_RDONLY),
            "plan must be read-only")
    data = file.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, "plan file digest differs")
    plan = validate_plan(strict_json(data), manifest)
    require(request.get("manifest_sha256") == fingerprint(manifest), "request manifest differs")
    plan = copy.deepcopy(plan)
    plan.update(_plan_path=str(file), _plan_sha256=sha256, _plan_bytes=data.decode())
    if "case_parallelism" in request or "gpu_group" in request:
        require(canonical(bind_request(request, plan)) == canonical(request), "parent group binding differs")
    return plan


def bind_request(request, plan):
    bound = copy.deepcopy(request)
    bound.pop("gpu", None)
    bound["case_parallelism"] = {"schema_version": 1, "plan_sha256": plan["_plan_sha256"],
                                 "assignment_sha256": plan["assignment_sha256"]}
    bound["gpu_group"] = copy.deepcopy(plan["bindings"])
    return bound


def _check_request(request, plan):
    require(canonical(request) == canonical(bind_request(request, plan)), "unbound parent request")
    require(request.get("phase") in _PHASES and isinstance(request.get("request_id"), str)
            and _TOKEN.fullmatch(request["request_id"]), "invalid request phase/ID")
    require(isinstance(request.get("package_sha256"), str) and _SHA.fullmatch(request["package_sha256"]),
            "missing producer package identity")


def prepare_assignments(root, manifest, request, plan, *, leg):
    """Prepare fresh envelopes; also used by the trusted container dispatcher."""
    _check_request(request, plan)
    require(leg in _LEGS, "invalid source leg")
    root = Path(root).resolve(strict=True)
    _sources(root, request)
    tree = input_tree_identity(root)
    base = root / "build" / "case-parallel" / request["request_id"]
    require(not base.exists() and not base.is_symlink(), "request directory exists; resume is forbidden")
    base.mkdir(parents=True)
    require(base.resolve() == base and all(not p.is_symlink() for p in base.parents), "private output escapes task")
    assignments = []
    for partition, binding in zip(plan["assignments"], plan["bindings"]):
        directory = base / ("worker-" + str(partition["worker_id"]))
        directory.mkdir(mode=0o700)
        assignments.append({"schema": "whole-case-worker-assignment-v1",
            "parent_request_sha256": fingerprint(request), "plan_sha256": plan["_plan_sha256"],
            "assignment_sha256": plan["assignment_sha256"], "worker_id": partition["worker_id"],
            "worker_nonce": secrets.token_hex(24), "phase": request["phase"], "leg": leg,
            "case_ids": partition["case_ids"], "expected_gpu": binding,
            "output_directory": directory.relative_to(root).as_posix(),
            "deadline_unix": plan["work_stop_unix"], "input_tree_sha256": tree})
    return assignments


def _assignment(request, plan, assignment):
    require(isinstance(assignment, dict) and assignment.get("schema") == "whole-case-worker-assignment-v1",
            "invalid worker assignment schema")
    index = assignment.get("worker_id")
    require(type(index) is int and 0 <= index < len(plan["bindings"]), "invalid worker ID")
    for key, expected in (("parent_request_sha256", fingerprint(request)), ("plan_sha256", plan["_plan_sha256"]),
                          ("assignment_sha256", plan["assignment_sha256"]), ("phase", request["phase"]),
                          ("case_ids", plan["assignments"][index]["case_ids"]), ("expected_gpu", plan["bindings"][index])):
        require(canonical(assignment.get(key)) == canonical(expected), "assignment differs: " + key)
    require(assignment.get("leg") in _LEGS and re.fullmatch(r"[0-9a-f]{48}", str(assignment.get("worker_nonce", ""))),
            "invalid worker leg/nonce")
    deadline = assignment.get("deadline_unix")
    require(type(deadline) in (int, float) and math.isfinite(deadline) and 0 < deadline <= plan["work_stop_unix"],
            "worker deadline exceeds grant")
    require(_SHA.fullmatch(str(assignment.get("input_tree_sha256", ""))), "missing independent input digest")
    expected_output = f"build/case-parallel/{request['request_id']}/worker-{index}"
    require(assignment.get("output_directory") == expected_output, "worker output is not private")


@dataclass(frozen=True)
class WorkerContext:
    root: Path
    manifest: dict
    request: dict
    assignment: dict
    cases: tuple
    output_directory: Path
    gpu_preflight: dict


def enter_worker(root, manifest, request, assignment_path):
    root = Path(root).resolve(strict=True)
    plan = load_plan(root, manifest, request)
    require(plan is not None, "worker requires explicit group plan")
    _check_request(request, plan)
    assignment = _read(assignment_path)
    _assignment(request, plan, assignment)
    require(time.time() < assignment["deadline_unix"], "worker deadline expired")
    output = root / assignment["output_directory"]
    require(output.is_dir() and output.resolve() == output
            and all(not p.is_symlink() for p in (output, *output.parents)), "unsafe worker output")
    require(not (output / "receipt.json").exists(), "worker receipt exists; resume is forbidden")
    require(input_tree_identity(root) == assignment["input_tree_sha256"], "worker input tree differs")
    _sources(root, request)
    expected = assignment["expected_gpu"]
    require(os.environ.get("ROCR_VISIBLE_DEVICES") == expected["rocr_uuid"]
            and all(os.environ.get(name) == "0" for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")),
            "worker visibility differs")
    preflight = {**gpu_binding.runtime_identity(), "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES")}
    gpu_binding.validate_preflight(preflight, expected)
    selected = set(assignment["case_ids"])
    return WorkerContext(root, manifest, copy.deepcopy(request), assignment,
                         tuple(case for case in manifest["cases"] if case["case_id"] in selected), output, preflight)


def _specializations(compiled, ids, sources):
    require(isinstance(compiled, list) and [row.get("case_id") for row in compiled] == ids,
            "per-case compiled specialization coverage differs")
    for row in compiled:
        # Existing Kimi and MiniMax formats retain their stronger native checks.
        kimi = (row.get("candidate_invoked") is True and row.get("independent_reference_invoked") is True
                and row.get("source_sha256") == sources)
        minimax = (row.get("invoked_and_synchronized") is True
                   and isinstance(row.get("candidate_launch_evidence"), dict)
                   and bool(row["candidate_launch_evidence"].get("engaged_kernels")))
        require(kimi or minimax, "per-case native compilation/engagement proof missing")


def _rows(manifest, request, rows, ids):
    require(isinstance(rows, list), "raw rows must be a list")
    if request["phase"] == "compile":
        require(rows == [], "compile worker must not publish measured rows")
        return
    require(all(isinstance(row, dict) and isinstance(row.get("case"), dict) for row in rows), "malformed raw row")
    require([row["case"].get("case_id") for row in rows] == ids, "worker row case sequence differs")
    original = {case["case_id"]: case for case in manifest["cases"]}
    require(all(canonical(row["case"]) == canonical(original[row["case"]["case_id"]]) for row in rows),
            "worker changed original case ABI")


def finish_worker(context, *, rows, compiled_specializations, evidence):
    assignment, request = context.assignment, context.request
    ids = assignment["case_ids"]
    require(isinstance(evidence, dict) and evidence.get("completed_case_ids") == ids, "incomplete case work")
    _rows(context.manifest, request, rows, ids)
    _specializations(compiled_specializations, ids, request["source_sha256"])
    require(time.time() < assignment["deadline_unix"], "worker completed after deadline")
    _sources(context.root, request)
    after = input_tree_identity(context.root)
    require(after == assignment["input_tree_sha256"], "worker input tree changed")
    receipt = {"schema": "whole-case-worker-result-v1", "status": "partition_complete", "full_task": False,
        "scoreable": False, "parent_request": request, "assignment_sha256": assignment["assignment_sha256"],
        "worker_assignment_sha256": fingerprint(assignment), "plan_sha256": assignment["plan_sha256"],
        "worker_nonce": assignment["worker_nonce"], "worker_id": assignment["worker_id"],
        "phase": assignment["phase"], "leg": assignment["leg"], "expected_gpu": assignment["expected_gpu"],
        "assigned_case_ids": ids, "completed_case_ids": ids, "gpu_preflight": context.gpu_preflight,
        "source_sha256": request["source_sha256"], "package_before_sha256": request["package_sha256"],
        "package_after_sha256": request["package_sha256"], "input_tree_before_sha256": assignment["input_tree_sha256"],
        "input_tree_after_sha256": after, "rows": rows, "compiled_specializations": compiled_specializations,
        "evidence": evidence, "completed_unix": time.time()}
    _write(context.output_directory / "receipt.json", receipt)
    return receipt


def collect_workers(manifest, request, plan, assignments, observed_exits, receipt_paths):
    """Reduce only an exact full union. Exit codes come from the coordinator."""
    _check_request(request, plan)
    count = len(plan["bindings"])
    require(len(assignments) == count and len(observed_exits) == count and len(receipt_paths) == count,
            "missing worker result")
    require(all(type(code) is int and code == 0 for code in observed_exits), "worker exited unsuccessfully")
    require([a.get("worker_id") for a in assignments] == list(range(count)), "worker assignment order differs")
    require(len({a.get("worker_nonce") for a in assignments}) == count, "duplicate worker nonce")
    require(len({a.get("input_tree_sha256") for a in assignments}) == 1, "workers used different input trees")
    require(len({a.get("leg") for a in assignments}) == 1, "workers used different source legs")
    rows, compiled, workers = [], [], []
    for assignment, path, code in zip(assignments, receipt_paths, observed_exits):
        _assignment(request, plan, assignment)
        file = _regular(path)
        data = file.read_bytes()
        receipt = strict_json(data)
        require(receipt.get("schema") == "whole-case-worker-result-v1" and receipt.get("status") == "partition_complete"
                and receipt.get("full_task") is False and receipt.get("scoreable") is False, "worker claimed scoring authority")
        require(not ({"test_cases", "task_result.yaml", "validation_report.yaml", "accepted_gain", "arithmetic_mean_speedup",
                      "returncode", "exit_code"} & set(receipt)), "worker published forbidden authority")
        expected = {"parent_request": request, "assignment_sha256": plan["assignment_sha256"],
            "worker_assignment_sha256": fingerprint(assignment), "plan_sha256": plan["_plan_sha256"],
            "worker_nonce": assignment["worker_nonce"], "worker_id": assignment["worker_id"],
            "phase": request["phase"], "leg": assignment["leg"], "expected_gpu": assignment["expected_gpu"],
            "assigned_case_ids": assignment["case_ids"], "completed_case_ids": assignment["case_ids"],
            "source_sha256": request["source_sha256"], "package_before_sha256": request["package_sha256"],
            "package_after_sha256": request["package_sha256"], "input_tree_before_sha256": assignment["input_tree_sha256"],
            "input_tree_after_sha256": assignment["input_tree_sha256"]}
        require(all(canonical(receipt.get(key)) == canonical(value) for key, value in expected.items()),
                "stale or foreign worker receipt identity")
        completed = receipt.get("completed_unix")
        require(type(completed) in (int, float) and math.isfinite(completed) and 0 < completed < assignment["deadline_unix"],
                "worker receipt missed deadline")
        gpu_binding.validate_preflight(receipt.get("gpu_preflight"), assignment["expected_gpu"])
        ids = assignment["case_ids"]
        require(isinstance(receipt.get("evidence"), dict) and receipt["evidence"].get("completed_case_ids") == ids,
                "worker evidence is incomplete")
        _rows(manifest, request, receipt.get("rows"), ids)
        _specializations(receipt.get("compiled_specializations"), ids, request["source_sha256"])
        rows.extend(receipt["rows"])
        compiled.extend(receipt["compiled_specializations"])
        workers.append({"assignment": assignment, "observed_exit": code,
                        "receipt_path": assignment["output_directory"] + "/receipt.json",
                        "receipt_sha256": hashlib.sha256(data).hexdigest()})
    order = {name: index for index, name in enumerate(_manifest_ids(manifest))}
    rows.sort(key=lambda row: order[row["case"]["case_id"]])
    compiled.sort(key=lambda row: order[row["case_id"]])
    metadata = {"schema": "whole-case-full-report-v1", "plan_sha256": plan["_plan_sha256"],
                "assignment_sha256": plan["assignment_sha256"], "parent_request_sha256": fingerprint(request),
                "plan_bytes": plan["_plan_bytes"], "workers": workers,
                "case_gpu_mapping": [{"case_id": name, "worker_id": a["worker_id"],
                    "gpu": plan["bindings"][a["worker_id"]]} for name in order for a in plan["assignments"] if name in a["case_ids"]]}
    return {"rows": rows, "compiled_specializations": compiled, "workers": workers, "case_parallelism": metadata}


def _stop_workers(processes):
    # Each worker owns a new process group. Always terminate descendants too,
    # including a compiler left behind by a worker which has already exited.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.monotonic() + 2
    while any(process.poll() is None for process in processes) and time.monotonic() < end:
        time.sleep(0.05)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_workers(root, manifest, request, plan, entrypoint, *, leg):
    """One new exec interpreter per GPU; never fork initialized HIP state."""
    root = Path(root).resolve(strict=True)
    assignments = prepare_assignments(root, manifest, request, plan, leg=leg)
    entrypoint = Path(entrypoint)
    entrypoint = _relative(root, entrypoint.relative_to(root).as_posix() if entrypoint.is_absolute() else str(entrypoint))
    base = root / "build" / "case-parallel" / request["request_id"]
    parent = base / "request.json"
    _write(parent, request)
    processes, logs, receipts = [], [], []
    handlers = {}
    def cancelled(signum, _frame):
        raise InterruptedError("group coordinator received signal " + str(signum))
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, cancelled)
        for assignment in assignments:
            output = root / assignment["output_directory"]
            envelope = output / "assignment.json"
            _write(envelope, assignment)
            env = os.environ.copy()
            env.update({PLAN_ENV: plan["_plan_path"], HASH_ENV: plan["_plan_sha256"],
                "ROCR_VISIBLE_DEVICES": assignment["expected_gpu"]["rocr_uuid"], "HIP_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "0", "GPU_DEVICE_ORDINAL": "0", "PYTHONDONTWRITEBYTECODE": "1"})
            for variable, directory in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("TRITON_CACHE_DIR", "triton"),
                                        ("TORCH_EXTENSIONS_DIR", "torch"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
                private = output / directory
                private.mkdir()
                env[variable] = str(private)
            # AITER may ship populated caches. Preserve their bytes while giving
            # every worker an independent writable cache directory.
            aiter = output / "aiter"
            if os.environ.get("AITER_JIT_DIR"):
                shutil.copytree(os.environ["AITER_JIT_DIR"], aiter)
            else:
                aiter.mkdir()
            env["AITER_JIT_DIR"] = str(aiter)
            log = (output / "worker.log").open("xb")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, "-B", str(entrypoint), request["phase"],
                "--request", str(parent), "--case-worker", str(envelope)], cwd=root, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            receipts.append(output / "receipt.json")
        while True:
            exits = [process.poll() for process in processes]
            require(time.time() < min(a["deadline_unix"] for a in assignments), "group deadline expired")
            require(all(code in (None, 0) for code in exits), "worker failed; cancelling owned siblings")
            if all(code is not None for code in exits):
                break
            time.sleep(0.1)
        require(input_tree_identity(root) == assignments[0]["input_tree_sha256"], "parent input tree changed")
        _sources(root, request)
        return collect_workers(manifest, request, plan, assignments, exits, receipts)
    except BaseException:
        _stop_workers(processes)
        raise
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
        for log in logs:
            log.close()


def validate_group_report(root, manifest, report, *, required=False):
    """Recheck the full receipt chain before framework scoring/finalization."""
    request = report.get("request", {})
    metadata = report.get("case_parallelism")
    active = bool(os.environ.get(PLAN_ENV) or os.environ.get(HASH_ENV))
    if not metadata and not request.get("case_parallelism") and not request.get("gpu_group"):
        require(not required and not active, "group report provenance was stripped")
        return
    require(isinstance(metadata, dict) and metadata.get("schema") == "whole-case-full-report-v1", "missing full group receipt")
    raw = metadata.get("plan_bytes")
    require(isinstance(raw, str) and hashlib.sha256(raw.encode()).hexdigest() == metadata.get("plan_sha256"),
            "embedded plan digest differs")
    # Archived reports can be revalidated after the cutoff; the worker completion
    # timestamps remain checked against the original deadline by the reducer.
    plan = validate_plan(strict_json(raw), manifest, now=0)
    plan.update(_plan_bytes=raw, _plan_sha256=metadata["plan_sha256"])
    if active:
        current = load_plan(root, manifest, request)
        require(current["_plan_sha256"] == plan["_plan_sha256"], "report belongs to a different active plan")
    workers = metadata.get("workers")
    require(isinstance(workers, list), "missing worker chain")
    paths = []
    for worker in workers:
        path = _relative(root, worker["receipt_path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == worker.get("receipt_sha256"), "worker receipt bytes changed")
        paths.append(path)
    result = collect_workers(manifest, request, plan, [w["assignment"] for w in workers],
                             [w["observed_exit"] for w in workers], paths)
    require(canonical(result["case_parallelism"]) == canonical(metadata), "full group provenance differs")
    require(canonical(result["rows"]) == canonical(report.get("cases")), "report rows differ from full worker union")
    require(canonical(result["compiled_specializations"]) == canonical(report.get("compiled_specializations")),
            "report compile evidence differs from full worker union")
    _sources(root, request)
