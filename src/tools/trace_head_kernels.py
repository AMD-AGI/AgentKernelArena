"""Bounded baseline device traces, separate from scoring and task validation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import time

import yaml

from src.scripts.top5_head_kernels import REPO_ROOT, plan_run
from src.tools.runtime_image_identity import verify_identity
from src.tools.verify_head_kernels import copy_task, fingerprint, run_command, write_json


PROBE = Path(__file__).with_name("head_kernel_trace_worker.py")
LIMITS = {"framework_task_validator": "NOT_RUN", "framework_PASS_claimed": False,
          "scoring_performed": False, "original_dispatch_equivalence_certified": False}


def retained_evidence(task, config):
    metadata = config.get("headkernel") or {}
    files = [task / "ut/selection_validation.json", task / "ut/provenance/selection_validation.json"]
    return {"declared_device_symbol": metadata.get("device_symbol"),
            "declared_profile_symbols": [row["info_kernel"] for row in metadata.get("info_rows", [])
                                         if row.get("info_kernel")],
            "python_target_callable": metadata.get("target_callable"),
            "files": {p.relative_to(task).as_posix(): fingerprint(p) for p in files if p.is_file()},
            "comparison": "not automated; callable names are not device-kernel evidence"}


def validate_trace_artifacts(workspace, report):
    if report.get("status") != "trace_recorded" or not report.get("cases"):
        raise ValueError("trace worker did not report GPU kernel evidence")
    selected = report.get("selected_case_indexes")
    if (not isinstance(selected, list) or [row.get("case_index") for row in report["cases"]] != selected
            or len(set(selected)) != len(selected)):
        raise ValueError("trace case coverage is inconsistent")
    for row in report["cases"]:
        if row.get("status") != "trace_recorded" or not row.get("kernel_events"):
            raise ValueError("selected case has no GPU kernel events")
        artifact = row.get("raw_trace") or {}
        path = (workspace / artifact.get("path", "")).resolve()
        if not path.is_relative_to(workspace.resolve()) or not path.is_file():
            raise ValueError("trace artifact is missing or escapes its workspace")
        actual = fingerprint(path)
        if actual["sha256"] != artifact.get("sha256") or actual["bytes"] != artifact.get("bytes"):
            raise ValueError("raw device trace changed after worker finalization")


def trace_task(source, workspace, repo, args, deadline):
    result = {"task": source.relative_to(repo / "tasks").as_posix(),
              "workspace": workspace.relative_to(repo).as_posix(), "status": "preparing"}
    try:
        result["source_identity"] = copy_task(source, workspace, repo)
        probe = workspace / "scripts/_device_trace.py"
        shutil.copy2(PROBE, probe)
        result["probe_identity"] = fingerprint(probe)
        config = yaml.safe_load((workspace / "config.yaml").read_text())
        result["retained_original_evidence"] = retained_evidence(workspace, config)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("device-trace timeout exhausted during workspace preparation")
        command = [sys.executable, "-u", str(probe), "--mode", args.mode,
                   "--max-cases", str(args.max_cases), "--replays", str(args.replays),
                   "--timeout", str(remaining)]
        for index in args.case_index:
            command.extend(["--case-index", str(index)])
        with (workspace / "device-trace.stdout").open("xb") as stdout, \
                (workspace / "device-trace.stderr").open("xb") as stderr:
            result["execution"] = run_command(shlex.join(command), workspace, stdout, stderr, remaining)
        report_path = workspace / "build/device_trace_report.json"
        report = {}
        if report_path.is_file():
            result["native_report"] = {"path": report_path.relative_to(workspace).as_posix(), **fingerprint(report_path)}
            report = json.loads(report_path.read_text())
        if result["execution"]["timed_out"]:
            raise TimeoutError("device-trace worker exceeded the remaining timeout")
        if result["execution"]["returncode"] != 0:
            raise ValueError("device-trace worker failed; inspect stdout/stderr and partial report")
        validate_trace_artifacts(workspace, report)
        result.update(status="trace_recorded", selected_case_ids=[row["case_id"] for row in report["cases"]],
                      untraced_case_ids=report["untraced_case_ids"])
    except Exception as error:
        result.update(status="trace_failed", error_type=type(error).__name__, error=str(error))
    return result


def trace(config, args, repo=REPO_ROOT):
    if (not 1 <= args.max_cases <= 16 or not 1 <= args.replays <= 5
            or not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600):
        raise ValueError("max-cases must be 1..16, replays 1..5, and timeout in (0, 3600]")
    if (len(args.case_index) > args.max_cases or len(set(args.case_index)) != len(args.case_index)
            or any(index < 0 for index in args.case_index)):
        raise ValueError("case-index values must be distinct, nonnegative, and within max-cases")
    repo = repo.resolve()
    plan = plan_run(config, repo)
    if os.environ.get("AGENT_KERNEL_ARENA_DOCKER") != "1":
        raise ValueError("Use top5_head_kernels.py trace through the Docker runner")
    proof = json.loads(os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IDENTITY", "{}"))
    identity = verify_identity(plan["image"], plan["expected_image_id"], {
        "Id": os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID"),
        "RepoDigests": json.loads(os.environ.get("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "[]")),
        "Descriptor": proof.get("descriptor"),
    })
    if (os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE") != plan["image"]
            or identity["verified_config_digest"] != os.environ.get("AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST")):
        raise ValueError("Docker identity differs from the declared trace runtime")
    run = Path(tempfile.mkdtemp(prefix="workspace_device_trace_", dir=repo))
    path = run / "device-trace.json"
    result = {"schema": "aka-device-trace-run-v1", **LIMITS, "status": "running",
              "plan": plan, "runtime_identity": identity, "mode": args.mode,
              "max_cases_per_task": args.max_cases, "profile_replays": args.replays,
              "timeout_seconds": args.timeout, "tasks": []}
    write_json(path, result)
    print(f"Device trace evidence: {path.relative_to(repo)}", flush=True)
    deadline = time.monotonic() + args.timeout
    for index, selector in enumerate(plan["tasks"]):
        if time.monotonic() >= deadline:
            result["unstarted_tasks"] = plan["tasks"][index:]
            break
        source = (repo / "tasks" / selector).resolve()
        result["tasks"].append(trace_task(source, run / f"{index:03d}-{source.name}", repo, args, deadline))
        write_json(path, result)
    success = (len(result["tasks"]) == len(plan["tasks"])
               and all(task["status"] == "trace_recorded" for task in result["tasks"]))
    result["status"] = "trace_recorded" if success else "trace_failed_or_incomplete"
    write_json(path, result)
    return (0 if success else 1), path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_name", required=True, type=Path)
    parser.add_argument("--mode", choices=("timed-graph", "eager"), default="timed-graph")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--case-index", action="append", type=int, default=[])
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    try:
        return trace(args.config_name, args)[0]
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
