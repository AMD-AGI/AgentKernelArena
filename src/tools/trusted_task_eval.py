"""Fresh-container evaluation for portable contracts and supported Qwen graphs.

This leaves the qualified native-quant evaluator unchanged. The invoking host,
Git database, image and Docker daemon remain trusted infrastructure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path, PurePosixPath

import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.native_baseline import scoring_policy, load_native_measurements, as_test_cases, metric_summary
from src.benchmark_quality import comparison_quality
from src.quality_measurements import portable_test_cases

if __package__:
    from ..task_contract import (
        canonical,
        fingerprint,
        require,
        strict_json,
        validate_manifest,
        validate_report,
    )
    from .gpu_binding import command_with_binding, select_gpu, validate_preflight
    from .seed_aiter_jit_cache import copy_cache, seed_image_cache, tree_manifest
    from .trusted_native_eval import docker_command, extract_task, hash_regular_file, read_regular, sha256
    from .trusted_fixtures import DEFAULT_MAX_BYTES, DEFAULT_MAX_FILES, load_fixture_manifest, materialize_fixtures
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Isolated Python omits the script directory, so add only this trusted
    # helper directory rather than depending on the caller's working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpu_binding import command_with_binding, select_gpu, validate_preflight
    from seed_aiter_jit_cache import copy_cache, seed_image_cache, tree_manifest
    from trusted_native_eval import docker_command, extract_task, hash_regular_file, read_regular, sha256
    from trusted_fixtures import DEFAULT_MAX_BYTES, DEFAULT_MAX_FILES, load_fixture_manifest, materialize_fixtures

    from task_contract import (
        canonical,
        fingerprint,
        require,
        strict_json,
        validate_manifest,
        validate_report,
    )


PORTABLE_CONTRACT = Path(__file__).resolve().parents[1] / "task_contract.py"
GPU_BINDING_HELPER = Path(__file__).resolve().with_name("gpu_binding.py")
PHASES = ("compile", "correctness", "performance")
EXHAUSTIVE_CORRECTNESS = "exhaustive_observed_values"


def distribution_phase_arguments(mode, phase):
    """Only a committed exhaustive opt-in may extend the protected command."""
    require(phase in PHASES, "unknown evaluation phase")
    require(mode in (None, EXHAUSTIVE_CORRECTNESS), "unsupported distribution correctness mode")
    return (["--distribution-correctness-mode", mode]
            if mode is not None and phase in ("compile", "correctness") else [])


def phase_timeout(config, phase, maximum):
    limit = config.get(phase + "_timeout", maximum)
    require(type(limit) is int and limit > 0, "phase timeout must be a positive integer")
    return min(maximum, limit)


def evaluation_timeouts(config, mode, maximum, overrides=None):
    """Apply explicit host caps without changing legacy task timeout policy."""
    require(type(maximum) is int and maximum > 0, "timeout must be a positive integer")
    overrides = {} if overrides is None else overrides
    require(isinstance(overrides, dict) and set(overrides) <= set(PHASES),
            "phase timeouts must name only compile, correctness or performance")
    for limit in overrides.values():
        require(type(limit) is int and limit > 0, "phase timeout must be a positive integer")
    return {phase: min(phase_timeout(config, phase, maximum) if mode is not None else maximum,
                       overrides.get(phase, maximum)) for phase in PHASES}


def distribution_contract(task, manifest, mode):
    """Read the manifest-bound registry without importing task executable code."""
    if mode is None:
        return None
    reference = manifest.get("observed_work_distributions", {})
    payload = read_regular(task / relative_file(reference.get("path")))
    require(sha256(payload) == reference.get("sha256"), "work distribution registry digest differs")
    registry = strict_json(payload)
    require(registry.get("schema") == "moe-observed-work-distributions-v1", "unsupported work distribution registry")
    groups = registry.get("groups", {})
    dynamic = {case["case_id"]: groups[case["distribution_group"]]
               for case in manifest["cases"] if case.get("provenance_kind") == "generated_observed_work_distribution"}
    require(dynamic and len(dynamic) == len(groups), "work distribution case coverage differs")
    for case_id, group in dynamic.items():
        rows = group["histogram"]
        require(group["case_id"] == case_id and rows
                and len({row["variant_id"] for row in rows}) == len(rows)
                and fingerprint(rows) == group["histogram_sha256"], "invalid work distribution histogram")
    return dynamic


def validate_distribution_report(report, manifest, request, distributions, mode):
    """Require exhaustive seed/control evidence while keeping timing sampled."""
    if distributions is None:
        require("distribution_correctness_mode" not in request, "unconfigured distribution mode")
        return
    phase = request["phase"]
    expected_mode = mode if phase in ("compile", "correctness") else None
    require(request.get("distribution_correctness_mode") == expected_mode, "request distribution mode differs")
    compiled = report.get("compiled_specializations", [])
    ids = [row.get("case_id") for row in compiled]
    require(len(ids) == len(manifest["cases"]) and set(ids) == {case["case_id"] for case in manifest["cases"]},
            "compiled specialization coverage differs")
    for receipt in compiled:
        case_id = receipt["case_id"]
        if case_id not in distributions:
            continue
        group = distributions[case_id]
        selected = expected_mode or group["correctness"]["default_mode"]
        require(receipt.get("invoked_and_synchronized") is True
                and receipt.get("distribution_compile_setting") == group["histogram"][0]["num_valid_ids"][0]
                and receipt.get("correctness_mode") == selected
                and receipt.get("all_observed_settings_correctness") is False,
                "distribution compile receipt mode/coverage differs")
    if phase != "correctness":
        return
    policy = manifest["measurement"]
    observed = {row["case"]["case_id"]: row for row in report["cases"]}
    for case_id, group in distributions.items():
        row = observed[case_id]
        expected = [{"variant_id": setting["variant_id"], "num_valid_ids": setting["num_valid_ids"],
                     "correct": True, "seeds": policy["correctness_seeds"],
                     "negative_controls": {control: True for control in policy["negative_controls"]}}
                    for setting in group["histogram"]]
        require(canonical(row.get("correctness_variants")) == canonical(expected),
                "exhaustive distribution seed/control coverage differs")
        proof = row.get("observed_work_distribution", {})
        require(proof.get("histogram_sha256") == group["histogram_sha256"]
                and proof.get("correctness_mode") == mode
                and type(proof.get("correctness_settings_checked")) is int
                and proof["correctness_settings_checked"] == len(expected)
                and type(proof.get("observed_setting_count")) is int
                and proof["observed_setting_count"] == len(expected)
                and proof.get("all_observed_settings_correctness") is True
                and proof.get("all_observed_settings_timed") is False
                and proof.get("actual_other_rank_routing_recovered") is False
                and proof.get("reference_generated_outputs_not_parent_goldens") is True,
                "exhaustive distribution scope/provenance differs")


def relative_file(value):
    require(isinstance(value, str) and bool(value), "task file paths must be nonempty strings")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in ("..", ".git", "build", "__pycache__") for part in path.parts),
            "task file path escapes the immutable package")
    return path.as_posix()


def package_contract(task):
    config = yaml.safe_load(read_regular(task / "config.yaml"))
    require(isinstance(config, dict), "invalid task configuration")
    if config.get("trusted_evaluation") is None and config.get("benchmark_quality") is not None:
        from src.tools import trusted_qwen_eval
        return trusted_qwen_eval.package_contract(task, config)
    descriptor = config.get("trusted_evaluation", {})
    require(isinstance(descriptor, dict) and type(descriptor.get("schema_version")) is int
            and descriptor["schema_version"] == 1,
            "task must opt into trusted_evaluation schema_version 1")
    require(config.get("harness_protection", {}).get("reject_new_source_symlinks") is True,
            "trusted tasks must reject new editable-source symlinks")
    sources = config.get("source_file_path")
    require(isinstance(sources, list) and sources, "explicit source_file_path list is required")
    sources = [relative_file(source) for source in sources]
    require(len(set(sources)) == len(sources), "duplicate editable source path")
    manifest_path = relative_file(descriptor.get("case_manifest", "cases.json"))
    helper_path = relative_file(descriptor.get("contract_file", "ut/evaluation_contract.py"))
    guard_path = relative_file(descriptor.get("source_guard", "ut/source_guard.py"))
    references = descriptor.get("reference_sources", {})
    require(isinstance(references, dict) and set(references) == set(sources), "every source needs a frozen reference")
    references = {source: relative_file(reference) for source, reference in references.items()}
    reserved = {"config.yaml", "scripts/task_runner.py", manifest_path, helper_path, guard_path, *references.values()}
    require(not set(sources) & reserved, "editable sources overlap a protected harness/reference file")
    mode = descriptor.get("distribution_correctness_mode")
    for phase in PHASES:
        arguments = distribution_phase_arguments(mode, phase)
        expected_command = " ".join(["python3", "scripts/task_runner.py", phase, *arguments])
        require(config.get(phase + "_command") == [expected_command],
                "trusted tasks must use the protected scripts/task_runner.py entrypoint")
        if mode is not None and phase + "_timeout" in config:
            phase_timeout(config, phase, config[phase + "_timeout"])
    require(read_regular(task / helper_path) == read_regular(PORTABLE_CONTRACT), "task contract helper differs from the trusted host version")
    manifest = validate_manifest(strict_json(read_regular(task / manifest_path)))
    require(config.get("headkernel", {}).get("docker") == manifest["runtime_image"], "task and cases name different runtime images")
    needs_cache = descriptor.get("requires_aiter_jit_cache", False)
    require(type(needs_cache) is bool, "requires_aiter_jit_cache must be boolean")
    for source, reference in references.items():
        require(read_regular(task / source) == read_regular(task / reference), "trusted source differs from its frozen reference")
    fixtures = None
    if "fixture_manifest" in descriptor:
        fixtures = load_fixture_manifest(task, descriptor["fixture_manifest"], manifest, reserved | set(sources))
    score_policy = scoring_policy(config)
    if score_policy is not None:
        read_regular(task / score_policy['native_source_manifest'])
    return {"sources": sources, "references": references, "manifest": manifest,
            "guard": guard_path, "needs_cache": needs_cache, "fixtures": fixtures,
            "scoring_policy": score_policy, "config": config,
            "distribution_correctness_mode": mode,
            "distributions": distribution_contract(task, manifest, mode)}


def prepare_reference(*, repo, commit, task_path, reference, staging, candidate_workspace,
                      output, scratch_explicit, fixture_local_mirror=None, fixture_oci_prefix=None,
                      fixture_max_files=DEFAULT_MAX_FILES, fixture_max_bytes=DEFAULT_MAX_BYTES, timeout=1800):
    from src.tools import trusted_qwen_eval
    qwen = task_path in {"tasks/headkernel/" + name for name in trusted_qwen_eval.PROFILES}
    if qwen:
        extract_task(repo, commit, task_path, reference, allow_internal_symlinks=True)
    else:
        extract_task(repo, commit, task_path, reference)
    contract = package_contract(reference)
    fixture_receipt = None
    if "qwen_profile" in contract:
        from src.tools import trusted_qwen_eval
        require(scratch_explicit, "trusted Qwen fixture/cache staging requires an explicit scratch directory")
        fixture_receipt = trusted_qwen_eval.prepare_fixtures(
            repo, commit, task_path, reference, output, contract,
            local_mirror=fixture_local_mirror, oci_prefix=fixture_oci_prefix,
            max_files=fixture_max_files, max_bytes=fixture_max_bytes, timeout=timeout)
        return contract, fixture_receipt
    if contract["fixtures"] is not None:
        require(scratch_explicit, "external fixtures require an explicit --scratch-dir with sufficient disk space")
        proof = materialize_fixtures(
            reference, contract["fixtures"], contract["manifest"], candidate_workspace=candidate_workspace,
            staging=staging, local_mirror=fixture_local_mirror, allowed_oci_prefix=fixture_oci_prefix,
            max_files=fixture_max_files, max_bytes=fixture_max_bytes, timeout=timeout)
        proof.update(trusted_commit=commit, task_path=task_path)
        encoded = canonical(proof).encode() + b"\n"
        (output / "fixtures_receipt.json").write_bytes(encoded)
        fixture_receipt = {"file": "fixtures_receipt.json", "sha256": sha256(encoded)}
    return contract, fixture_receipt


def guard_sources(reference, candidate, guard_path):
    code = (
        "import pathlib,runpy,sys; "
        "sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent)); "
        "guard=runpy.run_path(sys.argv[1]); "
        "guard['validate_sources'](pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3]))"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(reference / guard_path),
                    str(candidate), str(reference)], check=True, timeout=30)


def declared_failure_bundles(build):
    """Only bounded tensor files named and hashed by a failure manifest qualify."""
    bundles = {}
    for manifest in sorted(build.glob('*.tensor_failure.json')):
        require(manifest.stat().st_size <= 1 << 20, 'failure manifest exceeds metadata bound')
        record = strict_json(read_regular(manifest))
        require(isinstance(record, dict), 'tensor failure manifest must be a dictionary')
        require(record.get('schema') == 'trusted-tensor-failure-v1', 'unknown tensor failure schema')
        item = record.get('bundle')
        if item is None:
            continue
        require(isinstance(item, dict) and item.get('file') == manifest.with_suffix('.pt').name,
                'tensor failure bundle must share its manifest basename')
        require(type(item.get('bytes')) is int and 0 <= item['bytes'] <= 2 << 30
                and isinstance(item.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', item['sha256']),
                'invalid tensor failure bundle size or digest')
        require(len(bundles) < 1, 'only one tensor failure bundle is permitted per phase')
        source = build / item['file']
        require(source.is_file() and not source.is_symlink() and source.stat().st_size == item['bytes'],
                'declared tensor failure bundle is missing or differs in size')
        bundles[item['file']] = item
    return bundles


def preserve_diagnostics(build, output, *, recursive=False, excluded=()):
    output.mkdir()
    records = {}
    try:
        bundles = {}
        def selected_sources():
            # Keep the original textual failure context even if its tensor
            # bundle is missing or invalid.
            candidates = build.rglob("*") if recursive else build.iterdir()
            for source in sorted(candidates):
                if str(source.relative_to(build)) in excluded or source.is_dir():
                    continue
                if source.suffix in (".json", ".jsonl", ".log", *([".txt"] if recursive else [])):
                    yield source
            bundles.update(declared_failure_bundles(build))
            for name in sorted(bundles):
                yield build / name
        for source in selected_sources():
            relative = str(source.relative_to(build))
            require("\n" not in relative and "\r" not in relative,
                    "diagnostic filenames cannot contain line breaks")
            def diagnostic_digest(path):
                if source.name in bundles:
                    return hash_regular_file(path, max_bytes=bundles[source.name]['bytes'])
                return sha256(read_regular(path))
            digest = diagnostic_digest(source)
            if source.name in bundles:
                require(digest == bundles[source.name]['sha256'], 'tensor failure bundle digest changed')
            target = output / relative
            environment = os.environ.copy()
            environment["GOMAXPROCS"] = "1"
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="aka-diagnostic-") as listing:
                listing.write(relative + "\n")
                listing.flush()
                subprocess.run(["rclone", "copy", str(build), str(output), "--files-from-raw", listing.name,
                                "--transfers", "64000", "--progress", "--buffer-size", "0",
                                "--multi-thread-streams", "0", "--config", os.devnull, "--no-traverse"],
                               check=True, timeout=300, env=environment)
            require(diagnostic_digest(target) == digest, "diagnostic copy differs from phase output")
            records[relative] = digest
    finally:
        (output / "hashes.json").write_text(json.dumps(records, indent=2) + "\n")


def container_resource_arguments(resources):
    """Explicit host limits; these do not alter task measurement settings."""
    resources = resources or {}
    require(set(resources) <= {"cpus", "memory_gib"}, "unknown container resource limit")
    arguments = []
    cpus = resources.get("cpus")
    if cpus is not None:
        require(type(cpus) in (int, float) and math.isfinite(cpus) and cpus > 0,
                "container CPU limit must be positive and finite")
        arguments.extend(["--cpus", str(cpus)])
    memory = resources.get("memory_gib")
    if memory is not None:
        require(type(memory) is int and memory > 0, "container memory GiB must be a positive integer")
        arguments.extend(["--memory", str(memory) + "g"])
    return arguments


def process_memory_snapshot(proc=Path("/proc")):
    """Read only the invoking process's cgroup, without resetting counters."""
    try:
        group = next(line.split(":", 2)[2] for line in (proc / "self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        group = PurePosixPath(group)
        require(group.is_absolute() and ".." not in group.parts, "invalid process cgroup path")
        mounts = [line.split() for line in (proc / "self/mountinfo").read_text().splitlines()
                  if " - cgroup2 " in line]
        require(len(mounts) == 1, "one process cgroup2 mount is required")
        mount_root, mountpoint = PurePosixPath(mounts[0][3]), Path(mounts[0][4])
        suffix = PurePosixPath(".") if str(group) == "/" else group.relative_to(mount_root)
        directory = mountpoint / suffix
        counters = {}
        for name in ("memory.current", "memory.peak", "memory.max", "memory.events",
                     "memory.events.local", "memory.swap.current", "memory.swap.max"):
            path = directory / name
            if path.is_file():
                counters[name] = path.read_text().strip()
        require("memory.current" in counters and "memory.max" in counters,
                "process cgroup memory counters are unavailable")
        return {"status": "observed", "cgroup": str(directory), "counters": counters,
                "observed_unix": time.time()}
    except (OSError, ValueError, StopIteration) as error:
        return {"status": "unavailable", "error": str(error), "observed_unix": time.time()}


def preserve_container_exit(name, destination, before):
    """Retain OOM/limit evidence before cleanup; never dump Config.Env."""
    record = {"schema": "trusted-container-exit-v1", "name": name,
              "process_memory_before": before, "process_memory_after": process_memory_snapshot(),
              "observed_unix": time.time(), "container": None}
    try:
        result = subprocess.run(["docker", "inspect", name], capture_output=True,
                                text=True, check=False, timeout=10)
        require(result.returncode == 0, "container inspection failed: " + (result.stderr or "")[:2000])
        rows = json.loads(result.stdout)
        require(isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict)
                and rows[0].get("Name") == "/" + name,
                "container inspection identity differs")
        row = rows[0]
        require(isinstance(row.get("State"), dict) and isinstance(row.get("HostConfig"), dict),
                "container state or limits are missing")
        require(re.fullmatch(r"[a-f0-9]{64}", row.get("Id", "")) is not None,
                "container inspection ID is invalid")
        record["container"] = {
            "id": row["Id"], "name": row["Name"], "image_id": row.get("Image"),
            "state": {key: row.get("State", {}).get(key) for key in
                      ("Status", "OOMKilled", "ExitCode", "Error", "StartedAt", "FinishedAt")},
            "limits": {key: row.get("HostConfig", {}).get(key) for key in
                       ("Memory", "MemorySwap", "NanoCpus", "CpuQuota", "CpuPeriod", "CpusetCpus", "PidsLimit")},
        }
        record["status"] = "observed"
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        record["status"] = "unavailable"
        record["error"] = str(error)
    with Path(destination).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_phase(image, task, staging, output, leg, request, render_device, timeout, cache_source=None,
              container_resources=None, qwen_contract=None, worker_assignment=None, case_plan=None,
              group_control=None):
    phase = request["phase"]
    label = leg + "_" + phase
    if worker_assignment is not None:
        require(qwen_contract is None and case_plan is not None, "unsupported case worker contract")
        label += "_worker_" + str(worker_assignment["worker_id"])
    expected_gpu = request["gpu"] if worker_assignment is None else worker_assignment["expected_gpu"]
    build = task / "build" if qwen_contract is not None else staging / (label + "_build")
    build.mkdir()
    omitted = set()
    ut_runtime = None
    cache = None
    if cache_source is not None:
        cache = staging / (label + "_jit")
        copy_cache(cache_source, cache, timeout=timeout)
    request_path = staging / (label + "_request.json")
    request_path.write_text(canonical(request))
    gpu_path = staging / (label + "_gpu.json")
    gpu_path.write_text(canonical(expected_gpu))
    name = "aka-task-retest-" + uuid.uuid4().hex
    command = docker_command(image, task, build, render_device, name, cache)
    if qwen_contract is not None:
        from src.tools import trusted_qwen_eval
        ut_runtime = staging / (label + "_ut_runtime")
        mounts, omitted = trusted_qwen_eval.runtime_mounts(task, build, qwen_contract, runtime_root=ut_runtime)
        mounts += ["--mount", f"type=bind,src={trusted_qwen_eval.PHASE_BRIDGE},dst=/trusted_qwen_phase.py,readonly",
                   "--env", "HK_RUN_ROOT=/task/build/ledger-root"]
        command[command.index(image):command.index(image)] = mounts
        command[-1] = "/trusted_qwen_phase.py"
    # Existing finally cleanup owns removal. Automatic removal would discard
    # Docker's OOMKilled/ExitCode evidence before the host can inspect it.
    command = [argument for argument in command if argument != "--rm"]
    command = command_with_binding(command, image, expected_gpu, GPU_BINDING_HELPER, gpu_path)
    image_index = command.index(image)
    command[image_index:image_index] = container_resource_arguments(container_resources)
    image_index = command.index(image)
    if cache is not None:
        # AITER's import otherwise selects the image's root-owned FlyDSL cache
        # independently of AITER_JIT_DIR. Keep the verified AITER cache intact,
        # but compile FlyDSL from this phase's sources instead of image artifacts.
        fresh_flydsl_name = "fresh_flydsl_" + request["request_id"]
        fresh_flydsl_cache = cache / fresh_flydsl_name
        require(not fresh_flydsl_cache.exists() and not fresh_flydsl_cache.is_symlink(),
                "FlyDSL phase cache must start absent")
        command[image_index:image_index] = ["--env", "FLYDSL_RUNTIME_CACHE_DIR=/aiter-jit/" + fresh_flydsl_name]
        image_index = command.index(image)
    command[image_index:image_index] = ["--mount", f"type=bind,src={request_path},dst=/evaluation-request.json,readonly"]
    if worker_assignment is not None:
        from src.tools import case_parallel
        private_output = build / Path(worker_assignment["output_directory"]).relative_to("build")
        private_output.mkdir(parents=True)
        assignment_path = staging / (label + "_assignment.json")
        assignment_path.write_text(canonical(worker_assignment))
        image_index = command.index(image)
        command[image_index:image_index] = [
            "--mount", f"type=bind,src={assignment_path},dst=/case-worker-assignment.json,readonly",
            "--mount", f"type=bind,src={case_plan['_plan_path']},dst=/run/aka-case-gpu-plan.json,readonly",
            "--env", case_parallel.PLAN_ENV + "=/run/aka-case-gpu-plan.json",
            "--env", case_parallel.HASH_ENV + "=" + case_plan["_plan_sha256"]]
    if qwen_contract is not None:
        command += ["--task", "/task", "--request", "/evaluation-request.json", "--timeout", str(timeout)]
    else:
        command += [phase, "--request", "/evaluation-request.json"]
        command += distribution_phase_arguments(request.get("distribution_correctness_mode"), phase)
        if worker_assignment is not None:
            command += ["--case-worker", "/case-worker-assignment.json"]
    primary = None
    memory_before = process_memory_snapshot()
    try:
        if group_control is not None:
            group_control.register(name)
        with (output / (label + ".log")).open("xb") as log:
            if group_control is None:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
            else:
                run_case_container(command, log, timeout, group_control)
        validate_preflight(strict_json(read_regular(build / "gpu_preflight.json")), expected_gpu)
        if worker_assignment is not None:
            destination = task / worker_assignment["output_directory"] / "receipt.json"
            with destination.open("xb") as stream:
                stream.write(read_regular(private_output / "receipt.json"))
            return destination
        return strict_json(read_regular(build / (phase + "_report.json")))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = []
        try:
            preserve_container_exit(name, output / (label + ".container.json"), memory_before)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(exc)
        try:
            removal = subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, check=False, timeout=30)
            if ut_runtime is not None and ut_runtime.exists():
                require(removal.returncode == 0, "Qwen container removal failed; UT outputs remain private")
                ut_runtime.rename(build / ".runtime-ut")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(exc)
        try:
            options = {"recursive": True, "excluded": omitted} if qwen_contract is not None or worker_assignment is not None else {}
            preserve_diagnostics(build, output / (label + ".diagnostics"), **options)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(exc)
        for exc in failures:
            if primary is None:
                raise exc
            primary.add_note("Additional diagnostic/cleanup failure: " + str(exc))
        if group_control is not None:
            group_control.finished(name)


def run_case_container(command, log, timeout, control):
    """Observe the real Docker exit and interrupt the CLI on group failure."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            require(not control.cancelled.is_set(), "owned case group was cancelled")
            require(time.monotonic() < deadline, "owned case container timed out")
            time.sleep(0.1)
        require(process.returncode == 0, "owned case container exited with code " + str(process.returncode))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class CaseGroupControl:
    """Cancel only the fresh containers owned by this phase; never retry."""
    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.names = set()

    def register(self, name):
        with self.lock:
            require(not self.cancelled.is_set(), "case group was cancelled before container launch")
            self.names.add(name)

    def finished(self, name):
        with self.lock:
            self.names.discard(name)

    def stop(self):
        self.cancelled.set()
        with self.lock:
            names = sorted(self.names)
        for name in names:
            # Stop retains Docker ExitCode/OOMKilled evidence for run_phase's
            # existing inspection and final removal. Repeated stops also close
            # the short race between registration and Docker daemon creation.
            try:
                subprocess.run(["docker", "stop", "--time", "1", name], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            except (OSError, subprocess.SubprocessError):
                # Worker supervisors also terminate their CLI and force-remove
                # their container in run_phase's existing finally cleanup.
                pass


def run_group_phase(image, task, staging, output, leg, request, plan, timeout, cache_source=None):
    from src.tools import case_parallel
    assignments = case_parallel.prepare_assignments(task, plan["_manifest"], request,
        {k: v for k, v in plan.items() if k != "_manifest"}, leg=leg)
    stop = min(time.time() + min(timeout, plan["phase_timeouts"][request["phase"]]), plan["work_stop_unix"])
    for assignment in assignments:
        assignment["deadline_unix"] = min(assignment["deadline_unix"], stop)
    resources = {"cpus": plan["resources"]["cpus_per_worker"], "memory_gib": plan["resources"]["memory_gib_per_worker"]}
    control = CaseGroupControl()
    pool = ThreadPoolExecutor(max_workers=len(assignments))
    futures = []
    handlers = {}
    def cancelled(signum, _frame):
        raise InterruptedError("trusted case group received signal " + str(signum))
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, cancelled)
        for assignment in assignments:
            futures.append(pool.submit(run_phase, image, task, staging, output, leg, request,
                assignment["expected_gpu"]["render_device"], max(1, int(stop - time.time())), cache_source,
                container_resources=resources, worker_assignment=assignment, case_plan=plan, group_control=control))
        pending = set(futures)
        while pending:
            require(time.time() < stop, "case group phase deadline expired")
            completed, pending = wait(pending, timeout=min(0.2, max(0, stop - time.time())), return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
        paths = [future.result() for future in futures]
        require(case_parallel.input_tree_identity(task) == assignments[0]["input_tree_sha256"], "staged worker inputs changed")
        union = case_parallel.collect_workers(plan["_manifest"], request, plan, assignments, [0] * len(paths), paths)
        union_path = staging / (leg + "_" + request["phase"] + "_union.json")
        request_path = staging / (leg + "_" + request["phase"] + "_parent_request.json")
        report_path = staging / (leg + "_" + request["phase"] + "_full_report.json")
        union_path.write_text(canonical(union))
        request_path.write_text(canonical(request))
        require(time.time() < stop, "case group deadline expired before finalization")
        subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).with_name("case_finalize.py")),
            "--task", str(task), "--request", str(request_path), "--union", str(union_path),
            "--output", str(report_path)], check=True, timeout=max(1, stop - time.time()))
        report = strict_json(read_regular(report_path))
        case_parallel.validate_group_report(task, plan["_manifest"], report, required=True)
        # Preserve the exact chain under one per-leg root for later audits.
        for worker in union["workers"]:
            archived = output / (leg + "_group_receipts") / worker["receipt_path"]
            archived.parent.mkdir(parents=True, exist_ok=True)
            with archived.open("xb") as stream:
                stream.write(read_regular(task / worker["receipt_path"]))
        return report
    except BaseException:
        control.cancelled.set()
        while any(not future.done() for future in futures):
            control.stop()
            wait(futures, timeout=0.2)
        raise
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
        pool.shutdown(wait=True, cancel_futures=True)


def trusted_retest(*, repo, commit, task_path, candidate_workspace, output, render_device=None, scratch_dir=None, timeout=7200,
                   fixture_local_mirror=None, fixture_oci_prefix=None,
                   fixture_max_files=DEFAULT_MAX_FILES, fixture_max_bytes=DEFAULT_MAX_BYTES,
                   phase_timeouts=None, container_resources=None, case_gpu_plan=None, case_gpu_plan_sha256=None):
    # Reject malformed host budgets before staging fixtures or selecting a GPU.
    evaluation_timeouts({}, None, timeout, phase_timeouts)
    require(bool(render_device) != bool(case_gpu_plan) and bool(case_gpu_plan) == bool(case_gpu_plan_sha256),
            "select exactly one render device or complete case GPU plan/digest")
    repo, candidate_workspace = Path(repo).resolve(), Path(candidate_workspace).resolve()
    output = Path(output).absolute()
    scratch = Path(scratch_dir or tempfile.gettempdir()).resolve()
    for path in (repo, output.resolve(), scratch):
        require(not path.is_relative_to(candidate_workspace), "trusted repository/output/scratch must be outside the agent workspace")
    require(timeout > 0, "timeout must be positive")
    container_resource_arguments(container_resources)
    resource_options = {"container_resources": container_resources} if container_resources else {}
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, parents=True)
    with tempfile.TemporaryDirectory(prefix="aka-task-retest-", dir=scratch) as temporary:
        staging = Path(temporary)
        info = staging.stat()
        require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700, "scratch directory is not private")
        reference = staging / "reference"
        contract, fixture_receipt = prepare_reference(
            repo=repo, commit=commit, task_path=task_path, reference=reference, staging=staging,
            candidate_workspace=candidate_workspace, output=output, scratch_explicit=scratch_dir is not None,
            fixture_local_mirror=fixture_local_mirror, fixture_oci_prefix=fixture_oci_prefix,
            fixture_max_files=fixture_max_files, fixture_max_bytes=fixture_max_bytes, timeout=timeout)
        limits = evaluation_timeouts(contract["config"], contract["distribution_correctness_mode"],
                                     timeout, phase_timeouts)
        candidate = staging / "candidate"
        copy_cache(reference, candidate)
        qwen = "qwen_profile" in contract
        if qwen:
            from src.tools import trusted_qwen_eval
            trusted_qwen_eval.admit_sources(reference, candidate, candidate_workspace, contract, guard_sources)
        else:
            # Only these declared regular files are taken from the agent workspace.
            for source in contract["sources"]:
                (candidate / source).write_bytes(read_regular(candidate_workspace / source))
            guard_sources(reference, candidate, contract["guard"])
        manifest = contract["manifest"]
        image = manifest["runtime_image"]
        group_plan = None
        if case_gpu_plan is not None:
            from src.case_parallelism import enabled, verify_helpers
            from src.tools import case_parallel, case_gpu_plan as plan_admission
            require(not qwen and enabled(contract["config"]), "trusted case group task did not opt in")
            verify_helpers(reference, contract["config"])
            group_plan = case_parallel.load_plan(reference, manifest, {"manifest_sha256": fingerprint(manifest)},
                path=case_gpu_plan, sha256=case_gpu_plan_sha256)
            require(group_plan["task_path"] == task_path and group_plan["baseline_commit"] == commit,
                    "trusted group plan commit/task differs")
            require(group_plan["phase_timeouts"] == {phase: contract["config"].get(phase + "_timeout") for phase in PHASES},
                    "trusted plan differs from original phase caps")
            plan_admission.validate_scheduler(group_plan, plan_admission.scheduler_state(group_plan["job_id"]),
                                               plan_admission.socket.gethostname())
            for binding in group_plan["bindings"]:
                require(select_gpu(binding["render_device"]) == {k: v for k, v in binding.items() if k != "worker_id"},
                        "trusted group physical binding differs")
            granted_resources = {"cpus": group_plan["resources"]["cpus_per_worker"],
                                 "memory_gib": group_plan["resources"]["memory_gib_per_worker"]}
            require(container_resources is None or container_resources == granted_resources,
                    "container resources differ from the explicit group grant")
            group_plan["_manifest"] = manifest
            limits = {phase: min(value, group_plan["phase_timeouts"][phase]) for phase, value in limits.items()}
            gpu = None
        else:
            gpu = select_gpu(render_device)
        image_record = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
        require(any(item.endswith("@" + image.rsplit("@", 1)[1]) for item in image_record.get("RepoDigests", [])),
                "local image does not attest the requested digest")
        cache_source = None
        if contract["needs_cache"]:
            cache_root = staging / "image-jit"
            proof = seed_image_cache(image, cache_root, output / "cache_init.log", timeout=timeout)
            (output / "cache_manifest.json").write_text(canonical(proof))
            cache_source = cache_root / "jit"
        results, reports, sources, scored_native, scored_portable = {}, {}, {}, {}, {}
        phase_receipts = []
        challenge_seed = secrets.randbelow(2**30)
        for leg, task in (("reference", reference), ("candidate", candidate)):
            package = fingerprint(tree_manifest(task))
            sources[leg] = {source: sha256(read_regular(task / source)) for source in contract["sources"]}
            reports[leg] = {}
            if not qwen:
                (task / "build").mkdir()
            for phase in PHASES:
                request = {"schema_version": 1, "request_id": secrets.token_hex(24), "phase": phase,
                           "manifest_sha256": fingerprint(manifest), "package_sha256": package,
                           "source_sha256": sources[leg], "challenge_seed": challenge_seed, "gpu": gpu}
                if group_plan is not None:
                    request = case_parallel.bind_request(request, group_plan)
                mode = contract["distribution_correctness_mode"]
                if mode is not None and phase in ("compile", "correctness"):
                    request["distribution_correctness_mode"] = mode
                options = {**resource_options, **({"qwen_contract": contract} if qwen else {})}
                if group_plan is not None:
                    # Recheck the same grant before each complete phase; a
                    # preemption/restart invalidates the entire evaluation.
                    plan_admission.validate_scheduler(group_plan, plan_admission.scheduler_state(group_plan["job_id"]),
                                                       plan_admission.socket.gethostname())
                    report = run_group_phase(image, task, staging, output, leg, request, group_plan, limits[phase], cache_source)
                else:
                    report = run_phase(image, task, staging, output, leg, request, render_device,
                                       limits[phase], cache_source, **options)
                if qwen:
                    measured, quality_cases = trusted_qwen_eval.validate_phase(task, contract, phase, report, request)
                    phase_receipt = strict_json(read_regular(task / "build/host_phase_receipt.json"))
                    (task / "build").rename(staging / (leg + "_" + phase + "_completed_build"))
                else:
                    measured = validate_report(report, manifest, request)
                    validate_distribution_report(report, manifest, request, contract["distributions"], mode)
                # Build outputs are outside the staged input identity. The
                # serial path retains its original empty bind-mount point.
                current = tree_manifest(task)
                current.pop("build", None)
                if group_plan is not None:
                    current = {name: value for name, value in current.items() if not name.startswith("build/")}
                require(fingerprint(current) == package, "protected staged task files changed")
                encoded = canonical(report).encode() + b"\n"
                filename = leg + "_" + phase + ".json"
                (output / filename).write_bytes(encoded)
                reports[leg][phase] = {"file": filename, "sha256": sha256(encoded)}
                if qwen:
                    phase_receipt.update(status="verified", report=reports[leg][phase],
                                         diagnostics_directory=leg + "_" + phase + ".diagnostics")
                    receipt_name = leg + "_" + phase + ".phase_receipt.json"
                    receipt_bytes = canonical(phase_receipt).encode() + b"\n"
                    (output / receipt_name).write_bytes(receipt_bytes)
                    phase_receipts.append({"leg": leg, "phase": phase, "file": receipt_name,
                                           "sha256": sha256(receipt_bytes)})
                if phase == "performance":
                    results[leg] = measured
                    if contract['scoring_policy'] is not None:
                        evidence = load_native_measurements(task, contract['config'], report=report, request=request)
                        scored_native[leg] = as_test_cases(evidence, is_baseline=leg == 'reference')
                    elif qwen:
                        scored_portable[leg] = quality_cases
                    else:
                        scored_portable[leg] = portable_test_cases(report, manifest, request, sources[leg])
        cases = []
        for baseline, optimized in zip(results["reference"], results["candidate"]):
            require(baseline["case_sha256"] == optimized["case_sha256"], "reference/candidate ABI differs")
            ratio = baseline["execution_time_ms"] / optimized["execution_time_ms"]
            require(math.isfinite(ratio) and ratio > 0, "invalid speedup ratio")
            cases.append({"test_case_id": baseline["test_case_id"], "case_sha256": baseline["case_sha256"],
                          "reference_ms": baseline["execution_time_ms"], "candidate_ms": optimized["execution_time_ms"],
                          "speedup": ratio})
        result = {"schema_version": 1, "status": "measured", "trusted_commit": commit, "task_path": task_path,
                  "phase_timeouts_seconds": limits,
                  "image": image, "local_image_id": image_record["Id"], "source_sha256": sources,
                  "gpu": gpu,
                  "manifest_sha256": fingerprint(manifest), "full_case_coverage": True, "cases": cases,
                  "arithmetic_mean_speedup": math.fsum(row["speedup"] for row in cases) / len(cases),
                  "reports": reports, "framework_task_validator_status": "not_asserted"}
        if group_plan is not None:
            result.pop("gpu")
            result.update(gpu_group=group_plan["bindings"], case_parallelism={
                "plan_sha256": group_plan["_plan_sha256"], "assignment_sha256": group_plan["assignment_sha256"],
                "fresh_worker_containers": len(group_plan["bindings"]) * len(PHASES) * 2,
                "receipt_directories": {leg: leg + "_group_receipts" for leg in ("reference", "candidate")}})
        if container_resources:
            result["requested_container_resources"] = dict(container_resources)
        if qwen:
            require(len(phase_receipts) == 6, "all six independent Qwen phases are required")
            result.update(report_contract="qwen_graph", fresh_phase_count=6, phase_receipts=phase_receipts)
        if contract["distributions"] is not None:
            result["observed_work_correctness"] = {
                "mode": contract["distribution_correctness_mode"],
                "settings_per_leg": sum(len(group["histogram"]) for group in contract["distributions"].values()),
                "verified_legs": ["reference", "candidate"],
                "all_observed_settings_correctness": True,
                "all_observed_settings_timed": False,
                "actual_other_rank_routing_recovered": False,
                "scope": "recorded work counts with representative generated routes and activations"}
        if contract['scoring_policy'] is not None:
            summary = metric_summary(scored_native['reference'], scored_native['candidate'])
            result.update(summary)
            result['arithmetic_mean_speedup'] = summary['native_speedup_ratio']
            result['cases'] = [{**row, 'reference_ms': row['native_ms']}
                               for row in summary['native_baseline_cases']]
            quality = summary['benchmark_quality']
        else:
            quality = comparison_quality(scored_portable['reference'], scored_portable['candidate'])
            for row in result['cases']:
                row['raw_speedup'] = row['speedup']
                if not quality['gain_eligible']:
                    row['speedup'] = None
            result.update(benchmark_quality_required=True, benchmark_quality=quality,
                          gain_eligible=quality['gain_eligible'], accepted_gain=quality['accepted_gain'])
        result['raw_arithmetic_mean_speedup'] = quality['raw_arithmetic_mean_speedup']
        result['arithmetic_mean_speedup'] = quality['accepted_arithmetic_mean_speedup']
        if quality['status'] != 'pass':
            result['status'] = 'rejected_timing_quality'
        if fixture_receipt is not None:
            result["fixtures"] = fixture_receipt
        (output / "trusted_measurement.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result


def stage_trusted_task(*, repo, commit, task_path, output, scratch_dir, fixture_local_mirror=None,
                       fixture_oci_prefix=None, fixture_max_files=DEFAULT_MAX_FILES,
                       fixture_max_bytes=DEFAULT_MAX_BYTES, timeout=1800):
    """Prepare a verified original task for a validator; no GPU or agent ingress."""
    repo, output, scratch = Path(repo).resolve(), Path(output).absolute(), Path(scratch_dir).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="aka-task-stage-", dir=scratch) as temporary:
        staging = Path(temporary)
        reference = staging / "reference"
        _contract, receipt = prepare_reference(
            repo=repo, commit=commit, task_path=task_path, reference=reference, staging=staging,
            candidate_workspace=staging / "unused-candidate", output=output, scratch_explicit=True,
            fixture_local_mirror=fixture_local_mirror, fixture_oci_prefix=fixture_oci_prefix,
            fixture_max_files=fixture_max_files, fixture_max_bytes=fixture_max_bytes, timeout=timeout)
        copy_cache(reference, output / "task", timeout=timeout)
        result = {"schema_version": 1, "status": "staged_not_evaluated", "trusted_commit": commit,
                  "task_path": task_path, "package_sha256": fingerprint(tree_manifest(output / "task")),
                  "fixtures": receipt}
        (output / "staging_receipt.json").write_text(canonical(result) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "commit", "task", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--candidate-workspace")
    parser.add_argument("--render-device")
    parser.add_argument("--case-gpu-plan")
    parser.add_argument("--case-gpu-plan-sha256")
    parser.add_argument("--stage-only", action="store_true", help="prepare a verified original task for a validator without GPU execution")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--fixture-local-mirror", help="trusted mirror using portable object_key or original headkernel fixture paths")
    parser.add_argument("--fixture-oci-prefix", help="host-approved exact OCI prefix for fixture download")
    parser.add_argument("--fixture-max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--fixture-max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--timeout", type=int, default=7200)
    for phase in PHASES:
        parser.add_argument("--" + phase + "-timeout", type=int,
                            help="optional host cap for each " + phase + " phase; cannot exceed --timeout")
    parser.add_argument("--container-cpus", type=float, help="explicit CPU quota for each task phase")
    parser.add_argument("--container-memory-gib", type=int, help="explicit memory limit for each task phase")
    args = parser.parse_args()
    if Path("/.dockerenv").exists():
        parser.error("run from a trusted host outside the agent container")
    fixtures = dict(fixture_local_mirror=args.fixture_local_mirror, fixture_oci_prefix=args.fixture_oci_prefix,
                    fixture_max_files=args.fixture_max_files, fixture_max_bytes=args.fixture_max_bytes)
    phase_timeouts = {phase: getattr(args, phase + "_timeout") for phase in PHASES
                      if getattr(args, phase + "_timeout") is not None}
    if args.stage_only:
        if not args.scratch_dir or args.candidate_workspace or args.render_device or phase_timeouts or args.case_gpu_plan or args.case_gpu_plan_sha256:
            parser.error("--stage-only requires --scratch-dir and accepts no candidate workspace, GPU device or phase timeouts")
        result = stage_trusted_task(repo=args.repo, commit=args.commit, task_path=args.task, output=args.output,
                                   scratch_dir=args.scratch_dir, timeout=args.timeout, **fixtures)
        print(json.dumps(result))
        return
    if (not args.candidate_workspace or bool(args.render_device) == bool(args.case_gpu_plan)
            or bool(args.case_gpu_plan) != bool(args.case_gpu_plan_sha256)):
        parser.error("retesting requires --candidate-workspace and exactly one render device or case GPU plan/digest")
    resources = {key: value for key, value in
                 (("cpus", args.container_cpus), ("memory_gib", args.container_memory_gib))
                 if value is not None}
    result = trusted_retest(repo=args.repo, commit=args.commit, task_path=args.task,
                            candidate_workspace=args.candidate_workspace, output=args.output,
                            render_device=args.render_device, scratch_dir=args.scratch_dir, timeout=args.timeout,
                            phase_timeouts=phase_timeouts, container_resources=resources or None,
                            case_gpu_plan=args.case_gpu_plan, case_gpu_plan_sha256=args.case_gpu_plan_sha256, **fixtures)
    print(json.dumps({"measurement": str(Path(args.output) / "trusted_measurement.json"),
                      "arithmetic_mean_speedup": result["arithmetic_mean_speedup"]}))
    if result['status'] == 'rejected_timing_quality':
        raise SystemExit(1)


if __name__ == "__main__":
    main()
