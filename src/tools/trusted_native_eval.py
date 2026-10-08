"""Host-only, fresh-container retest for the guarded SG520 quant task contract.

Trust the invoking host, Git object database, Docker daemon and pinned image.
Never execute this tool from the permissive optimization worker. GPU driver
isolation is not a general security sandbox for hostile native programs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

import yaml

if __package__:
    from .seed_aiter_jit_cache import copy_cache, seed_image_cache
    from ..benchmark_quality import gate_native_measurement
else:
    from seed_aiter_jit_cache import copy_cache, seed_image_cache
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmark_quality import gate_native_measurement

SOURCE = "source/quant_kernels.cu"
MODES = ("compile", "correctness", "performance")
LEGS = ("production_native", "candidate_native")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_regular(path):
    """Open without following any symlink, including a redirected parent."""
    path = Path(os.path.abspath(path))
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024 * 1024:
                raise ValueError("input must be a regular file no larger than 64 MiB")
            data = handle.read(64 * 1024 * 1024 + 1)
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ) or len(data) != before.st_size:
                raise ValueError("input changed while being read")
            return data
    finally:
        os.close(directory)


def extract_task(repo, commit, task_path, destination):
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("--commit must be an explicit full Git commit hash")
    relative = PurePosixPath(task_path)
    if relative.is_absolute() or not relative.parts or any(p in ("..", ".git") for p in relative.parts):
        raise ValueError("task must be a repository-relative package path")
    git = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo)]
    resolved = subprocess.check_output(git + ["rev-parse", "--verify", commit + "^{commit}"], text=True).strip()
    if resolved != commit:
        raise ValueError("Git object is not the requested commit")
    archive = subprocess.check_output(git + ["archive", "--format=tar", commit, "--", str(relative)])
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if member.isdir() and (name == relative or name in relative.parents):
                continue
            if not name.is_relative_to(relative) or ".." in name.parts:
                raise ValueError("Git archive contains an escaping path")
            local = name.relative_to(relative)
            if any(p in {"build", "__pycache__", ".venv", ".task-venv", ".git"} for p in local.parts):
                raise ValueError("trusted task contains a generated cache or build tree")
            target = destination / local
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile() and not member.issym() and not member.islnk():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as handle:
                    target.write_bytes(handle.read())
            else:
                raise ValueError("unsupported task package: only regular files and directories are accepted")
    return resolved


def validate_contract(task):
    config = yaml.safe_load((task / "config.yaml").read_text())
    manifest = json.loads((task / "cases.json").read_text())
    if (config.get("task_type") != "hip2hip"
            or config.get("source_file_path") != [SOURCE]
            or config.get("target_file_path") != SOURCE
            or config.get("target_kernel_functions") != ["dynamic_per_group_scaled_quant_kernel"]
            or config.get("harness_protection", {}).get("reject_new_source_symlinks") is not True
            or config.get("platform_support", {}).get("required_arch") != "gfx950"):
        raise ValueError("unsupported task: requires the guarded SG520 native quant contract")
    for mode in MODES:
        if config.get(mode + "_command") != ["python3 scripts/task_runner.py " + mode]:
            raise ValueError("unsupported protected entrypoint")
    image = config["headkernel"]["docker"]
    if (not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image)
            or image != manifest.get("runtime_image")):
        raise ValueError("task and cases must declare the same digest-pinned image")
    cases = manifest.get("cases", [])
    if len(cases) != 3 or len({c["case_id"] for c in cases}) != 3:
        raise ValueError("unsupported workload: exactly three distinct cases are required")
    if any(not isinstance(c.get("shape"), list)
           or type(c.get("trace_call_count")) is not int or c["trace_call_count"] <= 0 for c in cases):
        raise ValueError("invalid frozen case shapes or weights")
    if (task / SOURCE).read_bytes() != (task / "ut/native/quant_kernels.reference.cu").read_bytes():
        raise ValueError("trusted source must be the frozen reference source")
    return image, manifest


def validate_candidate(task, candidate_bytes):
    candidate = task.parent / "submission.cu"
    candidate.write_bytes(candidate_bytes)
    # Only the guard extracted from the explicit trusted commit executes here.
    # -I/-B exclude caller PYTHONPATH, user packages and bytecode caches.
    code = (
        "import pathlib,runpy,sys; "
        "guard=runpy.run_path(sys.argv[1]); "
        "guard['validate_source'](pathlib.Path(sys.argv[2]).read_text(),"
        "pathlib.Path(sys.argv[3]).read_text())"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", code,
                    str(task / "ut/source_guard.py"), str(candidate),
                    str(task / "ut/native/quant_kernels.reference.cu")], check=True, timeout=30)


def copy_payload(source, destination):
    """Stage a local trusted payload with the required transfer settings."""
    destination.mkdir()
    subprocess.run([
        "rclone", "copy", str(source.resolve()), str(destination.resolve()),
        "--transfers", "64000", "--progress", "--config", os.devnull,
    ], check=True, timeout=300)
    if identities(destination) != identities(source):
        raise ValueError("staged task payload differs from its trusted reference")


def identities(task, *, candidate_bytes=None):
    # Offline quality review can bind a source-only candidate without writing
    # it into the trusted task or modifying any existing evidence.
    def contents(path):
        return candidate_bytes if candidate_bytes is not None and path == task / SOURCE else read_regular(path)
    paths = [task / "cases.json", task / "config.yaml"]
    for directory in ("source", "ut", "scripts", "provenance"):
        paths.extend(p for p in (task / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    package = hashlib.sha256()
    for path in sorted(paths):
        package.update(str(path.relative_to(task)).encode() + b"\0")
        package.update(contents(path))
    native = hashlib.sha256()
    sources = [task / SOURCE, task / "ut/native/quant_entry_pybind.cu"]
    sources += sorted(p for p in (task / "ut/native/include").rglob("*") if p.is_file())
    for path in sources:
        native.update(str(path.relative_to(task)).encode())
        native.update(contents(path))
    return {"package_sha256": package.hexdigest(), "source_tree_sha256": native.hexdigest()}


def docker_command(image, task, build, render_device, name, jit_cache=None):
    if not re.fullmatch(r"/dev/dri/renderD[0-9]+", str(render_device)):
        raise ValueError("one explicit /dev/dri/renderD<number> device is required")
    for path in (task, build, *([jit_cache] if jit_cache is not None else [])):
        if "," in str(path):
            raise ValueError("Docker bind paths cannot contain commas")
    command = [
        "docker", "run", "--rm", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--ipc=private", "--pids-limit=1024",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--device", "/dev/kfd", "--device", str(render_device),
        "--mount", f"type=bind,src={task},dst=/task,readonly",
        "--mount", f"type=bind,src={build},dst=/task/build",
        "--tmpfs", "/tmp:rw,exec,nosuid,mode=1777,size=8g",
        "--tmpfs", "/cache:rw,exec,nosuid,mode=1777,size=8g",
        "--shm-size=2g", "--workdir", "/task", "--entrypoint", "python3",
    ]
    for device in (Path("/dev/kfd"), Path(render_device)):
        command.extend(["--group-add", str(device.stat().st_gid)])
    for variable in (
        # AITER_JIT_DIR is set below only for a complete, verified image cache.
        "HOME=/tmp", "USER=aka-evaluator", "LOGNAME=aka-evaluator", "XDG_CACHE_HOME=/cache",
        "TORCH_EXTENSIONS_DIR=/cache/torch", "TORCHINDUCTOR_CACHE_DIR=/cache/inductor",
        "TRITON_CACHE_DIR=/cache/triton",
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "PYTHONPATH=", "LD_PRELOAD=",
        "ROCR_VISIBLE_DEVICES=0", "HIP_VISIBLE_DEVICES=0",
    ):
        command.extend(["--env", variable])
    if jit_cache is not None:
        command.extend(["--mount", f"type=bind,src={jit_cache},dst=/aiter-jit",
                        "--env", "AITER_JIT_DIR=/aiter-jit"])
    return command + [image, "-I", "-B", "/task/scripts/task_runner.py"]


def preserve_phase_diagnostics(build, mode, log_path):
    """Keep worker/compiler output after disposable build trees are removed."""
    destination = log_path.with_suffix(".diagnostics")
    destination.mkdir()
    manifest = {"mode": mode, "logs": {}, "reports": {}, "errors": []}
    try:
        if log_path.is_file():
            manifest["coordinator_log"] = {
                "file": log_path.name, "sha256": sha256(read_regular(log_path)),
            }
        sources = [build / f"{mode}_{leg}.log" for leg in LEGS]
        sources.append(build / f"{mode}_report.json")
        for source in sources:
            if not source.exists() and not source.is_symlink():
                continue
            expected = sha256(read_regular(source))
            target = destination / source.name
            subprocess.run([
                "rclone", "copyto", str(source.resolve()), str(target.resolve()),
                "--transfers", "64000", "--progress", "--config", os.devnull,
            ], check=True, timeout=300)
            actual = sha256(read_regular(target))
            if actual != expected:
                raise ValueError("preserved worker log differs from container output")
            category = "reports" if source.suffix == ".json" else "logs"
            manifest[category][source.name] = {"sha256": actual, "bytes": target.stat().st_size}
    except Exception as exc:
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_mode(image, task, build, render_device, mode, log_path, timeout, jit_source=None):
    build.mkdir()
    name = "aka-trusted-" + uuid.uuid4().hex
    jit_cache = None
    if jit_source is not None:
        jit_cache = build.parent / (build.name + "_jit")
        copy_cache(jit_source, jit_cache, timeout=timeout)
    command = docker_command(image, task, build, render_device, name, jit_cache) + [mode]
    primary_error = None
    try:
        with log_path.open("xb") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        return json.loads(read_regular(build / (mode + "_report.json")))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        # Also remove a container left running after a host timeout/interruption.
        errors = []
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(exc)
        try:
            preserve_phase_diagnostics(build, mode, log_path)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(exc)
        for exc in errors:
            if primary_error is not None:
                primary_error.add_note(f"Additional cleanup/diagnostic failure: {type(exc).__name__}: {exc}")
            else:
                raise exc


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _proof(proof, identity, leg):
    _require(isinstance(proof, dict) and proof.get("leg") == leg
             and proof.get("production_namespace_rebound") is False
             and SHA256.fullmatch(str(proof.get("extension_sha256", ""))), "invalid native build evidence")
    if leg == "candidate_native":
        _require(proof.get("fresh_compilation") is True
                 and proof.get("source_tree_sha256") == identity["source_tree_sha256"],
                 "native candidate was not freshly compiled from staged source")


def validate_report(report, mode, identity, cases):
    _require(isinstance(report, dict) and report.get("status") == "ok", "unsuccessful report")
    for key, expected in identity.items():
        _require(report.get(key) == expected, "foreign or stale report: " + key)
    _require(re.fullmatch(r"[0-9a-f]{48}", str(report.get("run_id", ""))), "missing fresh run ID")
    if mode != "performance":
        workers = report.get("workers", {})
        _require(set(workers) == set(LEGS), "incomplete native workers")
        for leg in LEGS:
            worker = workers[leg]
            _require(worker.get("status") == "ok" and worker.get("mode") == mode
                     and worker.get("leg") == leg and worker.get("run_id") == report["run_id"]
                     and all(worker.get(k) == v for k, v in identity.items()), "invalid worker identity")
            _proof(worker.get("native_build"), identity, leg)
            rows = worker.get("results")
            expected = [] if mode == "compile" else [c["case_id"] for c in cases]
            _require(isinstance(rows, list) and [r.get("case_id") for r in rows] == expected,
                     "incomplete or duplicate correctness cases")
            for case, row in zip(cases, rows):
                _require(row.get("shape") == case["shape"]
                         and row.get("trace_call_count") == case["trace_call_count"]
                         and row.get("correct") is True and row.get("input_immutable") is True
                         and row.get("negative_controls") is True and row.get("seeds") == [0, 1],
                         "correctness or negative controls failed")
        return None
    _require(report.get("benchmark_method") == "cuda_graph"
             and report.get("warmup_iterations") == 10 and report.get("benchmark_iterations") == 100
             and report.get("isolated_native_processes") is True
             and report.get("fresh_input_and_poisoned_output_each_replay") is True,
             "unsupported timing or replay validation")
    for leg in LEGS:
        _proof(report.get("native_builds", {}).get(leg), identity, leg)
    rows = report.get("paired_cases", [])
    _require([r.get("case_id") for r in rows] == [c["case_id"] for c in cases],
             "incomplete or duplicate performance cases")
    measurements = []
    for case, row in zip(cases, rows):
        _require(row.get("shape") == case["shape"]
                 and row.get("trace_call_count") == case["trace_call_count"]
                 and row.get("graph_correctness") is True, "changed or unchecked workload")
        means = {}
        for leg in LEGS:
            timing = row.get("timings", {}).get(leg, {})
            samples = timing.get("samples_ms")
            _require(isinstance(samples, list) and len(samples) == 100
                     and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples),
                     "invalid or incomplete device samples")
            mean = math.fsum(samples) / 100
            _require(timing == {"samples_ms": samples, "mean_ms": mean,
                                "min_ms": min(samples), "max_ms": max(samples)},
                     "forged timing summary")
            means[leg] = mean
        measurements.append(means)
    expected = [{"test_case_id": c["case_id"], "execution_time_ms": m["candidate_native"],
                 "shape": c["shape"], "params": {"trace_call_count": c["trace_call_count"]},
                 "metadata": {"benchmark_method": "cuda_graph"}}
                for c, m in zip(cases, measurements)]
    _require(report.get("test_cases") == expected, "score cases differ from fresh device samples")
    return measurements


def trusted_retest(*, repo, commit, task_path, candidate, agent_workspace, output, render_device,
                   timeout=7200, scratch_dir=None):
    repo, agent_workspace = Path(repo).resolve(), Path(agent_workspace).resolve()
    candidate = Path(os.path.abspath(candidate))
    output = Path(os.path.abspath(output))
    _require(candidate.is_relative_to(agent_workspace), "candidate must be in declared agent workspace")
    _require(not repo.is_relative_to(agent_workspace) and not output.resolve().is_relative_to(agent_workspace),
             "trusted repository and output must be outside the agent workspace")
    scratch_parent = Path(scratch_dir or tempfile.gettempdir()).resolve()
    _require(not scratch_parent.is_relative_to(agent_workspace), "scratch directory must be outside the agent workspace")
    scratch_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require(timeout > 0, "container timeout must be positive")
    candidate_bytes = read_regular(candidate)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="aka-native-retest-", dir=scratch_parent) as temporary:
        staging = Path(temporary)
        info = staging.stat()
        _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
                 "fresh scratch directory must be private and owned by the invoking UID")
        reference = staging / "reference"
        extract_task(repo, commit, task_path, reference)
        image, manifest = validate_contract(reference)
        validate_candidate(reference, candidate_bytes)
        baseline_hash = sha256((reference / SOURCE).read_bytes())
        edited = staging / "candidate"
        copy_payload(reference, edited)
        (edited / SOURCE).write_bytes(candidate_bytes)
        for task in (reference, edited):
            (task / "build").mkdir()  # Empty mount point, never an agent cache.
        inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
        digest = image.rsplit("@", 1)[1]
        _require(any(d.endswith("@" + digest) for d in inspected.get("RepoDigests", [])),
                 "local image does not attest requested registry digest")
        seed = staging / "image-jit"
        cache_manifest = seed_image_cache(image, seed, output / "cache_init.log", timeout=timeout)
        cache_bytes = json.dumps(cache_manifest, sort_keys=True, indent=2).encode() + b"\n"
        (output / "cache_manifest.json").write_bytes(cache_bytes)
        cases = manifest["cases"]
        measurements, reports, run_ids = {}, {}, set()
        for leg, task in (("reference", reference), ("candidate", edited)):
            identity = identities(task)
            reports[leg] = {}
            for mode in MODES:
                payload = run_mode(image, task, staging / (leg + "_" + mode), render_device,
                                   mode, output / (leg + "_" + mode + ".log"), timeout,
                                   jit_source=seed / "jit")
                measured = validate_report(payload, mode, identity, cases)
                _require(payload["run_id"] not in run_ids, "replayed report run ID")
                run_ids.add(payload["run_id"])
                _require(identities(task) == identity, "staged inputs changed during container execution")
                serialized = json.dumps(payload, indent=2, allow_nan=False).encode() + b"\n"
                report_name = leg + "_" + mode + ".json"
                (output / report_name).write_bytes(serialized)
                reports[leg][mode] = {"file": report_name, "sha256": sha256(serialized), **identity}
                if measured is not None:
                    measurements[leg] = measured
        rows = []
        for case, original, optimized in zip(cases, measurements["reference"], measurements["candidate"]):
            ratio = original["candidate_native"] / optimized["candidate_native"]
            _require(math.isfinite(ratio) and ratio > 0, "invalid speedup ratio")
            rows.append({"test_case_id": case["case_id"], "shape": case["shape"],
                         "reference_ms": original["candidate_native"],
                         "candidate_ms": optimized["candidate_native"], "speedup": ratio,
                         "production_diagnostic_ms": {"reference_run": original["production_native"],
                                                       "candidate_run": optimized["production_native"]}})
        result = {"schema_version": 1, "status": "measured", "task_path": task_path, "trusted_commit": commit,
                  "image": image, "local_image_id": inspected["Id"], "render_device": str(render_device),
                  "image_cache_manifest": {"file": "cache_manifest.json", "sha256": sha256(cache_bytes)},
                  "reference_source_sha256": baseline_hash, "candidate_source_sha256": sha256(candidate_bytes),
                  "full_case_coverage": True, "case_count": len(cases), "cases": rows,
                  "arithmetic_mean_speedup": math.fsum(row["speedup"] for row in rows) / len(rows),
                  "comparison": "same protected candidate_native entrypoint with reference then candidate source",
                  "reports": reports, "framework_task_validator_status": "not_asserted"}
        # Assess only after every phase/case has finished and its raw report is
        # retained. A quality failure never selects a subset or starts a retry.
        result = gate_native_measurement(result,
            json.loads(read_regular(output / reports['reference']['performance']['file'])),
            json.loads(read_regular(output / reports['candidate']['performance']['file'])))
        (output / "trusted_measurement.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "commit", "task", "candidate", "agent-workspace", "output", "render-device"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=int, default=7200, help="timeout per fresh container in seconds")
    parser.add_argument("--scratch-dir", help="local host scratch root; defaults to the host temporary directory")
    args = parser.parse_args()
    if Path("/.dockerenv").exists():
        parser.error("run from a trusted host, outside the agent container")
    result = trusted_retest(repo=args.repo, commit=args.commit, task_path=args.task, candidate=args.candidate,
                            agent_workspace=args.agent_workspace, output=args.output,
                            render_device=args.render_device, timeout=args.timeout, scratch_dir=args.scratch_dir)
    print(json.dumps({"measurement": str(Path(args.output) / "trusted_measurement.json"),
                      "arithmetic_mean_speedup": result["arithmetic_mean_speedup"]}))
    if result['status'] == 'rejected_timing_quality':
        raise SystemExit(1)


if __name__ == "__main__":
    main()
