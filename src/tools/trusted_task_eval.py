"""Fresh-container evaluation for tasks implementing the portable case contract.

This leaves the qualified native-quant evaluator unchanged. The invoking host,
Git database, image and Docker daemon remain trusted infrastructure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath

import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.native_baseline import scoring_policy, load_native_measurements, as_test_cases, metric_summary

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
    from .trusted_native_eval import docker_command, extract_task, read_regular, sha256
    from .trusted_fixtures import DEFAULT_MAX_BYTES, DEFAULT_MAX_FILES, load_fixture_manifest, materialize_fixtures
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gpu_binding import command_with_binding, select_gpu, validate_preflight
    from seed_aiter_jit_cache import copy_cache, seed_image_cache, tree_manifest
    from trusted_native_eval import docker_command, extract_task, read_regular, sha256
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


def relative_file(value):
    require(isinstance(value, str) and bool(value), "task file paths must be nonempty strings")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in ("..", ".git", "build", "__pycache__") for part in path.parts),
            "task file path escapes the immutable package")
    return path.as_posix()


def package_contract(task):
    config = yaml.safe_load(read_regular(task / "config.yaml"))
    require(isinstance(config, dict), "invalid task configuration")
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
    for phase in PHASES:
        require(config.get(phase + "_command") == [f"python3 scripts/task_runner.py {phase}"],
                "trusted tasks must use the protected scripts/task_runner.py entrypoint")
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
            "scoring_policy": score_policy, "config": config}


def prepare_reference(*, repo, commit, task_path, reference, staging, candidate_workspace,
                      output, scratch_explicit, fixture_local_mirror=None, fixture_oci_prefix=None,
                      fixture_max_files=DEFAULT_MAX_FILES, fixture_max_bytes=DEFAULT_MAX_BYTES, timeout=1800):
    extract_task(repo, commit, task_path, reference)
    contract = package_contract(reference)
    fixture_receipt = None
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


def preserve_diagnostics(build, output):
    output.mkdir()
    records = {}
    try:
        for source in sorted(build.iterdir()):
            if source.suffix not in (".json", ".jsonl", ".log"):
                continue
            require("\n" not in source.name and "\r" not in source.name,
                    "diagnostic filenames cannot contain line breaks")
            digest = sha256(read_regular(source))
            target = output / source.name
            environment = os.environ.copy()
            environment["GOMAXPROCS"] = "1"
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="aka-diagnostic-") as listing:
                listing.write(source.name + "\n")
                listing.flush()
                subprocess.run(["rclone", "copy", str(build), str(output), "--files-from-raw", listing.name,
                                "--transfers", "64000", "--progress", "--buffer-size", "0",
                                "--multi-thread-streams", "0", "--config", os.devnull, "--no-traverse"],
                               check=True, timeout=300, env=environment)
            require(sha256(read_regular(target)) == digest, "diagnostic copy differs from phase output")
            records[source.name] = digest
    finally:
        (output / "hashes.json").write_text(json.dumps(records, indent=2) + "\n")


def run_phase(image, task, staging, output, leg, request, render_device, timeout, cache_source=None):
    phase = request["phase"]
    label = leg + "_" + phase
    build = staging / (label + "_build")
    build.mkdir()
    cache = None
    if cache_source is not None:
        cache = staging / (label + "_jit")
        copy_cache(cache_source, cache, timeout=timeout)
    request_path = staging / (label + "_request.json")
    request_path.write_text(canonical(request))
    gpu_path = staging / (label + "_gpu.json")
    gpu_path.write_text(canonical(request["gpu"]))
    name = "aka-task-retest-" + uuid.uuid4().hex
    command = docker_command(image, task, build, render_device, name, cache)
    command = command_with_binding(command, image, request["gpu"], GPU_BINDING_HELPER, gpu_path)
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
    command += [phase, "--request", "/evaluation-request.json"]
    primary = None
    try:
        with (output / (label + ".log")).open("xb") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        validate_preflight(strict_json(read_regular(build / "gpu_preflight.json")), request["gpu"])
        return strict_json(read_regular(build / (phase + "_report.json")))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = []
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(exc)
        try:
            preserve_diagnostics(build, output / (label + ".diagnostics"))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(exc)
        for exc in failures:
            if primary is None:
                raise exc
            primary.add_note("Additional diagnostic/cleanup failure: " + str(exc))


def trusted_retest(*, repo, commit, task_path, candidate_workspace, output, render_device, scratch_dir=None, timeout=7200,
                   fixture_local_mirror=None, fixture_oci_prefix=None,
                   fixture_max_files=DEFAULT_MAX_FILES, fixture_max_bytes=DEFAULT_MAX_BYTES):
    repo, candidate_workspace = Path(repo).resolve(), Path(candidate_workspace).resolve()
    output = Path(output).absolute()
    scratch = Path(scratch_dir or tempfile.gettempdir()).resolve()
    for path in (repo, output.resolve(), scratch):
        require(not path.is_relative_to(candidate_workspace), "trusted repository/output/scratch must be outside the agent workspace")
    require(timeout > 0, "timeout must be positive")
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
        candidate = staging / "candidate"
        copy_cache(reference, candidate)
        # Only these declared regular files are taken from the agent workspace.
        for source in contract["sources"]:
            (candidate / source).write_bytes(read_regular(candidate_workspace / source))
        guard_sources(reference, candidate, contract["guard"])
        gpu = select_gpu(render_device)
        manifest = contract["manifest"]
        image = manifest["runtime_image"]
        image_record = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
        require(any(item.endswith("@" + image.rsplit("@", 1)[1]) for item in image_record.get("RepoDigests", [])),
                "local image does not attest the requested digest")
        cache_source = None
        if contract["needs_cache"]:
            cache_root = staging / "image-jit"
            proof = seed_image_cache(image, cache_root, output / "cache_init.log", timeout=timeout)
            (output / "cache_manifest.json").write_text(canonical(proof))
            cache_source = cache_root / "jit"
        results, reports, sources, scored_native = {}, {}, {}, {}
        challenge_seed = secrets.randbelow(2**30)
        for leg, task in (("reference", reference), ("candidate", candidate)):
            package = fingerprint(tree_manifest(task))
            sources[leg] = {source: sha256(read_regular(task / source)) for source in contract["sources"]}
            reports[leg] = {}
            (task / "build").mkdir()
            for phase in PHASES:
                request = {"schema_version": 1, "request_id": secrets.token_hex(24), "phase": phase,
                           "manifest_sha256": fingerprint(manifest), "package_sha256": package,
                           "source_sha256": sources[leg], "challenge_seed": challenge_seed, "gpu": gpu}
                report = run_phase(image, task, staging, output, leg, request, render_device, timeout, cache_source)
                measured = validate_report(report, manifest, request)
                # Ignore only the empty bind-mount point added after fingerprinting.
                current = tree_manifest(task)
                current.pop("build", None)
                require(fingerprint(current) == package, "protected staged task files changed")
                encoded = canonical(report).encode() + b"\n"
                filename = leg + "_" + phase + ".json"
                (output / filename).write_bytes(encoded)
                reports[leg][phase] = {"file": filename, "sha256": sha256(encoded)}
                if phase == "performance":
                    results[leg] = measured
                    if contract['scoring_policy'] is not None:
                        evidence = load_native_measurements(task, contract['config'], report=report, request=request)
                        scored_native[leg] = as_test_cases(evidence, is_baseline=leg == 'reference')
        cases = []
        for baseline, optimized in zip(results["reference"], results["candidate"]):
            require(baseline["case_sha256"] == optimized["case_sha256"], "reference/candidate ABI differs")
            ratio = baseline["execution_time_ms"] / optimized["execution_time_ms"]
            require(math.isfinite(ratio) and ratio > 0, "invalid speedup ratio")
            cases.append({"test_case_id": baseline["test_case_id"], "case_sha256": baseline["case_sha256"],
                          "reference_ms": baseline["execution_time_ms"], "candidate_ms": optimized["execution_time_ms"],
                          "speedup": ratio})
        result = {"schema_version": 1, "status": "measured", "trusted_commit": commit, "task_path": task_path,
                  "image": image, "local_image_id": image_record["Id"], "source_sha256": sources,
                  "gpu": gpu,
                  "manifest_sha256": fingerprint(manifest), "full_case_coverage": True, "cases": cases,
                  "arithmetic_mean_speedup": math.fsum(row["speedup"] for row in cases) / len(cases),
                  "reports": reports, "framework_task_validator_status": "not_asserted"}
        if contract['scoring_policy'] is not None:
            summary = metric_summary(scored_native['reference'], scored_native['candidate'])
            result.update(summary)
            result['arithmetic_mean_speedup'] = summary['native_speedup_ratio']
            result['cases'] = [{**row, 'reference_ms': row['native_ms']}
                               for row in summary['native_baseline_cases']]
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
    parser.add_argument("--stage-only", action="store_true", help="prepare a verified original task for a validator without GPU execution")
    parser.add_argument("--scratch-dir")
    parser.add_argument("--fixture-local-mirror", help="trusted local mirror laid out by manifest object_key")
    parser.add_argument("--fixture-oci-prefix", help="host-approved exact OCI prefix for fixture download")
    parser.add_argument("--fixture-max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--fixture-max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if Path("/.dockerenv").exists():
        parser.error("run from a trusted host outside the agent container")
    fixtures = dict(fixture_local_mirror=args.fixture_local_mirror, fixture_oci_prefix=args.fixture_oci_prefix,
                    fixture_max_files=args.fixture_max_files, fixture_max_bytes=args.fixture_max_bytes)
    if args.stage_only:
        if not args.scratch_dir or args.candidate_workspace or args.render_device:
            parser.error("--stage-only requires --scratch-dir and accepts no candidate workspace or GPU device")
        result = stage_trusted_task(repo=args.repo, commit=args.commit, task_path=args.task, output=args.output,
                                   scratch_dir=args.scratch_dir, timeout=args.timeout, **fixtures)
        print(json.dumps(result))
        return
    if not args.candidate_workspace or not args.render_device:
        parser.error("retesting requires --candidate-workspace and --render-device")
    result = trusted_retest(repo=args.repo, commit=args.commit, task_path=args.task,
                            candidate_workspace=args.candidate_workspace, output=args.output,
                            render_device=args.render_device, scratch_dir=args.scratch_dir, timeout=args.timeout, **fixtures)
    print(json.dumps({"measurement": str(Path(args.output) / "trusted_measurement.json"),
                      "arithmetic_mean_speedup": result["arithmetic_mean_speedup"]}))


if __name__ == "__main__":
    main()
