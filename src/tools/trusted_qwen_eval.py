"""Adapters for the four committed Qwen graph contracts, without GPU orchestration."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.quality_measurements import _QWEN_GRAPH_PROFILES, load_quality_measurements
from src.task_contract import canonical, fingerprint, require, strict_json
from src.tools import prepare_head_kernel_artifacts as artifacts
from src.tools.trusted_native_eval import read_regular, sha256


PROFILES = {
    "qwen3.8-2.4t__dense_bf16_gemm_cluster": "qwen_dense_bf16_graph_v1",
    "qwen3.8-2.4t__fused_moe_2stage_mxfp4": "qwen_mxfp4_moe_graph_v1",
    "qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode": "qwen_recurrent_graph_v1",
    "qwen3.8-2.4t__paged_attention_decode": "qwen_paged_attention_graph_v1",
}
PHASE_BRIDGE = Path(__file__).with_name("trusted_qwen_phase.py")


def validator(task, contract):
    profile = contract["qwen_profile"]
    relative = profile["validator"]
    code = read_regular(task / relative)
    require(sha256(code) == contract["config"]["benchmark_quality"]["validator_sha256"],
            "Qwen validator differs from the committed descriptor")
    namespace = {"__file__": str(task / relative), "__name__": "_trusted_qwen_validator"}
    exec(compile(code, str(task / relative), "exec"), namespace)
    support = profile.get("routing_contract")
    if support:
        code = read_regular(task / support)
        require(sha256(code) == namespace["ROUTING_CONTRACT_SHA256"], "Qwen routing helper changed")
        helper = {"__file__": str(task / support), "__name__": "_trusted_qwen_routing"}
        exec(compile(code, str(task / support), "exec"), helper)
        namespace["_routing_contract"] = lambda: SimpleNamespace(**helper)
    else:
        helper = namespace
    return namespace, helper


def package_contract(task, config):
    policy = config.get("benchmark_quality")
    require(isinstance(policy, dict) and policy.get("adapter") in PROFILES.values(),
            "unsupported trusted Qwen report contract")
    profile = _QWEN_GRAPH_PROFILES[policy["adapter"]]
    require(set(policy) == {"schema_version", "adapter", "validator_sha256"}
            and type(policy["schema_version"]) is int and policy["schema_version"] == 1,
            "invalid trusted Qwen quality descriptor")
    require(config.get("source_file_path") == profile["sources"]
            and config.get("target_kernel_functions") == profile["targets"]
            and config.get("task_type") == profile["task_type"], "Qwen target contract differs")
    guard = config.get("harness_protection", {})
    require(guard.get("freeze_task_inputs") is True and guard.get("reject_new_source_symlinks") is True,
            "trusted Qwen requires the complete source/harness guard")
    for phase in ("compile", "correctness", "performance"):
        require(config.get(phase + "_command") == ["python3 scripts/task_runner.py " + phase],
                "unsupported protected Qwen entrypoint")
    meta = strict_json(read_regular(task / "ut/meta.json"))
    require(meta.get("op_kind") == profile["op_kind"] and meta.get("num_cases") == profile["cases"],
            "Qwen case metadata differs from its registered profile")
    image = config.get("headkernel", {}).get("docker")
    require(isinstance(image, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image)
            and image == meta["source_provenance"]["runtime_image"], "Qwen image must be digest pinned")
    aliases = {}
    binding = meta.get("candidate_bind")
    if binding:
        relative = Path("ut") / binding["file"]
        target = Path(meta["source_provenance"].get("protected_native_source",
                                                 meta["source_provenance"]["candidate_source"]))
        require(relative.parent == Path("ut/kernel_src") and target.parent == Path("source"),
                "unsupported committed Qwen source alias")
        aliases[str(relative)] = os.path.relpath(target, relative.parent)
    observed_aliases = set()
    for path in task.rglob("*"):
        if path.is_symlink():
            relative = str(path.relative_to(task))
            require(relative in aliases and str(path.readlink()) == aliases[relative]
                    and path.resolve().is_relative_to(task.resolve()) and path.is_file(),
                    "undeclared or escaping Qwen source alias")
            observed_aliases.add(relative)
    require(observed_aliases == set(aliases), "declared Qwen source alias is missing or changed")
    result = {"qwen_profile": profile, "config": config, "sources": profile["sources"],
              "references": {}, "guard": None, "needs_cache": True, "fixtures": None,
              "scoring_policy": None, "distribution_correctness_mode": None, "distributions": None,
              "qwen_metadata": meta, "qwen_aliases": aliases}
    _, helper = validator(task, result)
    if profile.get("native_routing"):
        helper["source_context"](task, meta)
        case_ids = [row["sig"] + "|" + row["regime"] for row in helper["specs"](meta)]
    else:
        case_ids = [row["test_case_id"] for row in helper["contract"](task)["cases"]]
    require(len(case_ids) == len(set(case_ids)) == profile["cases"], "Qwen case coverage changed")
    result["manifest"] = {"schema": "trusted-qwen-graph-v1", "runtime_image": image,
                          "metadata_sha256": sha256(read_regular(task / "ut/meta.json")),
                          "cases": [{"case_id": name} for name in case_ids]}
    return result


def prepare_fixtures(repo, commit, task_path, task, output, contract, *, local_mirror,
                     oci_prefix, max_files, max_bytes, timeout):
    name = Path(task_path).name
    require(task_path == "tasks/headkernel/" + name
            and PROFILES.get(name) == contract["config"]["benchmark_quality"]["adapter"],
            "Qwen task path differs from its supported contract")
    encoded = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":tools/headkernel-artifacts.json"])
    registry = output / "original_fixture_registry.json"
    registry.write_bytes(encoded)
    manifest = artifacts.load_manifest(registry)
    selected = artifacts.select_artifacts(manifest, [name])
    require(len(selected) <= max_files and sum(a.size_bytes for a in selected) <= max_bytes,
            "Qwen original fixtures exceed approved host bounds")
    selected = [replace(a, path=str(Path(a.path).relative_to(name)), mirror_path=a.mirror_path or a.path)
                for a in selected]
    if selected:
        if local_mirror is not None:
            actions = artifacts.prepare_artifacts(selected, task, mirror=Path(local_mirror))
        else:
            require(oci_prefix is not None and oci_prefix == manifest.oci_remote_root,
                    "provide an original fixture mirror or approve its exact committed OCI prefix")
            download = task.parent / "qwen-fixture-download"
            download.mkdir(mode=0o700)
            listing = download / "selected.txt"
            listing.write_text("".join(a.mirror_path + "\n" for a in selected))
            subprocess.run(["rclone", "copy", oci_prefix, str(download), "--files-from-raw", str(listing),
                            "--transfers", "64000", "--progress", "--buffer-size", "0"],
                           env={**os.environ, "GOMAXPROCS": "1"}, check=True, timeout=timeout)
            actions = artifacts.prepare_artifacts(selected, task, mirror=download)
        artifacts.prepare_artifacts(selected, task, verify_only=True)
    else:
        actions = []
    materialize_perf_helpers_in_workspace(task)
    proof = {"schema": "trusted-qwen-original-fixtures-v1", "trusted_commit": commit,
             "task_path": task_path, "registry_sha256": sha256(encoded), "actions": actions,
             "artifacts": [{"path": a.path, "bytes": a.size_bytes, "sha256": a.sha256} for a in selected],
             "generated_replacement_inputs": False, "workload_scope": "original_capture_contract"}
    data = canonical(proof).encode() + b"\n"
    (output / "fixtures_receipt.json").write_bytes(data)
    return {"file": "fixtures_receipt.json", "sha256": sha256(data)}


def admit_sources(reference, candidate, source_workspace, contract, guard_sources):
    snapshot = snapshot_workspace_harness(candidate, task_root=reference)
    for name in contract["sources"]:
        (candidate / name).write_bytes(read_regular(source_workspace / name))
    verify_workspace_harness(snapshot)
    if contract["qwen_profile"].get("native_routing"):
        _, helper = validator(reference, contract)
        helper["source_context"](candidate, contract["qwen_metadata"])
    else:
        guard_sources(reference, candidate, "ut/source_guard.py")


def runtime_mounts(task, build, contract, *, runtime_root=None):
    """Protect complete UT entries against parent-directory replacement.

    Legacy atomic result writes need a writable UT root. Every other top-level
    entry is an immutable mount point, and its backing directory is never
    exposed through the writable build mount while the container runs.
    """
    outputs = contract["config"]["harness_protection"].get("runtime_outputs", [])
    outputs = [Path(name) for name in outputs if Path(name).parts[0] == "ut"]
    if not outputs:
        return [], set()
    require(all(len(name.parts) == 2 and name.parts[0] == "ut" and ".." not in name.parts for name in outputs),
            "invalid Qwen runtime output path")
    runtime = Path(runtime_root) if runtime_root is not None else build.with_name(build.name + "-ut-runtime")
    require(not runtime.resolve().is_relative_to(task.resolve())
            and not runtime.resolve().is_relative_to(build.resolve()),
            "UT runtime backing must not be exposed by the task or build mounts")
    runtime.mkdir()
    mounts = ["--mount", f"type=bind,src={runtime},dst=/task/ut"]
    omitted = set()
    for source in sorted((task / "ut").iterdir()):
        relative = source.relative_to(task)
        if relative in outputs:
            continue
        require(not source.is_symlink() and (source.is_file() or source.is_dir())
                and source.resolve().is_relative_to(task.resolve()), "invalid protected Qwen UT entry")
        target = runtime / source.name
        if source.is_dir():
            target.mkdir()
        else:
            target.touch()
            omitted.add(".runtime-ut/" + source.name)
        mounts += ["--mount", f"type=bind,src={source.resolve()},dst=/task/{relative},readonly"]
    return mounts, omitted


def validate_phase(task, contract, phase, report, request):
    build = task / "build"
    receipt = strict_json(read_regular(build / "host_phase_receipt.json"))
    require(receipt.get("schema") == "trusted-qwen-phase-v1" and receipt.get("host_request") == request
            and type(receipt.get("returncode")) is int and receipt["returncode"] == 0
            and receipt.get("producer_report_sha256") == sha256(read_regular(build / (phase + "_report.json"))),
            "Qwen phase is stale, incomplete or differs from its host request")
    require(report.get("status") == "ok", "Qwen protected phase did not pass")
    controller, helper = validator(task, contract)
    profile, meta = contract["qwen_profile"], contract["qwen_metadata"]
    if phase == "performance":
        measured = load_quality_measurements(task, contract["config"])
        require({c.test_case_id for c in measured} == {c["case_id"] for c in contract["manifest"]["cases"]},
                "trusted Qwen performance coverage differs")
        return [{"test_case_id": case.test_case_id,
                 "case_sha256": fingerprint(case.metadata["params"]),
                 "execution_time_ms": case.execution_time_ms} for case in measured], measured
    if profile.get("native_routing"):
        producer_request = helper["validate_request"](report.get("request", {}), task, phase)
        if phase == "correctness":
            if profile["routing_kind"] == "dense":
                raw = strict_json(read_regular(build / "native-routing-runs" / producer_request["run_id"] / "raw.json"))
                rows = controller["_validate_dense_payload"](SimpleNamespace(**helper), raw, meta, producer_request)
                require(report.get("case_results") == rows, "dense correctness publication differs from raw evidence")
            else:
                raw = strict_json(read_regular(build / "reports" / producer_request["run_id"] / "raw.json"))
                require(raw == report, "MoE correctness publication differs from raw evidence")
                rows = helper["validate_payload"](raw, meta, producer_request)
            require(len(rows) == profile["cases"], "Qwen correctness coverage is incomplete")
    elif phase == "correctness":
        require(report.get("exit_code") == 0, "original Qwen correctness checks did not pass")
        for case_id in meta["ledger_ids"]:
            ledger = strict_json(read_regular(build / "ledger-root/reports/ledger" / (case_id + ".json")))
            require(ledger.get("ledger_id") == case_id and ledger.get("status") == "PASS",
                    "original Qwen correctness ledger is missing or failed")
        raw = strict_json(read_regular(build / "_cpu_truth_correctness.json"))
        require(report.get("cpu_truth_challenges") == raw, "Qwen correctness raw/publication differs")
        helper["correctness_cases"](raw, task)
    if phase == "compile":
        require(report.get("parsed_sources") == contract["sources"]
                and set(report.get("symbols_found", {})) == set(contract["config"]["target_kernel_functions"]),
                "Qwen compilation source/symbol coverage differs")
        if profile["task_type"] == "hip2hip":
            native = strict_json(read_regular(build / "native_compile_report.json"))
            require(report.get("native_compile") == native and native.get("source_sha256") == request["source_sha256"],
                    "native Qwen compilation is not bound to the submitted device source")
    return [], []
