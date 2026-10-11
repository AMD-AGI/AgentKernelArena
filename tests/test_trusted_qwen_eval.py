"""Trusted Qwen admission and six-phase evidence; all device calls are mocked."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from types import SimpleNamespace

import pytest

from src.task_contract import canonical
from src.tools import trusted_qwen_eval as adapter, trusted_qwen_phase as bridge
from src.tools import trusted_task_eval as trusted

ROOT = Path(__file__).resolve().parents[1]
DENSE = "qwen3.8-2.4t__dense_bf16_gemm_cluster"


def clone(task, target):
    shutil.copytree(task, target, symlinks=True,
                    ignore=shutil.ignore_patterns("build", "__pycache__", "reference_io.pt"))
    return target


@pytest.mark.parametrize("name", adapter.PROFILES)
def test_each_supported_package_retains_original_cases_and_closed_source_admission(tmp_path, name):
    original = clone(ROOT / "tasks/headkernel" / name, tmp_path / "reference")
    candidate = clone(original, tmp_path / "candidate")
    contract = trusted.package_contract(original)
    adapter.admit_sources(original, candidate, original, contract, trusted.guard_sources)
    assert len(contract["manifest"]["cases"]) == contract["qwen_profile"]["cases"]
    source = candidate / contract["sources"][0]
    if source.suffix == ".py":
        source.write_text("import os\n" + source.read_text())
    else:
        source.write_text("#include <fstream>\n" + source.read_text())
    poisoned = clone(original, tmp_path / "poisoned")
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        adapter.admit_sources(original, poisoned, candidate, contract, trusted.guard_sources)


def test_readonly_UT_inputs_are_preserved_while_declared_outputs_are_fresh(tmp_path):
    task = ROOT / "tasks/headkernel/qwen3.8-2.4t__paged_attention_decode"
    contract = trusted.package_contract(task)
    build = tmp_path / "build"
    build.mkdir()
    runtime = tmp_path / "ut-runtime"
    mounts, omitted = adapter.runtime_mounts(task, build, contract, runtime_root=runtime)
    assert f"type=bind,src={runtime},dst=/task/ut" in mounts
    for relative in ("ut/meta.json", "ut/harness_lib.py", "ut/benchmark_contract.json", "ut/source_guard.py"):
        assert f"type=bind,src={task / relative},dst=/task/{relative},readonly" in mounts
        assert ".runtime-ut/" + relative.removeprefix("ut/") in omitted
    assert not any("dst=/task/ut/result.json,readonly" in value for value in mounts)
    assert not (runtime / "result.json").exists()
    assert f"type=bind,src={task / 'ut/kernel_src'},dst=/task/ut/kernel_src,readonly" in mounts


@pytest.mark.parametrize("target", ["../meta.json", "/etc/passwd"])
def test_undeclared_or_escaping_committed_alias_cannot_enter_package(tmp_path, target):
    task = clone(ROOT / "tasks/headkernel" / DENSE, tmp_path / "task")
    (task / "ut/extra_alias").symlink_to(target)
    with pytest.raises(ValueError, match="undeclared or escaping"):
        trusted.package_contract(task)


def test_phase_bridge_runs_existing_compile_and_binds_the_exact_host_request(tmp_path):
    task = clone(ROOT / "tasks/headkernel" / DENSE, tmp_path / "task")
    (task / "build").mkdir()
    contract = trusted.package_contract(task)
    request = {"phase": "compile", "request_id": "a" * 48,
               "source_sha256": {name: trusted.sha256(trusted.read_regular(task / name))
                                 for name in contract["sources"]}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    bridge.run(task, path, 10)
    report = json.loads((task / "build/compile_report.json").read_text())
    adapter.validate_phase(task, contract, "compile", report, request)
    with pytest.raises(ValueError, match="host request"):
        adapter.validate_phase(task, contract, "compile", report, dict(request, request_id="b" * 48))


@pytest.fixture
def committed_dense(tmp_path):
    repo = tmp_path / "repo"
    task_path = "tasks/headkernel/" + DENSE
    clone(ROOT / task_path, repo / task_path)
    (repo / "tools").mkdir()
    (repo / "tools/headkernel-artifacts.json").write_bytes((ROOT / "tools/headkernel-artifacts.json").read_bytes())
    for command in (["init", "-q"], ["add", "."],
                    ["-c", "user.name=CPU Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(repo), *command], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    candidate = tmp_path / "candidate"
    (candidate / "source").mkdir(parents=True)
    (candidate / "source/routing.py").write_bytes((repo / task_path / "source/routing.py").read_bytes())
    # Reports and executable harness files from an optimization workspace must
    # never enter either of the fresh phase tasks.
    (candidate / "config.yaml").write_text("untrusted: true")
    (candidate / "scripts").mkdir()
    (candidate / "scripts/task_runner.py").write_text("raise AssertionError('agent harness executed')")
    (candidate / "build").mkdir()
    (candidate / "build/performance_report.json").write_text('{"speedup":999999}')
    return repo, commit, task_path, candidate


def test_internal_alias_extraction_preserves_commit_return_value(committed_dense, tmp_path):
    repo, commit, task_path, _ = committed_dense
    extracted = tmp_path / "extracted"
    assert trusted.extract_task(repo, commit, task_path, extracted, allow_internal_symlinks=True) == commit
    assert (extracted / "ut/kernel_src/tuned_gemm_candidate.py").is_symlink()


def producer_phase(task, contract, phase, host_request, *, outlier=False):
    from test_qwen_dense_native_routing import payload_fixture
    controller, helper = adapter.validator(task, contract)
    producer_request = helper["make_request"](task, phase, host_request["request_id"])
    build = task / "build"
    build.mkdir()
    if phase == "compile":
        report = {"status": "ok", "request": producer_request, "parsed_sources": contract["sources"],
                  "symbols_found": {"select_routes": "source/routing.py"}}
    else:
        _, raw = payload_fixture(task)
        raw.update(request=producer_request, request_sha256=helper["fingerprint"](producer_request))
        if phase == "performance" and outlier:
            row = raw["cases"][0]
            row["samples_ms"][27] = 10000.0
            samples = row["samples_ms"]
            row.update(mean_ms=math.fsum(samples) / 100, median_ms=statistics.median(samples),
                       min_ms=min(samples), max_ms=max(samples))
        if phase == "correctness":
            for row in raw["cases"]:
                row["validation"].update(checked_invocations=14, graph_checked_invocations=9,
                    prior_outputs_unchanged=True, independent_output_storage=True,
                    negative_controls={"no_op": "rejected", "wrong_output": "rejected"})
        archive = build / "native-routing-runs" / producer_request["run_id"] / "raw.json"
        archive.parent.mkdir(parents=True)
        archive.write_text(json.dumps(raw))
        if phase == "performance":
            (build / "performance_request.json").write_text(json.dumps(producer_request))
            (build / "_bench_raw.json").write_text(json.dumps(raw))
            report = {"status": "ok", "request": producer_request, "protocol": helper["PROTOCOL"],
                      "timer": "cuda_graph", "warmup_iterations": 10, "benchmark_iterations": 100,
                      "complete_raw_report": str(archive.relative_to(task)),
                      "test_cases": controller["_routing_cases"](raw)}
        else:
            report = {"status": "ok", "request": producer_request, "case_results": raw["cases"]}
    (build / (phase + "_report.json")).write_text(json.dumps(report))
    receipt = {"schema": "trusted-qwen-phase-v1", "host_request": host_request, "returncode": 0,
               "producer_report_sha256": trusted.sha256(trusted.read_regular(build / (phase + "_report.json")))}
    (build / "host_phase_receipt.json").write_text(json.dumps(receipt))
    return report


@pytest.mark.parametrize("fault", [None, "missing_raw", "missing_case", "stale_host", "stale_source",
                                   "outlier", "route_change"])
def test_real_six_phase_loop_requires_all_current_producer_evidence(committed_dense, tmp_path, monkeypatch, fault):
    repo, commit, task_path, candidate = committed_dense
    if fault == "route_change":
        path = candidate / "source/routing.py"
        path.write_text(path.read_text().replace('"recorded"', '"torch"', 1))
    original_output = subprocess.check_output
    image = trusted.package_contract(repo / task_path)["manifest"]["runtime_image"]
    def output(command, *args, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"Id": "sha256:" + "a" * 64, "RepoDigests": [image]}])
        return original_output(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "check_output", output)
    gpu = {"render_device": "/dev/dri/renderD129", "rocr_uuid": "GPU-0000000000000001",
           "pci_bus_id": "0000:01:00.0"}
    monkeypatch.setattr(trusted, "select_gpu", lambda *a: gpu)
    def seed(image, cache, log, **kwargs):
        (cache / "jit").mkdir(parents=True)
        return {"complete_parity": True}
    monkeypatch.setattr(trusted, "seed_image_cache", seed)
    phases = []
    def run_phase(image, task, staging, output, leg, request, render, timeout, cache, **kwargs):
        phases.append((leg, request["phase"], request["request_id"]))
        contract = kwargs["qwen_contract"]
        assert "untrusted" not in (task / "config.yaml").read_text()
        assert "agent harness executed" not in (task / "scripts/task_runner.py").read_text()
        report = producer_phase(task, contract, request["phase"], request,
                                outlier=fault == "outlier" and leg == "candidate")
        if leg == "candidate" and request["phase"] == "performance" and fault not in (None, "outlier", "route_change"):
            build = task / "build"
            archive = build / report["complete_raw_report"].removeprefix("build/")
            if fault == "missing_raw":
                archive.unlink()
            elif fault == "missing_case":
                raw = json.loads(archive.read_text()); raw["cases"].pop()
                archive.write_text(json.dumps(raw)); (build / "_bench_raw.json").write_text(json.dumps(raw))
            elif fault == "stale_host":
                receipt = json.loads((build / "host_phase_receipt.json").read_text())
                receipt["host_request"]["request_id"] = "f" * 48
                (build / "host_phase_receipt.json").write_text(json.dumps(receipt))
            else:
                (task / "source/routing.py").write_text("import os\n")
        return report
    monkeypatch.setattr(trusted, "run_phase", run_phase)
    destination = tmp_path / "result"
    arguments = dict(repo=repo, commit=commit, task_path=task_path, candidate_workspace=candidate,
                     output=destination, render_device=gpu["render_device"], scratch_dir=tmp_path / "scratch")
    if fault not in (None, "outlier", "route_change"):
        with pytest.raises((ValueError, OSError)):
            trusted.trusted_retest(**arguments)
        assert not (destination / "trusted_measurement.json").exists()
        assert len(list(destination.glob("*.phase_receipt.json"))) == 5
    else:
        result = trusted.trusted_retest(**arguments)
        assert result["status"] == ("rejected_timing_quality" if fault == "outlier" else "measured")
        assert result["fresh_phase_count"] == 6
        assert result["full_case_coverage"] and len(result["cases"]) == 5
        if fault != "route_change":
            assert result["accepted_gain"] is False and result["arithmetic_mean_speedup"] is None
        else:
            assert result["benchmark_quality"]["source_changed"] is True
        assert len(result["phase_receipts"]) == 6
        for row in result["phase_receipts"]:
            assert trusted.sha256((destination / row["file"]).read_bytes()) == row["sha256"]
    assert [(leg, phase) for leg, phase, _ in phases] == [
        (leg, phase) for leg in ("reference", "candidate") for phase in trusted.PHASES]
    assert len({nonce for _, _, nonce in phases}) == 6


def test_candidate_symlink_is_rejected_before_any_gpu_access(committed_dense, tmp_path, monkeypatch):
    repo, commit, task_path, candidate = committed_dense
    source = candidate / "source/routing.py"
    source.unlink(); source.symlink_to(repo / task_path / "source/routing.py")
    monkeypatch.setattr(trusted, "select_gpu", lambda *a: pytest.fail("source escaped admission"))
    with pytest.raises(OSError):
        trusted.trusted_retest(repo=repo, commit=commit, task_path=task_path, candidate_workspace=candidate,
                              output=tmp_path / "result", render_device="/dev/dri/renderD129",
                              scratch_dir=tmp_path / "scratch")
