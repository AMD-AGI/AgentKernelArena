"""Generic trusted retest stages declared sources and checks complete case ABI."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from src.task_contract import finalize_report
from src.tools import trusted_task_eval as trusted
from src.tools.materialize_task_contract import materialize


def test_jsonl_failure_context_survives_disposable_build_cleanup(tmp_path, monkeypatch):
    if shutil.which("rclone") is None:
        pytest.skip("diagnostic preservation requires rclone")
    build = tmp_path / "build"
    build.mkdir()
    event = b'{"event":"verify_failure","leg":"candidate_port","seed":436841561,"iteration":61}\n'
    (build / "native_production_events.jsonl").write_bytes(event)
    (build / "performance_report.json").write_text('{"status":"failed"}\n')
    (build / "compiler.log").write_text("compiler evidence\n")
    (build / "unselected.bin").write_bytes(b"not a textual diagnostic")
    original = subprocess.run
    offered = []

    def run(command, **kwargs):
        names = Path(command[command.index("--files-from-raw") + 1]).read_text().splitlines()
        assert len(names) == 1
        assert command[command.index("--transfers") + 1] == "64000"
        assert command[command.index("--buffer-size") + 1] == "0"
        assert command[command.index("--multi-thread-streams") + 1] == "0"
        assert kwargs["env"]["GOMAXPROCS"] == "1"
        offered.extend(names)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "preserved"
    trusted.preserve_diagnostics(build, output)
    shutil.rmtree(build)
    assert (output / "native_production_events.jsonl").read_bytes() == event
    hashes = json.loads((output / "hashes.json").read_text())
    assert hashes["native_production_events.jsonl"] == trusted.sha256(event)
    assert set(offered) == {"native_production_events.jsonl", "performance_report.json", "compiler.log"}
    assert not (output / "unselected.bin").exists()


@pytest.fixture
def packaged_task(tmp_path):
    if shutil.which("rclone") is None:
        pytest.skip("trusted payload staging requires rclone")
    repo = tmp_path / "repo"
    task = repo / "tasks/example"
    for directory in ("source", "ut/reference", "scripts"):
        (task / directory).mkdir(parents=True)
    (task / "source/kernel.py").write_text("VALUE = 1\n")
    (task / "ut/reference/kernel.py").write_text("VALUE = 1\n")
    (task / "ut/source_guard.py").write_text(
        "def validate_sources(candidate_root, reference_root):\n"
        "    value = (candidate_root / 'source/kernel.py').read_text()\n"
        "    if value not in ('VALUE = 1\\n', 'VALUE = 2\\n'):\n"
        "        raise ValueError('source boundary violated')\n"
    )
    (task / "scripts/task_runner.py").write_text("# Protected test runner; GPU execution mocked by this test.\n")
    image = "registry/image@sha256:" + "a" * 64
    config = {"source_file_path": ["source/kernel.py"], "headkernel": {"docker": image},
              "harness_protection": {"reject_new_source_symlinks": True},
              "trusted_evaluation": {"schema_version": 1, "reference_sources": {"source/kernel.py": "ut/reference/kernel.py"}}}
    config.update({phase + "_command": [f"python3 scripts/task_runner.py {phase}"] for phase in trusted.PHASES})
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    tensor = {"role": "input", "shape": [8, 16], "strides": [16, 1], "storage_offset": 0,
              "dtype": "bfloat16", "device_type": "cuda"}
    manifest = {"schema_version": 1, "runtime_image": image,
                "cases": [{"case_id": name, "occurrences": count, "calls_per_sample": 1,
                           "tensors": {"x": tensor, "out": {**tensor, "role": "output"}},
                           "scalars": {"scale": scalar}}
                          for name, count, scalar in (("first", 7, 1.0), ("second", 19, 2.0))],
                "measurement": {"method": "cuda_graph", "warmup_iterations": 1, "benchmark_iterations": 2,
                                "correctness_seeds": [0, 1], "negative_controls": ["no_op", "wrong_output"],
                                "refresh_inputs": "each_replay", "initialize_outputs": "each_replay",
                                "validate_outputs": "each_replay"}}
    (task / "cases.json").write_text(json.dumps(manifest))
    materialize(task)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                                    "commit", "-qm", "trusted task"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    candidate = tmp_path / "agent"
    (candidate / "source").mkdir(parents=True)
    (candidate / "source/kernel.py").write_text("VALUE = 2\n")
    return repo, task, commit, candidate, manifest


def phase_report(request, manifest, timing):
    report = {"schema_version": 1, "status": "ok", "request": request, "cases": []}
    if request["phase"] == "compile":
        report["compiled"] = True
    else:
        for case in manifest["cases"]:
            row = {"case": copy.deepcopy(case), "correct": True}
            if request["phase"] == "correctness":
                row.update(seeds=[0, 1], negative_controls={"no_op": True, "wrong_output": True})
            else:
                row.update(samples_ms=[timing, timing], fresh_input_resets=2, output_initializations=2,
                           oracle_checks=2, warmup_iterations=1, benchmark_method="cuda_graph")
            report["cases"].append(row)
    return finalize_report(report, manifest, request)


@pytest.mark.parametrize("missing_case", [False, True])
def test_sources_only_retest_scores_complete_cases_or_rejects(packaged_task, tmp_path, monkeypatch, missing_case):
    repo, task, commit, candidate, manifest = packaged_task
    (candidate / "performance_report.json").write_text('{"speedup":999999}')
    (candidate / "ut").mkdir()
    (candidate / "ut/evaluation_contract.py").write_text("raise RuntimeError('agent helper must never run')")
    (task / "scripts/task_runner.py").write_text("raise RuntimeError('dirty checkout must never run')")
    check_output = subprocess.check_output

    def inspect(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"RepoDigests": [manifest["runtime_image"]], "Id": "sha256:" + "b" * 64}])
        return check_output(command, **kwargs)

    requests = []

    def run_phase(image, staged, staging, output, leg, request, *args):
        requests.append(request)
        assert "dirty checkout" not in (staged / "scripts/task_runner.py").read_text()
        assert (staged / "ut/evaluation_contract.py").read_bytes() == trusted.PORTABLE_CONTRACT.read_bytes()
        assert not (staged / "performance_report.json").exists()
        assert (staged / "source/kernel.py").read_text() == ("VALUE = 1\n" if leg == "reference" else "VALUE = 2\n")
        report = phase_report(request, manifest, 4.0 if leg == "reference" else 2.0)
        if missing_case and request["phase"] == "performance" and leg == "candidate":
            report["cases"].pop()
        return report

    monkeypatch.setattr(subprocess, "check_output", inspect)
    monkeypatch.setattr(trusted, "select_gpu", lambda render: {"render_device": render, "rocr_uuid": "GPU-000000000000abcd",
                                                              "pci_bus_id": "0000:83:00.0"})
    monkeypatch.setattr(trusted, "run_phase", run_phase)
    output = tmp_path / "accepted"

    def run():
        return trusted.trusted_retest(repo=repo, commit=commit, task_path="tasks/example", candidate_workspace=candidate,
                                      output=output, scratch_dir=tmp_path / "scratch", render_device="/dev/dri/renderD128")

    if missing_case:
        with pytest.raises(ValueError, match="missing, duplicate"):
            run()
        assert not (output / "trusted_measurement.json").exists()
    else:
        result = run()
        assert result["arithmetic_mean_speedup"] == 2
        assert len(result["cases"]) == 2 and result["full_case_coverage"] is True
        assert len(requests) == 6 and len({request["request_id"] for request in requests}) == 6
        assert len({request["challenge_seed"] for request in requests}) == 1
        assert not (output / "task_result.yaml").exists() and not (output / "validation_report.yaml").exists()


@pytest.mark.parametrize('stale_native_request', [False, True])
def test_native_baseline_retest_keeps_port_gain_secondary(packaged_task, tmp_path, monkeypatch, stale_native_request):
    import hashlib
    repo, task, _, candidate, manifest = packaged_task
    (task / 'provenance').mkdir()
    (task / 'provenance/NATIVE.json').write_text('{"native": "pinned"}\n')
    config = yaml.safe_load((task / 'config.yaml').read_text())
    config['scoring_baseline'] = {'schema_version': 1, 'kind': 'native_production',
                                  'native_source_manifest': 'provenance/NATIVE.json'}
    (task / 'config.yaml').write_text(yaml.safe_dump(config))
    subprocess.run(['git', '-C', str(repo), 'add', 'tasks/example'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'native scoring policy'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    check_output = subprocess.check_output
    def inspect(command, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect']:
            return json.dumps([{'RepoDigests': [manifest['runtime_image']], 'Id': 'sha256:' + 'b' * 64}])
        return check_output(command, **kwargs)
    def run_phase(image, staged, staging, output, leg, request, *args):
        timing = 4.0 if leg == 'reference' else 2.0
        report = phase_report(request, manifest, timing)
        if request['phase'] == 'performance':
            native_rows = phase_report(request, manifest, 1.0)['cases']
            native = {'schema_version': 1, 'schema': 'native-production-comparison-v1', 'status': 'ok',
                      'diagnostic_only': False, 'score_input': True, 'baseline_kind': 'native_production',
                      'request': copy.deepcopy(request), 'source_hashes': request['source_sha256'],
                      'source_sha256': next(iter(request['source_sha256'].values())),
                      'manifest_sha256': trusted.fingerprint(manifest), 'runtime_image': image,
                      'native_source_manifest_sha256': hashlib.sha256((staged / 'provenance/NATIVE.json').read_bytes()).hexdigest(),
                      'cases': [{'case_id': row['case']['case_id'], 'native_output_parity': True,
                                 'identical_captured_ABI_and_fresh_numeric_challenge_sequence': True,
                                 'legs': {'candidate_port': copy.deepcopy(row), 'native_production': native_row}}
                                for row, native_row in zip(report['cases'], native_rows)]}
            if stale_native_request and leg == 'candidate':
                native['request']['request_id'] = 'previous-run'
            report['native_production_comparison'] = native
        return report
    monkeypatch.setattr(subprocess, 'check_output', inspect)
    monkeypatch.setattr(trusted, 'select_gpu', lambda render: {'render_device': render, 'rocr_uuid': 'GPU-000000000000abcd',
                                                              'pci_bus_id': '0000:83:00.0'})
    monkeypatch.setattr(trusted, 'run_phase', run_phase)
    output = tmp_path / 'native-result'
    def run():
        return trusted.trusted_retest(repo=repo, commit=commit, task_path='tasks/example', candidate_workspace=candidate,
                                      output=output, scratch_dir=tmp_path / 'scratch', render_device='/dev/dri/renderD128')
    if stale_native_request:
        with pytest.raises(ValueError, match='Stale native comparison'):
            run()
        assert not (output / 'trusted_measurement.json').exists()
    else:
        result = run()
        assert result['arithmetic_mean_speedup'] == 0.5
        assert result['port_to_port_speedup_ratio'] == 2.0
        assert result['baseline_kind'] == 'native_production'
        assert result['production_kernel_improvement'] is False
        assert result['all_cases_faster_than_native'] is False
        assert len(result['regressed_case_ids']) == 2
        assert all(row['reference_ms'] == 1 and row['candidate_ms'] == 2 for row in result['cases'])


@pytest.mark.parametrize("attack", ["symlink", "body_escape"])
def test_bad_source_rejected_before_any_docker_call(packaged_task, tmp_path, monkeypatch, attack):
    repo, _task, commit, candidate, _manifest = packaged_task
    source = candidate / "source/kernel.py"
    if attack == "symlink":
        source.unlink()
        source.symlink_to("../../outside.py")
    else:
        source.write_text("import os; os._exit(0)\n")
    original = subprocess.check_output

    def forbid(command, **kwargs):
        assert command[0] != "docker"
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", forbid)
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        trusted.trusted_retest(repo=repo, commit=commit, task_path="tasks/example", candidate_workspace=candidate,
                               output=tmp_path / "output", render_device="/dev/dri/renderD128")


def test_materialized_helper_is_self_contained_and_edits_are_rejected(packaged_task):
    _repo, task, _commit, _candidate, _manifest = packaged_task
    path = materialize(task, check=True)
    subprocess.run(["python3", "-I", "-S", str(path)], check=True)
    path.write_text("# weakened contract\n")
    with pytest.raises(ValueError, match="differs from canonical"):
        materialize(task, check=True)
    with pytest.raises(ValueError, match="differs from the trusted host"):
        trusted.package_contract(task)


@pytest.fixture
def packaged_fixture_task(packaged_task, tmp_path):
    import hashlib
    from src.task_contract import fingerprint

    repo, task, _commit, candidate, manifest = packaged_task
    mirror = tmp_path / "fixture-mirror"
    (mirror / "objects").mkdir(parents=True)
    raw = b"captured kernel operand and expected output"
    blob_sha = hashlib.sha256(raw).hexdigest()
    fixture = {"schema": "served-tensor-fixture-v1", "payload": {"inputs": {"weight": {
        "storage_nbytes": len(raw), "segments": [{"blob": "blob.bin", "sha256": blob_sha,
        "bytes": len(raw), "offset_bytes": 0}]}}}}
    encoded = json.dumps(fixture).encode()
    fixture_sha = hashlib.sha256(encoded).hexdigest()
    (mirror / "objects/case.json").write_bytes(encoded)
    (mirror / "objects/data.bin").write_bytes(raw)
    for case in manifest["cases"]:
        case["live_fixture"] = {"path": "fixtures/case.json", "sha256": fixture_sha}
    (task / "cases.json").write_text(json.dumps(manifest))
    config = yaml.safe_load((task / "config.yaml").read_text())
    config["trusted_evaluation"]["fixture_manifest"] = "fixtures/EXTERNAL-MANIFEST.json"
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    (task / "fixtures").mkdir()
    assets = {"schema": "trusted-external-fixtures-v1", "case_manifest_fingerprint": fingerprint(manifest),
              "runtime_image": manifest["runtime_image"], "oci_prefix": "oci:test/bucket/fixture-release/", "assets": [
                  {"path": "fixtures/case.json", "object_key": "objects/case.json", "sha256": fixture_sha,
                   "bytes": len(encoded), "codec": "served-tensor-fixture-v1", "roles": ["case_metadata"]},
                  {"path": "fixtures/blob.bin", "object_key": "objects/data.bin", "sha256": blob_sha,
                   "bytes": len(raw), "codec": "raw-storage-segment-v1", "roles": ["kernel_weight", "oracle_output"]}]}
    (task / "fixtures/EXTERNAL-MANIFEST.json").write_text(json.dumps(assets))
    for args in (["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "pin fixture data"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return repo, task, commit, candidate, manifest, mirror, raw


def test_fixture_data_precedes_both_leg_hashes_and_ignores_agent_payloads(packaged_fixture_task, tmp_path, monkeypatch):
    from src.task_contract import fingerprint
    from src.tools.seed_aiter_jit_cache import tree_manifest

    repo, task, commit, candidate, manifest, mirror, raw = packaged_fixture_task
    (candidate / "fixtures").mkdir()
    (candidate / "fixtures/blob.bin").write_bytes(b"agent fixture must never enter either leg")
    (candidate / "config.yaml").write_text("trusted_evaluation: {fixture_manifest: evil.json}")
    # Dirty trusted working-tree data also cannot replace the committed artifact manifest.
    (task / "fixtures/EXTERNAL-MANIFEST.json").write_text("{}")
    original = subprocess.check_output
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: json.dumps([
        {"RepoDigests": [manifest["runtime_image"]], "Id": "sha256:" + "b" * 64}])
        if cmd[:3] == ["docker", "image", "inspect"] else original(cmd, **kw))
    monkeypatch.setattr(trusted, "select_gpu", lambda render: {"render_device": render})
    seen = []

    def phase(image, staged, staging, output, leg, request, *args):
        assert (staged / "fixtures/blob.bin").read_bytes() == raw
        assert "evil.json" not in (staged / "config.yaml").read_text()
        inventory = tree_manifest(staged)
        inventory.pop("build", None)
        assert request["package_sha256"] == fingerprint(inventory)
        assert "fixtures/blob.bin" in inventory
        assert (output / "fixtures_receipt.json").exists()
        seen.append(leg)
        return phase_report(request, manifest, 4.0 if leg == "reference" else 2.0)

    monkeypatch.setattr(trusted, "run_phase", phase)
    output = tmp_path / "fixture-evaluation"
    result = trusted.trusted_retest(repo=repo, commit=commit, task_path="tasks/example", candidate_workspace=candidate,
                                   output=output, render_device="/dev/dri/renderD128", scratch_dir=tmp_path / "scratch",
                                   fixture_local_mirror=mirror)
    assert seen == ["reference"] * 3 + ["candidate"] * 3
    assert result["arithmetic_mean_speedup"] == 2
    assert result["fixtures"]["sha256"] == trusted.sha256((output / "fixtures_receipt.json").read_bytes())


def test_missing_fixture_stops_before_gpu_or_docker(packaged_fixture_task, tmp_path, monkeypatch):
    repo, _task, commit, candidate, _manifest, mirror, _raw = packaged_fixture_task
    (mirror / "objects/data.bin").unlink()
    monkeypatch.setattr(trusted, "select_gpu", lambda *a: pytest.fail("GPU selected before fixture verification"))
    monkeypatch.setattr(trusted, "run_phase", lambda *a: pytest.fail("task ran before fixture verification"))
    original = subprocess.check_output

    def check(command, **kwargs):
        assert command[0] != "docker"
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", check)
    with pytest.raises(OSError):
        trusted.trusted_retest(repo=repo, commit=commit, task_path="tasks/example", candidate_workspace=candidate,
                               output=tmp_path / "rejected", render_device="/dev/dri/renderD128",
                               scratch_dir=tmp_path / "scratch", fixture_local_mirror=mirror)


def test_stage_only_prepares_original_fixture_task_without_gpu(packaged_fixture_task, tmp_path, monkeypatch):
    repo, _task, commit, _candidate, _manifest, mirror, raw = packaged_fixture_task
    monkeypatch.setattr(trusted, "select_gpu", lambda *a: pytest.fail("stage-only selected GPU"))
    monkeypatch.setattr(trusted, "run_phase", lambda *a: pytest.fail("stage-only ran task"))
    output = tmp_path / "validator-stage"
    result = trusted.stage_trusted_task(repo=repo, commit=commit, task_path="tasks/example", output=output,
                                       scratch_dir=tmp_path / "scratch", fixture_local_mirror=mirror)
    assert result["status"] == "staged_not_evaluated"
    assert (output / "task/fixtures/blob.bin").read_bytes() == raw
    assert (output / "task/source/kernel.py").read_text() == "VALUE = 1\n"
    assert (output / "staging_receipt.json").exists()
