"""Generic trusted retest stages declared sources and checks complete case ABI."""

import copy
import json
import shutil
import subprocess

import pytest
import yaml

from src.task_contract import finalize_report
from src.tools import trusted_task_eval as trusted
from src.tools.materialize_task_contract import materialize


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
