"""CPU checks of source-probe classification and non-launching smoke plans."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location("fp4_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure,expected,code", [
    ("reference", "invalid_reference", 2),
    ("candidate_setup", "setup_failure", 2),
    ("numerical", "candidate_rejected", 1),
    ("metadata", "setup_failure", 2),
    ("missing_launch", "setup_failure", 2),
    (None, "candidate_accepted", 0),
])
def test_probe_never_counts_invalid_reference_or_setup_as_rejection(monkeypatch, failure, expected, code):
    probe = script("check_source_binding")
    monkeypatch.setattr(probe, "load_dataset", lambda _: {"cases": [{"case_id": "case"}], "oracle_policy": {}})
    monkeypatch.setattr(probe, "validate_sources", lambda *_args: True)
    class State:
        def __init__(self, *args, leg="candidate", **kwargs):
            self.leg, self.proof = leg, {}
            self.calls = 0 if failure == "missing_launch" else 1
            self.probe = SimpleNamespace(launches=[] if failure == "missing_launch" else [
                {"kernel_name": "test_kernel", "kernel_hash": "compiled-test-kernel"}])
            if leg == "candidate" and failure == "candidate_setup":
                raise RuntimeError("compile failed")
        def capture_graph(self):
            self.proof["graph_captured"] = True
        def check_once(self, seed):
            self.proof["graph_replayed"] = True
            if failure == "reference" and self.leg == "reference":
                raise AssertionError("native reference failed")
            if self.leg == "candidate":
                if failure == "numerical":
                    raise AssertionError("independent packed-value oracle mismatch")
                if failure == "metadata":
                    raise AssertionError("wrong output stride")
    monkeypatch.setattr(probe, "FP4Case", State)
    actual_code, result = probe.probe(Path("dataset"), Path("submitted"), "case", "graph", 17)
    assert actual_code == code and result["status"] == expected
    assert result["scoreable"] is False and result["performance_samples"] == 0
    if code == 1:
        assert result["reference_calibrated"] and result["graph_captured"] and result["graph_replayed"]
        assert result["candidate_compiled_and_engaged"]


@pytest.mark.parametrize("bad_proof", [None, "compilation", "graph_replay"])
def test_source_controls_cover_all_actual_cases_and_require_compilation(tmp_path, monkeypatch, bad_proof):
    runner = script("task_runner")
    probe = script("check_source_binding")
    controls = script("make_source_controls")
    import sys
    monkeypatch.setitem(sys.modules, "check_source_binding", probe)
    monkeypatch.setitem(sys.modules, "make_source_controls", controls)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []
    def check(dataset, workspace, case_id, mode, seed):
        calls.append((workspace.name, case_id, mode, seed))
        return 1, {"status": "candidate_rejected", "reference_calibrated": True,
                   "candidate_compiled_and_engaged": bad_proof != "compilation", "case_id": case_id, "mode": mode,
                   "graph_captured": mode == "graph", "graph_replayed": mode == "graph" and bad_proof != "graph_replay"}
    monkeypatch.setattr(probe, "probe", check)
    manifest = json.loads((ROOT / "cases.json").read_text())
    if bad_proof:
        with pytest.raises(ValueError, match="candidate engagement"):
            runner.run_source_controls(Path("dataset"), manifest, 1)
        return
    reports = runner.run_source_controls(Path("dataset"), manifest, 1)
    expected = {(kind, case["case_id"], mode, 1) for kind in ("submitted_no_op", "submitted_wrong_output")
                for case in manifest["cases"] for mode in ("eager", "graph")}
    assert len(reports) == len(calls) == len(expected) == 56
    assert set(calls) == expected


def test_prepare_creates_frozen_command_without_executing_helper(tmp_path, monkeypatch):
    prepare = script("prepare_native_smoke")
    task, common = tmp_path / "task", tmp_path / "common"
    (task / "capture").mkdir(parents=True)
    common.mkdir()
    raw = b"raise RuntimeError('must not execute during preparation')\n"
    (common / "runtime_capture.py").write_bytes(raw)
    image = "registry/image@sha256:" + "a" * 64
    (task / "SOURCE-PROVENANCE.json").write_text(json.dumps({"runtime_image": image}))
    (task / "capture/INTEGRATION.json").write_text(json.dumps({"shared_recorder": {"runtime_capture_sha256": hashlib.sha256(raw).hexdigest()}}))
    binding = tmp_path / "binding.py"
    binding.write_bytes(raw)
    expectation = tmp_path / "gpu.json"
    expectation.write_text(json.dumps({"render_device": "/dev/dri/renderD128", "rocr_uuid": "GPU-0000000000000001"}))
    monkeypatch.setattr(prepare, "ROOT", task)
    output = tmp_path / "prepared"
    plan = prepare.prepare(output, common, binding, expectation)
    assert plan["GPU_actions"] is False and plan["scheduler_actions"] is False
    assert plan["status"] == "PREPARED_NOT_EXECUTED" and image in plan["command"]
    assert f"type=bind,src={output / 'task'},dst=/task,readonly" in plan["command"]
    assert f"type=bind,src={output / 'results'},dst=/results" in plan["command"]
    assert (output / "task/SOURCE-PROVENANCE.json").read_bytes() == (task / "SOURCE-PROVENANCE.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare.prepare(output, common, binding, expectation)
