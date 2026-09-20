"""Incomplete MoE sequence inputs must prevent qualification before GPU setup."""

import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted(
    path.parent.parent
    for path in (ROOT / "tasks/head_kernels/deepseek-v4-pro").glob(
        "*/*/moe*/ut/meta.json"
    )
)
LEDGER_HASHES = {
    "moe_stage1_grouped_gemm_silu_flydsl": "479d2a1e652bc91142b9c23d50aa30fe80f793e2dac3237096c0406c1c8bc84a",
    "moe_stage2_down_proj_reduce_opus_a8w4": "6281831ff9c45f3e083465ccfa306de89e4ed43b7d96909e506643ff3614ab3a",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(task):
    helper = load("_sequence_contract", task / "ut/generated_contract.py")
    meta = json.loads((task / "ut/meta.json").read_text())
    return helper, meta, helper.load_contract(task / "ut")


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_full_ledger_and_five_case_contract_are_retained(task):
    helper, meta, data = contract(task)
    assert canonical(meta["call_sequence"]) == LEDGER_HASHES[task.name]
    assert len(meta["call_sequence"]) == 256
    profiles = helper.profiles(meta)
    assert profiles["sequence"] == [row["sig"] for row in meta["call_sequence"]]
    assert len(profiles["sequence"]) == 256
    ids = [row["sig"] for row in data["records"]]
    assert len(ids) == meta["num_cases"] == 5
    assert profiles["eager"] == profiles["random"] == ids
    assert profiles["replay"] == [ids[1], ids[2], ids[1]]
    assert "256" in ids[1] and "64" in ids[2]
    assert meta["random_draws"] == 3
    assert meta["tol"] == (0.15 if data["task_kind"] == "moe1" else 0.05)
    assert meta["generated_inputs"]["scored_case_ids"] == ids
    assert [row["in_graph"] for row in meta["call_sequence"]] == (
        [False] * 122 + [True] * 61 + [False] * 73
    )
    assert meta["generated_inputs"]["qualification_status"] == (
        "blocked_missing_sequence_inputs"
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_exact_missing_signature_positions_and_recovery_provenance(task):
    helper, meta, data = contract(task)
    coverage = helper.profile_coverage(meta, data)
    assert coverage["status"] == "incomplete"
    assert coverage["scope"] == "input_availability_only_not_gpu_validation"
    assert coverage["required_sequence_calls"] == 256
    assert coverage["available_sequence_calls"] == 183
    assert coverage["missing_sequence_calls"] == 73
    missing = {
        "sig": meta["call_sequence"][183]["sig"],
        "positions_0based": list(range(183, 256)),
        "count": 73,
    }
    assert "1920" in missing["sig"]
    assert coverage["missing_inputs"] == {"sequence": [missing]}
    evidence = json.loads((task / "ut/sequence_coverage_evidence.json").read_text())
    assert evidence["missing"] == {**missing, "M": 1920}
    assert evidence["available"]["positions_0based"] == list(range(183))
    assert evidence["call_sequence_sha256"] == LEDGER_HASHES[task.name]
    assert evidence["recovery"]["source_reference_sha256"] == (
        meta["archival_capture"]["reference_io_sha256"]
    )
    assert evidence["original_harness"]["filter_line"] in (199, 204)
    assert len(evidence["original_harness"]["sha256"]) == 64
    assert len(evidence["recovery"]["evidence_sha256"]) == 6
    shapes = json.loads((task / "SHAPES.json").read_text())
    assert shapes["generated_input_draft"]["sequence_coverage"] == coverage
    for source in shapes["sources"]:
        assert helper.digest(task / source["path"]) == source["sha256"]


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
@pytest.mark.parametrize("mutation", ["filter", "drop", "append", "replace", "graph"])
def test_changed_or_filtered_ledger_is_rejected(task, mutation):
    helper, meta, data = contract(task)
    ledger = meta["call_sequence"]
    if mutation == "filter":
        meta["call_sequence"] = ledger[:183]
    elif mutation == "drop":
        ledger.pop()
    elif mutation == "append":
        ledger.append(ledger[-1])
    elif mutation == "replace":
        ledger[183]["sig"] = ledger[0]["sig"]
    else:
        ledger[0]["in_graph"] = True
    with pytest.raises(RuntimeError, match="ledger count or hash changed"):
        helper.profile_coverage(meta, data)


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_other_missing_inputs_cannot_be_silently_ignored(task):
    helper, meta, data = contract(task)
    reduced = copy.deepcopy(data)
    del reduced["records"][1]  # The M256 graph capture is mandatory too.
    coverage = helper.profile_coverage(meta, reduced)
    assert coverage["status"] == "incomplete"
    assert {"eager", "random", "sequence", "replay"} == set(coverage["missing_inputs"])
    assert coverage["missing_inputs"]["replay"][0]["positions_0based"] == [0, 2]
    meta["call_sequence"][0]["sig"] = "another uncaptured signature"
    # Even a deliberately repinned new declaration must resolve every input.
    meta["sequence_requirement"]["call_sequence_sha256"] = canonical(
        meta["call_sequence"]
    )
    coverage = helper.profile_coverage(meta, data)
    assert coverage["status"] == "incomplete"
    assert coverage["missing_sequence_calls"] == 74
    assert coverage["missing_inputs"]["sequence"][0]["positions_0based"] == [0]


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
@pytest.mark.parametrize("mode", ["correctness", "performance"])
def test_declared_commands_fail_before_runtime_torch_and_workers(
    task, mode, monkeypatch
):
    for alias in (
        "_deepseek_arena_runner",
        "_deepseek_generated_controller",
        "generated_contract",
    ):
        monkeypatch.setitem(sys.modules, alias, None)
        monkeypatch.delitem(sys.modules, alias)
    wrapper = load("_sequence_task_runner", task / "scripts/generated_task_runner.py")
    reports = {}
    monkeypatch.setattr(
        wrapper.runner,
        "write_report",
        lambda name, value: reports.update({name: value}),
    )
    monkeypatch.setattr(wrapper.runner, "load_config", lambda: {})
    monkeypatch.setattr(
        wrapper.runner, "run_worker", lambda *a, **kw: pytest.fail("GPU worker reached")
    )
    monkeypatch.setattr(sys, "argv", ["generated_task_runner.py", mode])
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name not in {"torch", "runtime_preflight"}, "GPU setup reached"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert wrapper.main() == 1
    report = reports[mode + "_report.json"]
    assert report["status"] == "fail" and report["test_cases"] == []
    assert "183/256 mandatory sequence calls" in report["error"]
    assert "73 are missing" in report["error"]
    assert reports["sequence_coverage_report.json"]["status"] == "incomplete"
    assert reports["sequence_coverage_report.json"]["missing_inputs"]["sequence"][0][
        "positions_0based"
    ] == list(range(183, 256))


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_direct_controller_also_fails_before_torch(task, monkeypatch):
    monkeypatch.setitem(sys.modules, "generated_contract", None)
    monkeypatch.delitem(sys.modules, "generated_contract")
    controller = load("_sequence_controller", task / "scripts/generated_correctness.py")
    reports = {}
    from types import SimpleNamespace

    runner = SimpleNamespace(
        UT_DIR=task / "ut",
        write_report=lambda name, value: reports.update({name: value}),
    )
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name != "torch", "Torch import reached"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    ok, error = controller.run_correctness(runner, {}, 60)
    assert not ok and "183/256" in error
    assert reports["correctness_report.json"]["status"] == "fail"
    assert reports["correctness_report.json"]["profiles"] == []


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_coverage_evidence_hash_and_full_requirement_are_enforced(task, tmp_path):
    helper, meta, _ = contract(task)
    for filename in (
        "meta.json",
        "generated_cases.json",
        "sequence_coverage_evidence.json",
    ):
        (tmp_path / filename).write_bytes((task / "ut" / filename).read_bytes())
    evidence = tmp_path / "sequence_coverage_evidence.json"
    original = evidence.read_bytes()
    evidence.write_bytes(original + b" ")
    with pytest.raises(RuntimeError, match="coverage evidence hash mismatch"):
        helper.load_contract(tmp_path)
    evidence.write_bytes(original)
    meta["call_sequence"] = meta["call_sequence"][:183]
    meta["sequence_requirement"].update(
        required_calls=183, call_sequence_sha256=canonical(meta["call_sequence"])
    )
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(
        RuntimeError, match="mandatory call-sequence requirement changed"
    ):
        helper.load_contract(tmp_path)
