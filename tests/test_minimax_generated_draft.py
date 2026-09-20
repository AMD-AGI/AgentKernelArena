"""CPU qualification of the isolated MiniMax generated-input draft.

Production geometry is inspected without allocating its multi-GB buffers. Small
CPU contracts exercise actual extraction, generation, workers and parent checks.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted(path.parent for path in (ROOT / "tasks/head_kernels/minimax-m3-mxfp4").rglob("config.yaml"))


def load(filename):
    spec = importlib.util.spec_from_file_location("test_" + filename, generated_helper("minimax", filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load("generated_contract")
controller = load("generated_correctness")
worker = load("generated_worker")
extractor = load("extract_contract")


def tensor_desc(shape, dtype="torch.float32", group="q", recipe="normal_unit", **extra):
    stride, count = [], 1
    for dim in reversed(shape):
        stride.insert(0, count)
        count *= dim
    return {"tensor": True, "shape": shape, "stride": stride, "dtype": dtype,
            "storage_offset": 0, "storage_group": group, "storage_numel": count,
            "tensor_attrs": {}, "recipe": recipe, **extra}


def simple_record(sig="case"):
    return {"sig": sig, "regime": "decode", "args": [], "kwargs": {
        "q": tensor_desc([2]),
        "req_to_token": tensor_desc([4, 3], "torch.int32", "rows", "captured_rows",
                                    row_indices=[2], rows=[[7, 2, 9]], max_slots=16),
        "slot_ids": tensor_desc([1], "torch.int64", "slots", "captured_values", values=[2]),
    }, "output_contract": {"sequence": "tuple", "items": [None,
        tensor_desc([2], group="out", recipe="runtime_reference"), None]}}


def install_contract(ut, records=None):
    records = records or [simple_record()]
    ut.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 1, "source_reference_sha256": "a" * 64,
            "correctness_case_count": len(records), "records": records}
    path = ut / "generated_cases.json"
    path.write_text(json.dumps(data))
    geometry = ut / "timing_geometry.json"
    geometry.write_text('{"records": []}')
    meta = {"reference_io_sha256": "a" * 64, "num_cases": len(records), "random_draws": 3,
            "tol": 0.02, "cases": [{"sig": row["sig"]} for row in records],
            "generated_inputs": {"contract_file": path.name, "contract_sha256": contract.digest(path),
                                 "geometry_file": geometry.name, "geometry_sha256": contract.digest(geometry)}}
    (ut / "meta.json").write_text(json.dumps(meta))
    return meta


def test_actual_compact_extraction_has_all_cases_and_original_abi():
    counts = {"decode_score_kernel": 5, "gqa_share_sparse_decode_kernel": 3,
              "gqa_share_sparse_fwd_kernel": 3}
    for task in TASKS:
        data = contract.load_contract(task / "ut")
        assert len(data["records"]) == counts[task.name]
        for row in data["records"]:
            kwargs = row["kwargs"]
            assert kwargs["req_to_token"]["shape"] == [4097, 11268]
            assert kwargs["req_to_token"]["stride"] == [11268, 1]
            assert kwargs["slot_ids"]["dtype"] == "torch.int64"
            assert kwargs["k_cache"]["shape"] == [4358330, 1, 128]
            assert kwargs["k_cache"]["dtype"] == "torch.bfloat16"
            assert "values" not in kwargs["k_cache"]
            if task.name != "decode_score_kernel":
                assert kwargs["topk_idx"]["recipe"] == "captured_values"
                assert kwargs["topk_idx"]["dtype"] == "torch.int32"
    decode = contract.load_contract(next(task for task in TASKS if task.name == "gqa_share_sparse_decode_kernel") / "ut")
    assert sum(torch.tensor(row["kwargs"]["topk_idx"]["values"], dtype=torch.int32).numel() * 4
               for row in decode["records"]) == 1024


def test_real_plan_preserves_scored_and_replay_geometry():
    expected = {"decode_score_kernel": (5, 4, 3), "gqa_share_sparse_decode_kernel": (3, 4, 3),
                "gqa_share_sparse_fwd_kernel": (3, 3, 3)}
    for task in TASKS:
        meta = json.loads((task / "ut/meta.json").read_text())
        plan = controller.expected_profiles(task / "ut", contract, meta)
        assert tuple(len(plan[k]) for k in ("recorded", "random", "replay")) == expected[task.name]
        assert plan["replay"][0] == plan["replay"][-1]


def test_generated_values_are_reproducible_and_structural_rows_are_real():
    record = simple_record()
    record["kwargs"]["q"].update(shape=[2], stride=[2], storage_offset=1, storage_numel=5,
                                     tensor_attrs={"captured_flag": True})
    record["kwargs"]["q_alias"] = copy.deepcopy(record["kwargs"]["q"])
    _, first = contract.build_record(record, 77, torch, "cpu")
    _, repeat = contract.build_record(record, 77, torch, "cpu")
    _, changed = contract.build_record(record, 78, torch, "cpu")
    assert first["q"].stride() == (2,)
    assert first["q"].storage_offset() == 1
    assert first["q"].captured_flag is True
    assert first["q"].data_ptr() == first["q_alias"].data_ptr()
    assert torch.equal(first["q"], repeat["q"])
    assert not torch.equal(first["q"], changed["q"])
    assert first["req_to_token"].shape == (4, 3)
    assert first["slot_ids"].tolist() == [2]
    assert first["req_to_token"][2].tolist() == [7, 2, 9]
    assert torch.equal(first["req_to_token"][2], changed["req_to_token"][2])


def test_clone_tree_preserves_layout_aliases_without_mutating_source():
    backing = torch.arange(12).reshape(3, 4)
    source = {"x": backing[:, 1:3], "y": backing.t()}
    source["x"].tag = "captured"
    result = contract.clone_tree(source, torch)
    assert result["x"].stride() == source["x"].stride()
    assert result["x"].storage_offset() == source["x"].storage_offset()
    assert result["x"].tag == "captured"
    assert result["x"].untyped_storage().data_ptr() == result["y"].untyped_storage().data_ptr()
    result["x"].fill_(-1)
    assert torch.equal(backing, torch.arange(12).reshape(3, 4))
    assert not torch.equal(result["y"], source["y"])


def test_extractor_keeps_metadata_and_routing_without_large_value_arrays():
    meta = {"num_cases": 1, "reference_io_sha256": "a" * 64,
            "cases": [{"sig": "case", "regime": "decode", "max_slots": 32}]}
    values = torch.randn(2, 3)
    blob = {"records": [{"sig": "case", "kwargs": {
        "q": {"__tensor__": True, "data": values},
        "k_cache": {"__tensor__": True, "data": values.t()},
        "req_to_token": {"__tensor__": True, "data": torch.arange(24).reshape(4, 6)},
        "slot_ids": {"__tensor__": True, "data": torch.tensor([2])},
        "topk_idx": {"__tensor__": True, "data": torch.tensor([[[1, -1]]], dtype=torch.int32)},
        "score_type": {"__repr__": "'max'"}}, "output": (None, torch.ones(1), None)}]}
    out = extractor.extract_blob(blob, meta, [], torch)["records"][0]
    assert "values" not in out["kwargs"]["q"]
    assert out["kwargs"]["k_cache"]["stride"] == [1, 3]
    assert out["kwargs"]["q"]["storage_group"] == out["kwargs"]["k_cache"]["storage_group"]
    assert out["kwargs"]["req_to_token"]["rows"] == [[12, 13, 14, 15, 16, 17]]
    assert out["kwargs"]["topk_idx"]["values"] == [[[1, -1]]]
    assert out["kwargs"]["score_type"] == "max"
    assert "values" not in out["output_contract"]["items"][1]


@pytest.mark.parametrize("attack", ["digest", "missing", "case_drop", "case_order"])
def test_compact_contract_corruption_fails_before_generation(tmp_path, attack):
    ut = tmp_path / "ut"
    meta = install_contract(ut, [simple_record("a"), simple_record("b")])
    path = ut / "generated_cases.json"
    if attack == "missing":
        path.unlink()
    elif attack == "digest":
        path.write_text(path.read_text() + " ")
    else:
        data = json.loads(path.read_text())
        data["records"] = data["records"][:1] if attack == "case_drop" else data["records"][::-1]
        path.write_text(json.dumps(data))
        meta["generated_inputs"]["contract_sha256"] = contract.digest(path)
        (ut / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        contract.load_contract(ut)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_wire_encoding_and_comparator_check_all_values(dtype):
    value = torch.tensor([1, 2, 3], dtype=dtype)
    encoded = contract.encode_output((None, value, None), torch)
    assert contract.compare_output(encoded, encoded, 0.02, torch)
    wrong = contract.encode_output((None, value + 1, None), torch)
    assert not contract.compare_output(wrong, encoded, 0.02, torch)
    wrong = copy.deepcopy(encoded)
    wrong["items"][1]["stride"] = [2]
    assert not contract.compare_output(wrong, encoded, 0.02, torch)
    wrong = copy.deepcopy(encoded)
    wrong["items"][0] = 0
    assert not contract.compare_output(wrong, encoded, 0.02, torch)


@pytest.mark.parametrize("attack", ["seed", "drop", "reorder", "extra_result", "failed"])
def test_parent_rejects_stale_partial_or_failed_worker_outputs(attack):
    result = {"schema_version": 1, "profile": "recorded", "seed": 77, "reference": False,
              "rows": [{"id": "a"}, {"id": "b"}]}
    code = 0
    if attack == "seed":result["seed"] += 1
    if attack == "drop":result["rows"].pop()
    if attack == "reorder":result["rows"].reverse()
    if attack == "failed":code = 1
    stdout = controller.PREFIX + json.dumps(result) + "\n"
    if attack == "extra_result":stdout *= 2
    with pytest.raises(RuntimeError):
        controller.parse_worker(SimpleNamespace(returncode=code, stdout=stdout, stderr="failed"),
                                "recorded", 77, False, ["a", "b"])


def test_parent_never_sends_reference_outputs_to_candidate(tmp_path, monkeypatch):
    ut = tmp_path / "ut"
    meta = install_contract(ut)
    for name in ("generated_contract",):
        (ut / (name + ".py")).write_bytes((generated_helper("minimax", name + ".py")).read_bytes())
    monkeypatch.setattr(controller, "expected_profiles", lambda *_: {
        "recorded": ["case"], "random": ["case"], "replay": ["a", "b", "a"]})
    calls, reports = [], []

    def run_worker(script, arguments, overlay, timeout, candidate, **kwargs):
        assert all("reference.pt" not in argument and "golden" not in argument for argument in arguments)
        assert kwargs.get("attest_files") is None
        profile = arguments[arguments.index("--profile") + 1]
        seed = int(arguments[arguments.index("--seed") + 1])
        reference = "--reference" in arguments
        assert reference is not candidate
        calls.append((reference, profile, seed))
        ids = ["a", "b", "a"] if profile == "replay" else ["case"]
        result = {"schema_version": 1, "profile": profile, "seed": seed, "reference": reference,
                  "rows": [{"id": name, "output": contract.encode_output(torch.tensor([seed % 100]), torch)} for name in ids]}
        return SimpleNamespace(returncode=0, stdout=controller.PREFIX + json.dumps(result), stderr="")

    runner = SimpleNamespace(UT_DIR=ut, TASK_DIR=tmp_path, overlays=lambda: ("frozen", "candidate"),
                             run_worker=run_worker, write_report=lambda name, report: reports.append(report))
    assert controller.run_correctness(runner, {}, 60)[0]
    assert len(calls) == 14  # three recorded draws, three random draws, one replay, two workers each
    assert all(calls[i][0] and not calls[i + 1][0] and calls[i][1:] == calls[i + 1][1:]
               for i in range(0, len(calls), 2))
    assert reports[-1]["status"] == "ok"
    assert not list(tmp_path.rglob("*.pt"))


def test_eager_checks_reject_input_mutation_and_persistent_outputs():
    q = torch.ones(2)
    def mutate(q):
        q.add_(1)
        return q.clone()
    with pytest.raises(RuntimeError, match="read-only"):
        worker.invoke_checked(mutate, (), {"q": q}, contract, torch, [])
    q = torch.ones(2)
    persistent = torch.ones(2)
    previous = []
    worker.invoke_checked(lambda q: persistent, (), {"q": q}, contract, torch, previous)
    with pytest.raises(RuntimeError, match="reused output"):
        worker.invoke_checked(lambda q: persistent, (), {"q": q}, contract, torch, previous)


def test_required_graph_capture_exception_is_not_a_pass(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():return True
        @staticmethod
        def Stream():return SimpleNamespace(wait_stream=lambda _: None)
        current_stream = Stream
        @staticmethod
        def stream(_):return nullcontext()
        @staticmethod
        def synchronize():pass
        @staticmethod
        def CUDAGraph():return object()
        @staticmethod
        def graph(_):raise RuntimeError("capture refused")
    monkeypatch.setattr(torch, "cuda", FakeCuda)
    values = [("a", {"q": torch.ones(2)}), ("b", {"q": torch.zeros(2)})]
    with pytest.raises(RuntimeError, match="capture refused"):
        worker.graph_sequence(lambda q: q.clone(), values, contract, torch)


def test_restricted_extraction_clears_safe_globals_and_never_runs_pickle_payload(tmp_path, monkeypatch):
    import os
    from src.tools.extract_head_kernel_contract import main as extract_main
    marker = tmp_path / "unsafe_was_executed"

    class UnsafePayload:
        def __reduce__(self):
            return (os.system, ("touch " + str(marker),))

    task = tmp_path / "task"
    (task / "ut").mkdir(parents=True)
    (task / "source").mkdir()
    (task / "source/kernel.py").write_text("def kernel(q): return q\n")
    archive = tmp_path / "reference_io.pt"
    torch.save({"payload": UnsafePayload()}, archive)
    (task / "ut/meta.json").write_text(json.dumps({"target_callable": "test:kernel",
        "reference_io_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
    torch.serialization.add_safe_globals([UnsafePayload])
    try:
        with pytest.raises(Exception, match="Weights only load failed"):
            extract_main(["--family", "minimax", "--task", str(task),
                          "--archive", str(archive), "--output", str(tmp_path / "output.json")])
        assert torch.serialization.get_safe_globals() == []
        assert not marker.exists()
        assert not (tmp_path / "output.json").exists()
    finally:
        torch.serialization.clear_safe_globals()


@pytest.mark.parametrize("stale", [False, True])
def test_actual_graph_sequence_changes_inputs_and_returns_to_first(monkeypatch, stale):
    class FakeGraph:
        callback = None
        def replay(self):self.callback()

    class FakeCuda:
        active = None
        @staticmethod
        def is_available():return True
        @staticmethod
        def Stream():return SimpleNamespace(wait_stream=lambda _: None)
        current_stream = Stream
        @staticmethod
        def stream(_):return nullcontext()
        @staticmethod
        def synchronize():pass
        @staticmethod
        def CUDAGraph():return FakeGraph()
        @staticmethod
        @contextmanager
        def graph(value):
            FakeCuda.active = value
            try:yield
            finally:FakeCuda.active = None

    monkeypatch.setattr(torch, "cuda", FakeCuda)
    values = [("a", {"q": torch.tensor([1., 2.])}), ("b", {"q": torch.tensor([3., 4.])})]

    def operation(q):
        result = q * 2
        if FakeCuda.active is not None:
            FakeCuda.active.callback = (lambda: None) if stale else lambda: result.copy_(q * 2)
        return result

    rows = worker.graph_sequence(operation, values, contract, torch)
    expected = [{"id": name, "output": contract.encode_output(kwargs["q"] * 2, torch)}
                for name, kwargs in values + values[:1]]
    if stale:
        with pytest.raises(RuntimeError, match="correctness mismatch"):
            controller.compare_workers(expected, rows, contract, 0.02, torch)
    else:
        controller.compare_workers(expected, rows, contract, 0.02, torch)
    # The static clone must not destroy A when filling B.
    assert torch.equal(values[0][1]["q"], torch.tensor([1., 2.]))


def test_real_worker_subprocess_uses_separate_baseline_and_changed_source(tmp_path):
    import os
    ut = tmp_path / "task/ut"
    install_contract(ut, [simple_record("a"), simple_record("b")])
    (ut / "generated_contract.py").write_bytes((generated_helper("minimax", "generated_contract.py")).read_bytes())
    (ut / "harness_lib.py").write_text("# CPU fixture: no tensor math is performed by this helper.\n")
    (ut / "cases.py").write_text('''import importlib.util, os

def _resolve():
    path = os.environ["TEST_KERNEL_FILE"]
    spec = importlib.util.spec_from_file_location("test_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel
''')
    baseline = tmp_path / "frozen.py"
    candidate = tmp_path / "task/source/kernel.py"
    candidate.parent.mkdir()
    baseline.write_text("def kernel(q, **kwargs): return (None, q * 2, None)\n")
    candidate.write_text(baseline.read_text())
    bootstrap = '''import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("worker",sys.argv[1])
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
result=w.run_profile(__import__("pathlib").Path(sys.argv[2]),"recorded",731,sys.argv[3]=="reference",device="cpu")
print(w.PREFIX+json.dumps(result))
'''

    def execute(path, reference):
        env = dict(os.environ, TEST_KERNEL_FILE=str(path), PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(generated_helper("minimax", "generated_worker.py")),
                               str(ut), "reference" if reference else "candidate"],
                              capture_output=True, text=True, env=env, timeout=30)
        return controller.parse_worker(proc, "recorded", 731, reference, ["a", "b"])

    expected = execute(baseline, True)
    observed = execute(candidate, False)
    controller.compare_workers(expected, observed, contract, 0.02, torch)
    candidate.write_text("def kernel(q, **kwargs): return (None, q * 3, None)\n")
    changed = execute(candidate, False)
    with pytest.raises(RuntimeError, match="correctness mismatch"):
        controller.compare_workers(expected, changed, contract, 0.02, torch)
    assert baseline.read_text() == "def kernel(q, **kwargs): return (None, q * 2, None)\n"
    assert not list(tmp_path.rglob("*.pt"))


def test_floating_tolerance_matches_captured_harness():
    path = TASKS[0] / "ut/harness_lib.py"
    spec = importlib.util.spec_from_file_location("original_minimax_harness", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    for dtype in (torch.float32, torch.bfloat16):
        reference = torch.tensor([100, 200], dtype=dtype)
        for delta in (0, 1, 10):
            observed = reference + delta
            expected_ok, _ = legacy.correct(observed, reference, 0.02)
            assert contract.compare_output(contract.encode_output(observed, torch),
                                           contract.encode_output(reference, torch), 0.02, torch) == expected_ok


def test_captured_paging_and_prefill_topk_match_original_compact_geometry():
    for task in TASKS:
        data = contract.load_contract(task / "ut")
        geometry = json.loads((task / "ut/timing_geometry.json").read_text())["records"]
        by_signature = {row["sig"]: row for row in geometry}
        for record in data["records"]:
            geo = by_signature[record["sig"]]
            assert geo["req_rows"]["values"] == record["kwargs"]["req_to_token"]["rows"]
            if "topk_idx" in geo:
                assert geo["topk_idx"]["values"] == record["kwargs"]["topk_idx"]["values"]


def test_candidate_output_must_remain_on_the_execution_device():
    with pytest.raises(RuntimeError, match="execution device"):
        worker.invoke_checked(lambda q: torch.empty(2, device="meta"), (), {"q": torch.ones(2)},
                              contract, torch, [])
