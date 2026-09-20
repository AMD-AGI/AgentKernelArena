"""CPU contract/binding regressions; these do not establish GPU correctness."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from head_kernel_test_utils import task_directory

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks/head_kernels"
BF16 = task_directory("glm-5.3-flash__gemm_a16w16_bf16_cijk")
FP8 = task_directory("glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle")


def load(task, filename):
    name = f"_glm_test_{task.name}_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, task / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task,count,families", [
    (BF16, 27, {(4096, 1536), (3072, 4096), (4096, 1024), (288, 4096),
                (128, 4096), (32, 4096), (8, 4096), (1024, 128), (4096, 512)}),
    (FP8, 21, {(2048, 1536), (4096, 1536), (3072, 4096), (512, 4096),
                (4096, 2048), (4096, 256), (2048, 4096)}),
])
def test_exact_upstream_cases(task, count, families):
    meta = json.loads((task / "ut/meta.json").read_text())
    original = json.loads((task / "ut/provenance/workload.json").read_text())
    cases = meta["cases"]
    assert len(cases) == count
    assert len({row["sig"] for row in cases}) == count
    assert {(row["n"], row["k"]) for row in cases} == families
    assert {(row["m"], row["n"], row["k"]) for row in cases} == {
        (m, n, k) for n, k in families for m in (1, 64, 8192)}
    assert [(row["sig"], row["m"], row["n"], row["k"], row["regime"])
            for row in cases] == [
        (row["name"], row["dims"][0][0], row["dims"][1][0], row["dims"][0][1], row["regime"])
        for row in original["cases"]]
    assert meta["synthesized"] is True
    assert not meta["reference_io_sha256"]


@pytest.mark.parametrize("task", [BF16, FP8])
def test_stock_function_body_and_signature(task):
    meta = json.loads((task / "ut/meta.json").read_text())
    stock = ast.parse((task / "ut/baseline_ref" / Path(meta["source_path"]).name).read_text())
    candidate = ast.parse((task / "source/kernel.py").read_text())
    original = next(node for node in stock.body if isinstance(node, ast.FunctionDef)
                    and node.name == meta["symbol"])
    edited = next(node for node in candidate.body if isinstance(node, ast.FunctionDef)
                  and node.name == meta["symbol"])
    # The stock FP8 torch.compile decorator is deliberately omitted for the
    # eager operator excerpt; its ABI and body remain verbatim.
    original.decorator_list = []
    assert ast.dump(original) == ast.dump(edited)
    harness = load(task, "ut/harness_lib.py")
    harness.verify_hash(task / "ut/baseline_ref" / Path(meta["source_path"]).name,
                        meta["source_sha256"])


def test_bf16_binding_reaches_import_time_solmap():
    harness = load(BF16, "ut/harness_lib.py")
    original = lambda *args: "native"
    candidate = lambda *args: "negative-control"
    other = lambda *args: "other-backend"
    module = SimpleNamespace(torch_gemm=original,
                             solMap={"torch": original, "other": other})
    # Demonstrate the trap that this regression guards.
    module.torch_gemm = candidate
    assert module.solMap["torch"]() == "native"
    harness.bind_candidate("bf16", original, candidate, module, module, {})
    assert module.solMap["torch"]() == "negative-control"
    assert module.solMap["other"] is other


def test_bf16_rejects_an_unexpected_dispatch_table():
    harness = load(BF16, "ut/harness_lib.py")
    module = SimpleNamespace(solMap={"torch": lambda: None})
    with pytest.raises(RuntimeError, match="solMap"):
        harness.bind_candidate("bf16", lambda: None, lambda: None, module, module, {})


def test_candidate_file_edit_is_the_function_executed_by_solmap(tmp_path, monkeypatch):
    """Exercise the real file loader and serving lookup, without GPU packages."""
    harness = load(BF16, "ut/harness_lib.py")
    task = tmp_path / "task"
    (task / "ut").mkdir(parents=True)
    (task / "source").mkdir()
    (task / "source/kernel.py").write_text("def torch_gemm(*args, **kwargs):\n    return 'edited-source'\n")
    original = lambda *args, **kwargs: "native"
    runtime = SimpleNamespace(torch_gemm=original, solMap={"torch": original})
    monkeypatch.setattr(harness, "HERE", task / "ut")
    monkeypatch.setattr(harness, "native_function", lambda: original)
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", runtime)
    candidate = harness.candidate_function()
    assert candidate() == "edited-source"
    assert runtime.solMap["torch"]() == "edited-source"
    assert runtime.torch_gemm is candidate


def test_fp8_preshuffle_byte_permutation_and_attribute(monkeypatch):
    """Execute the actual permutation using a small pure-Python tensor model."""
    import itertools
    import math

    class Tensor:
        dtype = "fp8"

        def __init__(self, data, shape):
            self.data, self.shape = list(data), tuple(shape)

        def view(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], str):
                return Tensor(self.data, self.shape)
            assert math.prod(shape) == len(self.data)
            return Tensor(self.data, shape)

        def permute(self, *axes):
            shape = [self.shape[axis] for axis in axes]
            reordered = []
            for indices in itertools.product(*(range(size) for size in shape)):
                original = [0] * len(axes)
                for axis, index in zip(axes, indices):
                    original[axis] = index
                offset = 0
                for size, index in zip(self.shape, original):
                    offset = offset * size + index
                reordered.append(self.data[offset])
            return Tensor(reordered, shape)

        def contiguous(self):
            return self

    harness = load(FP8, "ut/harness_lib.py")
    monkeypatch.setitem(sys.modules, "harness_lib", harness)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(uint8="uint8"))
    cases = load(FP8, "ut/cases.py")
    n, k = 32, 64
    output = cases.shuffle_weight(Tensor(range(n * k), (n, k)))
    # Independently enumerate each 16x16 tile, in tile-row/tile-column order.
    expected = [row * k + column for tile_row in range(0, n, 16)
                for tile_column in range(0, k, 16)
                for row in range(tile_row, tile_row + 16)
                for column in range(tile_column, tile_column + 16)]
    assert output.data == expected
    assert output.shape == (n, k)
    assert output.is_shuffled is True


def test_fp8_binding_reaches_package_and_sglang_aliases_without_wrapping_ck():
    harness = load(FP8, "ut/harness_lib.py")
    symbol = "gemm_a8w8_blockscale_bpreshuffle"
    original = lambda: "native"
    candidate = lambda: "negative-control"
    ck = lambda: "native-ck"
    package = SimpleNamespace(**{symbol: original})
    native = SimpleNamespace(**{symbol: original, symbol + "_ck": ck})
    sglang = SimpleNamespace(**{symbol: original})
    harness.bind_candidate("fp8", original, candidate, package, native,
                           {"sglang.srt.layers.quantization.fp8_utils": sglang})
    assert getattr(package, symbol)() == "negative-control"
    assert getattr(native, symbol)() == "negative-control"
    assert getattr(sglang, symbol)() == "negative-control"
    assert getattr(native, symbol + "_ck") is ck


def test_fp8_dispatch_is_frozen_to_the_native_revision(tmp_path, monkeypatch):
    harness = load(FP8, "ut/harness_lib.py")
    monkeypatch.setenv("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", "ignored-user-table")
    harness.configure_dispatch()
    import os
    assert Path(os.environ["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"]) == FP8 / "ut/dispatch.csv"
    bad = tmp_path / "bad.csv"
    bad.write_text("changed dispatch table")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        harness.verify_hash(bad, harness.META["dispatch_sha256"])


@pytest.mark.parametrize("task", [BF16, FP8])
def test_cases_reject_missing_or_reordered_cases(task, monkeypatch):
    harness = load(task, "ut/harness_lib.py")
    monkeypatch.setitem(sys.modules, "harness_lib", harness)
    cases = load(task, "ut/cases.py")
    assert len(cases.correctness_cases(cases.META)) in (21, 27)
    assert len(cases.selected_cases(cases.META, cases.META["ledger_ids"])) == 7
    with pytest.raises(RuntimeError, match="complete fixed"):
        cases.selected_cases(cases.META, cases.META["ledger_ids"][:-1])
    with pytest.raises(RuntimeError, match="complete fixed"):
        cases.selected_cases(cases.META, cases.META["ledger_ids"][::-1])


@pytest.mark.parametrize("task", [BF16, FP8])
def test_tasks_use_common_harness_and_declare_synthetic_values(task):
    config = yaml.safe_load((task / "config.yaml").read_text())
    assert config["headkernel"]["synthetic_values"] is True
    assert config["headkernel"]["captured_tensor_oracle"] is False
    assert config["platform_support"]["required_arch"] == "gfx950"
    assert config["source_file_path"] == ["source/kernel.py"]
    for filename in ("_bench.py", "task_runner.py", "runtime_preflight.py",
                     "runtime_integrity.py", "_trusted_worker.py"):
        assert (task / "scripts" / filename).read_bytes() == (TASKS / "_support" / filename).read_bytes()


OBSERVED_CASES = {
    "bf16": ["nk128x4096_m64_decode", "nk3072x4096_m64_decode",
             "nk288x4096_m64_decode", "nk1024x128_m64_decode",
             "nk8x4096_m64_decode", "nk4096x1024_m64_decode",
             "nk4096x1536_m64_decode"],
    "fp8": ["nk512x4096_m64_decode", "nk4096x256_m64_decode",
            "nk2048x4096_m64_decode", "nk2048x1536_m64_decode",
            "nk4096x2048_m64_decode", "nk3072x4096_m64_decode",
            "nk4096x1536_m64_decode"],
}


@pytest.mark.parametrize("task", [BF16, FP8])
def test_only_observed_rows_are_scored_and_counts_keep_their_scope(task):
    meta = json.loads((task / "ut/meta.json").read_text())
    original = json.loads((task / "ut/provenance/workload.json").read_text())
    assert meta["ledger_ids"] == OBSERVED_CASES[meta["kind"]]
    source_hash = hashlib.sha256((task / "ut/provenance/workload.json").read_bytes()).hexdigest()
    assert meta["benchmark_case_policy"]["source_sha256"] == source_hash
    assert meta["correctness_case_ids"] == [row["sig"] for row in meta["cases"]]
    for case, source in zip(meta["cases"], original["cases"]):
        evidence = case["scenario_evidence"]
        observed = case["sig"] in meta["ledger_ids"]
        assert evidence["scored"] is observed
        assert evidence["correctness_required"] is True
        assert evidence["source_case_id"] == source["name"]
        assert evidence["source_sha256"] == source_hash
        assert evidence["source_count"] == source["count"]
        assert evidence["source_weight_source"] == source["weight_source"]
        assert evidence["observed_invocation_count"] == (source["count"] if observed else None)
        if observed:
            assert case["m"] == 64
            assert evidence["classification"] == "observed_profile_shape"
        else:
            assert evidence["classification"] in ("inferred_m_bucket", "unprofiled_nk_family")
        if meta["kind"] == "fp8" and not observed:
            assert source["weight_source"] == "trace"
            assert "only M64 was observed" in evidence["source_label_caveat"]
    counts = [row["scenario_evidence"]["observed_invocation_count"]
              for row in meta["cases"] if row["scenario_evidence"]["scored"]]
    assert counts == ([None] * 7 if meta["kind"] == "bf16" else
                      [4410, 4410, 1155, 1155, 1155, 315, 315])
    assert meta["benchmark_case_policy"]["uses_observed_invocation_weights"] is False
    if meta["kind"] == "fp8":
        assert any("tuning table" in gap for gap in
                   meta["benchmark_case_policy"]["remaining_evidence_gaps"])


@pytest.mark.parametrize("task", [BF16, FP8])
def test_common_benchmark_adapter_gets_only_observed_cases(task, monkeypatch):
    harness = load(task, "ut/harness_lib.py")
    monkeypatch.setitem(sys.modules, "harness_lib", harness)
    cases = load(task, "ut/cases.py")
    bench = load(task, "scripts/_bench.py")
    # Exercise the unchanged adapter and real selector without building GPU inputs.
    monkeypatch.setattr(cases, "timing_case", lambda case: {"sig": case["sig"]})
    rows, call = bench.selected_cases(cases, harness, cases.META, None, True)
    assert [row["sig"] for row in rows] == OBSERVED_CASES[cases.META["kind"]]
    assert call is cases.baseline_call


@pytest.mark.parametrize("task", [BF16, FP8])
def test_robustness_coverage_cannot_be_dropped_or_enter_timing(task, monkeypatch):
    harness = load(task, "ut/harness_lib.py")
    monkeypatch.setitem(sys.modules, "harness_lib", harness)
    cases = load(task, "ut/cases.py")
    all_cases = cases.correctness_cases(cases.META)
    assert len(all_cases) == (27 if cases.META["kind"] == "bf16" else 21)
    for case in all_cases:
        if not case["scenario_evidence"]["scored"]:
            with pytest.raises(RuntimeError, match="unscored robustness"):
                cases.timing_case(case)
    incomplete = copy.deepcopy(cases.META)
    incomplete["cases"].pop()
    with pytest.raises(RuntimeError, match="complete fixed GLM correctness"):
        cases.correctness_cases(incomplete)
    with pytest.raises(RuntimeError, match="complete fixed observed"):
        cases.selected_cases(cases.META, cases.META["correctness_case_ids"])
    # Correctness's executable entrypoint must retain the all-case selector.
    tree = ast.parse((task / "ut/unittest.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "main")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "correctness_cases" for node in ast.walk(main))
