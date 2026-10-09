"""Functional-package timing protocol and candidate policy on CPU; simulated timing is not GPU evidence."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from src.perf_helper_materialization import (
    canonical_aka_helper,
    materialize_perf_helpers_in_workspace,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tasks/Aiter-task"
A8W8 = SUITE / "gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k256"
MHC = SUITE / "mhc_fused_post_pre_flat_rmsnorm_c4_d4096"


@contextmanager
def materialized(task, tmp_path, monkeypatch):
    workspace = tmp_path / "task"
    shutil.copytree(task, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    assert (workspace / "scripts/_aka_benchmark.py").read_text() == canonical_aka_helper(ROOT)
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(workspace))
        patch.syspath_prepend(str(workspace / "scripts"))
        for name in list(sys.modules):
            if name == "scripts" or name.startswith(("scripts.", "_sikl_")) or name == "_aka_benchmark":
                patch.delitem(sys.modules, name)
        try:
            yield workspace, importlib.import_module("scripts.task_runner"), importlib.import_module("_aka_benchmark")
        finally:
            for name in list(sys.modules):
                if name == "scripts" or name.startswith(("scripts.", "_sikl_")) or name == "_aka_benchmark":
                    sys.modules.pop(name, None)


def simulated_graph_benchmark(helper, kernel, repeats=1):
    """The canonical helper's contract on CPU: one replay per sample, prepared by
    ``prepare_fn`` before its start, ``after_sample`` after its end, and
    ``rerun_ms`` through the same path."""
    def benchmark(fn, *, warmup, repetition, target_ms, prepare_fn, timed_run):
        assert type(timed_run) is helper.TimedRun
        for _ in range(warmup):
            prepare_fn()
            fn()

        def sample():
            prepare_fn()
            return fn(), kernel.cost
        costs = []
        for _ in range(repetition):
            outputs, cost = sample()
            costs.append(cost)
            timed_run.after_sample(outputs)
        timed_run._bind(lambda: sample()[0], outputs, sample)
        return sum(costs) / len(costs), {"benchmark_method": "cuda_graph",
                                         "benchmark_effective_repeats": repeats}
    return benchmark


class Honest:
    def __init__(self, reference):
        self.reference, self.cost, self.calls, self.out = reference, None, 0, None

    def compute(self, values):
        self.cost = 1.0
        self.out = self.reference(**values)
        return self.out

    def __call__(self, **values):
        self.calls += 1
        return self.compute(values)


class ValueMemo(Honest):
    """Returns a stored, correct result for any call-varying draw seen before."""
    def __init__(self, reference):
        super().__init__(reference)
        self.store = {}

    def __call__(self, **values):
        import torch
        key = b"".join(v.contiguous().view(torch.uint8).numpy().tobytes()
                       for v in values.values() if isinstance(v, torch.Tensor))
        if key in self.store:
            self.cost = 0.1
            return self.store[key].clone()
        self.store[key] = self.compute(values).clone()
        return self.store[key]


class SkipsEveryThirdCall(Honest):
    def __call__(self, **values):
        self.calls += 1
        if self.out is not None and self.calls % 3 == 0:
            self.cost = 0.01
            return self.out
        return self.compute(values)


class SlightlyWrong(Honest):
    def compute(self, values):
        return super().compute(values) * 1.5


def patch_benchmark(helper, monkeypatch, kernel_type, repeats=1):
    """Route the canonical benchmark through the simulation, timing ``kernel_type``."""
    holder = {}

    def make(reference):
        holder["kernel"] = kernel_type(reference)
        return holder["kernel"]
    monkeypatch.setattr(helper, "benchmark_cuda_graph_or_events",
                        lambda *a, **kw: simulated_graph_benchmark(helper, holder["kernel"], repeats)(*a, **kw))
    return make


def run_time_row(workspace, runner, monkeypatch, kernel_type, *, role="candidate", diagnostic=False):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    from scripts.task_api import load_solution
    from scripts.task_inputs import make_inputs
    import scripts.task_timing as timing
    contract = json.loads((workspace / "scripts/workload.json").read_text())
    definition, policy, row = contract["definition"], contract["policy"], contract["rows"][1]
    reference = load_solution(workspace / "scripts/reference", "main.py::run")
    values = make_inputs(definition, row, policy, device="cpu")
    monkeypatch.setattr(timing, "choose_checked_samples", lambda repetition, count: list(range(count)))
    kernel = kernel_type(reference)
    return kernel, timing.time_row(kernel, reference, values, definition, row, policy, role=role,
                                   baseline_diagnostic=diagnostic, device="cpu")


@pytest.mark.parametrize("kernel_type, status, failure_kind", [
    (Honest, "PASS", None),
    (ValueMemo, "FAIL", "timing_input_memoized"),
    (SkipsEveryThirdCall, "FAIL", "numerical_mismatch"),
])
def test_timing_rejects_known_timed_path_exploits_and_accepts_honest_timing(
        kernel_type, status, failure_kind, tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        make = patch_benchmark(helper, monkeypatch, kernel_type)
        _, result = run_time_row(workspace, runner, monkeypatch, make)
        assert result["status"] == status, result.get("reason")
        assert result.get("failure_kind") == failure_kind
        if failure_kind != "timing_input_memoized":
            checked = result["metadata"]["timed_output_correctness"]["metadata"]["checked_invocations"]
            assert checked == 8 + 4


def test_honest_multi_output_timing_passes(tmp_path, monkeypatch):
    with materialized(MHC, tmp_path, monkeypatch) as (workspace, runner, helper):
        make = patch_benchmark(helper, monkeypatch, Honest)
        _, result = run_time_row(workspace, runner, monkeypatch, make)
        assert result["status"] == "PASS", result.get("reason")


def test_batched_capture_is_rejected(tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        make = patch_benchmark(helper, monkeypatch, Honest, repeats=8)
        _, result = run_time_row(workspace, runner, monkeypatch, make)
        assert result["status"] == "FAIL" and result["failure_kind"] == "timing_protocol"


@pytest.mark.parametrize("mutated", ["a", "b", "a_scale", "b_scale"])
def test_timing_rejects_mutation_of_each_input(mutated, tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        class Mutating(Honest):
            def compute(self, values):
                import torch
                out = super().compute(values)
                # An increment, unlike a bit flip, does not undo itself on the next call.
                values[mutated].view(torch.uint8).flatten()[0] += 1
                return out

        make = patch_benchmark(helper, monkeypatch, Mutating)
        with pytest.raises(RuntimeError, match=f"protected input tensor: {mutated}$"):
            run_time_row(workspace, runner, monkeypatch, make)


@pytest.mark.parametrize("role, diagnostic, status", [
    ("baseline", True, "PASS"), ("baseline", False, "FAIL"), ("candidate", True, "FAIL")])
def test_numerical_diagnostic_applies_only_to_a_declared_baseline(role, diagnostic, status, tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        make = patch_benchmark(helper, monkeypatch, SlightlyWrong)
        _, result = run_time_row(workspace, runner, monkeypatch, make, role=role, diagnostic=diagnostic)
        assert result["status"] == status
        checked = result["metadata"]["timed_output_correctness"]
        assert checked["status"] == "FAIL" and checked["failure_kind"] == "numerical_mismatch"
        if status == "PASS":
            assert result["metadata"]["baseline_numerical_diagnostic"]


@pytest.mark.parametrize("source", [
    "import aiter",
    "from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle",
    "import triton",
    "import scripts.task_api",
    "from . import helper",
    "import torch\ndef f(a, b):\n    return torch.matmul(a, b)",
    "def f(a, b):\n    return a @ b",
    "import torch\ndef f(a, b, s, t):\n    return torch._scaled_mm(a, b, s, t)",
    "import torch\ndef f(x):\n    return torch.sigmoid(x)",
    "from torch.nn.functional import rms_norm",
    "def f():\n    return __import__('aiter')",
])
def test_candidate_guard_rejects_protected_and_library_computation(source, tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        with pytest.raises(RuntimeError):
            runner.assert_source_independent(source)


def test_candidate_guard_allows_flydsl_and_plumbing(tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        runner.assert_source_independent(
            "import torch\nimport flydsl.expr as fx\nfrom flydsl.expr import math\n"
            "def build(**axes):\n    return lambda **inputs: torch.empty((axes['m'], axes['n']))\n"
            "def g(x):\n    return math.rsqrt(x)\n")


def test_initial_state_reads_the_declared_builder_without_executing_it(tmp_path, monkeypatch):
    import yaml
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        config = yaml.safe_load((workspace / "config.yaml").read_text())
        symbol = config["candidate"]["entrypoints"][0]["symbol"]
        kernel = workspace / "kernel.py"
        assert runner.initial_state(config) == "unimplemented"
        with pytest.raises(RuntimeError, match="unimplemented"):
            runner.load_builder(config)
        kernel.write_text(f"raise SystemExit('executed')\ndef {symbol}(**axes):\n    pass\n")
        assert runner.initial_state(config) == "implemented"
        kernel.write_text("import torch\nX = 1\n")
        with pytest.raises(RuntimeError, match="does not define its builder"):
            runner.initial_state(config)


def test_launch_takes_row_axes_and_positional_inputs_in_definition_order(tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        contract = json.loads((workspace / "scripts/workload.json").read_text())
        definition, row = contract["definition"], contract["rows"][3]
        seen = {}

        def builder(**axes):
            seen["axes"] = axes
            return lambda *inputs: inputs
        launch = runner.build_launch(builder, definition, row)
        values = {name: f"<{name}>" for name in reversed(list(definition["inputs"]))}
        assert launch(**values) == tuple(f"<{name}>" for name in definition["inputs"])
        assert seen["axes"] == {"m": 8, "n": 4096, "k": 256, "sn": 32, "sk": 2}


def test_exported_binding_derives_axes_from_shapes_and_calls_positionally(tmp_path, monkeypatch):
    import types
    import yaml
    with materialized(MHC, tmp_path, monkeypatch) as (workspace, runner, helper):
        exporter = importlib.import_module("scripts.export_solution")
        config = yaml.safe_load((workspace / "config.yaml").read_text())
        contract = json.loads((workspace / "scripts/workload.json").read_text())
        definition = contract["definition"]
        entry = config["candidate"]["entrypoints"][0]
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "kernel.py").write_text(
            f"def {entry['symbol']}(**axes):\n    return lambda *inputs: (axes, inputs)\n")
        (artifact / "sikl_entry.py").write_text(exporter.tensor_entry(entry, definition))
        spec = importlib.util.spec_from_file_location("mhc_binding_unit_test", artifact / "sikl_entry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = contract["rows"][2]
        from scripts.task_api import dimensions, shape_of
        dims = dimensions(definition, row)
        args = [types.SimpleNamespace(shape=shape_of(spec_, dims)) if spec_.get("shape") is not None
                else row["workload"]["inputs"][name]["value"] for name, spec_ in definition["inputs"].items()]
        axes, inputs = module.run(*args)
        assert axes == dims and list(inputs) == args


def test_export_requires_a_framework_accepted_complete_result(tmp_path, monkeypatch):
    with materialized(A8W8, tmp_path, monkeypatch) as (workspace, runner, helper):
        exporter = importlib.import_module("scripts.export_solution")
        accepted = {"pass_compilation": True, "pass_correctness": True, "pass_tool_gate": True,
                    "workload_consistent": True, "benchmark_method_consistent": True,
                    "valid_baseline_cases": 13, "valid_optimized_cases": 13,
                    "best_optimized_execution_time": 0.01}
        exporter.accepted_result(accepted, 13)
        for field, value in (("pass_correctness", False), ("valid_optimized_cases", 12),
                             ("best_optimized_execution_time", float("nan"))):
            with pytest.raises(ValueError):
                exporter.accepted_result({**accepted, field: value}, 13)
