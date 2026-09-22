"""Exercise the real RoPE harness without claiming CPU fixtures are GPU runs."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_flydsl_task_migration_v2 import (
    ROOT, _harness_functions, _RemoveStandardQuantChecks, module,
)

TASK = ROOT / "tasks/torch2flydsl/rope_2d_fwd_kernel"
ORIGINAL_HARNESS_SHA256 = {
    "_make_inputs": "3adbbc326b883377ff2c0183ec720141fc98beaca189e999df433ff463c8c811",
    "_aiter_op": "b4fd592e43ecca5a89dffe04b6b8ae741187a937f3c5bceb87ebf7aeb434c3be",
    "run_correctness": "d6076447e989e1fb05d19222c0c5134aa850c5d9aaa5c49cf255d63458ef58a3",
    "_mean_ms": "d577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be",
    "run_benchmark": "47063e42afc6f8481fe7a03d7f2ee3fdb933793695c934ed0c4291264fe34067",
    "arena_benchmark": "e5c1e6fab69d69a3444736df6950b0c357292c24473b12f633005069d3b477b6",
}


def comparator():
    ns = {"REL_TOL": 0.01, "PASS_PCT": 99.9}
    _harness_functions(TASK, {"_compare"}, ns)
    return ns["_compare"]


@pytest.mark.parametrize("defect", ["sparse_outlier", "low_pass_rate"])
def test_rope_2d_requires_both_original_error_bounds(defect):
    reference = torch.ones(2000, dtype=torch.bfloat16)
    actual = reference.clone()
    if defect == "sparse_outlier":
        actual[0] = 1e20
    else:
        reference[0] = actual[0] = 1000
        actual[1:] = 1.1
    ok, worst, percentage = comparator()(reference, actual)
    assert not ok
    # The previous OR accepted both defects. Neither threshold changed.
    assert worst <= 0.01 or percentage >= 99.9
    assert comparator()(reference, reference)[0]


@pytest.mark.parametrize("defect", ["shape", "dtype", "device", "none", "nan", "inf"])
def test_rope_2d_rejects_invalid_output_contract(defect):
    reference = torch.ones((2, 4), dtype=torch.bfloat16)
    actual = reference.clone()
    if defect == "shape": actual = actual[:1]
    if defect == "dtype": actual = actual.float()
    if defect == "device": actual = actual.to("meta")
    if defect == "none": actual = None
    if defect in {"nan", "inf"}: actual[0, 0] = float(defect)
    if defect in {"nan", "inf"}:
        assert not comparator()(reference, actual)[0]
    else:
        with pytest.raises(AssertionError, match="shape/dtype/device"):
            comparator()(reference, actual)


@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("provided", [True, False])
@pytest.mark.parametrize("defect", ["none", "wrong_timed", "wrong_replay", "cached",
                                  "unwritten", "input_mutated", "angle_mutated", "raise_replay"])
def test_rope_2d_actual_timing_and_replay(function, provided, defect, monkeypatch, tmp_path):
    model_module = module(TASK / "model.py")
    checks = module(TASK / "scripts/replay_checks.py")
    collector = module(ROOT / "src/tools/perf/aka_benchmark.py").TimedRun
    shape = {"name": "controlled", "b": 1, "height": 2, "width": 2, "h": 1, "d": 8}
    x = torch.arange(32).reshape(1, 4, 1, 8).to(torch.bfloat16)
    ch, sh = model_module._build_cos_sin_2d(2, 8)
    cw, sw = model_module._build_cos_sin_2d(2, 8)
    inputs = (x, ch, sh, cw, sw)
    pristine = tuple(value.clone() for value in inputs)
    oracle = model_module.Model(2, 2)
    cached = oracle(*inputs)
    phase = {"name": "preparation"}

    def compute(is_model):
        output = oracle(*inputs)
        if is_model == provided:
            if defect == "wrong_timed" and phase["name"] == "timed": output.zero_()
            if phase["name"] == "replay":
                if defect == "wrong_replay": output.zero_()
                if defect == "cached": output = cached.clone()
                if defect == "input_mutated": inputs[0].add_(1)
                if defect == "angle_mutated": inputs[1].add_(1)
        return output

    class Model:
        def __init__(self, *args): pass
        def to(self, *args): return self
        def __call__(self, *args): return compute(True)

    mmod = SimpleNamespace(Model=Model)
    kmod = SimpleNamespace(flydsl_rope_2d_fwd=lambda *args: compute(False))
    calls = []

    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition))
        phase["name"] = "timed"
        output = fn()
        phase["name"] = "preparation"

        def replay():
            assert torch.isnan(output).all(), "Actual measured output must be poisoned"
            if defect == "raise_replay": raise RuntimeError("controlled replay failure")
            phase["name"] = "replay"
            try:
                if defect != "unwritten": output.copy_(fn())
            finally:
                phase["name"] = "preparation"
            return output

        timed_run._bind(replay, output)
        return 0.1, {"benchmark_method": "cuda_graph"}

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    ns = dict(TimedRun=collector, benchmark_cuda_graph_or_events=benchmark,
              require_unchanged=checks.require_unchanged, verify_timed_run=checks.verify_timed_run,
              _make_inputs=lambda *args: inputs, _aiter_op=lambda *args: oracle(*args[:5]),
              _load_module=lambda directory, filename, alias: mmod if filename == "model.py" else None if provided else kmod,
              _KERNEL_DIR=str(tmp_path), MODEL_FILE="model.py", KERNEL_FILE="kernel.py",
              KERNEL_ENTRY="flydsl_rope_2d_fwd", SHAPES=[shape], REL_TOL=0.01,
              PASS_PCT=99.9, math=math, json=json, Path=Path)
    _harness_functions(TASK, {function, "_compare", "_mean_ms", "_rope_replay_validator"}, ns)
    if defect == "none":
        report = ns[function](verbose=False)
        if function == "run_benchmark":
            report = json.loads((tmp_path / "build/performance_report.json").read_text())
        assert calls == [(10, 100)] * (2 if provided else 3)
        assert report[0]["timed_output_correctness"] == report[0]["replay_correctness"] == "PASS"
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            ns[function](verbose=False)
    checks.require_unchanged(inputs, pristine)


def test_rope_2d_original_inputs_model_and_timing_are_preserved():
    path = TASK / "test_kernel_harness.py"
    current = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    for name, digest in ORIGINAL_HARNESS_SHA256.items():
        restored = _RemoveStandardQuantChecks().visit(current[name])
        assert hashlib.sha256(ast.dump(restored, include_attributes=False).encode()).hexdigest() == digest, name
    files = {
        "model.py": "dfcd006f67b8386be3a11f314ed3f64547a3b3413c60ae05d70a1a515cb55c6a",
        "cases.json": "bb61e2c7be56c7eac0d8f61fcb1646e54bd378c12030126f2c071696b68ac5d4",
    }
    for relative, digest in files.items():
        assert hashlib.sha256((TASK / relative).read_bytes()).hexdigest() == digest
