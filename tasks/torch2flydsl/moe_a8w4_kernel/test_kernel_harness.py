#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Protected A8W4 MoE oracle, input contract and cold-cache measurement."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
REL_TOL = 1e-2
SEED = 20260401


def load_kernel():
    spec = importlib.util.spec_from_file_location("arena_a8w4_candidate", ROOT / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest():
    return json.loads((ROOT / "cases.json").read_text())["cases"]


def reference_controls():
    """Untimed independent known answers, including the BF16 output range."""
    import torch
    from model import Model, _mxfp4_dequant
    values = torch.full((1, 32), 1.5, dtype=torch.bfloat16)
    values[0, 0] = 6
    expected = torch.full((1, 32), 2.0)
    expected[0, 0] = 6
    if not torch.equal(_mxfp4_dequant(values), expected):
        raise AssertionError("FP4 scale boundary known answer failed")
    model = Model(64, 64, 2, 2).eval()
    with torch.no_grad():
        model.w1.fill_(1)
        model.w2.fill_(1)
        actual = model.forward_with_routing(
            torch.ones(1, 64, dtype=torch.bfloat16), torch.full((1, 2), 0.5),
            torch.tensor([[0, 1]], dtype=torch.int32))
    expected = torch.full((1, 64), 262144, dtype=torch.bfloat16)
    if not torch.equal(actual, expected):
        raise AssertionError("BF16 wide-range known answer failed")
    for wrong in (torch.zeros_like(expected), expected * 2,
                  torch.full_like(expected, float("inf"))):
        try:
            compare(wrong, expected)
        except AssertionError:
            continue
        raise AssertionError("Negative output was accepted")
    return [{"name": "fp4_scale_and_bf16_range", "known_answer": "PASS",
             "negative_output": "rejected"}]


def compare(actual, expected):
    import torch
    if not isinstance(actual, torch.Tensor):
        raise AssertionError("MoE must return one Tensor")
    if (actual.shape, actual.dtype, actual.device) != (expected.shape, expected.dtype, expected.device):
        raise AssertionError("MoE output shape/dtype/device mismatch")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Nonfinite MoE output/reference")
    scale = max(expected.float().abs().max().item(), 1e-9)
    error = (actual.float() - expected.float()).abs().max().item() / scale
    if error > REL_TOL:
        raise AssertionError(f"normalized max error {error} exceeds {REL_TOL}")
    return {"normalized_max_error": error}


class Fixture:
    def __init__(self, row):
        import torch
        from model import Model, route_topk
        s = row["params"]
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        self.model = Model(s["model_dim"], s["inter_dim"], s["experts"], s["topk"]).eval()
        hidden = torch.randn(s["tokens"], s["model_dim"], device="cuda", dtype=torch.bfloat16)
        if s.get("input") == "ones":
            with torch.no_grad():
                hidden.fill_(1)
                self.model.w1.fill_(1)
                self.model.w2.fill_(1)
        # Routing and the independent oracle run on CPU, outside all timing.
        # A GPU BLAS loader is not part of the FlyDSL operator contract.
        with torch.no_grad():
            weights, ids = route_topk(self.model.gate(hidden.cpu()), s["topk"])
            self.inputs = (hidden, self.model.w1.detach().cuda(), self.model.w2.detach().cuda(),
                           weights.cuda(), ids.cuda())
            self.originals = tuple(x.clone() for x in self.inputs)
            self.expected = self.reference()
        self.signatures = tuple((x.shape, x.stride(), x.dtype, x.device, x.data_ptr())
                                for x in self.inputs)

    def reference(self, *, weight_draw=False):
        import torch
        hidden, _, _, weights, ids = self.inputs
        with torch.no_grad():
            factors = (-1.0, 0.5) if weight_draw else (1.0, 1.0)
            return self.model.forward_with_routing(
                hidden.cpu(), weights.cpu(), ids.cpu(), weight_factors=factors).cuda()

    def call(self, kernel):
        # Exactly the same reduction path is used for correctness and timing.
        return kernel.flydsl_moe_a8w4(*self.inputs, mode="reduce")

    def unchanged(self, snapshots=None):
        import torch
        snapshots = self.originals if snapshots is None else snapshots
        for value, pristine, signature in zip(self.inputs, snapshots, self.signatures):
            if (value.shape, value.stride(), value.dtype, value.device, value.data_ptr()) != signature:
                raise AssertionError("Input metadata/storage changed")
            if not torch.equal(value.view(torch.uint8), pristine.view(torch.uint8)):
                raise AssertionError("Read-only MoE input changed")

    def check(self, output, expected=None):
        import torch
        for value in self.inputs:
            if isinstance(output, torch.Tensor) and torch._C._overlaps(output, value):
                raise AssertionError("Output aliases a read-only input")
        return compare(output, self.expected if expected is None else expected)

    def verify_replay(self, timed):
        import torch
        if not timed.bound:
            raise AssertionError("Actual measured graph/output is unavailable")
        self.unchanged()
        self.check(timed.outputs)
        for change in ("hidden", "routing", "weights"):
            try:
                if change == "hidden":
                    self.inputs[0].neg_()
                elif change == "routing":
                    self.inputs[4].add_(1).remainder_(self.model.experts)
                    self.inputs[3].copy_(self.inputs[3].roll(1, dims=-1))
                else:
                    self.inputs[1].neg_()
                    self.inputs[2].mul_(0.5)
                pristine = tuple(x.clone() for x in self.inputs)
                expected = self.reference(weight_draw=change == "weights")
                timed.outputs.fill_(float("nan"))
                self.check(timed.rerun(), expected)
                self.unchanged(pristine)
            finally:
                for value, original in zip(self.inputs, self.originals):
                    value.copy_(original)
        return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
                "replay_inputs_perturbed": True, "replay_output_poisoned": True,
                "replay_input_draws": 3, "oracle_device": "cpu"}


class ObservedInvocation:
    """Retain capture-time buffers before the timer binds its final collector."""
    def __init__(self, fixture, kernel):
        self.fixture, self.kernel, self.output = fixture, kernel, None

    def call(self):
        self.output = self.fixture.call(self.kernel)
        return self.output

    def prepare(self):
        if self.output is not None:
            self.output.fill_(float("nan"))


def run_case(row, action, kernel):
    import torch
    with torch.no_grad():
        fixture = Fixture(row)
        if action == "correctness":
            output = fixture.call(kernel)
            torch.cuda.synchronize()
            metrics = fixture.check(output)
            fixture.unchanged()
            return {"status": "PASS", "metrics": metrics}
        from _aka_benchmark import ReadOnlyCacheFlush, TimedRun, benchmark_cuda_graph_cold_cache_samples
        timed = TimedRun()
        timed.after_sample = fixture.check
        invocation = ObservedInvocation(fixture, kernel)
        samples, metadata = benchmark_cuda_graph_cold_cache_samples(
            invocation.call, cache_flush=ReadOnlyCacheFlush(),
            warmup=10, repetition=100, prepare_fn=invocation.prepare, timed_run=timed)
        metadata.update(fixture.verify_replay(timed))
        metadata["raw_samples_ms"] = samples
        return {"status": "PASS", "execution_time_ms": sum(samples) / len(samples),
                "benchmark_method": metadata["benchmark_method"], "metadata": metadata}
