#!/usr/bin/env python3
"""Runnable cold-cache A/A or A/B example using the paged-attention task.

Run inside the project's GPU Docker runtime from the repository root. This is
an authoring reference, not a task evaluator and never writes task_result.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

TASK = REPO / "tasks/triton2triton/vllm/triton_chunked_prefill_paged_decode"
SOURCE = TASK / "source/triton_chunked_prefill_paged_decode.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # The original task runner changes cwd on import. Keep that task's sources
    # intact, while ensuring relative example output paths retain their meaning.
    previous = Path.cwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AttentionFixture:
    """Original task geometry/oracle/gate, with fresh operands for every call."""

    def __init__(self, harness, checks, shape, seed):
        import torch
        self.torch, self.harness, self.checks = torch, harness, checks
        self.seed = seed
        torch.manual_seed(seed)
        self.args = harness.make_test_data(*shape)
        self.values = dict(enumerate(self.args)) | {"filter_by_query_len": False}
        self.readonly = [i for i, value in enumerate(self.args)
                         if isinstance(value, torch.Tensor) and i != 1]
        self.initial = {i: self.args[i].clone() for i in self.readonly}
        self.metadata = {i: (self.args[i].shape, self.args[i].stride(), self.args[i].dtype,
                             self.args[i].device, self.args[i].data_ptr())
                         for i in self.readonly + [1]}
        self.generator = torch.Generator(device="cuda")
        self.before_arm("baseline", 0)

    def before_arm(self, role, index):
        del role
        # Identical draw sequence for both arms of this paired round. The helper
        # makes the same warmup/capture/sample/rerun calls on each arm.
        self.generator.manual_seed(self.seed + index)

    def prepare(self):
        for i, original in self.initial.items():
            self.args[i].copy_(original)
        for i in (0, 2, 3):  # query, K cache and V cache change on every invocation
            self.args[i].normal_(generator=self.generator)
        self.args[1].fill_(float("nan"))
        self.pristine = {i: self.args[i].clone() for i in self.readonly}
        # Compute the oracle BEFORE executing either implementation; a candidate
        # cannot mutate its inputs and thereby change the expected answer.
        # Keep the independent task formula on CPU, outside GPU timing. The
        # timing reference must not require a working GPU BLAS implementation.
        oracle_values = {key: value.to("cpu") if isinstance(value, self.torch.Tensor) else value
                         for key, value in (self.values | self.pristine).items()}
        self.expected = self.checks.expected_outputs(
            self.harness, oracle_values)[0].to(self.args[1].device)

    def call(self, module):
        return self.checks.checked_candidate_call(
            module, module.chunked_prefill_paged_decode,
            *self.args, filter_by_query_len=False)

    def check(self, role, output):
        del role
        torch = self.torch
        if output is not self.args[1]:
            raise AssertionError("The task wrapper must return the supplied output buffer")
        for i, metadata in self.metadata.items():
            value = self.args[i]
            if (value.shape, value.stride(), value.dtype, value.device, value.data_ptr()) != metadata:
                raise AssertionError(f"Changed tensor metadata/storage: argument {i}")
        for i, pristine in self.pristine.items():
            if not torch.equal(self.args[i].view(torch.uint8), pristine.view(torch.uint8)):
                raise AssertionError(f"Changed read-only input: argument {i}")
            if torch._C._overlaps(output, self.args[i]):
                raise AssertionError("Output aliases a read-only input")
        # Preserve the task's full-output FP16 gate. NaN poison cannot pass.
        torch.testing.assert_close(output, self.expected, atol=0.01, rtol=0.01, equal_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, default=SOURCE)
    parser.add_argument("--candidate-source", type=Path, default=SOURCE)
    parser.add_argument("--case", type=int, choices=range(1, 6), action="append",
                        help="Original task case, 1 through 5; default: all five")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.output.exists():
        parser.error("output already exists; use a new artifact path")
    if options.rounds < 4 or options.rounds % 2 or options.samples < 1 or options.warmup < 1:
        parser.error("use even rounds >= 4, samples >= 1 and warmup >= 1")

    import torch
    import triton
    from src.tools.perf.aka_benchmark import (
        ReadOnlyCacheFlush, benchmark_paired_cuda_graph_cold_cache,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA/ROCm GPU")
    harness = load_module(TASK / "scripts/task_runner.py", "_cold_cache_task_harness")
    checks = load_module(TASK / "_arena_checks.py", "_cold_cache_attention_checks")
    modules = {role: load_module(path.resolve(), f"_cold_cache_{role}")
               for role, path in (("baseline", options.baseline_source),
                                  ("candidate", options.candidate_source))}
    flush = ReadOnlyCacheFlush()  # one initialized 1 GiB buffer, shared by A/B
    report = {
        "example": "paged_attention_cold_cache_graph",
        "task": TASK.relative_to(REPO).as_posix(),
        "comparison": ("A/A sanity check" if sha256(options.baseline_source) ==
                       sha256(options.candidate_source) else "A/B comparison"),
        "source_sha256": {role: sha256(path) for role, path in
                          (("baseline", options.baseline_source),
                           ("candidate", options.candidate_source))},
        "harness_sha256": sha256(TASK / "scripts/task_runner.py"),
        "checks_sha256": sha256(TASK / "_arena_checks.py"),
        "timer_sha256": sha256(REPO / "src/tools/perf/aka_benchmark.py"),
        "runtime": {"torch": torch.__version__, "triton": triton.__version__,
                    "rocm": torch.version.hip, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name()},
        "cache_flush_bytes": flush.nbytes,
        "seed": options.seed,
        "oracle_device": "cpu",
        "cases": [],
    }
    for case in options.case or range(1, 6):
        shape = harness.TEST_SHAPES[case - 1]
        fixture = AttentionFixture(harness, checks, shape, options.seed)
        result = benchmark_paired_cuda_graph_cold_cache(
            lambda: fixture.call(modules["baseline"]),
            lambda: fixture.call(modules["candidate"]),
            cache_flush=flush, rounds=options.rounds, repetition=options.samples,
            warmup=options.warmup, seed=options.seed, prepare_fn=fixture.prepare,
            check_fn=fixture.check, before_arm=fixture.before_arm,
        )
        report["cases"].append({"test_case_id": f"perf{case}", "shape": list(shape), **result})
        print(json.dumps({"case": case, **result["summary"]}, allow_nan=False), flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing user artifacts, including on a race.
    with options.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
