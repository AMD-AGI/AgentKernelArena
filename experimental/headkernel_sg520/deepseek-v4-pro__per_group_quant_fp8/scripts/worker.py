"""One native leg per process; only the coordinator emits scoreable reports."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from contract import ITERATIONS, SEEDS, WARMUP, package_identity, sample_statistics, write_report
from native import baseline, build_proof, candidate, source_identity
from oracle import compare, generate, invoke, load_cases, reference, validate_output_metadata


def check_negative_controls(actual, expected, x):
    """Prove that wrong FP8 bytes and wrong physical scales are rejected."""
    import torch

    for value, replacement in ((actual[0].view(torch.uint8), 255), (actual[1], -1.0)):
        flat = value.reshape(-1)
        saved = flat[0].clone()
        flat[0] = replacement
        try:
            compare(actual, expected, x)
        except AssertionError:
            pass
        else:
            raise AssertionError("oracle accepted a corrupted output")
        finally:
            flat[0] = saved


def cpu_snapshot(tensor):
    """Own the saved bytes independently of all candidate-visible GPU storage."""
    return tensor.detach().to(device="cpu", copy=True)


def validate_completed(actual, x, before_cpu, torch, *, negative_controls=False):
    """Called only after native work synchronizes; snapshot before the oracle."""
    validate_output_metadata(actual, x)
    actual_cpu = tuple(cpu_snapshot(value) for value in actual)
    after_cpu = cpu_snapshot(x)
    if not torch.equal(after_cpu, before_cpu):
        raise AssertionError("native kernel modified its input")
    # No current expected output exists while the candidate runs. Use the
    # independent host input, and capture actual outputs before these GPU ops.
    oracle_input = before_cpu.to(device=x.device, copy=True)
    expected_gpu = reference(oracle_input)
    expected_cpu = tuple(cpu_snapshot(value) for value in expected_gpu)
    compare(actual_cpu, expected_cpu, before_cpu)
    if negative_controls:
        check_negative_controls(actual_cpu, expected_cpu, before_cpu)


def correctness(function, case):
    import torch

    for seed in SEEDS:
        x = generate(case, seed, "cuda")
        before_cpu = cpu_snapshot(x)
        actual = invoke(function, x)
        torch.cuda.synchronize()
        validate_completed(actual, x, before_cpu, torch, negative_controls=True)
    return {"seeds": list(SEEDS), "negative_controls": True}


def replay_with_fresh_input(graph, result, x, fresh, torch):
    """Copies and output poisoning are outside the device timing interval."""
    before_cpu = cpu_snapshot(fresh)
    x.copy_(fresh)
    result[0].view(torch.uint8).fill_(255)
    result[1].fill_(float("nan"))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    validate_completed(result, x, before_cpu, torch)
    return start.elapsed_time(end)


def performance(function, case, seed):
    import torch

    x = generate(case, seed, "cuda")
    before_cpu = cpu_snapshot(x)
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(WARMUP):
            invoke(function, x)
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = invoke(function, x)
    graph.replay()
    torch.cuda.synchronize()
    validate_completed(result, x, before_cpu, torch)
    del before_cpu
    samples = []
    for iteration in range(ITERATIONS):
        fresh = generate(case, seed + iteration + 1, "cuda")
        samples.append(replay_with_fresh_input(graph, result, x, fresh, torch))
    return {"benchmark_method": "cuda_graph", "warmup_iterations": WARMUP,
            "fresh_input_replays": ITERATIONS, "poisoned_output_replays": ITERATIONS,
            "oracle_checks": ITERATIONS + 1, "timings": sample_statistics(samples)}


def main():
    if len(sys.argv) != 3:
        raise ValueError("worker requires its coordinator request and result paths")
    request_path, result_path = map(Path, sys.argv[1:])
    if result_path.exists():
        raise ValueError("worker result must not exist before evaluation")
    request = json.loads(request_path.read_text())
    if request["package_sha256"] != package_identity() or request["source_tree_sha256"] != source_identity():
        raise ValueError("worker inputs do not match coordinator request")
    _, cases = load_cases(ROOT / "cases.json")
    import torch
    from aiter.jit.utils.chip_info import get_gfx

    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        raise RuntimeError("the observed SG520 gfx950 runtime is required")
    if request["leg"] == "production_native":
        function = baseline()
    elif request["leg"] == "candidate_native":
        function = candidate()
    else:
        raise ValueError("unknown native leg")
    rows = []
    if request["mode"] != "compile":
        for case in cases:
            if request["mode"] == "correctness":
                details = correctness(function, case)
            elif request["mode"] == "performance":
                details = performance(function, case, request["challenge_seed"])
            else:
                raise ValueError("unknown worker mode")
            rows.append({"case_id": case["case_id"], "shape": case["shape"],
                         "trace_call_count": case["trace_call_count"], "correct": True,
                         "input_immutable": True, **details})
    if request["package_sha256"] != package_identity() or request["source_tree_sha256"] != source_identity():
        raise ValueError("worker inputs changed during evaluation")
    write_report(result_path, {**request, "status": "ok", "pid": os.getpid(),
                               "native_build": build_proof(), "results": rows})


if __name__ == "__main__":
    main()
