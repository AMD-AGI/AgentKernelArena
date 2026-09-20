"""Task-local baseline device trace; copied into a diagnostic workspace only.

This reuses the task's scored-case adapter and canonical timing helper. It never
writes a performance report or changes the task's scoring/benchmark controls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_KERNEL_EVENTS = 4096
GPU_KERNEL_CATEGORIES = {"kernel", "gpu_kernel", "cuda_kernel", "hip_kernel"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_indices(rows, maximum, requested=()):
    if not 1 <= maximum <= 16:
        raise ValueError("max-cases must be between 1 and 16")
    if not rows:
        raise ValueError("the scored case builder returned no cases")
    if requested:
        selected = list(requested)
        if (len(selected) > maximum or len(set(selected)) != len(selected)
                or any(index < 0 or index >= len(rows) for index in selected)):
            raise ValueError("case indexes must be distinct, in range, and within max-cases")
        return selected
    # Cover each regime first, then spread remaining selections over the scored
    # list. Selection never creates shapes or admits correctness-only cases.
    selected, regimes = [], set()
    for index, row in enumerate(rows):
        regime = row.get("regime", "")
        if regime not in regimes:
            regimes.add(regime)
            selected.append(index)
        if len(selected) == maximum:
            return selected
    spread = [round(index * (len(rows) - 1) / max(1, maximum - 1)) for index in range(maximum)]
    for index in [*spread, *range(len(rows))]:
        if index not in selected:
            selected.append(index)
        if len(selected) == min(maximum, len(rows)):
            break
    return selected


def kernel_events(trace):
    """Project actual GPU kernel events; CPU op/callable names are never used."""
    events = trace.get("traceEvents") if isinstance(trace, dict) else None
    if not isinstance(events, list):
        raise ValueError("profiler output does not contain traceEvents")
    kernels = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        categories = {item.strip().lower() for item in str(event.get("cat", "")).split(",")}
        if not categories.intersection(GPU_KERNEL_CATEGORIES):
            continue
        if not isinstance(event.get("name"), str) or not event["name"]:
            continue
        args = event.get("args") or {}
        normalized = {str(key).lower().replace("_", " "): value for key, value in args.items()}
        aliases = {
            "grid": ("grid", "grid dimensions", "grid dims"),
            "block": ("block", "block dimensions", "block dims"),
            "device": ("device", "device id"),
            "stream": ("stream", "stream id"),
            "correlation": ("correlation", "correlation id"),
            "graph_node_id": ("graph node id", "node id"),
        }
        metadata = {name: next((normalized[key] for key in keys if key in normalized), None)
                    for name, keys in aliases.items()}
        duration = event.get("dur")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            duration = None
        kernels.append({"symbol": event["name"], "category": event.get("cat"),
                        "profiler_duration_us": duration, **metadata,
                        "missing_metadata": [name for name, value in metadata.items() if value is None],
                        "raw_args": args})
        if len(kernels) > MAX_KERNEL_EVENTS:
            raise ValueError("profiler kernel-event limit exceeded; reduce trace replays")
    return kernels


def tensor_contracts(row, bench, torch):
    result = []
    for path, tensor in bench.tensors(row["args"], torch):
        attributes = {name: value if isinstance(value, (str, int, float, bool, type(None)))
                      else {"type": type(value).__name__, "value_recorded": False}
                      for name, value in tensor.__dict__.items()}
        result.append({"argument": path, **bench.tensor_signature(tensor), "attributes": attributes})
    return result


def timed_baseline(row, call, module, h, meta, torch, bench, timer, warmup, iterations):
    """Capture the replay object passed through the unchanged scored-case path."""
    state = bench.InputState(row["args"], torch, bench.replay_probe(row["args"], torch))
    expected = {}
    for probe, label in ((False, "base"), (True, "probe")):
        state.probe_enabled = probe
        output = bench.collect_output(lambda: call(row["args"]), state.restore,
                                      bench.output_transform(module, meta, row, torch),
                                      int(row.get("validation_replays", 1)), torch)
        expected[label] = bench.cpu_copy(output, torch)
    state.probe_enabled = False
    state.restore()
    captured = []

    def observe_timer(function, **kwargs):
        captured.append(kwargs["timed_run"])
        return timer(function, **kwargs)

    measurement = bench.measure_case(row, call, expected, module, h, meta, torch,
                                     warmup, iterations, observe_timer)
    if len(captured) != 1:
        raise RuntimeError("the scored helper did not expose exactly one timed replay")
    return captured[0].replay, measurement


def profile_call(callback, torch, destination, replays):
    activities = torch.profiler.supported_activities()
    if torch.profiler.ProfilerActivity.CUDA not in activities:
        raise RuntimeError("torch.profiler does not expose CUDA/HIP device activity")
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, with_stack=False) as profiler:
        for _ in range(replays):
            callback()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(destination))
    if destination.stat().st_size > MAX_TRACE_BYTES:
        raise RuntimeError("raw profiler trace exceeds the 64 MiB diagnostic limit")
    kernels = kernel_events(json.loads(destination.read_text()))
    if not kernels:
        raise RuntimeError("profiler recorded no GPU kernel events; CPU launch names are insufficient")
    return kernels


def worker(task, args):
    import torch
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("torch.profiler does not expose CUDA/HIP device activity")
    bench = load("_headkernel_trace_bench", task / "scripts/_bench.py")
    runner = load("_headkernel_trace_runner", task / "scripts/task_runner.py")
    ut = task / "ut"
    meta = json.loads((ut / "meta.json").read_text())
    if meta.get("dispatch_config"):
        os.environ["AITER_CONFIG_GEMM_BF16"] = str(ut / meta["dispatch_config"])
    h = sys.modules["harness_lib"]
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ut]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    module = bench.load_module("_headkernel_cases", ut / ("cases.py" if (ut / "cases.py").is_file()
                                                         else "unittest.py"))
    rows, call = bench.selected_cases(module, h, meta, torch, reference=True)
    identities = bench.validate_cases(rows)
    selected = select_indices(rows, args.max_cases, args.case_index)
    report = {"schema": "aka-baseline-device-trace-v1", "status": "recording",
              "scope": "baseline diagnostic; not scoring or framework validation",
              "mode": args.mode, "profile_replays": args.replays,
              "all_scored_case_ids": identities, "selected_case_indexes": selected,
              "untraced_case_ids": [identity for index, identity in enumerate(identities) if index not in selected],
              "runtime": json.loads((task / "build/runtime_preflight.json").read_text()),
              "case_builder": {"file": "scripts/_bench.py", "sha256": digest(task / "scripts/_bench.py"),
                               "entrypoint": "selected_cases(..., reference=True)"},
              "warmup_iterations": runner.WARMUP_ITERATIONS,
              "benchmark_iterations": runner.BENCHMARK_ITERATIONS if args.mode == "timed-graph" else None,
              "correctness_scope": "baseline replay self-consistency only; full task correctness not run",
              "original_dispatch_equivalence": "not_evaluated", "cases": []}
    report_path = task / "build/device_trace_report.json"
    trace_dir = task / "build/device-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for index in selected:
        row = rows[index]
        entry = {"case_index": index, "case_id": identities[index],
                 "input_contract": tensor_contracts(row, bench, torch),
                 "status": "recording", "input_restore_in_trace": True,
                 "graph_node_attribution": "only where native profiler metadata exposes it"}
        report["cases"].append(entry)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if args.mode == "timed-graph":
            callback, measurement = timed_baseline(
                row, call, module, h, meta, torch, bench,
                sys.modules["_aka_benchmark"].benchmark_cuda_graph_or_events_samples,
                runner.WARMUP_ITERATIONS, runner.BENCHMARK_ITERATIONS)
            entry["graph_relation"] = "same graph object timed in this diagnostic by the unchanged scored-case helper"
            entry["diagnostic_measurement"] = measurement
        else:
            state = bench.InputState(row["args"], torch)
            def callback():
                state.restore()
                return call(row["args"])
            for _ in range(runner.WARMUP_ITERATIONS):
                callback()
            entry["graph_relation"] = "eager baseline trace; not a timed-graph trace"
        trace = trace_dir / f"case-{index:04d}.chrome.json"
        entry["kernel_events"] = profile_call(callback, torch, trace, args.replays)
        entry.update(status="trace_recorded", kernel_launch_count=len(entry["kernel_events"]),
                     raw_trace={"path": trace.relative_to(task).as_posix(), "sha256": digest(trace),
                                "bytes": trace.stat().st_size})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "trace_recorded"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("timed-graph", "eager"), default="timed-graph")
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--case-index", action="append", type=int, default=[])
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if (not 1 <= args.replays <= 5 or not 1 <= args.max_cases <= 16
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("replays must be 1..5, max-cases 1..16, and timeout positive")
    task = Path(__file__).resolve().parents[1]
    if args.worker:
        try:
            return worker(task, args)
        except Exception as error:
            path = task / "build/device_trace_report.json"
            report = json.loads(path.read_text()) if path.is_file() else {}
            report.update(status="trace_failed", error_type=type(error).__name__, error=str(error))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            raise
    runner = load("_headkernel_trace_parent", task / "scripts/task_runner.py")
    runner.verify_fixtures()
    baseline, _ = runner.overlays()
    report = task / "build/device_trace_report.json"
    tail = ["--worker", "--mode", args.mode, "--max-cases", str(args.max_cases),
            "--replays", str(args.replays), "--timeout", str(args.timeout)]
    for index in args.case_index:
        tail += ["--case-index", str(index)]
    proc = runner.run_worker(Path(__file__), tail, baseline, args.timeout, False,
                             attest_files=(report,))
    print(proc.stdout, end="")
    print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
