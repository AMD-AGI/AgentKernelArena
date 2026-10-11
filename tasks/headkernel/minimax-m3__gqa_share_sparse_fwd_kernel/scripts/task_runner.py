"""Protected MiniMax evaluation with complete cases and fresh replay inputs."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from evaluation_contract import checked_replays, finalize_report, fingerprint, observe_case, require, strict_json
from served_contract import load_cases, sha
from source_guard import validate_sources


def configure_cpu_threads():
    """Bound host tensor work when multiple GPU validators share one CPU node."""
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    return {"intraop_threads_before": previous, "intraop_threads": torch.get_num_threads(),
            "scope": "CPU geometry and snapshot comparisons; native graph and oracles unchanged"}


def tree_identity():
    entries = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"build", ".validator_audit", ".validator_torch_extensions"}:
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            require(path.resolve().is_relative_to(ROOT.resolve()), "task symlink escapes package")
            entries[relative.as_posix()] = {"type": "symlink", "target": str(path.readlink())}
        elif path.is_dir():
            entries[relative.as_posix()] = {"type": "directory"}
        elif path.is_file():
            entries[relative.as_posix()] = {"type": "file", "bytes": path.stat().st_size, "sha256": sha(path)}
    return fingerprint(entries)


def write_report(phase, report):
    path = ROOT / "build" / (phase + "_report.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class CaseEvaluation:
    def __init__(self, case, definition, operator, *, paired=False, paired_seed=None):
        import torch
        from minimax_data import Inputs
        self.case, self.definition, self.operator = case, definition, operator
        self.root = ROOT
        if "launch_contract" in case:
            operator.select_launch_contract(case["launch_contract"])
        self.inputs = Inputs(case, definition, root=ROOT)
        require(not paired or (type(paired_seed) is int and paired_seed >= 0), "paired graph setup seed missing")
        initial = self.inputs.reset(paired_seed if paired else 0)
        if paired:
            self.paired_setup_seed = paired_seed
            self.paired_initial_truth = initial
            self.result = self.graph = self.baseline = None
            return
        del initial
        self.result = operator(self.inputs.args)
        torch.cuda.synchronize()
        # Capturing this same frozen wrapper preserves its allocation/work
        # boundary; replay is one complete invocation, not a precomputed output.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.result = operator(self.inputs.args)
        self.baseline = None
        self.observe()

    def native_reference(self, storage):
        import torch
        from minimax_native import Operator
        from minimax_data import snapshot_output
        if self.baseline is None:
            self.baseline = Operator(ROOT, self.definition, reference=True)
            if "launch_contract" in self.case:
                self.baseline.select_launch_contract(self.case["launch_contract"])
        expected = self.baseline(self.inputs.reference_args(storage))
        torch.cuda.synchronize()
        return snapshot_output(expected)

    def compare_outputs(self, actual, expected):
        import torch
        from minimax_reference import mixed_close
        require(set(actual) == set(expected), "output structure differs")
        for name, value in actual.items():
            if value is None or expected[name] is None:
                require(value is None and expected[name] is None, "optional output differs")
            elif value.is_floating_point():
                mixed_close(value, expected[name], self.definition["tolerance"])
            elif not torch.equal(value, expected[name]):
                raise AssertionError("native/captured integer-output parity failed")

    def observe(self):
        tensors, scalars = self.inputs.observe_arguments(self.result)
        return observe_case(self.case, tensors, scalars)

    def initialize(self):
        from minimax_data import initialize_outputs
        initialize_outputs(self.result)

    def verify(self, before, *, corrupt=False):
        import torch
        from minimax_data import snapshot_output
        from minimax_reference import mixed_close, score_reference, check_topk
        from minimax_reference import sparse_attention
        torch.cuda.synchronize()
        actual = snapshot_output(self.result)
        self.inputs.assert_immutable(before)
        # Create oracle-only device storage after candidate outputs are saved
        # and input immutability has passed. Both references use the same frozen
        # snapshot, avoiding repeated host gathers for every sparse query.
        reference_storage = {alias: value.to("cuda", copy=True) for alias, value in before.items()}
        args = self.inputs.reference_args(reference_storage)
        if corrupt:
            first = next(value for value in actual.values() if value is not None)
            first.reshape(-1)[0] = 1e4 if first.is_floating_point() else -999
        if self.definition["kind"] == "decode_score":
            score, counts, topk = score_reference(args, device="cuda")
            check_topk(actual["result.1"], score, counts, topk, tolerance=self.definition["tolerance"])
            # Independent cutoff math establishes validity; exact native parity
            # also preserves the existing integer-index contract and tie choice.
        else:
            expected, roundoff = sparse_attention(args, self.definition["kind"], device="cuda", return_roundoff=True)
            mixed_close(actual["result"], expected, self.definition["tolerance"], arithmetic_error=roundoff)
        self.compare_outputs(actual, self.native_reference(reference_storage))

    def check_independent_output(self, actual, args):
        from minimax_reference import mixed_close, score_reference, check_topk
        from minimax_reference import sparse_attention
        if self.definition["kind"] == "decode_score":
            score, counts, topk = score_reference(args, device="cuda")
            check_topk(actual["result.1"], score, counts, topk, tolerance=self.definition["tolerance"])
            # Independent cutoff math establishes validity; exact native parity
            # also preserves the existing integer-index contract and tie choice.
        else:
            expected, roundoff = sparse_attention(args, self.definition["kind"], device="cuda", return_roundoff=True)
            mixed_close(actual["result"], expected, self.definition["tolerance"], arithmetic_error=roundoff)

    def measure(self, replay):
        import torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); replay(); end.record(); end.synchronize()
        return start.elapsed_time(end)

    def correctness(self, policy):
        from minimax_data import snapshot_output
        recorded = 0
        if self.inputs.external:
            for index in range(len(self.inputs.states)):
                before = self.inputs.restore_recorded(index)
                self.observe()
                self.initialize(); self.graph.replay(); self.verify(before)
                self.compare_outputs(snapshot_output(self.result), self.inputs.recorded_outputs(index))
                recorded += 1
        for seed in policy["correctness_seeds"]:
            before = self.inputs.reset(seed)
            self.initialize(); self.graph.replay(); self.verify(before)
        before = self.inputs.reset(policy["correctness_seeds"][0] + 1000)
        self.initialize()
        try:
            self.verify(before)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("no-op replay was accepted")
        self.graph.replay(); self.verify(before)
        try:
            self.verify(before, corrupt=True)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output was accepted")
        return {"case": self.observe(), "correct": True, "seeds": policy["correctness_seeds"],
                "negative_controls": {"no_op": True, "wrong_output": True},
                "recorded_representatives_checked": recorded,
                "independent_math_and_frozen_native_parity": True}

    def performance(self, policy, seed, *, manifest, request):
        require(request["phase"] == "performance" and request["challenge_seed"] == seed
                and manifest["measurement"] == policy, "paired prefill request differs")
        from minimax_paired import performance
        return performance(self, manifest, request)



def evaluate_cases(manifest, definition, request, *, cases=None, worker=False):
    """Run complete original cases; only the full-task caller finalizes reports."""
    if cases is None:
        cases = manifest["cases"]
    ids = [case["case_id"] for case in cases]
    require(ids and len(ids) == len(set(ids)), "case selection must be nonempty and distinct")
    selected = [case for case in manifest["cases"] if case["case_id"] in set(ids)]
    require(fingerprint(cases) == fingerprint(selected), "case selection changed original records or order")
    from minimax_native import Operator
    operator = Operator(ROOT, definition)
    rows, specializations, completed = [], [], []
    for case in cases:
        started = time.monotonic()
        print(json.dumps({"phase": request["phase"], "case_id": case["case_id"], "event": "start"}), flush=True)
        evaluation = CaseEvaluation(case, definition, operator, paired=request["phase"] == "performance",
            paired_seed=request["challenge_seed"] + manifest["measurement"]["warmup_iterations"] + manifest["measurement"]["benchmark_iterations"])
        if request["phase"] == "correctness":
            rows.append(evaluation.correctness(manifest["measurement"]))
        elif request["phase"] == "performance":
            rows.append(evaluation.performance(manifest["measurement"], request["challenge_seed"], manifest=manifest, request=request))
            specializations.append(evaluation.paired_specialization)
        if worker and request["phase"] != "performance":
            proof = operator.proof()
            require(proof["source_sha256"] == request["source_sha256"][definition["source_file"]]
                    and set(proof["engaged_kernels"]) == set(definition["kernels"]),
                    "whole-case native engagement is incomplete")
            specializations.append({"case_id": case["case_id"], "source_sha256": request["source_sha256"],
                "candidate_invoked": True, "invoked_and_synchronized": True, "candidate_launch_evidence": proof})
        del evaluation
        completed.append(case["case_id"])
        print(json.dumps({"phase": request["phase"], "case_id": case["case_id"], "event": "complete",
                          "wall_seconds": time.monotonic() - started}), flush=True)
    return {"rows": rows, "compiled_specializations": specializations,
            "kernel_engagement": operator.proof(), "completed_case_ids": completed}


def parent_engagement(manifest, request, union):
    """Retain an actual worker launch proof after complete specialization coverage."""
    compiled = union["compiled_specializations"]
    require([row["case_id"] for row in compiled] == [case["case_id"] for case in manifest["cases"]],
            "full-task compiled specialization coverage differs")
    # This is the actual final original case's proof, matching the serial field.
    # All worker proofs remain in the full compiled-specialization records.
    return compiled[-1]["candidate_launch_evidence"]


def complete_report(manifest, request, payload, cpu_runtime, *, parallel=None):
    report = {"schema_version": 1, "status": "ok", "request": request, "cases": payload["rows"],
              "kernel_engagement": payload["kernel_engagement"], "task_validator_status": "pending",
              "cpu_runtime": cpu_runtime, "capture_scope": manifest["capture"], "speedup_claim": False}
    if request["phase"] == "compile":
        report["compiled"] = True
    if parallel is not None:
        report["case_parallelism"] = parallel
        report["compiled_specializations"] = payload["compiled_specializations"]
    if request["phase"] == "performance":
        from paired_reference import attach_comparison
        report["compiled_specializations"] = payload["compiled_specializations"]
        attach_comparison(ROOT, report, manifest, request)
    return finalize_report(report, manifest, request)


def worker_requested(arguments):
    return any(arg == "--case-worker" or arg.startswith("--case-worker=") for arg in arguments)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("phase", choices=("compile", "correctness", "performance"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--case-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case_worker is None:
        (ROOT / "build").mkdir(exist_ok=True)
        (ROOT / "build" / (args.phase + "_report.json")).unlink(missing_ok=True)
    else:
        require(args.request is not None, "internal worker requires the unchanged parent request")
    manifest, definition = load_cases(ROOT)
    validate_sources(ROOT, ROOT)
    package = tree_identity()
    sources = {definition["source_file"]: sha(ROOT / definition["source_file"])}
    request = (strict_json(args.request.read_text()) if args.request else
               {"schema_version": 1, "request_id": secrets.token_hex(24), "phase": args.phase,
                "manifest_sha256": fingerprint(manifest), "package_sha256": package,
                "source_sha256": sources, "challenge_seed": secrets.randbelow(2**30)})
    require(request.get("phase") == args.phase and request.get("manifest_sha256") == fingerprint(manifest)
            and request.get("source_sha256") == sources and request.get("package_sha256") == package,
            "request does not match current source/package/cases")
    for name, value in manifest.get("environment", {}).items():
        require(name == "SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX" and value in ("0", "1"),
                "unknown evaluation environment control")
        require(name not in os.environ or os.environ[name] == value, "runtime environment conflicts with captured path")
        os.environ[name] = value
    context = plan = parallel = None
    enabled = args.case_worker is not None or any(
        request.get(name) is not None for name in ("case_parallelism", "gpu_group")) or any(
        name in os.environ for name in ("AKA_CASE_GPU_PLAN", "AKA_CASE_GPU_PLAN_SHA256"))
    if enabled:
        # Both paths enter the shared verifier before importing Torch or native code.
        from case_parallel import bind_request, enter_worker, finish_worker, load_plan, run_workers
        if args.case_worker is not None:
            context = enter_worker(ROOT, manifest, request, args.case_worker)
        else:
            plan = load_plan(ROOT, manifest, request)
            require(plan is not None, "explicit whole-case mode requires an admitted GPU plan")
            request = bind_request(request, plan)
    if plan is not None:
        union = run_workers(ROOT, manifest, request, plan, Path(__file__).resolve(), leg="framework")
        payload = {**union, "kernel_engagement": parent_engagement(manifest, request, union)}
        parallel = union["case_parallelism"]
        cpu_runtime = {"scope": "Per-worker CPU policies are retained in the whole-case receipts"}
    else:
        cpu_runtime = configure_cpu_threads()
        diagnostics = context.output_directory if context is not None else ROOT / "build"
        (diagnostics / ("cpu_runtime_" + args.phase + ".json")).write_text(json.dumps(cpu_runtime) + "\n")
        payload = evaluate_cases(manifest, definition, request,
            cases=context.cases if context is not None else None, worker=context is not None)
    after = tree_identity()
    require(after == package and sha(ROOT / definition["source_file"]) == sources[definition["source_file"]],
            "source or protected package changed during evaluation")
    if context is not None:
        finish_worker(context, rows=payload["rows"], compiled_specializations=payload["compiled_specializations"],
            evidence={"completed_case_ids": payload["completed_case_ids"],
                "kernel_engagement": payload["kernel_engagement"], "cpu_runtime": cpu_runtime,
                "source_sha256": sources, "package_before_sha256": package, "package_after_sha256": after})
        return 0
    report = complete_report(manifest, request, payload, cpu_runtime, parallel=parallel)
    write_report(args.phase, report)
    print(args.phase.capitalize() + ": PASS")
    return 0


def entrypoint():
    try:
        return main()
    except Exception as error:
        # A child failure must never overwrite its coordinator's public report.
        if not worker_requested(sys.argv[1:]):
            phase = sys.argv[1] if len(sys.argv) > 1 else "unknown"
            (ROOT / "build").mkdir(exist_ok=True)
            write_report(phase, {"schema_version": 1, "status": "fail", "error": f"{type(error).__name__}: {error}", "cases": []})
        raise


if __name__ == "__main__":
    raise SystemExit(entrypoint())
