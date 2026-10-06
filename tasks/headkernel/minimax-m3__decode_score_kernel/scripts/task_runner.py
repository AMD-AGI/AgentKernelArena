"""Protected MiniMax evaluation with complete cases and fresh replay inputs."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from evaluation_contract import checked_replays, finalize_report, fingerprint, observe_case, require, strict_json
from served_contract import load_cases, sha
from source_guard import validate_sources


def tree_identity():
    entries = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "build" or "__pycache__" in relative.parts or path.suffix == ".pyc":
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
    def __init__(self, case, definition, operator):
        import torch
        from minimax_data import Inputs
        self.case, self.definition, self.operator = case, definition, operator
        if "launch_contract" in case:
            operator.select_launch_contract(case["launch_contract"])
        self.inputs = Inputs(case, definition, root=ROOT)
        self.inputs.reset(0)
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

    def performance(self, policy, seed):
        return checked_replays(self.case, policy, reset_inputs=self.inputs.reset,
            initialize_outputs=self.initialize, replay=self.graph.replay, verify=self.verify,
            measure=self.measure, observe=self.observe, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("compile", "correctness", "performance"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    (ROOT / "build").mkdir(exist_ok=True)
    report_path = ROOT / "build" / (args.phase + "_report.json")
    report_path.unlink(missing_ok=True)
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
    from minimax_native import Operator
    operator = Operator(ROOT, definition)
    rows = []
    for case in manifest["cases"]:
        evaluation = CaseEvaluation(case, definition, operator)
        if args.phase == "correctness":
            rows.append(evaluation.correctness(manifest["measurement"]))
        elif args.phase == "performance":
            rows.append(evaluation.performance(manifest["measurement"], request["challenge_seed"]))
        del evaluation
    require(tree_identity() == package and sha(ROOT / definition["source_file"]) == sources[definition["source_file"]],
            "source or protected package changed during evaluation")
    report = {"schema_version": 1, "status": "ok", "request": request, "cases": rows,
              "kernel_engagement": operator.proof(), "task_validator_status": "pending",
              "capture_scope": manifest["capture"], "speedup_claim": False}
    if args.phase == "compile":
        report["compiled"] = True
    report = finalize_report(report, manifest, request)
    write_report(args.phase, report)
    print(args.phase.capitalize() + ": PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        phase = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        (ROOT / "build").mkdir(exist_ok=True)
        write_report(phase, {"schema_version": 1, "status": "fail", "error": f"{type(error).__name__}: {error}", "cases": []})
        raise
