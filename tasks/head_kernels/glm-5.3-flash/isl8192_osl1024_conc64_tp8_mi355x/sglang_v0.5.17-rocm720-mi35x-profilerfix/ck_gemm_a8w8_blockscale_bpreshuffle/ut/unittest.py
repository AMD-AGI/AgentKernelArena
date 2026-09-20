#!/usr/bin/env python3
"""Protected eager and graph checks of every fixed GLM GEMM case."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
# This filename must not shadow the standard-library unittest imported by torch.
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h = load("harness_lib", HERE / "harness_lib.py")
cases = load("_glm_cases", HERE / "cases.py")


def assert_inputs(args, saved):
    import torch
    for name, value in args.items():
        signature, attributes, data = saved[name]
        actual = (value.shape, value.stride(), value.dtype, value.data_ptr())
        if actual != signature or value.__dict__ != attributes:
            raise AssertionError(f"input contract changed: {name}")
        if not torch.equal(value.contiguous().view(torch.uint8), data):
            raise AssertionError(f"input mutated: {name}")


def save_inputs(args):
    import torch
    # A column-major scale's byte view needs a final contiguous axis. Preserve
    # the actual shape/stride separately and compare a logical contiguous copy.
    return {name: ((value.shape, value.stride(), value.dtype, value.data_ptr()),
                   dict(value.__dict__), value.contiguous().view(torch.uint8).clone())
            for name, value in args.items()}


def assert_output(output, reference, args, tol):
    import torch
    ok, error = h.correct(output, reference, tol)
    if not ok:
        raise AssertionError(f"numerical output mismatch: error={error}")
    if not output.is_contiguous():
        raise AssertionError("output must preserve the fresh row-major layout")
    if output.untyped_storage().data_ptr() in {
            value.untyped_storage().data_ptr() for value in args.values()}:
        raise AssertionError("output aliases an input")


def copy_inputs(destination, source):
    for key, value in destination.items():
        value.copy_(source[key])


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GLM GEMM correctness requires a gfx950 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = cases.META
    selected = cases.selected_cases(meta, meta["ledger_ids"])
    for index, case in enumerate(selected):
        previous = None
        for draw in range(1 + int(meta["random_draws"])):
            args, reference = cases.make_args(case, seed=1234 + index + draw * 100000,
                                               with_reference=True)
            saved = save_inputs(args)
            native = cases.baseline_call(args)
            assert_output(native, reference, args, meta["tol"])
            candidate = cases.candidate_call(args)
            assert_output(candidate, reference, args, meta["tol"])
            assert_output(candidate, native, args, meta["tol"])
            assert_inputs(args, saved)
            if previous is not None:
                output, snapshot = previous
                if (output.untyped_storage().data_ptr() == candidate.untyped_storage().data_ptr()
                        or not torch.equal(output, snapshot)):
                    raise AssertionError("candidate reused or changed a previous output")
            previous = candidate, candidate.clone()

        # Eager serving is the historical regime. These additional graph checks
        # qualify the task's isolated device-time benchmark replay contract.
        first, first_ref = cases.make_args(case, seed=2000, with_reference=True)
        second, second_ref = cases.make_args(case, seed=2001, with_reference=True)
        cases.candidate_call(first)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = cases.candidate_call(first)
        for inputs, reference in ((second, second_ref), (first, first_ref)):
            # Preserve the original A values for the final A/B/A replay.
            if inputs is first:
                inputs = cases.make_args(case, seed=2000)
            copy_inputs(first, inputs)
            graph.replay()
            torch.cuda.synchronize()
            assert_output(graph_output, reference, first, meta["tol"])
        print(f"PASS {case['sig']}: FP32, native, fresh-output, graph replay")
    print(json.dumps({"status": "ok", "case_count": len(selected),
                      "oracle": meta["oracle"], "captured_tensor_oracle": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
