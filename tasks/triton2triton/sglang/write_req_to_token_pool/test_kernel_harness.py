"""Protected independent reference, fixtures and cold-cache graph evaluation."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
POISON = -(2**31)


def manifest():
    return json.loads((ROOT / "cases.json").read_text())["cases"]


def load_kernel():
    from triton.runtime import JITFunction
    spec = importlib.util.spec_from_file_location("arena_token_pool_candidate", ROOT / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module.write_req_to_token_pool_triton
    if not isinstance(kernel, JITFunction):
        raise TypeError("Entrypoint must be a Triton JIT function")
    return kernel


def reference(initial, rows, pre, seq, extend, cache, prefixes, logical_width):
    """CPU-only oracle; inputs are trusted snapshots, never candidate results."""
    import torch
    tensors = [initial, rows, pre, seq, extend, cache, *prefixes]
    if any(x.device.type != "cpu" for x in tensors):
        raise ValueError("Reference inputs must be on CPU")
    batch = len(rows)
    if not (len(pre) == len(seq) == len(extend) == len(prefixes) == batch):
        raise ValueError("Mismatched request counts")
    if len(set(rows.tolist())) != batch or not 0 <= logical_width <= initial.shape[1]:
        raise ValueError("Invalid row mapping or logical width")
    result, start = initial.clone(), 0
    for i in range(batch):
        row, pl, sl, ex = (int(x) for x in (rows[i], pre[i], seq[i], extend[i]))
        if not (0 <= row < initial.shape[0] and 0 <= pl <= sl <= logical_width
                and ex == sl - pl and pl <= len(prefixes[i]) and start + ex <= len(cache)):
            raise ValueError("Invalid request lengths or row")
        result[row, :pl] = prefixes[i][:pl].to(torch.int32)
        result[row, pl:sl] = cache[start:start + ex].to(torch.int32)
        start += ex
    return result


def compare(actual, expected):
    import torch
    if not isinstance(actual, torch.Tensor) or (
        actual.shape, actual.dtype, actual.device
    ) != (expected.shape, expected.dtype, expected.device):
        raise AssertionError("Output shape/dtype/device mismatch")
    if not torch.equal(actual, expected):
        raise AssertionError("Exact table comparison failed (including untouched regions)")
    return {"exact_match": True}


def reference_controls():
    import torch
    initial = torch.full((6, 1041), -17, dtype=torch.int32)
    rows = torch.tensor([4, 1, 5])
    pre, extend = torch.tensor([0, 511, 513]), torch.tensor([0, 1, 512])
    seq = pre + extend
    cache = torch.arange(513, dtype=torch.int64) + 10000
    prefixes = [torch.arange(int(n), dtype=torch.int64) + i * 1000 for i, n in enumerate(pre)]
    actual = reference(initial, rows, pre, seq, extend, cache, prefixes, 1030)
    # Scalar known answer does not share the reference's slicing/cumsum logic.
    expected = initial.clone()
    for i, row in enumerate(rows.tolist()):
        for column in range(int(seq[i])):
            value = (prefixes[i][column] if column < pre[i] else
                     cache[sum(int(x) for x in extend[:i]) + column - int(pre[i])])
            expected[row, column] = int(value)
    compare(actual, expected)
    wrong_written, wrong_untouched = expected.clone(), expected.clone()
    wrong_written[1, 0] += 1
    wrong_untouched[0, -1] += 1
    for wrong in (initial, torch.zeros_like(expected), wrong_written, wrong_untouched):
        try:
            compare(wrong, expected)
        except AssertionError:
            continue
        raise AssertionError("Negative output control was accepted")
    return [{"name": "mixed_0_511_512_513_and_guard_regions", "known_answer": "PASS",
             "negative_output": "rejected"}]


class Fixture:
    def __init__(self, row, device="cuda"):
        import torch
        p = row["params"]
        self.width, self.fill = p["max_context_len"], p.get("initial_value", 0)
        self.batch, self.rows = p["B"], p["max_batch"]
        self.shape = (self.rows, self.width + p.get("row_padding", 0))
        generator = torch.Generator(device=device).manual_seed(p["seed"] + self.batch)
        self.output = torch.full(self.shape, self.fill, dtype=torch.int32, device=device)
        self.table = self.output[:, :self.width]
        rows = torch.randperm(self.rows, generator=generator, device=device)[:self.batch].to(torch.int64)
        pre = torch.tensor(p["pre_lens"], dtype=torch.int64, device=device)
        extend = torch.tensor(p["extend_lens"], dtype=torch.int64, device=device)
        seq = pre + extend
        # Preserve the original scored cases' RNG order: rows, cache, prefixes.
        cache = torch.randint(0, self.width, (int(extend.sum()),), dtype=torch.int64,
                              device=device, generator=generator)
        prefixes = [torch.randint(0, self.width, (int(n),), dtype=torch.int64,
                                  device=device, generator=generator) for n in pre.tolist()]
        pointers = torch.tensor([x.data_ptr() for x in prefixes], dtype=torch.uint64, device=device)
        self.inputs = (rows, pointers, pre, seq, extend, cache, *prefixes)
        self.originals = tuple(x.clone() for x in self.inputs)
        self.signatures = tuple(self.signature(x) for x in self.inputs)
        self.set_expected()

    @staticmethod
    def signature(value):
        return value.shape, value.stride(), value.dtype, value.device, value.data_ptr()

    def set_expected(self):
        import torch
        rows, _, pre, seq, extend, cache, *prefixes = [x.cpu().clone() for x in self.inputs]
        initial = torch.full(self.shape, self.fill, dtype=torch.int32)
        expected = reference(initial, rows, pre, seq, extend, cache, prefixes, self.width)
        poisoned = initial
        for row, sl in zip(rows.tolist(), seq.tolist()):
            poisoned[row, :sl] = POISON
        self.expected = expected.to(self.output.device)
        self.poisoned = poisoned.to(self.output.device)
        self.current_snapshots = tuple(x.clone() for x in self.inputs)

    def prepare(self):
        # Always restore/poison, including warmup and samples before TimedRun binds.
        self.output.copy_(self.poisoned)

    def call(self, kernel):
        kernel[(self.batch,)](self.table, *self.inputs[:6], self.table.stride(0))
        return self.output

    def unchanged(self, snapshots=None):
        import torch
        snapshots = self.current_snapshots if snapshots is None else snapshots
        for value, pristine, signature in zip(self.inputs, snapshots, self.signatures):
            if self.signature(value) != signature or not torch.equal(
                value.view(torch.uint8), pristine.view(torch.uint8)
            ):
                raise AssertionError("Read-only input or pointed-to prefix changed")

    def check(self, output):
        if output is not self.output:
            raise AssertionError("Measured output must be the caller's actual backing table")
        metrics = compare(output, self.expected)
        self.unchanged()
        return metrics

    def verify_replay(self, timed):
        if not timed.bound:
            raise AssertionError("Actual measured graph is unavailable")
        self.check(timed.outputs)
        for draw in ("values", "layout", "combined"):
            try:
                rows, _, pre, seq, extend, cache, *prefixes = self.inputs
                if draw in {"values", "combined"}:
                    for value in (cache, *prefixes):
                        value.add_(1).remainder_(self.width)
                if draw in {"layout", "combined"}:
                    rows.copy_(rows.roll(1))
                    pre.copy_(pre // 2)
                    extend.copy_(extend // 2)
                    seq.copy_(pre + extend)
                self.set_expected()
                self.check(timed.rerun())
            finally:
                for value, original in zip(self.inputs, self.originals):
                    value.copy_(original)
                self.set_expected()
        return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
                "replay_inputs_perturbed": True, "replay_output_poisoned": True,
                "replay_input_draws": 3, "oracle_device": "cpu",
                "untouched_regions_checked": True, "prefix_pointees_checked": True}


def run_case(row, action, kernel):
    import torch
    with torch.no_grad():
        fixture = Fixture(row)
        fixture.prepare()
        if action == "correctness":
            output = fixture.call(kernel)
            torch.cuda.synchronize()
            return {"status": "PASS", "metrics": fixture.check(output)}
        from _aka_benchmark import ReadOnlyCacheFlush, TimedRun, benchmark_cuda_graph_cold_cache_samples
        timed = TimedRun()
        timed.after_sample = fixture.check
        samples, metadata = benchmark_cuda_graph_cold_cache_samples(
            lambda: fixture.call(kernel), cache_flush=ReadOnlyCacheFlush(),
            warmup=10, repetition=100, prepare_fn=fixture.prepare, timed_run=timed)
        metadata.update(fixture.verify_replay(timed))
        metadata["raw_samples_ms"] = samples
        return {"status": "PASS", "execution_time_ms": sum(samples) / len(samples),
                "benchmark_method": metadata["benchmark_method"], "metadata": metadata}
