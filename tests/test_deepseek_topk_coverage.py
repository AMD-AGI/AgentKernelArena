"""CPU regressions for Top-k coverage and comparison, not GPU qualification."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/Aiter-task/topk_transform_paged_paged_k512_page_size64"
REQUIRED_LENGTHS = {
    0, 1, 63, 64, 65, 511, 512, 513, 1024, 2048, 4096, 8192,
    65536, 131072, 262207, 262208, 37, 65537,
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_covers_every_length_for_every_original_batch():
    contract = json.loads((TASK / "scripts/workload.json").read_text())
    covered = {}
    originals = []
    assert len(contract["rows"]) == len(contract["cases"]) == 107
    for row, case in zip(contract["rows"], contract["cases"]):
        workload = row["workload"]
        batch = workload["axes"]["batch"]
        regime = workload["runtime_lengths"]
        assert case["test_case_id"] == workload["uuid"]
        assert case["params"]["runtime_lengths"] == regime
        assert case["checks"] == ["correctness", "performance"]
        assert case["shape"]["scores"] == [batch, 262208]
        pattern = regime["seq_lens"]
        covered.setdefault(batch, set()).update(pattern[i % len(pattern)] for i in range(batch))
        assert len(pattern) == len(regime["replay_seq_lens"])
        assert all(a != b for a, b in zip(pattern, regime["replay_seq_lens"]))
        if regime["name"] == "original":
            originals.append(workload["uuid"])
            # Original lengths use this Python seed arithmetic, not device RNG.
            seed = int.from_bytes(hashlib.sha256(
                f"{contract['policy']['seed']}:{workload['uuid']}".encode()
            ).digest()[:8], "little") % (2**63)
            choices = [0, 262208 // 4, 262208 // 2, 262208]
            assert pattern == [choices[(index + seed) % 4]
                               for index in range(min(batch, 4))]
    assert len(originals) == 13
    assert set(covered) == {2**i for i in range(13)}
    assert all(REQUIRED_LENGTHS <= lengths for lengths in covered.values())
    for callback in ("initialize", "reference", "compare"):
        assert contract["definition"][callback] == (
            TASK / f"scripts/{callback}/main.py").read_text()
    assert "compare(actual, expected, *, seq_lens)" in contract["definition"]["description"]
    assert contract["bundle_readme"] == (TASK / "BUNDLE_README.md").read_text()


@pytest.fixture(autouse=True)
def assert_task_import_path_restored():
    # A leaked regular scripts package masks other tasks' namespace packages
    # even when their task roots are inserted earlier on the import path.
    original = sys.path.count(str(TASK))
    yield
    assert sys.path.count(str(TASK)) == original


@pytest.fixture
def topk(monkeypatch):
    torch = pytest.importorskip("torch")
    # The runner inserts its task root at import time. Snapshot sys.path so
    # monkeypatch also restores that direct mutation after this fixture.
    monkeypatch.syspath_prepend(str(TASK))
    package = types.ModuleType("scripts")
    package.__path__ = [str(TASK / "scripts")]
    monkeypatch.setitem(sys.modules, "scripts", package)
    for name in ("scripts.task_api", "scripts.task_inputs"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    runner = load_module("topk_test_runner", TASK / "scripts/task_runner.py")
    compare = load_module("topk_test_compare", TASK / "scripts/compare/main.py")
    reference = load_module("topk_test_reference", TASK / "scripts/reference/main.py")
    yield torch, runner, compare, reference
    # Remove task-local modules imported during the test before monkeypatch restores
    # any modules that predated this fixture.
    for name in ("scripts.task_api", "scripts.task_inputs"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("length", [511, 512, 513])
@pytest.mark.parametrize("dual", [False, True])
def test_order_boundary_uses_lengths_even_without_padding(topk, length, dual):
    torch, _, comparator, _ = topk
    raw = torch.arange(512, dtype=torch.int32).reshape(1, -1)
    mapped = raw * 3 + 1024
    if length < 512:
        raw[:, length:] = -1
        mapped[:, length:] = -1
    actual_raw, actual_mapped = raw.clone(), mapped.clone()
    active = min(length, 512)
    actual_raw[:, :active] = raw[:, :active].flip(-1)
    actual_mapped[:, :active] = mapped[:, :active].flip(-1)
    expected = {"out_page_indices": mapped, "out_raw_indices": raw} if dual else mapped
    actual = ({"out_page_indices": actual_mapped, "out_raw_indices": actual_raw}
              if dual else actual_mapped)
    lengths = torch.tensor([length], dtype=torch.int32)
    if length <= 512:
        with pytest.raises(AssertionError, match="short-row order"):
            comparator.run(actual, expected, seq_lens=lengths)
    else:
        comparator.run(actual, expected, seq_lens=lengths)


def test_comparator_rejects_wrong_selection_padding_and_missing_lengths(topk):
    torch, _, comparator, _ = topk
    expected = torch.arange(512, dtype=torch.int32).reshape(1, -1)
    actual = expected.clone()
    actual[0, -1] = 900
    with pytest.raises(AssertionError, match="sets differ"):
        comparator.run(actual, expected, seq_lens=torch.tensor([1024], dtype=torch.int32))
    expected[0, -1] = -1
    with pytest.raises(AssertionError, match="padding positions"):
        comparator.run(expected.roll(1, -1), expected,
                       seq_lens=torch.tensor([511], dtype=torch.int32))
    with pytest.raises(TypeError, match="seq_lens"):
        comparator.run(expected, expected)
    empty = torch.full_like(expected, -1)
    comparator.run(empty, empty, seq_lens=torch.tensor([0], dtype=torch.int32))


def small_contract():
    contract = json.loads((TASK / "scripts/workload.json").read_text())
    row = next(row for row in contract["rows"] if row["workload"]["axes"]["batch"] == 4)
    row["workload"]["axes"].update(width=1024, pages=16)
    row["workload"]["runtime_lengths"] = {
        "name": "test", "seq_lens": [0, 512, 513, 1024],
        "replay_seq_lens": [1, 513, 514, 0],
    }
    return contract, row


def test_refill_preserves_explicit_regime_and_graph_bound_storage(topk):
    torch, runner, _, _ = topk
    contract, row = small_contract()
    definition, policy = contract["definition"], contract["policy"]
    values = runner.make_inputs(definition, row, policy, device="cpu")
    pointers = {name: value.data_ptr() for name, value in values.items()
                if isinstance(value, torch.Tensor)}
    scores = values["scores"].clone()
    assert values["seq_lens"].tolist() == [0, 512, 513, 1024]
    runner.refill_inputs(values, definition, row, policy, device="cpu")
    assert not torch.equal(scores, values["scores"])
    assert values["seq_lens"].tolist() == [0, 512, 513, 1024]
    runner.apply_runtime_lengths(values, row, replay=True)
    assert values["seq_lens"].tolist() == [1, 513, 514, 0]
    assert values["metadata"][0].tolist() == [2**31 - 1, 0]
    assert not values["metadata"][1:].any()
    assert pointers == {name: values[name].data_ptr() for name in pointers}


@pytest.mark.parametrize("cache_lengths", [False, True])
def test_measured_replay_checks_changed_lengths(topk, monkeypatch, cache_lengths):
    torch, runner, _, reference = topk
    contract, row = small_contract()
    definition, policy = contract["definition"], contract["policy"]
    values = runner.make_inputs(definition, row, policy, device="cpu")
    destination = torch.empty((4, 512), dtype=torch.int32)
    saved_lengths = values["seq_lens"].clone()
    replaying = False

    def launch(**kwargs):
        if cache_lengths and replaying:
            kwargs = {**kwargs, "seq_lens": saved_lengths}
        reference.topk_reference(**kwargs, out_page_indices=destination)
        return destination

    def fake_benchmark(launch_fn, *, timed_run, **kwargs):
        nonlocal replaying
        result = launch_fn()
        replaying = True
        timed_run._bind(launch_fn, result)
        return 1.0, {"benchmark_method": "cpu-test"}

    module = types.ModuleType("_aka_benchmark")
    # Exercise the same replay holder that the materialized GPU helper exports.
    canonical = load_module("topk_canonical_benchmark", ROOT / "src/tools/perf/aka_benchmark.py")
    module.TimedRun = canonical.TimedRun
    module.benchmark_cuda_graph_or_events = fake_benchmark
    monkeypatch.setitem(sys.modules, "_aka_benchmark", module)
    if cache_lengths:
        with pytest.raises(AssertionError, match="sets differ"):
            runner.measure_case(launch, reference.run, values, definition, row, policy, device="cpu")
    else:
        result = runner.measure_case(launch, reference.run, values, definition, row, policy, device="cpu")
        assert result["metadata"]["changed_length_replay_validated"]
        assert result["metadata"]["refilled_input_replay_validated"]
        assert result["metadata"]["exact_graph_replay_validated"]


def test_production_loader_isolates_role_sources_and_content_caches(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from torch.utils import cpp_extension

    builds, calls = [], []
    monkeypatch.setattr(torch.version, "hip", "test-runtime")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "cache"))

    def build(**kwargs):
        staged = Path(kwargs["sources"][0])
        assert staged.parent == Path(kwargs["build_directory"])
        assert staged.is_relative_to(tmp_path / "cache")
        assert kwargs["with_cuda"] is True
        assert "--offload-arch=gfx950" in kwargs["extra_cuda_cflags"]
        builds.append(kwargs)
        return types.SimpleNamespace(run=lambda *args: calls.append(args))

    monkeypatch.setattr(cpp_extension, "load", build)
    values = {name: object() for name in (
        "scores", "seq_lens", "metadata", "page_tables", "out_page_indices")}
    values["page_size"] = 64
    sources = []
    for relative in ("scripts/baseline/main.py", "source/implementation/main.py"):
        path = TASK / relative
        sources.append(path.with_name("topk_kernel.cu").read_bytes())
        adapter = load_module("topk_adapter_test", path)
        assert adapter.run(**values) is None
        assert adapter.run(**values) is None  # The process builds only once per role.
    assert sources[0] == sources[1]
    assert len(builds) == 2
    assert builds[0]["name"] == builds[1]["name"]
    assert Path(builds[0]["sources"][0]).read_bytes() == sources[0]
    assert all(args == (values["scores"], values["seq_lens"], values["page_tables"],
                        values["out_page_indices"], 64, None) for args in calls)

    # An edited initial implementation cannot reuse the frozen baseline binary.
    changed = tmp_path / "candidate"
    changed.mkdir()
    (changed / "main.py").write_text((TASK / "source/implementation/main.py").read_text())
    (changed / "topk_kernel.cu").write_bytes(sources[1] + b"\n// candidate change\n")
    edited = load_module("topk_edited_adapter_test", changed / "main.py")
    edited.run(**values)
    assert builds[-1]["name"] != builds[0]["name"]
    assert (TASK / "scripts/baseline/topk_kernel.cu").read_bytes() == sources[0]


def test_gpu_production_overflow_boundary_and_refilled_graph(topk):
    torch, _, comparator, reference = topk
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("Requires MI355X and the pinned ROCm build dependencies")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("This production task is qualified for gfx950")
    adapter = load_module("topk_gpu_adapter_test", TASK / "scripts/baseline/main.py")
    width = 65536
    # Every score falls in the same FP16 coarse bin but has a distinct FP32 key.
    values = {
        "scores": (0.5 + torch.arange(width, dtype=torch.float32, device="cuda") / 2**24)
                  .repeat(4, 1),
        "seq_lens": torch.tensor([6144, 6145, 8192, width], dtype=torch.int32, device="cuda"),
        "page_tables": torch.arange(width // 64, dtype=torch.int32, device="cuda")
                       .flip(0).repeat(4, 1),
        "metadata": torch.zeros((5, 2), dtype=torch.int32, device="cuda"),
        "page_size": 64,
    }
    values["metadata"][0, 0] = 2**31 - 1
    destination = torch.empty((4, 512), dtype=torch.int32, device="cuda")
    launch = lambda: adapter.run(**values, out_page_indices=destination)
    expected = reference.run(**values)
    launch()  # Compile before capture.
    comparator.run(destination, expected, seq_lens=values["seq_lens"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    values["seq_lens"].copy_(torch.tensor([6145, 6144, width, 8192], dtype=torch.int32, device="cuda"))
    values["scores"].neg_()
    expected = reference.run(**values)
    destination.fill_(-99)
    graph.replay()
    comparator.run(destination, expected, seq_lens=values["seq_lens"])
