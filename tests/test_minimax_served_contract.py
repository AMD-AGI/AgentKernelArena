"""CPU regression tests for MiniMax's protected served-case implementation.

Synthetic tensors exercise the adapter and independent reference only. They are
never task fixtures or evidence of native GPU execution.
"""
import ast
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib

import pytest

REPO = Path(__file__).resolve().parents[1]
TASKS = [REPO / "tasks/headkernel" / name for name in (
    "minimax-m3__decode_score_kernel", "minimax-m3__gqa_share_sparse_decode_kernel",
    "minimax-m3__gqa_share_sparse_fwd_kernel")]


@pytest.fixture(scope="module")
def modules():
    names = ("evaluation_contract", "served_contract", "source_guard", "minimax_work", "minimax_fixtures",
             "minimax_coverage", "minimax_data", "minimax_reference")
    previous = {name: sys.modules.get(name) for name in names}
    loaded = {}
    for name in names:
        spec = importlib.util.spec_from_file_location(name, TASKS[0] / "ut" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    yield loaded
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def args_for(torch, kind):
    gen = torch.Generator().manual_seed(42)
    args = {"k_cache": torch.randn(16, 1, 2, generator=gen),
            "v_cache": torch.randn(16, 1, 3, generator=gen),
            "req_to_token": (torch.arange(40).reshape(4, 10) % 16).to(torch.int32),
            "sm_scale": 0.7, "q_scale": 0.5, "k_scale": 1.5, "v_scale": 0.8,
            "use_tma": True, "sink": torch.randn(2, 2, generator=gen)}
    if kind == "sparse_decode":
        args.update(q=torch.randn(3, 2, 2, generator=gen), block_size=2,
                    seq_lens=torch.tensor([7, 4, 0], dtype=torch.int32),
                    slot_ids=torch.tensor([0, 2, 3], dtype=torch.int64),
                    topk_idx=torch.tensor([[[0, 0, 1, 3], [0, 1, -1, -1], [-1, -1, -1, -1]]], dtype=torch.int32))
    else:
        args.update(q=torch.randn(8, 2, 2, generator=gen), block_size_q=2, block_size_k=2,
                    seq_lens=torch.tensor([9, 7], dtype=torch.int32),
                    slot_ids=torch.tensor([0, 2], dtype=torch.int64),
                    cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
                    prefix_lens=torch.tensor([4, 4], dtype=torch.int32),
                    cu_seqblocks_q=torch.tensor([0, 3, 5], dtype=torch.int32),
                    max_seqlen=5, num_q_loop=1,
                    topk_idx=torch.tensor([[[0, 0, 2, 4], [0, 1, 2, 4], [0, 1, 2, 4],
                                           [0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.int32))
    return args


def case_for(torch, modules, args, kind="sparse_decode"):
    contract, data = modules["served_contract"], modules["minimax_data"]
    tensors, scalars, aliases, storage = {}, {}, {}, {}
    for name, value in args.items():
        if torch.is_tensor(value):
            tensors[name] = {"role": "input", "shape": list(value.shape), "strides": list(value.stride()),
                             "storage_offset": value.storage_offset(), "dtype": str(value.dtype).removeprefix("torch."),
                             "device_type": "cuda"}
            key = value.untyped_storage()._cdata
            aliases[name] = storage.setdefault(key, "storage-" + str(len(storage)))
        else:
            scalars[name] = value
    shape = [args["q"].shape[0], args["q"].shape[1], args["q"].shape[-1] if kind == "sparse_decode" else args["v_cache"].shape[-1]]
    output = torch.empty(shape)
    tensors["result"] = {"role": "output", "shape": shape, "strides": list(output.stride()),
                         "storage_offset": 0, "dtype": "float32", "device_type": "cuda"}
    scalars.update(data.work_controls(args))
    geometry = {name: contract.encode_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())
                for name, value in args.items() if torch.is_tensor(value) and name not in contract.FLOAT_INPUTS}
    case = {"kind": kind, "production_mode": "graph", "tensors": tensors, "scalars": scalars,
            "input_aliases": aliases, "output_structure": [{"name": "result", "kind": "tensor"}],
            "calls_per_sample": 1, "occurrences": 8,
            "states": [{"state_id": "synthetic-unit-test", "fixture_sha256": "a"*64,
                        "served": True, "startup_values": False, "graph_id": "graph64",
                        "served_context": {"tp_rank": 0}, "geometry": geometry}]}
    case["case_id"] = contract.case_identity(case)
    definition = {"kind": kind, "runtime_image": "test/image@sha256:"+"b"*64,
                  "source_sha256": "c"*64, "arguments": list(args), "max_replay_storage_bytes": 1 << 20}
    return case, definition


def manifest_for(case, definition):
    return {"schema_version": 1, "runtime_image": definition["runtime_image"], "cases": [case],
            "measurement": {"method": "cuda_graph", "warmup_iterations": 10, "benchmark_iterations": 100,
                            "correctness_seeds": [0, 1, 2], "refresh_inputs": "each_replay",
                            "initialize_outputs": "each_replay", "validate_outputs": "each_replay",
                            "negative_controls": ["no_op", "wrong_output"],
                            "refresh_addressing": "translated_pages_and_legal_block_permutations"},
            "capture": {"served_only": True, "startup_values": False, "complete": True,
                        "dropped_case_count": 0, "source_sha256": definition["source_sha256"],
                        "run_id": "unit-test-only", "manifest_sha256": "d"*64,
                        "required_case_ids": [case["case_id"]], "represented_calls": 8, "target_calls": 8,
                        "graph_replays_by_rank": {"0": {"graph64": 2}}}}


def test_common_helper_copies_and_canonical_contract_match():
    for name in ("served_contract.py", "source_guard.py", "minimax_data.py", "minimax_reference.py", "minimax_native.py",
                 "minimax_work.py", "minimax_fixtures.py", "minimax_coverage.py"):
        assert len({(task/"ut"/name).read_bytes() for task in TASKS}) == 1
    for task in TASKS:
        assert (task/"ut/evaluation_contract.py").read_bytes() == (REPO/"src/task_contract.py").read_bytes()


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_shipped_source_passes_guard(modules, task):
    assert modules["source_guard"].validate_sources(task, task)


def candidate_with_body(tmp_path, task, body):
    definition = json.loads((task/"task_definition.json").read_text())
    text = (task/definition["source_file"]).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == definition["kernels"][0])
    lines = text.splitlines(keepends=True)
    lines[node.body[0].lineno-1:node.end_lineno] = ["    "+line+"\n" for line in body.splitlines()]
    path = tmp_path/definition["source_file"]
    path.parent.mkdir(parents=True)
    path.write_text("".join(lines))
    return path


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_noop_body_reaches_numerical_gate(modules, tmp_path, task):
    candidate_with_body(tmp_path, task, "return")
    assert modules["source_guard"].validate_sources(tmp_path, task)


@pytest.mark.parametrize("body", ["import os", "os._exit(0)", "print('fake pass')", "tl.__dict__",
                                  "eval('1')", "(lambda: 0)()", "x = q_ptr\nx.__class__",
                                  "def nested():\n    return 1", "torch.cuda.synchronize()"])
def test_host_effects_rejected(modules, tmp_path, body):
    candidate_with_body(tmp_path, TASKS[0], body)
    with pytest.raises(ValueError):
        modules["source_guard"].validate_sources(tmp_path, TASKS[0])


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
@pytest.mark.parametrize("body", [
    'tl.core.builtins.exec("raise RuntimeError(\'guard_escape_compile_probe\')")',
    'min = tl.core.builtins.exec\nmin("raise RuntimeError(\'guard_escape_compile_probe\')")',
    "tl = torch\ntl.cuda.synchronize()",
    "namespace = tl.core\nreturn",
    "namespace = tl\nreturn",
    "callback = tl.max\nreturn",
    "compiler_function = tl.max.fn\nreturn",
    "namespace = torch\nreturn",
    "tl.core.reshape(0)",
    "tl.static_print('fake pass')",
])
def test_host_namespace_escapes_rejected(modules, tmp_path, task, body):
    candidate_with_body(tmp_path, task, body)
    with pytest.raises(ValueError):
        modules["source_guard"].validate_sources(tmp_path, task)


@pytest.mark.parametrize("name", ["tl", "range", "min", "max", "int", "float", "abs", "len"])
@pytest.mark.parametrize("binding", [
    "{name} = 0", "{name}: tl.constexpr = 0", "{name} += 1", "({name}, other) = (0, 1)",
    "*{name}, other = (0, 1)", "for {name} in range(1):\n    pass",
    "values = [0 for {name} in range(1)]", "if ({name} := 0):\n    pass", "del {name}",
])
def test_callable_and_namespace_bindings_are_frozen(modules, tmp_path, name, binding):
    candidate_with_body(tmp_path, TASKS[0], binding.format(name=name))
    with pytest.raises(ValueError):
        modules["source_guard"].validate_sources(tmp_path, TASKS[0])


@pytest.mark.parametrize("body", [
    "torch = 0\nreturn", "namespace = torch\ntorch = 0", "exec = 0\nreturn",
    "callback = min\nreturn", "callback = q_ptr.to\nreturn",
    "tl.float32 = 0", "q_ptr.dtype = tl.float32", "q_ptr.dtype.element_ty = tl.float32",
    "metadata = q_ptr.device", "tl.constexpr(0)",
    "match q_ptr:\n    case tl:\n        pass",
])
def test_global_aliases_callable_inspection_and_mutation_rejected(modules, tmp_path, body):
    candidate_with_body(tmp_path, TASKS[0], body)
    with pytest.raises(ValueError):
        modules["source_guard"].validate_sources(tmp_path, TASKS[0])


def test_public_dsl_tensor_methods_and_metadata_remain_editable(modules, tmp_path):
    candidate_with_body(tmp_path, TASKS[0], "\n".join([
        "width: tl.constexpr = 16",
        "offsets = tl.arange(0, width)",
        "values = tl.load(q_ptr + offsets, mask=offsets < head_dim, other=0)",
        "values = values.to(tl.float32).reshape((4, 4)).trans(1, 0)",
        "values = values.astype(tl.float16)",
        "dtype = score_ptr.dtype.element_ty",
        "shape = values.shape",
        "values = values.to(dtype)",
        "for idx in range(min(1, max(1, len(shape)))):",
        "    values = tl.maximum(values, float('-inf')) + abs(int(0))",
        "tl.static_assert(width == 16)",
        "return",
    ]))
    assert modules["source_guard"].validate_sources(tmp_path, TASKS[0])


@pytest.mark.parametrize("edit", [lambda s: "import os\n"+s,
    lambda s: s.replace("def flash_decode_with_topk_idx(", "def changed_wrapper("),
    lambda s: s.replace("num_warps=4, num_stages=1", "num_warps=8, num_stages=1")])
def test_protected_host_and_decorators_rejected(modules, tmp_path, edit):
    path = candidate_with_body(tmp_path, TASKS[0], "return")
    path.write_text(edit(path.read_text()))
    with pytest.raises(ValueError):
        modules["source_guard"].validate_sources(tmp_path, TASKS[0])


def test_codec_roundtrip_empty_hash_bounds_and_extra_stream(modules):
    c = modules["served_contract"]
    for raw in (b"", b"\0"*200, bytes(range(256))*50):
        assert c.decode_bytes(c.encode_bytes(raw)) == raw
    item = c.encode_bytes(b"abc")
    with pytest.raises(ValueError):
        c.decode_bytes(item, limit=2)
    wrong = {**item, "sha256": "0"*64}
    with pytest.raises(ValueError):
        c.decode_bytes(wrong)
    extra = {**item, "data": base64.b64encode(base64.b64decode(item["data"])*2).decode()}
    with pytest.raises(ValueError):
        c.decode_bytes(extra)


def test_chunked_geometry_shares_repeated_pages_and_checks_every_byte(modules, tmp_path):
    c = modules["served_contract"]
    data = bytearray(2 << 20)
    data[4096:8192] = bytes(range(256))*16
    first = c.encode_geometry(data, tmp_path)
    assert first["encoding"] == "zlib-chunk-sha256-v1"
    assert c.decode_bytes(first, root=tmp_path) == data
    assert len(list((tmp_path/"cases/geometry").glob("*.zlib"))) == 2
    assert len(json.dumps(first)) < 2000
    data[8192:12288] = b"x"*4096
    second = c.encode_geometry(data, tmp_path)
    assert c.decode_bytes(second, root=tmp_path) == data
    assert len(list((tmp_path/"cases/geometry").glob("*.zlib"))) == 3
    with pytest.raises(ValueError, match="missing task root"):
        c.decode_bytes(second)
    checksum = hashlib.sha256(bytes(range(256))*16).hexdigest()
    (tmp_path/"cases/geometry"/(checksum+".zlib")).write_bytes(zlib.compress(b"z"*4096))
    with pytest.raises(ValueError, match="chunk hash"):
        c.decode_bytes(first, root=tmp_path)


def test_chunked_geometry_rejects_symlink_paths(modules, tmp_path):
    c = modules["served_contract"]
    task, outside = tmp_path/"task", tmp_path/"outside"
    task.mkdir(); outside.mkdir()
    (task/"cases").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes task"):
        c.encode_geometry(b"x"*70000, task)


def test_chunked_paging_geometry_loads_through_case_validation_and_inputs(torch, modules, tmp_path):
    case, definition = case_for(torch, modules, args_for(torch, "sparse_decode"))
    c = modules["served_contract"]
    geometry = case["states"][0]["geometry"]
    geometry["req_to_token"] = c.encode_geometry(c.decode_bytes(geometry["req_to_token"]), tmp_path, inline_limit=0)
    c.validate_cases(manifest_for(case, definition), definition, tmp_path)
    inputs = modules["minimax_data"].Inputs(case, definition, device="cpu", root=tmp_path)
    before = inputs.reset(1)
    inputs.assert_immutable(before)


def test_one_state_per_abi_retains_two_actual_graph_replays(torch, modules):
    case, definition = case_for(torch, modules, args_for(torch, "sparse_decode"))
    manifest = manifest_for(case, definition)
    assert modules["served_contract"].validate_cases(manifest, definition)
    manifest["capture"]["graph_replays_by_rank"]["0"]["graph64"] = 1
    with pytest.raises(ValueError, match="two actual served"):
        modules["served_contract"].validate_cases(manifest, definition)


@pytest.mark.parametrize("change", [lambda c: c.update(production_mode="eager"),
    lambda c: c["tensors"]["q"].update(storage_offset=1),
    lambda c: c["tensors"]["q"].update(strides=[5, 2, 1]),
    lambda c: c["scalars"].update(sm_scale=1),
    lambda c: c["input_aliases"].update(q="other")])
def test_identity_preserves_full_mode_geometry_and_scalar_types(torch, modules, change):
    case, _ = case_for(torch, modules, args_for(torch, "sparse_decode"))
    before = case["case_id"]
    change(case)
    assert modules["served_contract"].case_identity(case) != before


@pytest.mark.parametrize("kind", ["sparse_decode", "sparse_prefill"])
def test_refresh_preserves_work_and_partial_causal_boundaries(torch, modules, kind):
    args = args_for(torch, kind)
    case, definition = case_for(torch, modules, args, kind)
    inputs = modules["minimax_data"].Inputs(case, definition, device="cpu")
    original = inputs.states[0]
    seen = set()
    for seed in range(10):
        state = inputs._fresh_geometry(original, seed)
        inputs.validate_geometry(state)
        seen.add(tuple(state["req_to_token"].reshape(-1).tolist()))
        assert modules["minimax_data"].work_controls({**args, **state}) == modules["minimax_data"].work_controls(args)
        block = args["block_size"] if kind == "sparse_decode" else args["block_size_k"]
        batch_for = list(range(args["seq_lens"].numel())) if kind == "sparse_decode" else [0, 0, 0, 1, 1]
        for index, batch in enumerate(batch_for):
            old, new = original["topk_idx"][0, index], state["topk_idx"][0, index]
            length = int(args["seq_lens"][batch])
            work = lambda x: sorted(min(block, length-int(i)*block) for i in x if i >= 0)
            assert work(old) == work(new)
            mult = lambda x: sorted(int((x == i).sum()) for i in x.unique() if i >= 0)
            assert mult(old) == mult(new)
            if kind == "sparse_prefill":
                local = index - (0 if batch == 0 else 3)
                for qpos in range(int(args["prefix_lens"][batch])+2*local,
                                  min(int(args["prefix_lens"][batch])+2*local+2, length)):
                    causal = lambda x: sorted(max(0, min(block, length-int(i)*block, qpos-int(i)*block+1)) for i in x if i >= 0)
                    assert causal(old) == causal(new)
    assert len(seen) > 1


@pytest.mark.parametrize("bad", [lambda s: s["seq_lens"].fill_(-1),
                                 lambda s: s["seq_lens"].fill_(11),
                                 lambda s: s["slot_ids"].fill_(7),
                                 lambda s: s["topk_idx"].fill_(7),
                                 lambda s: s["topk_idx"][0, 0].copy_(__import__("torch").tensor([0, -1, 1, -1]))])
def test_bad_geometry_rejected_before_remap(torch, modules, bad):
    case, definition = case_for(torch, modules, args_for(torch, "sparse_decode"))
    inputs = modules["minimax_data"].Inputs(case, definition, device="cpu")
    state = {name: value.clone() for name, value in inputs.states[0].items()}
    bad(state)
    with pytest.raises(ValueError):
        inputs._fresh_geometry(state, 0)


def test_reset_preserves_alias_offsets_strides_and_detects_padding_write(torch, modules):
    args = args_for(torch, "sparse_decode")
    backing = torch.empty(100)
    args["q"] = backing.as_strided((3, 2, 2), (10, 3, 1), 5)
    args["sink"] = backing.as_strided((2, 2), (3, 1), 50)
    case, definition = case_for(torch, modules, args)
    inputs = modules["minimax_data"].Inputs(case, definition, device="cpu")
    before = inputs.reset(4)
    q, sink = inputs.tensors["q"], inputs.tensors["sink"]
    assert q.untyped_storage()._cdata == sink.untyped_storage()._cdata
    assert q.stride() == (10, 3, 1) and q.storage_offset() == 5 and sink.storage_offset() == 50
    inputs.assert_immutable(before)
    view = inputs.reference_args(before)["q"]
    assert torch.equal(q, view) and view.stride() == q.stride()
    old = q.clone()
    inputs.reset(5)
    assert not torch.equal(old, q)
    before = inputs.reset(4)
    inputs.storage[inputs.aliases["q"]][0] = 1
    with pytest.raises(AssertionError, match="padding"):
        inputs.assert_immutable(before)


def test_runtime_alias_drift_is_rejected(torch, modules):
    args = args_for(torch, "sparse_decode")
    args["v_cache"] = args["k_cache"]
    case, definition = case_for(torch, modules, args)
    inputs = modules["minimax_data"].Inputs(case, definition, device="cpu")
    inputs.reset(0)
    inputs.observe_arguments(torch.empty(3, 2, 2))
    inputs.tensors["v_cache"] = inputs.tensors["v_cache"].clone()
    with pytest.raises(ValueError, match="alias"):
        inputs.observe_arguments(torch.empty(3, 2, 2))


def test_fresh_sink_is_observable_at_long_context(torch, modules):
    args = {"q": torch.empty(1, 2, 128), "sink": torch.empty(2, 128),
            "k_cache": torch.empty(2048, 1, 128), "v_cache": torch.empty(2048, 1, 128),
            "req_to_token": torch.arange(2048, dtype=torch.int32).reshape(1, -1),
            "seq_lens": torch.tensor([2048], dtype=torch.int32), "slot_ids": torch.tensor([0]),
            "topk_idx": torch.arange(16, dtype=torch.int32).reshape(1, 1, 16),
            "block_size": 128, "sm_scale": None, "q_scale": None, "k_scale": None, "v_scale": None}
    case, definition = case_for(torch, modules, args)
    definition["max_replay_storage_bytes"] = 8 << 20
    inputs = modules["minimax_data"].Inputs(case, definition, device="cpu")
    before = inputs.reset(1)
    actual_args = inputs.reference_args(before)
    assert torch.equal(actual_args["sink"], actual_args["q"][0])
    ref = modules["minimax_reference"]
    expected = ref.sparse_attention(actual_args, "sparse_decode")
    ignored = ref.sparse_attention({**actual_args, "sink": None}, "sparse_decode")
    with pytest.raises(AssertionError):
        ref.mixed_close(ignored, expected)


def test_package_identity_matches_trusted_host_before_build_mount(modules, tmp_path):
    from src.tools.seed_aiter_jit_cache import tree_manifest
    (tmp_path/"source").mkdir()
    (tmp_path/"source/kernel.py").write_text("# synthetic package-identity fixture\n")
    (tmp_path/"config.yaml").write_text("schema: test-only\n")
    path = TASKS[0]/"scripts/task_runner.py"
    spec = importlib.util.spec_from_file_location("minimax_runner_identity_test", path)
    runner = importlib.util.module_from_spec(spec)
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        spec.loader.exec_module(runner)
        runner.ROOT = tmp_path
        expected = modules["evaluation_contract"].fingerprint(tree_manifest(tmp_path))
        (tmp_path/"build").mkdir()
        (tmp_path/"build/ignored-report.json").write_text("{}\n")
        assert runner.tree_identity() == expected
        (tmp_path/"source/kernel.py").write_text("# modified source\n")
        assert runner.tree_identity() != expected
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def scalar_attention(torch, args, kind):
    q, keys, values = args["q"], args["k_cache"], args["v_cache"]
    dim = q.shape[-1] if kind == "sparse_decode" else values.shape[-1]
    out = torch.empty(q.shape[0], q.shape[1], dim)
    for token in range(q.shape[0]):
        if kind == "sparse_decode":
            batch, qblock, limit, block = token, token, 10**9, args["block_size"]
        else:
            batch = 0 if token < int(args["cu_seqlens"][1]) else 1
            local = token-int(args["cu_seqlens"][batch])
            qblock = int(args["cu_seqblocks_q"][batch])+local//args["block_size_q"]
            limit, block = int(args["prefix_lens"][batch])+local, args["block_size_k"]
        row, length = int(args["slot_ids"][batch]), int(args["seq_lens"][batch])
        positions = [int(i)*block+j for i in args["topk_idx"][0, qblock] if i >= 0
                     for j in range(block) if int(i)*block+j < length and int(i)*block+j <= limit]
        for head in range(q.shape[1]):
            logits, vals = [], []
            scale = args["sm_scale"]*args["q_scale"]
            for pos in positions:
                physical = int(args["req_to_token"][row, pos]) % keys.shape[0]
                logits.append((q[token, head]*keys[physical, 0]).sum()*scale*args["k_scale"])
                vals.append(values[physical, 0, :dim]*args["v_scale"])
            if args["sink"] is not None:
                logits.append((q[token, head]*args["sink"][head]).sum()*scale)
                vals.append(torch.zeros(dim))
            out[token, head] = ((torch.softmax(torch.stack(logits), 0)[:, None]*torch.stack(vals)).sum(0)
                                if logits else torch.full((dim,), float("nan")))
    return out


@pytest.mark.parametrize("kind", ["sparse_decode", "sparse_prefill"])
@pytest.mark.parametrize("has_sink", [True, False])
@pytest.mark.parametrize("chunk", [1, 2, 256])
def test_sparse_reference_matches_scalar_softmax_with_duplicates_and_causality(torch, modules, kind, has_sink, chunk):
    args = args_for(torch, kind)
    if not has_sink:
        args["sink"] = None
    actual = modules["minimax_reference"].sparse_attention(args, kind, query_blocks_per_chunk=chunk)
    expected = scalar_attention(torch, args, kind)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_score_reference_bias_and_cutoff_reject_wrong_selection(torch, modules, score_type):
    ref = modules["minimax_reference"]
    args = {"q": torch.tensor([[[1.0, 0.0]]]),
            "k_cache": torch.tensor([[[float(i), 0.0]] for i in range(8)]),
            "req_to_token": torch.arange(8, dtype=torch.int32).reshape(1, 8),
            "seq_lens": torch.tensor([8]), "slot_ids": torch.tensor([0]),
            "block_size": 2, "topk": 2, "sm_scale": 1.0, "q_scale": 1.0, "k_scale": 1.0,
            "score_type": score_type, "init_blocks": 1, "local_blocks": 1}
    scores, counts, topk = ref.score_reference(args)
    assert scores[0, 0, 0] == 1e30 and scores[0, 0, 3] == 1e29
    ref.check_topk(torch.tensor([[[0, 3]]], dtype=torch.int32), scores, counts, topk)
    for wrong in ([[[1, 3]]], [[[0, 0]]], [[[3, 0]]], [[[0, -1]]]):
        with pytest.raises(AssertionError):
            ref.check_topk(torch.tensor(wrong, dtype=torch.int32), scores, counts, topk)
    args.update(init_blocks=0, local_blocks=0)
    scores, _, _ = ref.score_reference(args)
    expected = torch.arange(8).float().reshape(4, 2)
    expected = expected.amax(-1) if score_type == "max" else torch.logsumexp(expected, -1)
    torch.testing.assert_close(scores[0, 0], expected/torch.log(torch.tensor(2.0)))


def test_output_poison_and_mixed_oracle_reject_stale_or_wrong_values(torch, modules):
    ref, data = modules["minimax_reference"], modules["minimax_data"]
    expected = torch.tensor([1.0, -2.0, 0.3])
    output = expected.clone()
    ref.mixed_close(output, expected)
    data.initialize_outputs(output)
    with pytest.raises(AssertionError):
        ref.mixed_close(output, expected)
    with pytest.raises(AssertionError):
        ref.mixed_close(torch.zeros_like(expected), expected)
    indices = torch.tensor([0, 1, 3], dtype=torch.int32)
    data.initialize_outputs((None, indices))
    assert (indices == torch.iinfo(torch.int32).min).all()


@pytest.mark.parametrize("task", TASKS, ids=lambda p: p.name)
def test_sealed_cases_preserve_all_variants_calls_and_provenance(modules, task):
    manifest = json.loads((task/"cases.json").read_text())
    definition = json.loads((task/"task_definition.json").read_text())
    certificate = modules["minimax_coverage"].validate_coverage(manifest, definition, task)
    expected = {"decode_score": (16, 0), "sparse_decode": (24, 8), "sparse_prefill": (40, 20)}
    cases, transfers = expected[definition["kind"]]
    assert len(manifest["cases"]) == cases
    assert sum("fixture_transfer" in case for case in manifest["cases"]) == transfers
    assert certificate["total_observed_calls"] == 178009*8
    assert certificate["supplemental_observed_calls"] == 412338
    assert manifest["capture"]["target_calls"] == sum(case["occurrences"] for case in manifest["cases"])
    for case in manifest["cases"]:
        assert modules["served_contract"].case_identity(case) == case["case_id"]
        modules["served_contract"].validate_distribution(case)


@pytest.mark.parametrize("edit", [
    lambda row: row.update(tolerance=0.03),
    lambda row: row["actual_native_launch"]["last_launches"][0].update(num_warps=64),
    lambda row: row["actual_native_launch"]["last_launches"][0].update(shared=0),
    lambda row: row["actual_native_launch"].update(production_namespace_rebound=True),
    lambda row: row["recorded_states"][0].update(fixture_sha256="0"*64),
    lambda row: row["recorded_states"][0].update(captured_output_parity=False),
    lambda row: row["recorded_states"][0].update(independent_math_parity=False),
    lambda row: row["recorded_states"].pop(),
    lambda row: row.update(representative_case_key="invented-capture"),
])
def test_supplemental_replay_rejects_changed_identity_launch_or_parity(modules, edit):
    task = TASKS[1]
    certificate = json.loads((task/"provenance/COVERAGE.json").read_text())
    entry = next(row for row in certificate["variants"].values() if "native_replay" in row)
    row = deepcopy(entry["native_replay"])
    modules["minimax_coverage"].validate_replay(row, entry["schema"], row["representative_case_key"], entry["states"])
    key = row["representative_case_key"]
    edit(row)
    with pytest.raises(ValueError):
        modules["minimax_coverage"].validate_replay(row, entry["schema"], key, entry["states"])


def test_raw_bundle_retains_byte_exact_source_and_full_physical_abi(modules, tmp_path):
    codec = modules["minimax_fixtures"]
    meta = {"shape": [2, 3], "stride": [5, 1], "storage_offset": 4,
            "dtype": "torch.float32", "alias": "input-0", "storage_nbytes": 128}
    fixture = {"source_sha256": "a"*64, "family": "sparse_decode", "startup_values": False,
               "origin": "served_graph", "case_key": "original-case", "served": {"tp_rank": 0},
               "tensor_controls": {}, "inputs": {"q": meta}, "outputs": {"result": None},
               "controls": {"operator_scalars": {"sm_scale": 1.0}}}
    source = json.dumps(fixture, indent=2)+"\n"
    state = {"fixture_sha256": hashlib.sha256(source.encode()).hexdigest(),
             "served_context": fixture["served"], "tensor_controls": {}}
    case = {"case_id": "synthetic-bundle-test", "production_mode": "graph", "source_capture_key": "original-case",
            "states": [state], "tensors": {"q": codec.specification(meta, "input")},
            "input_aliases": {"q": "input-0"}, "original_storage_nbytes": {"q": 128},
            "scalars": {"sm_scale": 1.0, "result": None}}
    bundle = {"schema": "served-tensor-fixture-v1", "bundle_schema": "minimax-representatives-v1",
              "case_id": case["case_id"], "representatives": [{"source_json": source,
                  "source_sha256": state["fixture_sha256"], "storage_directory": "rank-0"}]}
    path = tmp_path/"fixtures/case.json"; path.parent.mkdir(); path.write_text(json.dumps(bundle))
    case["fixture"] = {"path": "fixtures/case.json", "sha256": codec.file_hash(path)}
    definition = {"source_sha256": "a"*64, "kind": "sparse_decode", "arguments": ["q", "sm_scale"]}
    assert codec.load_bundle(tmp_path, case, definition)[0]["fixture"] == fixture
    case["original_storage_nbytes"]["q"] = 124
    with pytest.raises(ValueError, match="physical input ABI"):
        codec.load_bundle(tmp_path, case, definition)
    case["original_storage_nbytes"]["q"] = 128
    bundle["representatives"][0]["source_json"] = source+" "
    path.write_text(json.dumps(bundle)); case["fixture"]["sha256"] = codec.file_hash(path)
    with pytest.raises(ValueError, match="original served fixture bytes"):
        codec.load_bundle(tmp_path, case, definition)
