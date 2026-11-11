"""Multi-GPU (distributed) schema-v3 tasks: contract, workspace, scheduling, audit and 2-rank evaluation.

CPU only. A toy two-rank task (a row-parallel product whose partial sums are all-reduced through
symmetric peer memory) is written into tmp_path. The evaluation tests run the real launcher,
spawn processes and gloo group through the evaluator's CPU test platform, which emulates symmetric
memory with shared file mappings; the GPU path (HIP IPC, events, graphs) is not exercised here.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import yaml

from src.task_contract import (
    RESERVED_INPUTS, hip_op_schema, load_task_contract, render_workspace, validate_task_contract,
)
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TOY_ID = "triton/other/toy_allreduce/w2"

TASK_YAML = """\
schema_version: 3
description: >-
  Toy row-parallel product out = x @ w^T whose per-rank partial sums are all-reduced through
  symmetric peer memory; out is replicated, out_shard is this rank's column block.
operator: {name: toy_allreduce, category: other}
interface:
  axes:
    n: {kind: var, min: 1, max: 4096}
    k: {kind: const, value: 8}
    h: {kind: const, value: 16}
  derived_axes:
    h_shard: "h // world_size"
  inputs:
    x: {kind: tensor, shape: [n, k], dtype: float32, distribution: {kind: generator}}
    w: {kind: tensor, shape: [h, k], dtype: float32, role: weight, distribution: {kind: generator}}
  outputs:
    out: {shape: [n, h], dtype: float32}
    out_shard: {shape: [n, h_shard], dtype: float32}
  generator: {file: inputs.py, symbol: make_inputs}
distributed:
  world_size: 2
  symmetric_bytes: "64 + world_size * n * h * 4"
  shard: {file: inputs.py, symbol: shard_inputs}
  expected: {file: inputs.py, symbol: rank_expected}
  ranks_agree: [out]
reference: {file: reference.py, symbol: reference}
baseline:
  language: triton
  entry: baseline/kernel.py
  provenance: {source: task-original, revision: v1, path: baseline/kernel.py, license: Apache-2.0}
compare:
  out: {bound: baseline_relative}
  out_shard: {bound: baseline_relative}
workloads: workloads.json
"""

WORKLOADS = {"description": "Toy token counts.", "cases": [
    {"id": "n1", "axes": {"n": 1}, "checks": ["correctness"], "source": "minimum"},
    {"id": "n6", "axes": {"n": 6}, "checks": ["correctness"], "source": "non-power-of-two"},
    {"id": "n33", "axes": {"n": 33}, "checks": ["correctness"], "source": "unaligned"},
    {"id": "n8", "axes": {"n": 8}, "checks": ["correctness", "performance"], "source": "toy"},
    {"id": "n64", "axes": {"n": 64}, "checks": ["correctness", "performance"], "source": "toy"},
    {"id": "n256", "axes": {"n": 256}, "checks": ["correctness", "performance"], "source": "toy"},
    {"id": "n1024", "axes": {"n": 1024}, "checks": ["correctness", "performance"], "source": "toy"},
]}

REFERENCE = """\
import torch


def reference(*, x, w):
    out = x.double() @ w.double().T
    return {"out": out, "out_shard": out}
"""

INPUTS = """\
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    width = axes["k"] * axes["world_size"]
    x = torch.randn((axes["n"], width), generator=generator, device=device)
    w = torch.randn((axes["h"], width), generator=generator, device=device)
    return {"x": x, "w": w}


def shard_inputs(inputs, *, rank, world_size):
    k = inputs["x"].shape[1] // world_size
    return {"x": inputs["x"][:, rank * k:(rank + 1) * k], "w": inputs["w"][:, rank * k:(rank + 1) * k]}


def rank_expected(outputs, *, rank, world_size):
    width = outputs["out_shard"].shape[1] // world_size
    return {"out": outputs["out"], "out_shard": outputs["out_shard"][:, rank * width:(rank + 1) * width]}
"""

# A statically valid Triton baseline (the CPU pre-check audits it but never runs it).
TRITON_BASELINE = """\
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(x, w, out, n, K: tl.constexpr, H: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in range(K):
        acc += tl.load(x + row * K + k) * tl.load(w + cols * K + k)
    tl.store(out + row * H + cols, acc)


def run(*, x, w, out, out_shard, rank, world_size, peers, symmetric):
    _partial[(x.shape[0],)](x, w, out, x.shape[0], K=x.shape[1], H=w.shape[0])
"""

# The CPU stand-in: partial sums are stored into every peer's symmetric buffer through the
# ``peers`` addresses, then each rank polls its own buffer's flags and reduces in rank order.
CPU_BASELINE = """\
import ctypes
import threading
import time

import torch

HEADER = 64
_FENCE = threading.Lock()


def _fence():
    with _FENCE:  # a lock round trip orders this process's stores and loads
        pass


def run(*, x, w, out, out_shard, rank, world_size, peers, symmetric):
    partial = (x.double() @ w.double().T).float().contiguous()
    nbytes = partial.numel() * 4
    # Peers may already be writing into this buffer; only this rank writes its own flag and slot.
    mine = HEADER + rank * nbytes
    if int(symmetric[4 * rank:4 * rank + 4].view(torch.int32)[0]) or bool(symmetric[mine:mine + nbytes].any()):
        raise RuntimeError("the symmetric buffer was not zeroed before this call")
    for peer in range(world_size):
        ctypes.memmove(int(peers[peer]) + HEADER + rank * nbytes, partial.data_ptr(), nbytes)
    _fence()
    for peer in range(world_size):
        ctypes.c_int32.from_address(int(peers[peer]) + 4 * rank).value = 1
    flags = symmetric[:4 * world_size].view(torch.int32)
    deadline = time.monotonic() + 60
    while int(flags.sum()) != world_size:
        if time.monotonic() > deadline:
            raise RuntimeError("peers never wrote their partial sums")
        time.sleep(0.0002)
    _fence()
    total = torch.zeros_like(partial)
    for peer in range(world_size):
        total += symmetric[HEADER + peer * nbytes:HEADER + (peer + 1) * nbytes].view(torch.float32).view_as(partial)
    # HOOK
    out.copy_(total)
    width = out_shard.shape[1]
    out_shard.copy_(total[:, rank * width:(rank + 1) * width])
"""


def write_toy(root: Path, *, baseline: str = TRITON_BASELINE, task_yaml: str = TASK_YAML,
              inputs: str = INPUTS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.yaml").write_text(task_yaml)
    (root / "workloads.json").write_text(json.dumps(WORKLOADS, indent=1))
    (root / "reference.py").write_text(REFERENCE)
    (root / "inputs.py").write_text(inputs)
    (root / "README.md").write_text("# Toy all-reduce\n\nA framework test task.\n")
    (root / "baseline").mkdir(exist_ok=True)
    (root / "baseline" / "kernel.py").write_text(baseline)
    return root


def _raw():
    return yaml.safe_load(TASK_YAML), copy.deepcopy(WORKLOADS)


def _validate(raw, workloads=None):
    return validate_task_contract(raw, workloads or copy.deepcopy(WORKLOADS), task_id=TOY_ID)


# ------------------------------------------------------------------------------------------------
# task.yaml validation


def test_a_distributed_task_validates_and_exposes_its_world_size():
    contract = _validate(*_raw())
    assert contract.world_size == 2
    assert contract.call_names == ["x", "w", "out", "out_shard", "rank", "world_size", "peers", "symmetric"]
    assert contract.distributed["ranks_agree"] == ["out"]


def test_single_gpu_tasks_keep_their_call_and_world_size():
    pilot = ROOT / "tasks/triton/normalization/rms_norm/bf16_h8192"
    contract = load_task_contract(pilot / "task.yaml", task_id="triton/normalization/rms_norm/bf16_h8192")
    assert contract.distributed is None and contract.world_size == 1
    assert contract.call_names == ["x", "weight", "epsilon", "out"]


@pytest.mark.parametrize("world_size", [1, 9, 0, "4", 2.0, True])
def test_world_size_is_bounded(world_size):
    raw, workloads = _raw()
    raw["distributed"]["world_size"] = world_size
    with pytest.raises(TaskConfigError, match="world_size must be an integer from 2 to 8"):
        _validate(raw, workloads)


@pytest.mark.parametrize("update, message", [
    ({"symmetric_bytes": "n * hidden"}, "reads unknown axes"),
    ({"symmetric_bytes": "n ** 2"}, "unsupported syntax"),
    ({"symmetric_bytes": -4}, "negative dimension"),
    ({"symmetric_bytes": 0}, "between 1 and"),
    ({"symmetric_bytes": "1 << 40"}, "unsupported syntax"),
    ({"symmetric_bytes": "n * 1099511627776"}, "between 1 and"),
    ({"shard": {"file": "baseline/inputs.py", "symbol": "shard_inputs"}}, "task root"),
    ({"expected": {"file": "inputs.py"}}, "missing fields"),
    ({"ranks_agree": ["x"]}, "ranks_agree"),
    ({"ranks_agree": ["out", "out"]}, "ranks_agree"),
    ({"ranks_agree": "out"}, "ranks_agree"),
    ({"ranks_agree": [["out"]]}, "ranks_agree"),
    ({"topology": "ring"}, "unknown fields"),
])
def test_malformed_distributed_blocks_are_rejected(update, message):
    raw, workloads = _raw()
    raw["distributed"].update(update)
    with pytest.raises(TaskConfigError, match=message):
        _validate(raw, workloads)


@pytest.mark.parametrize("field", ["world_size", "symmetric_bytes", "shard", "expected"])
def test_required_distributed_fields(field):
    raw, workloads = _raw()
    del raw["distributed"][field]
    with pytest.raises(TaskConfigError, match="missing fields"):
        _validate(raw, workloads)


def test_symmetric_bytes_may_be_an_integer():
    raw, workloads = _raw()
    raw["distributed"]["symmetric_bytes"] = 4 << 20
    assert _validate(raw, workloads).distributed["symmetric_bytes"] == 4 << 20


@pytest.mark.parametrize("name", RESERVED_INPUTS)
def test_reserved_call_inputs_cannot_be_declared(name):
    raw, workloads = _raw()
    raw["interface"]["inputs"][name] = {"kind": "scalar", "dtype": "int32", "default": 0}
    with pytest.raises(TaskConfigError, match="reserved multi-GPU call input"):
        _validate(raw, workloads)
    raw, workloads = _raw()
    raw["interface"]["outputs"][name] = {"shape": ["n"], "dtype": "float32"}
    raw["compare"][name] = {"bound": "baseline_relative"}
    with pytest.raises(TaskConfigError, match="reserved multi-GPU call input"):
        _validate(raw, workloads)


def test_world_size_is_a_reserved_shape_symbol():
    raw, workloads = _raw()
    raw["interface"]["axes"]["world_size"] = {"kind": "const", "value": 2}
    with pytest.raises(TaskConfigError, match="axis world_size is reserved"):
        _validate(raw, workloads)
    raw, workloads = _raw()
    raw["interface"]["derived_axes"]["world_size"] = "2"
    with pytest.raises(TaskConfigError, match="derived axis world_size is reserved"):
        _validate(raw, workloads)
    # Without a distributed block, world_size is not a symbol at all.
    raw, workloads = _raw()
    del raw["distributed"]
    with pytest.raises(TaskConfigError, match="unknown axes"):
        _validate(raw, workloads)


def test_single_gpu_tasks_may_still_use_reserved_names():
    # LoRA tasks have an axis named rank; only multi-GPU tasks reserve the call input names.
    raw, workloads = _raw()
    del raw["distributed"]
    raw["interface"]["derived_axes"] = {"h_shard": "h // 2"}
    raw["interface"]["axes"]["rank"] = {"kind": "const", "value": 16}
    raw["interface"]["inputs"]["peers"] = {"kind": "scalar", "dtype": "int32", "default": 1}
    assert _validate(raw, workloads).distributed is None


# ------------------------------------------------------------------------------------------------
# Resolution into the workspace


def test_execution_config_and_contract_json_carry_the_distributed_block(tmp_path):
    package = write_toy(tmp_path / "task")
    spec = load_task_spec(package / "task.yaml", task_id=TOY_ID, target_language="hip")
    assert spec.world_size == 2 and spec.contract["world_size"] == 2
    again = TaskSpec.from_mapping(spec.to_mapping(), task_id=TOY_ID)
    assert again.world_size == 2 and again.to_mapping() == spec.to_mapping()
    workspace = tmp_path / "ws"
    shutil.copytree(package, workspace)
    render_workspace(package, workspace, task_id=TOY_ID, target_language="triton")
    resolved = json.loads((workspace / "_aka_eval/contract.json").read_text())
    assert resolved["distributed"] == {
        "world_size": 2, "symmetric_bytes": "64 + world_size * n * h * 4",
        "shard": {"file": "inputs.py", "symbol": "shard_inputs"},
        "expected": {"file": "inputs.py", "symbol": "rank_expected"},
        "ranks_agree": ["out"], "timeout_s": 3600}
    assert (workspace / "_aka_eval/distributed.py").is_file()
    rendered = TaskSpec.from_mapping(yaml.safe_load((workspace / "config.yaml").read_text()), task_id=TOY_ID)
    assert rendered.world_size == 2


def test_task_spec_rejects_an_invalid_world_size_record(tmp_path):
    package = write_toy(tmp_path / "task")
    mapping = load_task_spec(package / "task.yaml", task_id=TOY_ID, target_language="hip").to_mapping()
    for value in (1, 9, "2"):
        mapping["contract"]["world_size"] = value
        with pytest.raises(TaskConfigError, match="world_size"):
            TaskSpec.from_mapping(mapping, task_id=TOY_ID)


def test_resolved_contract_reads_the_distributed_block(tmp_path):
    from src.tools.arena_eval.contract import Contract, ContractError

    package = write_toy(tmp_path / "task")
    resolved = load_task_contract(package / "task.yaml", task_id=TOY_ID).resolved_contract("triton")
    contract = Contract(copy.deepcopy(resolved))
    assert contract.world_size == 2 and contract.ranks_agree == ("out",)
    assert contract.call_keys[-4:] == RESERVED_INPUTS
    assert contract.written_tensors == ("out", "out_shard", "symmetric")
    case = contract.case("n33")
    assert contract.symbols(case)["world_size"] == 2 and contract.symbols(case)["h_shard"] == 8
    assert contract.symmetric_bytes(case) == 64 + 2 * 33 * 16 * 4
    single = copy.deepcopy(resolved)
    del single["distributed"]
    assert Contract(single).call_keys == ("x", "w", "out", "out_shard")
    bad = copy.deepcopy(resolved)
    bad["distributed"]["world_size"] = 1
    with pytest.raises(ContractError):
        Contract(bad)


def test_candidate_signature_and_hip_schema_append_the_reserved_inputs(tmp_path):
    package = write_toy(tmp_path / "task")
    contract = load_task_contract(package / "task.yaml", task_id=TOY_ID)
    assert hip_op_schema(contract) == (
        "run(Tensor x, Tensor w, Tensor(a!) out, Tensor(b!) out_shard, int rank, int world_size, "
        "Tensor peers, Tensor(c!) symmetric) -> ()")
    for language, entry in (("hip", "solution/kernel.hip"), ("flydsl", "solution/kernel.py")):
        workspace = tmp_path / language
        shutil.copytree(package, workspace)
        render_workspace(package, workspace, task_id=TOY_ID, target_language=language)
        template = (workspace / entry).read_text()
        document = (workspace / "_aka_eval/CANDIDATE.md").read_text()
        if language == "hip":
            assert "int64_t rank, int64_t world_size, const at::Tensor& peers, at::Tensor& symmetric" in template
        else:
            assert "def run(*, x, w, out, out_shard, rank, world_size, peers, symmetric) -> None:" in template
        assert "## Multi-GPU" in document and "**2 GPUs**" in document
        assert "| `peers` | reserved multi-GPU input | [world_size] | int64 |" in document
        assert "| `symmetric` | reserved multi-GPU input | [64 + world_size * n * h * 4] | uint8 |" in document
        assert "maximum over ranks" in document and "`out`" in document


# ------------------------------------------------------------------------------------------------
# CPU pre-check


@pytest.fixture()
def tasks_root(tmp_path, monkeypatch):
    from src.tools import check_task

    root = tmp_path / "repo"
    monkeypatch.setattr(check_task, "ROOT", root)
    return root


def test_check_task_accepts_a_valid_distributed_task(tasks_root):
    from src.tools.check_task import check

    task = write_toy(tasks_root / "tasks" / TOY_ID)
    result = check(task, smoke_limit=1 << 16)
    assert result["errors"] == [] and result["ok"], result
    ranks = result["smoke"][0]["ranks"]
    assert [view["rank"] for view in ranks] == [0, 1]
    assert ranks[1]["inputs"] == {"x": [1, 8], "w": [16, 8]}
    assert ranks[1]["expected"] == {"out": [1, 16], "out_shard": [1, 8]}


@pytest.mark.parametrize("inputs, message", [
    (INPUTS.replace('return {"x": inputs["x"][:, rank * k:(rank + 1) * k], ',
                    'return {"x": inputs["x"], '), "Input x for rank 0 is (1, 16)"),
    (INPUTS.replace(', "w": inputs["w"][:, rank * k:(rank + 1) * k]}', "}"), "must return every declared input"),
    (INPUTS.replace('"out_shard": outputs["out_shard"][:, rank * width:(rank + 1) * width]',
                    '"out_shard": outputs["out_shard"]'), "Expected out_shard for rank 0 is (1, 16)"),
])
def test_check_task_rejects_bad_shard_or_expected_functions(tasks_root, inputs, message):
    from src.tools.check_task import check

    task = write_toy(tasks_root / "tasks" / TOY_ID, inputs=inputs)
    result = check(task, smoke_limit=1 << 16)
    assert not result["ok"]
    assert any("per-rank shard/expected check failed" in e and message in e for e in result["errors"]), result


def test_check_task_reports_a_malformed_distributed_block(tasks_root):
    from src.tools.check_task import check

    task = write_toy(tasks_root / "tasks" / TOY_ID, task_yaml=TASK_YAML.replace("world_size: 2", "world_size: 16"))
    result = check(task, smoke_limit=1 << 16)
    assert not result.get("ok") and "world_size must be an integer from 2 to 8" in result["errors"][0]


# ------------------------------------------------------------------------------------------------
# Scheduling: GPU groups must match the world size


def _repo_with_tasks(root: Path) -> Path:
    write_toy(root / "tasks" / TOY_ID)
    pilot = ROOT / "tasks/triton/normalization/rms_norm/bf16_h8192"
    shutil.copytree(pilot, root / "tasks/triton/normalization/rms_norm/bf16_h8192")
    four = root / "tasks/triton/other/toy_allreduce/w4"
    write_toy(four, task_yaml=TASK_YAML.replace("world_size: 2", "world_size: 4"))
    return root


def test_kernel_world_size_and_group_check(tmp_path):
    from src.serving_runtime import check_group_sizes, check_pool_size, gpu_groups, kernel_world_size

    root = _repo_with_tasks(tmp_path)
    assert kernel_world_size({"tasks": ["triton/normalization"]}, root) == 1
    assert kernel_world_size({"tasks": [TOY_ID]}, root) == 2
    assert kernel_world_size({"tasks": [{"path": TOY_ID, "target_language": "hip"}]}, root) == 2
    with pytest.raises(ValueError, match="different GPU counts"):
        kernel_world_size({"tasks": ["triton/other"]}, root)
    with pytest.raises(ValueError, match="different GPU counts"):
        kernel_world_size({"tasks": [TOY_ID, "triton/normalization"]}, root)
    pool = [str(i) for i in range(8)]
    # Single-GPU runs keep their one-GPU workers and any explicit groups.
    check_group_sizes(gpu_groups({}, pool), 1)
    check_group_sizes(gpu_groups({"resources": {"gpu_groups": [[0, 1, 2, 3]]}}, pool), 1)
    check_group_sizes(gpu_groups({"resources": {"gpu_groups": [[0, 1], [2, 3], [4, 5]]}}, pool), 2)
    with pytest.raises(ValueError, match="fewer than world_size.*groups of exactly 2"):
        check_group_sizes(gpu_groups({}, pool), 2)
    with pytest.raises(ValueError, match="group 2,3,4,5 has 4 GPU"):
        check_group_sizes(gpu_groups({"resources": {"gpu_groups": [[0, 1], [2, 3, 4, 5]]}}, pool), 2)
    # A short allocation (7 of 8 GPUs on some burst nodes) fails clearly, naming what was allocated.
    short = [str(i) for i in range(7)]
    check_pool_size(short, 1)
    check_pool_size(short, 4)
    with pytest.raises(ValueError, match=r"allocated only 7 GPU\(s\) \(0,1,2,3,4,5,6\).*at least 8"):
        check_pool_size(short, 8)
    with pytest.raises(ValueError, match="escape the allocated device pool \\(7 GPU"):
        gpu_groups({"resources": {"gpu_groups": [[0, 1, 2, 3], [4, 5, 6, 7]]}}, short)
    with pytest.raises(ValueError, match="group 4,5,6 has 3 GPU.*fewer than world_size"):
        check_group_sizes(gpu_groups({"resources": {"gpu_groups": [[0, 1, 2, 3], [4, 5, 6]]}}, short), 4)


def test_groups_metadata_command_fails_before_any_container(tmp_path):
    root = _repo_with_tasks(tmp_path)
    config = tmp_path / "run.yaml"

    def metadata(mode, payload, devices="0,1,2,3"):
        config.write_text(yaml.safe_dump(payload))
        return subprocess.run([sys.executable, "-m", "src.serving_runtime", mode, "--config", str(config),
                               "--root", str(root), "--devices", devices],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    ok = metadata("groups", {"tasks": [TOY_ID], "resources": {"gpu_groups": [[0, 1], [2, 3]]}})
    assert ok.returncode == 0 and ok.stdout.split() == ["0,1", "2,3"], ok.stderr
    assert metadata("world-size", {"tasks": [TOY_ID]}).stdout.strip() == "2"
    assert metadata("world-size", {"tasks": ["triton/normalization"]}).stdout.strip() == "1"
    bad = metadata("groups", {"tasks": [TOY_ID]})
    assert bad.returncode != 0 and "groups of exactly 2" in bad.stderr
    four = "triton/other/toy_allreduce/w4"
    short = metadata("groups", {"tasks": [four], "resources": {"gpu_groups": [[0, 1, 2, 3]]}}, devices="0,1,2")
    assert short.returncode != 0 and "allocated only 3 GPU(s) (0,1,2)" in short.stderr


def test_serial_runner_assigns_one_group_for_multi_gpu_kernel_tasks():
    script = (ROOT / "src/scripts/docker_benchmark.sh").read_text()
    body = script[script.index("assign_serial_gpu_group() {"):]
    body = body[:body.index("\n}\n")]
    assert 'serving_runtime_metadata world-size "$config"' in body
    assert body.index("world-size") < body.index("resolve_gpu_ids")


# ------------------------------------------------------------------------------------------------
# Candidate audit: no host-side communication


@pytest.mark.parametrize("snippet", [
    "import torch.distributed as dist",
    "from torch import distributed",
    "import torch.multiprocessing",
    "g = torch.distributed.all_reduce",
    "q = torch.multiprocessing.Queue",
    "h = out.untyped_storage()._share_cuda_()",
    "out.share_memory_()",
    "import ctypes",
])
def test_python_candidates_cannot_communicate_on_the_host(snippet):
    from src.tools.arena_eval.audit import PolicyError, audit_python_sources

    with pytest.raises(PolicyError):
        audit_python_sources({"kernel.py": TRITON_BASELINE + "\n" + snippet + "\n"}, "triton")


def test_python_candidates_may_read_peers_and_symmetric():
    from src.tools.arena_eval.audit import audit_python_sources

    audit_python_sources({"kernel.py": TRITON_BASELINE}, "triton")


HIP_OK = """\
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>
__global__ void k(const float* x, float* y, const long long* peers, int rank) {
  float* remote = reinterpret_cast<float*>(peers[1 - rank]);
  __hip_atomic_store(remote, x[0], __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  y[0] = x[0];
}
void run(const at::Tensor& x, at::Tensor& out, int64_t rank, int64_t world_size, const at::Tensor& peers,
         at::Tensor& symmetric) {
  hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, at::hip::getCurrentHIPStream(), x.data_ptr<float>(),
                     out.data_ptr<float>(), (const long long*)peers.data_ptr<int64_t>(), (int)rank);
}
#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) { m.def("run(Tensor x, Tensor(a!) out) -> ()", &run); }
"""


def test_hip_candidates_may_store_to_peers_from_kernels():
    from src.tools.arena_eval.audit import audit_hip_sources

    audit_hip_sources({"kernel.hip": HIP_OK}, "kernel.hip")


@pytest.mark.parametrize("call", [
    "hipIpcOpenMemHandle(&p, h, 1);", "hipIpcGetMemHandle(&h, p);", "hipDeviceEnablePeerAccess(1, 0);",
    "hipDeviceCanAccessPeer(&ok, 0, 1);", "hipExtMallocWithFlags(&p, 64, 3);", "ncclAllReduce(a, b, n, t, o, c, s);",
    "rcclGetVersion(&v);", "hipMemcpyPeerAsync(a, 1, b, 0, n, s);", "hipMemExportToShareableHandle(&h, g, t, 0);",
    "MPI_Barrier(c);", "roc_shmem_barrier_all();",
])
def test_hip_candidates_cannot_communicate_on_the_host(call):
    from src.tools.arena_eval.audit import PolicyError, audit_hip_sources

    source = HIP_OK.replace("void run(", f"void helper() {{ {call} }}\nvoid run(")
    with pytest.raises(PolicyError, match="is not allowed in a HIP candidate"):
        audit_hip_sources({"kernel.hip": source}, "kernel.hip")


PROBE = r"""
import os, sys, tempfile
import torch, torch.distributed as dist
from src.tools.arena_eval.audit import preparation_only
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
dist.init_process_group("gloo", store=dist.FileStore(os.path.join(tempfile.mkdtemp(), "store"), 1),
                        rank=0, world_size=1)
for name, op in (("all_reduce", lambda: dist.all_reduce(torch.ones(4))), ("barrier", dist.barrier)):
    try:
        with preparation_only("candidate"):
            op()
        print(name, "ALLOWED")
    except RuntimeError as exc:
        print(name, "REJECTED", exc)
"""


def test_runtime_policy_rejects_host_collectives_during_a_candidate_call():
    process = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=120)
    lines = dict(line.split(" ", 1) for line in process.stdout.splitlines() if line.startswith(("all_reduce", "barrier")))
    assert set(lines) == {"all_reduce", "barrier"}, process.stdout + process.stderr
    for line in lines.values():
        assert line.startswith("REJECTED") and "host-side communication" in line


# ------------------------------------------------------------------------------------------------
# Two-rank evaluation on the CPU test platform

LAUNCH = r"""
import json, sys
from _aka_eval import distributed
from _aka_eval.contract import load_contract
job, role, action, timeout = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
result, failure = distributed.launch(load_contract(), job, role, action, platform="cpu-test", timeout_s=timeout)
print("LAUNCH=" + json.dumps({"result": result, "failure": failure}))
"""


def _workspace(tmp_path: Path, hook: str = "") -> Path:
    package = write_toy(tmp_path / "task", baseline=CPU_BASELINE.replace("    # HOOK\n", hook))
    workspace = tmp_path / "ws"
    shutil.copytree(package, workspace)
    render_workspace(package, workspace, task_id=TOY_ID, target_language="triton")
    contract = json.loads((workspace / "_aka_eval/contract.json").read_text())
    contract["timing"].update(warmup=2, repetition=6)  # keep the CPU test short
    (workspace / "_aka_eval/contract.json").write_text(json.dumps(contract))
    return workspace


def _launch(workspace: Path, role: str, action: str, *, job: str = "evaluate", timeout: float = 120):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARENA_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run([sys.executable, "-c", LAUNCH, job, role, action, str(timeout)], cwd=workspace,
                             env=env, capture_output=True, text=True, timeout=timeout + 120)
    lines = [line for line in process.stdout.splitlines() if line.startswith("LAUNCH=")]
    assert lines, process.stdout[-3000:] + process.stderr[-3000:]
    return json.loads(lines[-1].split("=", 1)[1])


def test_two_rank_correctness_checks_every_rank_and_rank_agreement(tmp_path):
    workspace = _workspace(tmp_path)
    launched = _launch(workspace, "baseline", "correctness")
    assert launched["failure"] is None, launched["failure"]
    report = launched["result"]
    assert report["status"] == "PASS" and report["metadata"]["world_size"] == 2
    assert [row["test_case_id"] for row in report["cases"]] == [c["id"] for c in WORKLOADS["cases"]]
    for row in report["cases"]:
        ranks = row["metrics"]["ranks"]
        assert len(ranks) == 2 and all(len(rank["calls"]) == 2 for rank in ranks)
        for call in range(2):
            digests = {rank["calls"][call]["ranks_agree"]["out"] for rank in ranks}
            assert len(digests) == 1
            for rank in ranks:
                assert set(rank["calls"][call]["outputs"]) == {"out", "out_shard"}
                assert rank["calls"][call]["outputs"]["out"]["bound"] == "baseline_sanity"


def test_two_rank_performance_reports_max_over_ranks_with_raw_samples(tmp_path):
    workspace = _workspace(tmp_path)
    launched = _launch(workspace, "baseline", "performance")
    assert launched["failure"] is None, launched["failure"]
    report = launched["result"]
    assert report["status"] == "PASS"
    assert [row["test_case_id"] for row in report["cases"]] == ["n8", "n64", "n256", "n1024"]
    for row in report["cases"]:
        metadata = row["metadata"]
        assert row["benchmark_method"] == "cpu_wall_clock_test_only"  # never a valid score
        raw = metadata["benchmark_rank_samples_ms"]
        assert len(raw) == 2 and all(len(samples) == 6 for samples in raw)
        maxima = [max(column) for column in zip(*raw)]
        assert row["execution_time_ms"] == pytest.approx(sorted(maxima)[2] / 2 + sorted(maxima)[3] / 2)
        assert len(metadata["benchmark_release_skew_us"]) == 6
        assert metadata["benchmark_rank_aggregation"] == "max_over_ranks"
        assert metadata["rank_checked_samples"] == [6, 6]
        assert sum(metadata["benchmark_slowest_rank_counts"]) == 6


def test_ranks_that_disagree_on_a_replicated_output_fail(tmp_path):
    workspace = _workspace(tmp_path, hook="    if rank == 1:\n        total[0, 0] += 1e-3\n")
    report = _launch(workspace, "baseline", "correctness")["result"]
    assert report["status"] == "FAIL" and report["failure_kind"] == "numerical_mismatch"
    for row in report["cases"]:
        assert row["status"] == "FAIL"
        assert row["metadata"]["ranks_disagree"][0]["output"] == "out"


def test_a_failing_rank_fails_the_action_and_stops_its_peers(tmp_path):
    workspace = _workspace(tmp_path)
    path = workspace / "baseline" / "kernel.py"
    path.write_text(path.read_text().replace(
        '    partial = (x.double()',
        '    if rank == 1:\n        raise RuntimeError("boom on rank 1")\n    partial = (x.double()'))
    started = time.monotonic()
    failure = _launch(workspace, "baseline", "correctness")["failure"]
    assert failure is not None and failure["kind"] == "execution_error"
    assert "rank 1 of 2 failed" in failure["reason"] and "boom on rank 1" in failure["reason"]
    assert time.monotonic() - started < 90  # rank 0 was spinning on its flags and was killed


def test_tampering_found_on_one_rank_fails_the_job_as_harness_tampering(tmp_path):
    # TamperError is a BaseException, so no per-case handler may turn it into an ordinary failure
    workspace = _workspace(tmp_path, hook=(
        "    if rank == 1:\n"
        "        from _aka_eval.integrity import TamperError\n"
        "        raise TamperError('Harness tampering detected after a candidate call: [test]')\n"))
    failure = _launch(workspace, "baseline", "correctness")["failure"]
    assert failure is not None and failure["kind"] == "harness_tampering"
    assert "rank 1 of 2 failed" in failure["reason"] and "Harness tampering detected" in failure["reason"]


def test_a_hanging_rank_is_killed_at_the_deadline(tmp_path):
    workspace = _workspace(tmp_path)
    path = workspace / "baseline" / "kernel.py"
    path.write_text(path.read_text().replace(
        '    partial = (x.double()', '    if rank == 1:\n        time.sleep(600)\n    partial = (x.double()'))
    started = time.monotonic()
    failure = _launch(workspace, "baseline", "correctness", timeout=10)["failure"]
    assert failure["kind"] == "timeout" and "exceeded 10s" in failure["reason"]
    assert time.monotonic() - started < 80


def test_the_command_line_reports_one_line_without_gpus(tmp_path):
    workspace = _workspace(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARENA_")}
    process = subprocess.run([sys.executable, "-m", "_aka_eval.evaluate", "baseline", "correctness"],
                             cwd=workspace, env=env, capture_output=True, text=True, timeout=300)
    lines = [line for line in process.stdout.splitlines() if line.startswith("ARENA_EVAL_RESULT=")]
    assert process.returncode == 1 and len(lines) == 1
    report = json.loads(lines[0].split("=", 1)[1])
    assert report["status"] == "FAIL" and report["failure_kind"] == "environment"
    assert "GPU" in report["reason"]
    validate = subprocess.run([sys.executable, "-m", "_aka_eval.evaluate", "validate-task"],
                              cwd=workspace, env=env, capture_output=True, text=True, timeout=300)
    assert validate.returncode == 0, validate.stdout[-2000:]


def test_the_qa_control_suite_runs_in_lockstep_on_every_rank(tmp_path):
    workspace = _workspace(tmp_path)
    launched = _launch(workspace, "task", "controls", job="controls")
    assert launched["failure"] is None, launched["failure"]
    report = launched["result"]
    assert report["survived"] == []
    assert all(control["rejected"] for control in report["controls"].values())
    assert {item.get("rank") for item in report["calibration_probe"] if "rank" in item} == {0, 1}
