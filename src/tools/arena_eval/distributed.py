# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Multi-GPU (multi-rank) evaluation: one process per rank, symmetric peer memory, one report.

A task with a ``distributed`` block runs every correctness and performance action (and the QA
control suite) on ``world_size`` GPUs of one node. The launcher in this module starts one
``spawn`` process per rank; rank ``r`` binds ``cuda:r`` of the worker's visible GPU group. The
ranks form a ``torch.distributed`` gloo group over a FileStore in a private temporary directory
(loopback only, no network use) that carries the IPC-handle exchange, barriers and result
gathering. Each rank allocates one uncached symmetric buffer, exchanges its HIP IPC handle and maps
every peer's buffer; the candidate receives ``rank``, ``world_size``, ``peers`` (every rank's buffer
address in this process) and ``symmetric`` (its own buffer) by keyword.

Rank 0 merges the ranks' results into the action's report, which the launcher prints as the
single ``ARENA_EVAL_RESULT=`` line. A rank that raises, exits or outlives the action deadline fails
the whole action, and the launcher kills every remaining rank: a peer may be spinning on the
device for that rank's writes, so no rank can recover on its own.
"""
from __future__ import annotations

import ctypes
import functools
import json
import multiprocessing
import multiprocessing.connection
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
import traceback
from typing import Any

from .contract import Case, Contract, ContractError, load_contract

# Device allocations map into peers in 2 MiB granules, so a partial last granule would fault there.
GRANULE = 2 << 20
_LAUNCH_POLL_S = 0.2
_TIMEOUT_MARGIN_S = 30.0
_MIN_TIMEOUT_S = 30.0


class RankFailure(RuntimeError):
    """One rank failed; the whole multi-rank action fails."""


class EnvironmentFailure(RuntimeError):
    """The worker cannot run the task's ranks (no GPU, too few GPUs, no HIP runtime)."""

    kind = "environment"


# --------------------------------------------------------------------------------------------------
# Host-side communication between the ranks (gloo, CPU only).


class GlooComm:
    """Barriers, a deadline broadcast and object gathering over the ranks' gloo group."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier()

    def broadcast_int(self, value: int) -> int:
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([int(value)], dtype=torch.int64)
        dist.broadcast(tensor, src=0)
        return int(tensor[0])

    def all_gather_object(self, obj: Any) -> list:
        import torch.distributed as dist

        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered


# --------------------------------------------------------------------------------------------------
# Symmetric peer memory.


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


_IPC_HANDLE_BYTES = 64
_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
_DEVICE_MALLOC_UNCACHED = 0x3
# Buffers and their peer mappings stay alive until the process exits: on MI355X, uncached pages
# that a kernel used, once freed and reused by ordinary allocations, gave later kernels stale lines.
_RETAINED: list = []


@functools.cache
def _hip_runtime():
    """The HIP runtime PyTorch loaded into this process, not a second copy."""
    path = None
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            for line in maps:
                if "libamdhip64" in line:
                    path = line.split()[-1]
                    break
    except OSError:
        pass
    library = ctypes.CDLL(path or "libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    signatures = {
        "hipExtMallocWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        "hipMemset": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        "hipDeviceSynchronize": [],
        "hipIpcGetMemHandle": [ctypes.POINTER(_IpcHandle), ctypes.c_void_p],
        "hipIpcOpenMemHandle": [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint],
        "hipGetErrorString": [ctypes.c_int],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ctypes.c_char_p if name == "hipGetErrorString" else ctypes.c_int
    return library


def _hip_check(error: int, what: str) -> None:
    if error:
        message = _hip_runtime().hipGetErrorString(error)
        raise RuntimeError(f"{what}: {message.decode() if message else error}")


class _DeviceBytes:
    """A non-owning ``__cuda_array_interface__`` view of raw device memory."""

    def __init__(self, pointer: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (pointer, False),
                                         "version": 3}


def _rounded(nbytes: int) -> int:
    return -(-int(nbytes) // GRANULE) * GRANULE


class HipPeerMemory:
    """One uncached symmetric buffer per rank, mapped into every peer through HIP IPC.

    Adapted from sgl-amd-mono (Apache-2.0) ``sgl_amd_mono/common/peer.py``, itself adapted from
    ROCm/ATOM at 97359d6 (MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.):
    ``atom/mono/runtime/peer_memory.py``. Construction is collective; all ranks fail together.
    """

    def __init__(self, nbytes: int, comm: Any, device: Any):
        import torch

        if torch.version.hip is None:
            raise EnvironmentFailure("multi-GPU tasks need a ROCm PyTorch: peer memory uses HIP IPC")
        hip = _hip_runtime()
        self.nbytes = _rounded(nbytes)
        pointer = ctypes.c_void_p()
        _hip_check(hip.hipExtMallocWithFlags(ctypes.byref(pointer), self.nbytes, _DEVICE_MALLOC_UNCACHED),
                   "hipExtMallocWithFlags")
        self.local = int(pointer.value)
        _hip_check(hip.hipMemset(ctypes.c_void_p(self.local), 0, self.nbytes), "hipMemset")
        _hip_check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        handle = _IpcHandle()
        _hip_check(hip.hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(self.local)), "hipIpcGetMemHandle")
        offered = comm.all_gather_object((self.nbytes, bytes(ctypes.string_at(ctypes.byref(handle),
                                                                              _IPC_HANDLE_BYTES))))
        addresses, opened, error = [], [], None
        try:
            for peer, (size, raw) in enumerate(offered):
                if size != self.nbytes:
                    raise RuntimeError(f"rank {peer} offers {size} bytes, this rank {self.nbytes}")
                if peer == comm.rank:
                    addresses.append(self.local)
                    continue
                peer_handle = _IpcHandle()
                ctypes.memmove(ctypes.byref(peer_handle), raw, _IPC_HANDLE_BYTES)
                mapped = ctypes.c_void_p()
                _hip_check(hip.hipIpcOpenMemHandle(ctypes.byref(mapped), peer_handle,
                                                   _IPC_MEM_LAZY_ENABLE_PEER_ACCESS), "hipIpcOpenMemHandle")
                opened.append(int(mapped.value))
                addresses.append(int(mapped.value))
        except RuntimeError as exc:  # every rank must still reach the agreement below
            error = f"{exc}"
        _RETAINED.append((self.local, opened))
        failures = [(rank, message) for rank, message in enumerate(comm.all_gather_object(error)) if message]
        if failures:
            raise RuntimeError(f"peer memory exchange failed: {failures[:4]}")
        self.addresses = addresses
        self.symmetric = torch.as_tensor(_DeviceBytes(self.local, self.nbytes), device=device)
        self.peers = torch.tensor(addresses, dtype=torch.int64, device=device)


class SharedFilePeerMemory:
    """CPU emulation of symmetric memory for framework tests: shared file mappings per rank.

    ``peers[p]`` is the address of rank ``p``'s mapping in this process, exactly as an opened IPC
    handle is on a GPU, so a toy CPU implementation can store into peer buffers through it.
    """

    def __init__(self, nbytes: int, comm: Any, device: Any, scratch: Path):
        import torch

        self.nbytes = _rounded(nbytes)
        mine = Path(scratch) / f"symmetric-{comm.rank}.bin"
        with open(mine, "wb") as handle:
            handle.truncate(self.nbytes)
        comm.barrier()
        self._views = [torch.from_file(str(Path(scratch) / f"symmetric-{peer}.bin"), shared=True,
                                       size=self.nbytes, dtype=torch.uint8) for peer in range(comm.world_size)]
        self.addresses = [view.data_ptr() for view in self._views]
        self.symmetric = self._views[comm.rank]
        self.peers = torch.tensor(self.addresses, dtype=torch.int64, device=device)


# --------------------------------------------------------------------------------------------------
# Platforms: the real GPU path, and a CPU seam that only framework tests select.


class GpuPlatform:
    name = "gpu"

    def __init__(self):
        self._flush = None

    def bind(self, rank: int, world_size: int):
        import torch

        if not torch.cuda.is_available():
            raise EnvironmentFailure("A GPU is required: torch.cuda.is_available() is false")
        visible = torch.cuda.device_count()
        if visible < world_size:
            raise EnvironmentFailure(f"this task runs on {world_size} GPUs, but the worker sees {visible}; "
                                   "run it with resources.gpu_groups of that size")
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()

    def peer_memory(self, nbytes: int, comm: Any, device: Any, scratch: Path):
        return HipPeerMemory(nbytes, comm, device)

    def cache_flush(self, nbytes: int):
        from ._aka_benchmark import ReadOnlyCacheFlush

        if self._flush is None or self._flush.nbytes != nbytes:
            self._flush = ReadOnlyCacheFlush(nbytes)
        return self._flush

    def benchmark(self, fn, **kwargs):
        from ._aka_benchmark import benchmark_distributed_cold_cache_samples

        return benchmark_distributed_cold_cache_samples(fn, **kwargs)


class CpuTestPlatform:
    """Framework-test seam: CPU tensors, file-backed peer memory and host-clock spans.

    The command line never selects it; its timings are labelled as test-only and are not a valid
    benchmark method, so the framework would reject them as a score.
    """

    name = "cpu-test"
    METHOD = "cpu_wall_clock_test_only"

    def bind(self, rank: int, world_size: int):
        import torch

        return torch.device("cpu")

    def synchronize(self) -> None:
        return None

    def peer_memory(self, nbytes: int, comm: Any, device: Any, scratch: Path):
        return SharedFilePeerMemory(nbytes, comm, device, scratch)

    def cache_flush(self, nbytes: int):
        return None

    def benchmark(self, fn, *, cache_flush, comm, warmup, repetition, prepare_fn=None, after_sample=None):
        from ._aka_benchmark import synchronized_release_samples

        comm.barrier()  # as on the GPU path: no rank resets its buffer while a peer still uses it
        for _ in range(warmup):
            if prepare_fn is not None:
                prepare_fn()
            comm.barrier()
            fn()
            comm.barrier()

        def run_sample() -> float:
            started = time.perf_counter_ns()
            fn()
            return max(1, time.perf_counter_ns() - started) / 1.0e6

        samples, metadata = synchronized_release_samples(run_sample, comm=comm, repetition=repetition,
                                                         prepare=prepare_fn, after_sample=after_sample)
        metadata.update(benchmark_method=self.METHOD, benchmark_samples=repetition, benchmark_warmup=warmup)
        return samples, metadata


PLATFORMS = {GpuPlatform.name: GpuPlatform, CpuTestPlatform.name: CpuTestPlatform}


# --------------------------------------------------------------------------------------------------
# One rank's view of the action.


class RankContext:
    """What one rank's evaluation needs: its rank, group, device and symmetric buffer."""

    def __init__(self, contract: Contract, rank: int, comm: Any, device: Any, platform: Any, memory: Any):
        self.contract = contract
        self.rank = rank
        self.world_size = contract.world_size
        self.comm = comm
        self.device = device
        self.platform = platform
        self.memory = memory
        self._peers_host = list(memory.addresses)
        import torch

        self._peers_source = torch.tensor(self._peers_host, dtype=torch.int64)

    def call_inputs(self, case: Case) -> dict:
        """The reserved keyword inputs of one call of ``case``."""
        nbytes = self.contract.symmetric_bytes(case)
        if nbytes > self.memory.nbytes:
            raise ContractError(f"case {case.case_id} needs {nbytes} symmetric bytes; {self.memory.nbytes} allocated")
        return {"rank": self.rank, "world_size": self.world_size, "peers": self.memory.peers,
                "symmetric": self.memory.symmetric[:nbytes]}

    def reset(self) -> None:
        """Zero this rank's whole symmetric buffer and restore the peer address table (stream-ordered)."""
        self.memory.symmetric.zero_()
        self.memory.peers.copy_(self._peers_source)

    def peers_intact(self) -> bool:
        return [int(v) for v in self.memory.peers.cpu().tolist()] == self._peers_host

    def fence(self) -> None:
        """Every rank's queued device work is complete."""
        self.platform.synchronize()
        self.comm.barrier()

    def before_call(self) -> None:
        """No rank calls before every rank finished its previous call and zeroed its buffer."""
        self.fence()
        self.reset()
        self.fence()

    def after_call(self) -> None:
        self.platform.synchronize()

    def gather(self, obj: Any) -> list:
        return self.comm.all_gather_object(obj)

    def agree(self, value: Any, what: str) -> None:
        values = self.gather(value)
        if any(other != values[0] for other in values[1:]):
            raise ContractError(f"ranks disagree on {what}: every rank must derive it identically from the seed")


# --------------------------------------------------------------------------------------------------
# Merging the ranks' results (rank 0).


def _merge_correctness_row(rows: list[dict], contract: Contract) -> dict:
    merged = {key: rows[0][key] for key in ("test_case_id", "params") if key in rows[0]}
    per_rank = [row.get("metrics", {}) for row in rows]
    merged["metrics"] = {"ranks": per_rank}
    failed = [rank for rank, row in enumerate(rows) if row["status"] != "PASS"]
    disagree = []
    calls = [metrics.get("calls", []) for metrics in per_rank]
    for index in range(min(map(len, calls)) if calls else 0):
        for name in contract.ranks_agree:
            digests = [rank_calls[index].get("ranks_agree", {}).get(name) for rank_calls in calls]
            if any(digest != digests[0] for digest in digests[1:]):
                disagree.append({"output": name, "seed": calls[0][index].get("seed"),
                                 "ranks_matching_rank0": [r for r, d in enumerate(digests) if d == digests[0]]})
    if failed:
        kinds = {rows[rank].get("failure_kind") for rank in failed}
        merged.update(status="FAIL", failure_kind="input_mutation" if "input_mutation" in kinds
                      else "numerical_mismatch", metadata={"failed_ranks": failed})
        if disagree:
            merged["metadata"]["ranks_disagree"] = disagree
    elif disagree:
        merged.update(status="FAIL", failure_kind="numerical_mismatch", metadata={"ranks_disagree": disagree})
    else:
        merged["status"] = "PASS"
    return merged


def _merge_performance_row(rows: list[dict]) -> dict:
    failed = [rank for rank, row in enumerate(rows) if row["status"] != "PASS"]
    if not failed:
        merged = dict(rows[0])
        times = {row["execution_time_ms"] for row in rows}
        if len(times) != 1:
            raise RankFailure(f"ranks report different aggregated times {sorted(times)}")
        metadata = dict(rows[0].get("metadata", {}))
        metadata["rank_checked_samples"] = [row.get("metadata", {}).get("checked_samples") for row in rows]
        merged["metadata"] = metadata
        return merged
    merged = {key: rows[0][key] for key in ("test_case_id", "params") if key in rows[0]}
    merged.update(status="FAIL", failure_kind="numerical_mismatch",
                  metadata={"failed_ranks": failed,
                            "measured_output_failures": {str(rank): rows[rank].get("metadata", {}).get(
                                "measured_output_failures") for rank in failed}})
    return merged


def merge_evaluation(reports: list[dict], contract: Contract) -> dict:
    """One action report from every rank's report of the same action."""
    from . import evaluate

    first = reports[0]
    role, action = first["role"], first["action"]
    if any((r["role"], r["action"]) != (role, action) or len(r["cases"]) != len(first["cases"]) for r in reports):
        raise RankFailure("ranks evaluated different actions or cases")
    rows = []
    for case_rows in zip(*(report["cases"] for report in reports)):
        if len({row["test_case_id"] for row in case_rows}) != 1:
            raise RankFailure("ranks evaluated cases in different orders")
        rows.append(_merge_correctness_row(list(case_rows), contract) if action == "correctness"
                    else _merge_performance_row(list(case_rows)))
    report = evaluate.correctness_report(role, rows) if action == "correctness" \
        else evaluate.performance_report(role, rows)
    report.setdefault("metadata", {})["world_size"] = contract.world_size
    return report


def _merge_first(reports: list[dict], contract: Contract) -> dict:
    """Jobs whose ranks already agreed on one result (the QA control suite)."""
    return reports[0]


# --------------------------------------------------------------------------------------------------
# Rank processes.


def _run_job(job: str, role: str, action: str, context: RankContext) -> dict:
    from . import evaluate

    contract = context.contract
    if job == "evaluate":
        # Rank 0 builds (HIP extensions, imports) first, so ranks never race on one build directory.
        if context.rank == 0:
            evaluate._role_calls(contract, role)
        context.comm.barrier()
        if action == "correctness":
            return evaluate.correctness_action(contract, role, rank=context)
        if action == "performance":
            return evaluate.performance_action(contract, role, rank=context)
        raise ValueError(f"no multi-rank {action} action")
    if job == "controls":
        from . import controls

        if context.rank == 0:
            evaluate._load(contract, "baseline", audit=False)
        context.comm.barrier()
        return controls.run_controls(rank=context)
    raise ValueError(f"unknown multi-rank job {job!r}")


_MERGES = {"evaluate": merge_evaluation, "controls": _merge_first}


def _die_with_parent() -> None:
    """A rank never outlives its launcher (Linux); the launcher's process group covers the rest."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _failure_kind(exc: BaseException) -> str:
    from .audit import PolicyError

    kind = getattr(exc, "kind", None)
    if isinstance(kind, str):
        return kind
    from .integrity import TamperError

    if isinstance(exc, TamperError):
        return "harness_tampering"
    if isinstance(exc, ContractError):
        return "task_error"
    if isinstance(exc, PolicyError):
        return "policy_violation"
    if type(exc).__name__ == "BuildError":
        return "compile_error"
    text = str(exc)
    if "PyTorch operation" in text or "kernel launch" in text or "launched kernels" in text \
            or "host-side communication" in text:
        return "policy_violation"
    return "execution_error"


def _write_json(path: Path, obj: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _worker(job: str, role: str, action: str, rank: int, world_size: int, scratch: str, platform_name: str,
            timeout_s: float) -> None:
    _die_with_parent()
    os.environ.setdefault("USER", "aka")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Rendezvous through the FileStore; rank-to-rank traffic stays on the loopback interface.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    code = 0
    try:
        from datetime import timedelta

        import torch.distributed as dist

        contract = load_contract()
        platform = PLATFORMS[platform_name]()
        device = platform.bind(rank, world_size)
        store = dist.FileStore(str(Path(scratch) / "store"), world_size)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=max(1.0, timeout_s)))
        comm = GlooComm(rank, world_size)
        nbytes = max(contract.symmetric_bytes(case) for case in contract.cases)
        memory = platform.peer_memory(nbytes, comm, device, Path(scratch))
        context = RankContext(contract, rank, comm, device, platform, memory)
        local = _run_job(job, role, action, context)
        reports = comm.all_gather_object(local)
        if rank == 0:
            _write_json(Path(scratch) / "result.json", _MERGES[job](reports, contract))
        comm.barrier()
    except BaseException as exc:  # noqa: BLE001 - every failure is reported, then the rank exits
        code = 1
        try:
            _write_json(Path(scratch) / f"error-{rank}.json", {
                "rank": rank, "time_ns": time.monotonic_ns(), "kind": _failure_kind(exc),
                "message": f"{type(exc).__name__}: {exc}"[:3000],
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # Skip interpreter teardown: a peer may still hold this rank's device memory mapped, and
            # HIP teardown with a spinning kernel would hang instead of exiting.
            os._exit(code)


def _stop(processes: list) -> None:
    for process in processes:
        if process.exitcode is None:
            process.kill()
    for process in processes:
        process.join(timeout=10)


def _primary_error(scratch: Path) -> tuple[dict | None, list[dict]]:
    errors = []
    for path in sorted(scratch.glob("error-*.json")):
        try:
            errors.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    errors.sort(key=lambda item: item.get("time_ns", 0))
    return (errors[0] if errors else None), errors


def launch(contract: Contract, job: str, role: str, action: str, *, platform: str = "gpu",
           timeout_s: float | None = None) -> tuple[dict | None, dict | None]:
    """Run ``job`` on every rank; return (rank 0's merged result, None) or (None, failure).

    A failure is ``{"reason": ..., "kind": ...}``: the first rank error, a rank that died without
    one, or the launcher deadline. Every remaining rank is killed before this returns.
    """
    if contract.distributed is None:
        raise ContractError("launch() is for multi-GPU tasks")
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}")
    world_size = contract.world_size
    if timeout_s is None:
        timeout_s = max(_MIN_TIMEOUT_S, float(contract.distributed.get("timeout_s", 3600)) - _TIMEOUT_MARGIN_S)
    scratch = Path(tempfile.mkdtemp(prefix="aka-ranks-"))
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker, name=f"aka-rank-{rank}",
                                 args=(job, role, action, rank, world_size, str(scratch), platform, timeout_s))
                 for rank in range(world_size)]
    failure = None
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout_s
        while True:
            codes = [process.exitcode for process in processes]
            if all(code == 0 for code in codes):
                break
            dead = [rank for rank, code in enumerate(codes) if code not in (None, 0)]
            if dead:
                # Give the failing rank's peers a moment to record their own errors, then stop all.
                time.sleep(0.5)
                _stop(processes)
                primary, errors = _primary_error(scratch)
                if primary is None:
                    failure = {"reason": f"rank {dead[0]} exited with code {codes[dead[0]]} without a report",
                               "kind": "execution_error"}
                else:
                    others = sorted({item["rank"] for item in errors} - {primary["rank"]})
                    failure = {"reason": f"rank {primary['rank']} of {world_size} failed: {primary['message']}"
                               + (f" (ranks {others} also failed)" if others else "")
                               + "; the remaining ranks were stopped",
                               "kind": primary.get("kind", "execution_error"),
                               "traceback": primary.get("traceback", "")}
                break
            if time.monotonic() > deadline:
                _stop(processes)
                waiting = [rank for rank, code in enumerate(codes) if code is None]
                failure = {"reason": f"the {world_size}-rank action exceeded {timeout_s:.0f}s; ranks {waiting} "
                                     "were still running and were killed", "kind": "timeout"}
                break
            alive = [process.sentinel for process in processes if process.exitcode is None]
            multiprocessing.connection.wait(alive, timeout=min(_LAUNCH_POLL_S, max(0.0, deadline - time.monotonic())))
        if failure is not None:
            return None, failure
        try:
            return json.loads((scratch / "result.json").read_text(encoding="utf-8")), None
        except (OSError, ValueError) as exc:
            return None, {"reason": f"rank 0 wrote no result: {exc}", "kind": "execution_error"}
    finally:
        _stop(processes)
        shutil.rmtree(scratch, ignore_errors=True)


__all__ = ["GRANULE", "GlooComm", "HipPeerMemory", "RankContext", "SharedFilePeerMemory", "launch",
           "merge_evaluation"]
