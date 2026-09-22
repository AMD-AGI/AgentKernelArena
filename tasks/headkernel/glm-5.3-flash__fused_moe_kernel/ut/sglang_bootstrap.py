"""Publish a process-wide sglang runtime context so the extracted MoE seam is callable standalone.

`fused_experts_impl` -> `_prepare_fused_moe_run` -> `try_get_optimal_moe_config` reads
`get_exec().deterministic.enable_deterministic_inference`, and `RuntimeContext.config_bag` fails
CLOSED ("config namespace 'exec' not published") in any process that never booted a server. The
extracted op is otherwise a pure tensor->tensor function, so the only thing missing outside the
server is this config projection. Publish the SAME ServerArgs the deployment runs with, so the
Triton config selection the op makes here is the config selection it makes online.

Both `build_oracle.py` (golden) and `cases.py` (baseline + candidate legs) call `ensure()`, so all
three paths agree. Kept tiny and dependency-free on purpose: it must not drag in a GPU/dist init.
"""
import os

MODEL_PATH = os.environ.get("GEAK_MODEL_PATH", "/shared_nfs/models/GLM-5.3-Flash")

# Mirrors the deployment launch flags that reach the MoE path (see regime.json / CURRENT_FLAGS).
_SERVER_ARGS = dict(
    model_path=MODEL_PATH,
    tokenizer_path=MODEL_PATH,
    trust_remote_code=True,
    tp_size=8,
    context_length=11264,
    kv_cache_dtype="bfloat16",
    moe_runner_backend="triton",
    disable_cuda_graph=True,
    disable_radix_cache=True,
    mem_fraction_static=0.8,
)

_done = False


def ensure():
    global _done
    if _done:
        return
    from sglang.srt import runtime_context as rc
    if rc._CONTEXT.is_config_namespace_published("exec"):
        _done = True
        return
    from sglang.srt.server_args import ServerArgs
    rc.publish(ServerArgs(**_SERVER_ARGS), role="scheduler")
    _init_parallel()
    _done = True


def _free_port():
    import socket
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _init_parallel():
    """Bring up a WORLD-SIZE-1 tensor-parallel group on this one GPU.

    `_fused_moe_kernel_sequence` calls `get_tp_group()` (for the NCCL symmetric-memory allocation
    context around the MoE output) and asserts if no TP group exists. The captured tensors are
    ALREADY this rank's TP=8 shard (per-TP intermediate size 256, w1 [288,512,4096]), so the op is
    replayed shard-local: world size 1 here is not a different problem size, it just makes the
    intra-op group trivial. Any collective on it is a no-op, matching the fact that the deployment's
    MoE all-reduce happens OUTSIDE this seam, in the caller.
    """
    import torch
    from sglang.srt.distributed import parallel_state as ps
    if ps.model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # ALWAYS take a fresh ephemeral port, never setdefault. unittest.py bootstraps in the PARENT
    # process (correctness runs in-process) and then measure_legs forks timing subprocesses that
    # INHERIT its environ -- with setdefault every child reused the parent's MASTER_PORT, collided
    # with the TCPStore the parent still owns, and raised inside cases.call. time_op swallows that
    # into ms=None, so every bucket silently reported baseline_ms=optimized_ms=null.
    os.environ["MASTER_PORT"] = os.environ.get("GEAK_MASTER_PORT") or str(_free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    ps.init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="env://",
        backend="nccl" if torch.cuda.is_available() else "gloo")
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
