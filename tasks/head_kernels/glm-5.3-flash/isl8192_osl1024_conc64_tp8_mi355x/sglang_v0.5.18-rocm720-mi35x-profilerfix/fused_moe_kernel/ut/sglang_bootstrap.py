"""Protected model-free context for the captured TP8 MoE shard.

The official SGLang override_server_args API publishes normal config bags via
its dummy-model boundary. The op still requires those bags and a one-rank TP
group; model configuration, model weights and architecture patches are unused.
"""
import atexit
from pathlib import Path
import tempfile

# Recorded non-model settings plus the unchanged defaults of the two config
# leaves read by the MoE dispatcher/config selector.
_SERVER_ARGS = dict(
    trust_remote_code=True,
    tp_size=8,
    context_length=11264,
    kv_cache_dtype="bfloat16",
    moe_runner_backend="triton",
    disable_cuda_graph=True,
    disable_radix_cache=True,
    mem_fraction_static=0.8,
    enable_deterministic_inference=False,
    enable_fused_moe_sum_all_reduce=False,
)
_done = False
_override = None
_owns_parallel = False
_rendezvous = None


def _assert_context(rc):
    config = rc.get_exec()
    if (config.deterministic.enable_deterministic_inference is not False
            or config.moe.enable_fused_moe_sum_all_reduce is not False
            or config.moe.moe_runner_backend != "triton"):
        raise RuntimeError("GLM runtime config differs from the captured MoE contract")
    args = rc.get_context().server_args
    if any(getattr(args, key) != value for key, value in _SERVER_ARGS.items()):
        raise RuntimeError("GLM runtime settings differ from the captured deployment")


def ensure():
    global _done, _override
    from sglang.srt import runtime_context as rc
    if _done:
        _assert_context(rc)
        return
    override = rc.get_context().override_server_args(**_SERVER_ARGS)
    override.install()
    _override = override
    try:
        _assert_context(rc)
        _init_parallel()
    except BaseException:
        cleanup()
        raise
    _done = True
    atexit.register(cleanup)


def cleanup():
    global _done, _override, _owns_parallel, _rendezvous
    try:
        if _owns_parallel:
            from sglang.srt.distributed import parallel_state as ps
            ps.destroy_model_parallel()
            ps.destroy_distributed_environment()
    finally:
        _owns_parallel = False
        if _rendezvous is not None:
            _rendezvous.cleanup()
            _rendezvous = None
        if _override is not None:
            _override.restore()
            _override = None
        _done = False


def _init_parallel():
    """Bring up a WORLD-SIZE-1 tensor-parallel group on this one GPU.

    `_fused_moe_kernel_sequence` calls `get_tp_group()` (for the NCCL symmetric-memory allocation
    context around the MoE output) and asserts if no TP group exists. The captured tensors are
    ALREADY this rank's TP=8 shard (per-TP intermediate size 256, w1 [288,512,4096]), so the op is
    replayed shard-local: world size 1 here is not a different problem size, it just makes the
    intra-op group trivial. Any collective on it is a no-op, matching the fact that the deployment's
    MoE all-reduce happens OUTSIDE this seam, in the caller.
    """
    global _owns_parallel, _rendezvous
    import torch
    from sglang.srt.distributed import parallel_state as ps
    if ps.model_parallel_is_initialized():
        if ps.get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("GLM standalone replay requires a one-rank TP group")
        return
    # Each worker/container gets its own task-local file store. Shared host
    # networking and inherited MASTER_PORT values cannot collide here.
    build = Path(__file__).resolve().parents[1] / "build"
    build.mkdir(parents=True, exist_ok=True)
    _rendezvous = tempfile.TemporaryDirectory(prefix="glm-rendezvous-", dir=build)
    store = (Path(_rendezvous.name) / "store").as_uri()
    torch.cuda.set_device(0)
    ps.init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=store,
        backend="nccl")
    _owns_parallel = True
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
