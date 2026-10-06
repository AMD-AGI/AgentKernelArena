"""Owner-installed adapter for the shared V4 served-tensor recorder.

Import is inert. The owner supplies recorder/served/graph lifecycle context;
this adapter neither starts a server nor invents served replay notifications.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading


ROOT = Path(__file__).resolve().parents[1]
KERNEL = "_gemm_afp4wfp4_kernel"
COMMON_SHA256 = "8cff65ca8a74c4ee7565de5fddfbab92f4a5bb564c6348b6449eb2086da741ea"
ARGS = ("a_ptr", "b_ptr", "c_ptr", "a_scales_ptr", "b_scales_ptr", "M", "N", "K",
        "stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_ck", "stride_cm", "stride_cn",
        "stride_asm", "stride_ask", "stride_bsn", "stride_bsk")


class KernelProbe:
    def __init__(self, original, context, cap):
        self.original, self.context, self.cap = original, context, cap

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]
        def run(*args, **kwargs):
            values = dict(zip(ARGS, args))
            values.update(kwargs)
            result = launch(*args, **kwargs)
            events = getattr(self.context, "events", None)
            if events is not None:
                controls = {key: self.cap.controls_json(value) for key, value in values.items()
                            if key not in ARGS[:5]}
                resolved_grid = grid(values) if callable(grid) else grid
                events.append({"kernel": KERNEL, "arguments": controls,
                               "grid": self.cap.controls_json(resolved_grid),
                               "compiled_name": getattr(result, "name", None),
                               "compiled_hash": getattr(result, "hash", None)})
            return result
        return run


def install(cap, quark_module, basic_module, kernel_module, context):
    """context() -> None or {recorder, served, graph_id, slot_id, bucket}.

    Before CUDA graph capture the owner must initialize the recorder and admit
    this family's slots/budgets. For eager calls served is mandatory. For graph
    calls the owner later reports actual served replays and finalizes/seals V4.
    """
    if hashlib.sha256(Path(cap.__file__).read_bytes()).hexdigest() != COMMON_SHA256:
        raise ValueError("FP4 adapter requires the pinned shared V4 recorder")
    pins = json.loads((ROOT / "SOURCE-PROVENANCE.json").read_text())
    for module, key in ((quark_module, "ut/native/quark_linear.py"), (basic_module, "ut/native/wrapper.py"),
                        (kernel_module, "source/kernel.py")):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != pins["sources"][key]["sha256"]:
            raise ValueError("FP4 capture source differs from the pinned native image")
    original = quark_module._gemm_afp4wfp4_orig
    if original is not basic_module.gemm_afp4wfp4:
        raise ValueError("Quark global is not bound to the pinned basic GEMM wrapper")
    signature = inspect.signature(original)
    if list(signature.parameters) != ["x", "w", "x_scales", "w_scales", "dtype", "y", "config", "skip_reduce"]:
        raise ValueError("Quark native FP4 callable signature changed")
    local = threading.local()
    kernel_original = basic_module._triton_gemm_afp4wfp4_kernel
    if kernel_original is not kernel_module._gemm_afp4wfp4_kernel:
        raise ValueError("native wrapper kernel global does not match the pinned kernel module")
    basic_module._triton_gemm_afp4wfp4_kernel = KernelProbe(kernel_original, local, cap)

    @functools.wraps(original)
    def capture(*args, **kwargs):
        active = context()
        if active is None:
            if cap.capturing():
                raise ValueError("FP4 graph capture lacks the owner lifecycle context")
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        inputs = {name: values[name] for name in ("x", "w", "x_scales", "w_scales", "y")}
        roles = {name: cap.Role("readonly" if name in ("w", "w_scales") else "mutable",
                                footprint="full_storage", optional=name == "y") for name in inputs}
        family = cap.Family("minimax_fp4_gemm", pins["sources"]["source/kernel.py"]["sha256"],
                            roles, {"result": cap.Role("mutable", footprint="full_storage")})
        controls = {"dtype": None if values["dtype"] is None else str(values["dtype"]),
                    "config_requested": cap.controls_json(values["config"]),
                    "skip_reduce": values["skip_reduce"], "use_splitk_bf16": basic_module._USE_GEMM_SPLITK_BF16,
                    "packing": "e2m1_low_nibble_first", "scale_format": "e8m0_group32_unshuffled"}
        if getattr(local, "events", None) is not None:
            raise ValueError("nested FP4 capture is unsupported")
        rec = active["recorder"]
        if cap.capturing():
            handle = rec.begin_graph(family, inputs, controls, graph_id=active["graph_id"],
                                     slot_id=active["slot_id"], bucket=active["bucket"])
        else:
            handle = rec.begin_eager(family, inputs, controls, active["served"])
        local.events = []
        try:
            result = original(*args, **kwargs)
            if len(local.events) != 1:
                raise ValueError("Quark call did not execute exactly one declared FP4 GEMM launch")
            handle.controls["launches"] = local.events
            rec.finish(handle, {"result": result})
            return result
        except BaseException as error:
            rec.abort(handle, error)
            raise
        finally:
            local.events = None

    # The already registered Quark custom-op implementation reads this global
    # when invoked. Replacing the function's module attribute alone is inert.
    quark_module._gemm_afp4wfp4_orig = capture
    return capture
