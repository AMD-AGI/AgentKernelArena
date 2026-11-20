# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next monokernel geometry (``amd/Qwen3.8-Flash-Next-Quark-MXFP4-
AttnFP8-PLEFP8``, SGLang's Qwen4-Exp) and the build key of one launch."""

import dataclasses

# the text config's shapes, which the kernel builds in
HIDDEN = 2560
HC = 4
HC_DIM = HIDDEN * HC  # 10240: the hyper-connection residual streams
HC_LOWRANK = 320
EPS = 1e-6
# Gated DeltaNet
GDN_HK = 16
GDN_HV = 48
GDN_DK = 128
GDN_DV = 128
GDN_KEY_DIM = GDN_HK * GDN_DK  # 2048
GDN_VAL_DIM = GDN_HV * GDN_DV  # 6144
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VAL_DIM  # 10240
GDN_QKVZ = 2 * GDN_KEY_DIM + 2 * GDN_VAL_DIM  # 16384
# MoE: 512 routed MXFP4 experts, top-10, and the shared expert fused at index 512
N_EXPERTS = 512
TOPK = 10
MOE_INTER = 640

# a launch's rows: R requests x L verify tokens (row t: request t // L, step t % L)
MAX_R = 8
MAX_L = 4
MAX_T = 32


@dataclasses.dataclass(frozen=True)
class ConvStrides:
    """Element strides (draft step, channel, window, request row) of a GDN layer's
    intermediate conv-window view (SGLang's ``intermediate_conv_window[0]``)."""

    step: int = 30720
    channel: int = 3
    window: int = 1
    row: int = 4 * 30720


@dataclasses.dataclass(frozen=True)
class MonoBuild:
    """One launch: its layers and rows, and the cache strides it is compiled for;
    every field is a scalar of the JIT cache key (``key_tuple``)."""

    plan: str  # the launch's layer kinds, comma-separated (gdn | qsa)
    nreq: int  # R requests
    ntok: int  # of L verify tokens each
    rtt_stride: int = 0  # req_to_token row stride (int32 elements)
    # element stride between the 3 MRoPE position axes (0: 1-D text positions)
    pos_stride: int = 1
    map_i64: bool = False  # the request -> mamba slot map is int64
    ple_first: bool = False  # the launch starts with layer 1's PLE
    final_mix: bool = False  # it ends with the model-level mixer
    prelude: bool = False  # it embeds the tokens (launch 0)
    eos_token_id: int = 0
    # SGLang's verify scratch strides: the intermediate conv window, and elements
    # per request row of intermediate_ssm, of the PLE intermediate short-conv state
    # (bf16) and of the n-gram intermediate context (int64)
    iconv: ConvStrides = dataclasses.field(default_factory=ConvStrides)
    issm_row: int = 4 * GDN_HV * GDN_DK * GDN_DV
    ple_ist_row: int = 4 * HC_DIM * 9
    ictx_row: int = 8
    # layer 0's input: "global" the previous launch, "embed" the prelude's token
    # embedding, "mtp" the MTP input fusion
    first_src: str = "global"
    # the GDN SSM states are bf16 (SGLANG_MAMBA_SSM_DTYPE=bfloat16): the recurrence
    # stays f32 and the states are rounded when stored, as SGLang does
    ssm_bf16: bool = False
    # each request's first sequence position is its seq_lens entry (a target verify,
    # whose RoPE positions may be MRoPE's), not its first row's position
    seq_p0: bool = False
    # index-key ring slots a request in SGLang's QSA pool: 4 (one group), or 8 (two
    # groups, so a verify window never overwrites the pending group's members)
    ring: int = 4

    def __post_init__(self) -> None:
        kinds = self.layers
        if not kinds or any(k not in ("gdn", "qsa") for k in kinds):
            raise ValueError(f"plan {self.plan!r}: layer kinds are gdn or qsa")
        if not (1 <= self.nreq <= MAX_R and 1 <= self.ntok <= MAX_L):
            raise ValueError(
                f"{self.nreq} requests x {self.ntok} tokens: the kernel takes 1..{MAX_R}"
                f" requests of 1..{MAX_L} tokens"
            )
        if self.nreq * self.ntok > MAX_T:
            raise ValueError(f"{self.nreq * self.ntok} rows, past {MAX_T}")
        if self.first_src not in ("global", "embed", "mtp"):
            raise ValueError(f"first_src {self.first_src!r}")

    @property
    def layers(self) -> tuple:
        return tuple(self.plan.split(",")) if self.plan else ()


def key_tuple(key) -> tuple:
    """Every build parameter of ``key`` as a (name, value) pair, nested dataclasses
    flattened: FlyDSL keys a kernel on the scalars it closes over."""
    return _flat_items(dataclasses.asdict(key))


def _flat_items(fields: dict, prefix: str = "") -> tuple:
    # dataclasses.asdict renders a nested dataclass as a dict; no field is a dict itself.
    out = []
    for name, value in fields.items():
        if isinstance(value, dict):
            out += _flat_items(value, f"{prefix}{name}.")
        else:
            out.append((f"{prefix}{name}", value))
    return tuple(out)
