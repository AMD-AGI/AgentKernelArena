# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Logical inputs of one MiniMax-M3.1-Flash FFN step on a TP4 group, each rank's shard, the kernel-facing
weight layouts and each rank's expected outputs.

Logical inputs (what make_inputs returns and reference.py reads; ffn_numerics.py defines the numerics):
  replicated, drawn by the framework from task.yaml: res [S, 6144] bf16 (the residual stream entering
    the layer), norm_w [6144] bf16 (post_attention_layernorm, applied as 1 + w), bias [128] fp32 (the
    routing correction bias).
  replicated, from make_inputs: router_w [128, 6144] fp32 (the gate, planted below).
  rank-specific, stacked on a leading TP axis (rank r's slice of the model, as SGLang shards it):
    attn [4, S, 2048] bf16: the rank's attention output (its 16 heads of 128), o_proj's input;
    wo [4, 6144, 2048] E4M3 + wos [4, 6144] fp32: the rank's o_proj input columns as SGLang's rowwise-FP8
      decode copy (per output row: scale = amax / 448);
    w13 [4, 128, 1536, 3072] uint8 + s13 [4, 128, 1536, 192] uint8: MXFP4 gate (rows 0-767) and up
      (768-1535) rows of the rank's intermediate slice of the 128 routed experts, two E2M1 codes a byte
      along the hidden axis (low nibble first), one E8M0 byte per 32;
    w2 [4, 128, 6144, 384] uint8 + s2 [4, 128, 6144, 24] uint8: the down projection's slice, MXFP4 along
      the intermediate axis;
    w13s [4, 1536, 6144] E4M3 + s13s [4, 1536, 192] uint8, w2s [4, 6144, 768] E4M3 + s2s [4, 6144, 24]
      uint8: the shared expert's slice in the checkpoint's MXFP8 (E4M3, E8M0 per 32 along K).

shard_inputs picks rank r's slices; to_kernel_layout prepares them as SGLang / AITER leave them after
loading and as the launch reads them (outside timing): AITER shuffle_weight (16, 16) on the o_proj FP8
and the MXFP4 codes, e8m0_shuffle on the MXFP4 scales, and the launch's a16w8 layout for the shared
expert (FP8 bytes in lane order, the E8M0 scales as fp32 powers of two). rank_expected gives every rank
the replicated rows: out and res_out are the all-reduced results every rank writes.

Value ranges follow sgl-amd-mono's own kernel test (test/kernels/test_minimax_m31_ffn.py): attention
output 3 N(0, 1), residual 4 N(0, 1), o_proj 0.02 N(0, 1) before its rowwise FP8, random MXFP4 codes with
scales 2^-8 .. 2^-5, the shared expert 2 N(0, 1) in FP8 with the same scales, norm gain 0.1 N(0, 1).

Planted margin (the one discrete decision, top-4 routing, must not flip between the reference and a
correct kernel): the router weight is a random 0.02 N(0, 1) matrix plus the minimum-norm correction in
the span of the rows' normalized inputs xn (computed exactly as the reference does) that gives every row
4 random winners with logits in [1.5, 3] and the other 124 experts logits in [-4, -0.25]; with the bias
in [-0.03, 0.03], sigmoid + bias separates each row's 4th and 5th choices by >= 0.32 (checked against
ROUTE_MARGIN). The winners' sigmoids, and so the routing weights, still vary continuously.
"""
import torch

from ffn_numerics import ATTN_K, EXPERTS, FP8_MAX, GROUP, HIDDEN, INTER, TOPK, TP, route, seam

ATTN_STD, OPROJ_STD, SHARED_STD = 3.0, 0.02, 2.0
E8M0_RANGE = (119, 123)  # expert scale bytes 119 .. 122: 2^-8 .. 2^-5
GATE_STD = 0.02
WIN_LOGITS, LOSE_LOGITS = (1.5, 3.0), (-4.0, -0.25)
ROUTE_MARGIN = 0.1
RANK_SPECIFIC = ("attn", "wo", "wos", "w13", "s13", "w2", "s2", "w13s", "s13s", "w2s", "s2s")


def _uniform(shape, low, high, g, device):
    return low + (high - low) * torch.rand(shape, generator=g, device=device, dtype=torch.float64)


def _e8m0(shape, g, device):
    return torch.randint(*E8M0_RANGE, shape, generator=g, device=device, dtype=torch.uint8)


def _plant_router(xn, g, device):
    """An fp32 router weight whose logits for the rows' normalized inputs ``xn`` have 4 clear winners."""
    rows = xn.shape[0]
    logits = _uniform((rows, EXPERTS), *LOSE_LOGITS, g, device)
    for t in range(rows):
        win = torch.randperm(EXPERTS, generator=g, device=device)[:TOPK]
        logits[t, win] = _uniform((TOPK,), *WIN_LOGITS, g, device)
    base = GATE_STD * torch.randn(EXPERTS, HIDDEN, generator=g, device=device, dtype=torch.float64)
    # base + C xn with (base + C xn) xn^T = logits^T: the minimum-norm correction in the rows' span
    corr = torch.linalg.solve(xn @ xn.T, (logits.T - base @ xn.T).T).T
    return (base + corr @ xn).to(torch.float32)


def _check_routing(xn, router_w, bias):
    _, _, logits = route(xn, router_w, bias)
    key = (torch.sigmoid(logits) + bias.double()).sort(dim=-1, descending=True).values
    margin = float((key[:, TOPK - 1] - key[:, TOPK]).min())
    if margin < ROUTE_MARGIN:
        raise RuntimeError(f"routing margin too small: {margin:.4f}")


def make_inputs(case, inputs, generator, device):
    g = generator
    rows = case["axes"]["rows"]
    assert case["axes"]["world_size"] == TP, case["axes"]
    out = {"attn": (ATTN_STD * torch.randn(TP, rows, ATTN_K, generator=g, device=device)).to(torch.bfloat16)}
    # o_proj as SGLang keeps it for decode on gfx950: each rank's input columns in rowwise FP8
    dense = OPROJ_STD * torch.randn(TP, HIDDEN, ATTN_K, generator=g, device=device)
    out["wos"] = dense.abs().amax(-1).clamp_min(1e-12) / FP8_MAX
    out["wo"] = (dense / out["wos"].unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    del dense
    for name, (n, k) in (("w13s", (2 * INTER, HIDDEN)), ("w2s", (HIDDEN, INTER))):
        out[name] = (SHARED_STD * torch.randn(TP, n, k, generator=g, device=device)).to(torch.float8_e4m3fn)
        out["s" + name[1:]] = _e8m0((TP, n, k // GROUP), g, device)
    for name, (n, k) in (("w13", (2 * INTER, HIDDEN)), ("w2", (HIDDEN, INTER))):
        out[name] = torch.randint(0, 256, (TP, EXPERTS, n, k // 2), generator=g, device=device,
                                  dtype=torch.uint8)
        out["s" + name[1:]] = _e8m0((TP, EXPERTS, n, k // GROUP), g, device)
    _, xn = seam(out["attn"], inputs["res"], out["wo"], out["wos"], inputs["norm_w"])
    out["router_w"] = _plant_router(xn, g, device)
    _check_routing(xn, out["router_w"], inputs["bias"])
    return out


# ------------------------------------------------------------------------------------------ layouts
def _bytes(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _like(u8, t):
    return u8.view(torch.float8_e4m3fn) if t.dtype == torch.float8_e4m3fn else u8


def shuffle_weight(t):
    """aiter.ops.shuffle.shuffle_weight(t, layout=(16, 16)) of a byte matrix [..., N, K bytes]: 16-row x
    32-byte tiles in 16-byte halves, so lane l of an MFMA reads row l % 16, bytes 16 (l / 16) .. + 16 of a
    64-byte K step."""
    x = _bytes(t)
    n, kb = x.shape[-2:]
    assert n % 16 == 0 and kb % 32 == 0, x.shape
    y = x.reshape(-1, n // 16, 16, kb // 32, 2, 16).permute(0, 1, 3, 4, 2, 5).contiguous().reshape(x.shape)
    return _like(y, t)


def e8m0_shuffle(t):
    """aiter e8m0_shuffle (shuffle_scale) of E8M0 scales [..., rows, groups] taken as one [rows', groups]
    matrix (rows' a multiple of 256 and groups of 8 here, so AITER's padding is empty)."""
    groups = t.shape[-1]
    x = t.reshape(-1, groups)
    r = x.shape[0]
    assert r % 256 == 0 and groups % 8 == 0, t.shape
    y = x.view(r // 32, 2, 16, groups // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()
    return y.reshape(t.shape)


def a16w8_layout(weight, scale):
    """sgl_amd_mono/minimax_m31/a16w8.py ``layout`` at 52486703: ``weight`` [R, K] FP8 and its E8M0 group-32
    ``scale`` [R, K / 32] -> (uint8 [R/16, K/128, 64, 32], f32 [R/16, K/128, 64]), lane = kg * 16 + row % 16
    for the K group kg = (k % 128) // 32 of the step."""
    rows, k = weight.shape
    w = weight.view(torch.uint8).view(rows // 16, 16, k // 128, 4, 32)
    w = w.permute(0, 2, 3, 1, 4).reshape(rows // 16, k // 128, 64, 32).contiguous()
    s = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    s = s.view(rows // 16, 16, k // 128, 4).permute(0, 2, 3, 1).reshape(rows // 16, k // 128, 64)
    return w, s.contiguous()


def shard_inputs(logical, *, rank, world_size):
    """Rank ``rank``'s inputs before to_kernel_layout: its slices and the replicated tensors."""
    assert world_size == TP, world_size
    return {name: (value[rank] if name in RANK_SPECIFIC else value) for name, value in logical.items()}


def to_kernel_layout(case, inputs):
    """A rank's weights as SGLang / AITER prepare them at load time and the launch reads them."""
    inputs = dict(inputs)
    for name in ("wo", "w13", "w2"):
        inputs[name] = shuffle_weight(inputs[name])
    for name in ("s13", "s2"):
        inputs[name] = e8m0_shuffle(inputs[name])
    for w, s in (("w13s", "s13s"), ("w2s", "s2s")):
        inputs[w], inputs[s] = a16w8_layout(inputs[w], inputs[s])
    return inputs


def rank_expected(logical_outputs, *, rank, world_size):
    """Rank ``rank``'s expected tensors: the all-reduced rows, the same on every rank."""
    assert world_size == TP, world_size
    return {"out": logical_outputs["out"], "res_out": logical_outputs["res_out"]}
