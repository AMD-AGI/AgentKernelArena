"""Protected paged decode inputs and an independent vectorized FP32 oracle."""
from __future__ import annotations

import math

import torch


def validate_params(p):
    required = {"sequences", "query_rows", "heads", "kv_heads", "head_size", "block_size",
                "context", "cache_blocks", "table_cols", "ragged", "layout"}
    if (set(p) != required or p["sequences"] < 1 or p["query_rows"] < p["sequences"]
            or p["head_size"] != 128 or p["block_size"] != 16
            or p["heads"] % p["kv_heads"] or p["heads"] // p["kv_heads"] not in (4, 8, 16)
            or p["context"] < 1 or p["table_cols"] < math.ceil(p["context"] / 16)
            or p["cache_blocks"] < p["sequences"] * math.ceil(p["context"] / 16)
            or type(p["ragged"]) is not bool or p["layout"] not in ("contiguous", "permuted")):
        raise ValueError(f"Unsupported paged-attention parameters: {p}")


def make_inputs(p, seed=42, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    seqs, heads, kv, dim, block = (p[k] for k in ("sequences", "heads", "kv_heads", "head_size", "block_size"))
    partitions = math.ceil(p["context"] / 256)
    blocks_per_seq = math.ceil(p["context"] / block)
    pages = torch.arange(p["cache_blocks"], device=device, dtype=torch.int32)
    if p["layout"] == "permuted":
        pages = torch.randperm(p["cache_blocks"], generator=generator, device=device).to(torch.int32)
    indices = torch.arange(seqs, device=device)[:, None] * blocks_per_seq
    indices = (indices + torch.arange(p["table_cols"], device=device)[None, :]).clamp_max(p["cache_blocks"] - 1)
    tables = pages[indices].contiguous()
    lengths = torch.full((seqs,), p["context"], dtype=torch.int32, device=device)
    if p["ragged"]:
        choices = [p["context"], 1, min(16, p["context"]), min(17, p["context"]), max(1, p["context"] - 1)]
        lengths.copy_(torch.tensor([choices[i % len(choices)] for i in range(seqs)], device=device, dtype=torch.int32))
    return {
        "query": torch.randn((p["query_rows"], heads, dim), generator=generator, device=device, dtype=torch.bfloat16),
        "key": torch.randn((p["cache_blocks"], kv, dim // 8, block, 8), generator=generator, device=device, dtype=torch.bfloat16),
        "value": torch.randn((p["cache_blocks"], kv, dim, block), generator=generator, device=device, dtype=torch.bfloat16),
        "tables": tables, "lengths": lengths,
        "query_start": torch.arange(seqs + 1, device=device, dtype=torch.int32),
        "k_scale": torch.ones((), device=device, dtype=torch.float32),
        "v_scale": torch.ones((), device=device, dtype=torch.float32),
        "out": torch.zeros((p["query_rows"], heads, dim), device=device, dtype=torch.bfloat16),
        "exp_sums": torch.empty((seqs, heads, partitions), device=device, dtype=torch.float32),
        "max_logits": torch.empty((seqs, heads, partitions), device=device, dtype=torch.float32),
        "tmp_out": torch.empty((seqs, heads, partitions, dim), device=device, dtype=torch.bfloat16),
    }


def readonly(values):
    return {name: values[name] for name in
            ("query", "key", "value", "tables", "lengths", "query_start", "k_scale", "v_scale")}


def draw(values, p, seed):
    query = values["query"]
    generator = torch.Generator(device=query.device).manual_seed(seed)
    return {"query": torch.randn(query.shape, generator=generator, device=query.device, dtype=query.dtype)}


def reference(values, p):
    seqs, kv, dim, block, length = (p[k] for k in ("sequences", "kv_heads", "head_size", "block_size", "context"))
    blocks = values["tables"][:, :math.ceil(length / block)].long()
    # Gather only live pages before widening. Unused captured cache capacity
    # is preserved in the task interface but does not bloat the FP32 oracle.
    key = values["key"][blocks].permute(0, 1, 4, 2, 3, 5).reshape(seqs, -1, kv, dim)[:, :length].float()
    value = values["value"][blocks].permute(0, 1, 4, 2, 3).reshape(seqs, -1, kv, dim)[:, :length].float()
    query = values["query"][:seqs].float().reshape(seqs, kv, p["heads"] // kv, dim)
    scores = torch.einsum("shgd,slhd->shgl", query, key) / math.sqrt(dim)
    mask = torch.arange(length, device=query.device)[None, :] < values["lengths"][:, None]
    scores.masked_fill_(~mask[:, None, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    active = torch.einsum("shgl,slhd->shgd", probabilities, value).reshape(seqs, p["heads"], dim)
    output = torch.zeros_like(values["out"])
    output[:seqs].copy_(active.to(torch.bfloat16))
    return output


def check_output_contract(got, expected):
    if (not isinstance(got, torch.Tensor) or got.shape != expected.shape
            or got.dtype != expected.dtype or got.device != expected.device):
        raise AssertionError("Attention output shape, dtype or device mismatch")
    if not torch.isfinite(got).all() or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite attention output or reference")


def compare(got, expected, p):
    check_output_contract(got, expected)
    if not torch.allclose(got.float(), expected.float(), atol=5e-2, rtol=5e-2):
        error = (got.float() - expected.float()).abs().max().item()
        raise AssertionError(f"Paged attention differs from FP32 oracle: max_abs={error}")
    if not torch.equal(got[p["sequences"]:], expected[p["sequences"]:]):
        raise AssertionError("Kernel modified inactive query rows")


def extra_negative_checks(expected, p):
    if p["query_rows"] > p["sequences"]:
        wrong = expected.clone()
        wrong[p["sequences"], 0, 0] = 1
        try:
            compare(wrong, expected, p)
        except AssertionError:
            pass
        else:
            raise AssertionError("Comparator ignored inactive output rows")


def poison_output(output, p):
    output[:p["sequences"]].fill_(float("nan"))


def timing_options():
    from build_kernel import timing_options as options
    return options()


def load_candidate():
    from build_kernel import load_extension
    op = load_extension("extracted_paged_attention").paged_attention

    def invoke(v, p):
        op(v["out"], v["exp_sums"], v["max_logits"], v["tmp_out"],
           v["query"], v["key"], v["value"], p["kv_heads"], 1 / math.sqrt(p["head_size"]),
           v["tables"], v["lengths"], v["query_start"], p["block_size"], p["context"],
           None, "auto", v["k_scale"], v["v_scale"], None, "f16")
        return v["out"]

    return invoke
