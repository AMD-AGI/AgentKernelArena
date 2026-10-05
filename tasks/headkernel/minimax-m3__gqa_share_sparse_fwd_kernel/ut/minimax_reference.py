"""Independent dense gather/dot/softmax reference for the sparse attention ABI."""
import math


def unit(value):
    return 1.0 if value is None else float(value)


def gather_rows(tensor, ids):
    """Bit-preserving gather also works for FP8 CPU snapshots."""
    import torch
    raw = tensor.view(torch.uint8).index_select(0, ids.to(device=tensor.device, dtype=torch.int64))
    return raw.view(tensor.dtype)


def _index_rows(args):
    import torch
    maximum = args["k_cache"].shape[0]
    rows = (args["slot_ids"].to(torch.int64) + maximum) % maximum
    lengths = args["seq_lens"].to(torch.int64)
    if bool(((lengths > 0) & ((rows < 0) | (rows >= args["req_to_token"].shape[0]))).any()):
        raise ValueError("active request selects an out-of-bounds paging row")
    if bool(((lengths < 0) | (lengths > args["req_to_token"].shape[1])).any()):
        raise ValueError("sequence lengths escape physical paging capacity")
    return rows, lengths


def _selected_positions(indices, block_size, length, *, device):
    import torch
    indices = indices.to(device=device, dtype=torch.int64)
    valid = indices >= 0
    # Native kernels count nonnegative entries and then consume that prefix.
    if bool((valid[..., 1:] & ~valid[..., :-1]).any()):
        raise ValueError("top-k indices must be right padded with -1")
    positions = indices[..., None] * block_size + torch.arange(block_size, device=device)
    mask = valid[..., None] & (positions < length)
    if bool((valid & (indices * block_size >= max(length, 1))).any()) and length > 0:
        raise ValueError("selected sparse block starts outside the active sequence")
    return positions.flatten(-2), mask.flatten(-2)


def sparse_attention(args, kind, *, device="cpu", query_blocks_per_chunk=256):
    """Use dense FP32 math on the selected logical tokens; no SGLang/Triton calls.

    Inputs are views of immutable storage snapshots. The caller may copy those
    snapshots to the reference device after candidate outputs have been saved.
    Chunked gathers bound intermediate memory and preserve physical row IDs.
    """
    import torch
    q = args["q"]
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("FP8-Q probability quantization requires a separate reference")
    pool_k, pool_v = args["k_cache"], args["v_cache"]
    heads, dim, kv_heads = q.shape[1], q.shape[2], pool_k.shape[1]
    out_dim = dim if kind == "sparse_decode" else pool_v.shape[2]
    if heads % kv_heads or pool_v.shape[1] != kv_heads or pool_k.shape[2] < dim or pool_v.shape[2] < out_dim:
        raise ValueError("GQA head geometry differs")
    group = heads // kv_heads
    scale = unit(args.get("sm_scale")) if args.get("sm_scale") is not None else dim ** -0.5
    scale *= unit(args.get("q_scale"))
    k_scale, v_scale = unit(args.get("k_scale")), unit(args.get("v_scale"))
    rows, lengths = _index_rows(args)
    if kind == "sparse_decode":
        starts = list(range(q.shape[0])); sizes = [1] * q.shape[0]
        block_q = 1; block_k = args["block_size"]
        block_starts = starts; prefixes = [0] * len(starts)
    else:
        cu = args["cu_seqlens"].to(torch.int64).tolist()
        if cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])) or cu[-1] > q.shape[0]:
            raise ValueError("ragged query offsets escape the physical query buffer")
        starts, sizes = cu[:-1], [b-a for a,b in zip(cu,cu[1:])]
        block_q, block_k = args["block_size_q"], args["block_size_k"]
        prefixes = args["prefix_lens"].to(torch.int64).tolist()
        if args.get("cu_seqblocks_q") is None:
            block_starts = [0]
            for size in sizes[:-1]:
                block_starts.append(block_starts[-1] + (size+block_q-1)//block_q)
        else:
            block_starts = args["cu_seqblocks_q"].to(torch.int64).tolist()[:-1]
    expected = torch.full((q.shape[0], heads, out_dim), float("nan"), dtype=torch.float32, device=device)
    for batch, (start, count) in enumerate(zip(starts, sizes)):
        length = int(lengths[batch]); row = int(rows[batch])
        if count == 0:
            continue
        num_blocks = (count + block_q - 1) // block_q
        for kv_head in range(kv_heads):
            lo, hi = kv_head*group, (kv_head+1)*group
            for begin in range(0, num_blocks, query_blocks_per_chunk):
                number = min(query_blocks_per_chunk, num_blocks-begin)
                topk = args["topk_idx"][kv_head, block_starts[batch]+begin:block_starts[batch]+begin+number]
                positions, valid = _selected_positions(topk, block_k, length, device=device)
                safe = positions.clamp(0, args["req_to_token"].shape[1]-1).to(args["req_to_token"].device)
                if length:
                    physical = args["req_to_token"][row].to(torch.int64)[safe]
                    physical = (physical + pool_k.shape[0]) % pool_k.shape[0]
                else:
                    physical = torch.zeros_like(safe)
                flat_ids = physical.reshape(-1)
                # Widen FP8 caches through the query representation, matching
                # the public dequantization contract rather than kernel code.
                k = gather_rows(pool_k[:, kv_head, :dim], flat_ids).to(q.dtype).float()
                v = gather_rows(pool_v[:, kv_head, :out_dim], flat_ids).to(q.dtype).float()
                k = k.reshape(number, -1, dim).to(device)
                v = v.reshape(number, -1, out_dim).to(device)
                token_ids = start + (torch.arange(begin, begin+number)[:, None] * block_q + torch.arange(block_q)[None, :])
                queries = q[token_ids.clamp_max(q.shape[0]-1).to(q.device), lo:hi].float().to(device)
                logits = torch.bmm(queries.reshape(number, block_q*group, dim), k.transpose(1, 2)) * (scale*k_scale)
                mask = valid[:, None, :].expand(number, block_q, positions.shape[-1])
                if kind == "sparse_prefill":
                    query_positions = prefixes[batch] + torch.arange(begin*block_q, (begin+number)*block_q, device=device).reshape(number, block_q)
                    mask = mask & (positions[:, None, :] <= query_positions[:, :, None])
                mask = mask[:, :, None, :].expand(number, block_q, group, positions.shape[-1]).reshape_as(logits)
                logits = logits.masked_fill(~mask, -float("inf"))
                sink = args.get("sink")
                if sink is not None:
                    sink_logits = (queries * sink[lo:hi].float().to(device)[None, None]).sum(-1) * scale
                    logits = torch.cat((logits, sink_logits.reshape(number, block_q*group, 1)), dim=-1)
                    v = torch.cat((v, torch.zeros((number, 1, v.shape[-1]), device=device)), dim=1)
                result = torch.bmm(torch.softmax(logits, dim=-1), v) * v_scale
                result = result.reshape(number*block_q, group, out_dim)
                good = min(count-begin*block_q, number*block_q)
                offset = start + begin*block_q
                expected[offset:offset+good, lo:hi] = result[:good]
    return expected.cpu()


def score_reference(args, *, device="cpu"):
    """Full independent block-score matrix in base-2 units used by the public op."""
    import torch
    rows, lengths = _index_rows(args)
    q, cache = args["q"], args["k_cache"]
    block, topk = args["block_size"], args["topk"]
    counts = [(int(n)+block-1)//block for n in lengths]
    scores = torch.full((q.shape[1], q.shape[0], max(counts, default=0)), -float("inf"), device=device)
    scale = (q.shape[-1] ** -0.5 if args.get("sm_scale") is None else float(args["sm_scale"]))
    scale *= unit(args.get("q_scale")) * unit(args.get("k_scale")) * math.log2(math.e)
    group = q.shape[1] // cache.shape[1]
    for batch, count in enumerate(counts):
        length = int(lengths[batch])
        if length == 0:
            continue
        physical = args["req_to_token"][int(rows[batch]), :length].to(torch.int64)
        physical = (physical + cache.shape[0]) % cache.shape[0]
        keys = gather_rows(cache[..., :q.shape[-1]], physical).to(q.dtype).float()
        for kv_head in range(cache.shape[1]):
            h0, h1 = kv_head*group, (kv_head+1)*group
            logits = q[batch, h0:h1].float().to(device) @ keys[:, kv_head].to(device).T * scale
            padded = torch.full((group, count*block), -float("inf"), device=device)
            padded[:, :length] = logits
            chunks = padded.reshape(group, count, block)
            if args["score_type"] == "max":
                value = chunks.amax(-1)
            elif args["score_type"] == "lse":
                value = torch.logsumexp(chunks * math.log(2), dim=-1) / math.log(2)
            else:
                raise ValueError("unrepresented score reduction")
            value[:, :args["init_blocks"]] = 1e30
            if args["local_blocks"]:
                value[:, max(0, count-args["local_blocks"]):] = 1e29
            scores[h0:h1, batch, :count] = value
    return scores.cpu(), counts, topk


def check_topk(indices, scores, counts, topk, *, tolerance=0.02):
    """Validate integer structure and the mathematical top-k cutoff, including ties."""
    import torch
    if tuple(indices.shape) != (scores.shape[0], len(counts), topk) or indices.dtype not in (torch.int32, torch.int64):
        raise AssertionError("top-k output ABI differs")
    for head in range(scores.shape[0]):
        for batch, count in enumerate(counts):
            real = min(count, topk)
            selected = indices[head, batch, :real].to(torch.int64)
            if (real and (bool((selected < 0).any()) or bool((selected >= count).any())
                          or bool((selected[1:] <= selected[:-1]).any()))) or bool((indices[head,batch,real:] != -1).any()):
                raise AssertionError("invalid, duplicate, unsorted or incorrectly padded block indices")
            if not real:
                continue
            values = scores[head, batch, :count]
            cutoff = torch.topk(values, real).values[-1]
            margin = tolerance * (1 + cutoff.abs())
            if bool((values[selected] < cutoff-margin).any()):
                raise AssertionError("selected block falls below independent top-k cutoff")
            mandatory = torch.where(values > cutoff+margin)[0]
            if not set(mandatory.tolist()).issubset(selected.tolist()):
                raise AssertionError("mandatory high-scoring block was omitted")


def mixed_close(actual, expected, tolerance=0.02):
    import torch
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError("output shape differs")
    a, e = actual.float(), expected.float()
    finite = torch.isfinite(e)
    if not torch.equal(torch.isnan(a), torch.isnan(e)) or not torch.equal(torch.isinf(a), torch.isinf(e)):
        raise AssertionError("unexpected nonfinite output")
    if not torch.equal(a[torch.isinf(e)], e[torch.isinf(e)]):
        raise AssertionError("infinite output sign differs")
    if not bool(finite.any()):
        raise AssertionError("case has no defined finite output to validate")
    floor = tolerance * e[finite].square().mean().sqrt().clamp_min(1e-6)
    if not bool(((a[finite]-e[finite]).abs() <= floor + tolerance*e[finite].abs()).all()):
        raise AssertionError("independent mixed-tolerance oracle failed")
