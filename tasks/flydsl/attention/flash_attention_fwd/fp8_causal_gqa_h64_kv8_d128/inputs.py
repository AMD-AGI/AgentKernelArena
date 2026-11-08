"""Per-tensor FP8 (e4m3fn) Q, K and V with their descales, quantized as AITER's per_tensor_quant does."""
import torch

FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
ROW_CHUNK = 1 << 20


def _quantize(master):
    """descale = amax / 448, code = fp8(x / descale); dequantized value = code * descale."""
    descale = (master.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(1)
    codes = torch.empty(master.shape, dtype=torch.float8_e4m3fn, device=master.device)
    flat_master = master.reshape(-1, master.shape[-1])
    flat_codes = codes.view(-1, master.shape[-1])
    for start in range(0, flat_master.shape[0], ROW_CHUNK):
        stop = min(start + ROW_CHUNK, flat_master.shape[0])
        flat_codes[start:stop] = (flat_master[start:stop].float() / descale).clamp(-FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn)
    return codes, descale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, seq_len = axes["batch"], axes["seq_len"]
    head_dim = axes["head_dim"]
    produced = {}
    for name, heads in (("q", axes["n_heads"]), ("k", axes["n_kv_heads"]), ("v", axes["n_kv_heads"])):
        master = torch.randn((batch, seq_len, heads, head_dim), dtype=torch.bfloat16, generator=generator,
                             device=device)
        produced[name], produced[name + "_descale"] = _quantize(master)
        del master
    return produced
