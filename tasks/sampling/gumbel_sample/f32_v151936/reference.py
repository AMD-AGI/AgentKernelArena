"""Gumbel-max sampling reference with an independent integer Philox-4x32-10 implementation.

The noise stream is Triton's: the per-row stream key is `randint(seed[state], pos[row])` (the
64-bit seed split into two key words, the position into two counter words), and the uniform draw
of vocabulary index v is the first Philox word for counter v keyed by that stream key, folded to
a non-negative int32 and scaled by 4.6566127342e-10 in float32.
"""
import torch

_M32 = 0xFFFFFFFF
_ROUND_A, _ROUND_B = 0xD2511F53, 0xCD9E8D57
_KEY_A, _KEY_B = 0x9E3779B9, 0xBB67AE85
_RAND_MIN = 4.6566127342e-10
_CHUNK_ELEMENTS = 1 << 21


def _mulhilo(constant, word):
    """High and low 32 bits of constant * word for 32-bit unsigned values held in int64."""
    low16 = constant * (word & 0xFFFF)
    high16 = constant * (word >> 16)
    carry = high16 + (low16 >> 16)
    return carry >> 16, ((carry & 0xFFFF) << 16) | (low16 & 0xFFFF)


def _philox_first_word(c0, c1, key_lo, key_hi):
    """First output word of Philox-4x32-10 for counter (c0, c1, 0, 0); all int64 holding uint32."""
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    for _ in range(10):
        hi_b, lo_b = _mulhilo(_ROUND_B, c2)
        hi_a, lo_a = _mulhilo(_ROUND_A, c0)
        c0, c1, c2, c3 = hi_b ^ c1 ^ key_lo, lo_b, hi_a ^ c3 ^ key_hi, lo_a
        key_lo = (key_lo + _KEY_A) & _M32
        key_hi = (key_hi + _KEY_B) & _M32
    return c0


def _noise(seed, pos, offsets):
    """float64 Gumbel noise [rows, len(offsets)] for per-row int64 seeds and positions."""
    stream = _philox_first_word(pos & _M32, (pos >> 32) & _M32, seed & _M32, (seed >> 32) & _M32)
    bits = _philox_first_word(offsets[None, :].expand(seed.shape[0], -1), torch.zeros_like(offsets)[None, :],
                              stream[:, None], torch.zeros_like(stream)[:, None])
    signed = torch.where(bits >= (1 << 31), bits - (1 << 32), bits)
    folded = torch.where(signed < 0, -signed - 1, signed)
    uniform = (folded.float() * torch.tensor(_RAND_MIN, dtype=torch.float32, device=bits.device)).double()
    uniform = torch.clamp(uniform, min=float(torch.tensor(_RAND_MIN, dtype=torch.float32)))
    return -torch.log(-torch.log1p(-uniform))


def reference(*, logits, expanded_idx_mapping, temperature, seed, pos):
    n_tokens, vocab_size = logits.shape
    device = logits.device
    state = expanded_idx_mapping.long()
    row_temperature = temperature[state].double()
    row_seed = seed[state].long()
    row_pos = pos.long()
    offsets = torch.arange(vocab_size, dtype=torch.int64, device=device)
    out = torch.empty(n_tokens, dtype=torch.int64, device=device)
    rows = max(1, _CHUNK_ELEMENTS // vocab_size)
    for start in range(0, n_tokens, rows):
        end = min(start + rows, n_tokens)
        scores = logits[start:end].double()
        noisy = scores + _noise(row_seed[start:end], row_pos[start:end], offsets)
        scores = torch.where((row_temperature[start:end] != 0)[:, None], noisy, scores)
        out[start:end] = scores.argmax(dim=-1)
    return out
