# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inputs for topk_topp_mask: logits rows, heterogeneous per-row k and p, and the pivot-estimator tables.

The two 200-entry tables are the constants of the vLLM Triton kernel's Gaussian pivot estimator
(vllm/v1/sample/ops/topk_topp_triton.py at the baseline's revision, Apache-2.0). They are inputs of
the operator interface because a kernel launch cannot create device constants; the reference ignores them.

Cut points are made unambiguous: the logit gap after the k-th largest value, and after the last
element top-p keeps, is widened, and p is placed in the middle of the probability interval of that
last element. A float32 kernel and the float64 reference then select the same elements however they
accumulate the probability mass.
"""
import math

import torch

# fmt: off
_NORMAL_CDF_TO_SIGMA_TABLE = (
    3.656, 3.650, 3.650, 3.650, 3.626, 3.626, 3.626, 3.514, 3.514, 3.503,
    3.503, 3.434, 3.434, 3.428, 3.428, 3.387, 3.380, 3.380, 3.376, 3.373,
    3.373, 3.356, 3.354, 3.354, 3.291, 3.249, 3.234, 3.214, 3.198, 3.198,
    3.185, 3.177, 3.177, 3.165, 3.164, 3.161, 3.138, 3.120, 3.115, 3.113,
    3.093, 3.066, 3.054, 3.043, 3.037, 3.023, 2.993, 2.991, 2.976, 2.970,
    2.952, 2.946, 2.932, 2.908, 2.902, 2.895, 2.886, 2.874, 2.861, 2.844,
    2.836, 2.810, 2.801, 2.790, 2.784, 2.779, 2.767, 2.757, 2.745, 2.733,
    2.723, 2.716, 2.693, 2.678, 2.671, 2.656, 2.649, 2.629, 2.611, 2.595,
    2.592, 2.585, 2.574, 2.550, 2.543, 2.534, 2.521, 2.518, 2.497, 2.485,
    2.468, 2.450, 2.441, 2.430, 2.412, 2.402, 2.389, 2.383, 2.377, 2.364,
    2.349, 2.338, 2.332, 2.319, 2.310, 2.301, 2.282, 2.274, 2.266, 2.250,
    2.242, 2.236, 2.226, 2.215, 2.207, 2.196, 2.179, 2.171, 2.162, 2.147,
    2.135, 2.121, 2.109, 2.095, 2.085, 2.073, 2.063, 2.045, 2.030, 2.016,
    2.003, 1.992, 1.983, 1.972, 1.960, 1.949, 1.940, 1.928, 1.912, 1.897,
    1.881, 1.869, 1.854, 1.838, 1.824, 1.807, 1.792, 1.779, 1.764, 1.751,
    1.739, 1.726, 1.711, 1.697, 1.685, 1.668, 1.652, 1.636, 1.622, 1.603,
    1.585, 1.568, 1.551, 1.534, 1.513, 1.499, 1.480, 1.464, 1.441, 1.422,
    1.394, 1.373, 1.347, 1.320, 1.296, 1.270, 1.246, 1.219, 1.190, 1.163,
    1.135, 1.104, 1.073, 1.041, 1.006, 0.969, 0.931, 0.894, 0.851, 0.806,
    0.757, 0.702, 0.643, 0.574, 0.498, 0.405, 0.288, 0.134, -0.110, -3.813,
)

_PERCENTILE_TO_STD_TABLE = (
    2.576, 2.319, 2.178, 2.064, 1.968, 1.892, 1.819, 1.757, 1.708, 1.659,
    1.616, 1.568, 1.526, 1.492, 1.456, 1.420, 1.382, 1.342, 1.309, 1.280,
    1.249, 1.221, 1.193, 1.169, 1.145, 1.121, 1.095, 1.073, 1.050, 1.030,
    1.008, 0.987, 0.966, 0.945, 0.926, 0.910, 0.891, 0.871, 0.854, 0.837,
    0.819, 0.803, 0.784, 0.767, 0.753, 0.734, 0.719, 0.702, 0.690, 0.675,
    0.658, 0.640, 0.625, 0.609, 0.595, 0.578, 0.564, 0.550, 0.537, 0.521,
    0.509, 0.495, 0.481, 0.466, 0.453, 0.439, 0.424, 0.410, 0.397, 0.383,
    0.370, 0.356, 0.343, 0.330, 0.316, 0.302, 0.289, 0.274, 0.261, 0.247,
    0.235, 0.223, 0.209, 0.196, 0.184, 0.172, 0.159, 0.149, 0.137, 0.124,
    0.112, 0.100, 0.086, 0.074, 0.062, 0.050, 0.035, 0.023, 0.009, -0.003,
    -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122,
    -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248,
    -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382,
    -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521,
    -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673,
    -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838,
    -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027,
    -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272,
    -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658,
    -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813,
)
# fmt: on

_K_CHOICES = (1, 8, 20, 50, 100, 400, 2000, 10000)
_P_CHOICES = (0.5, 0.8, 0.9, 0.95, 0.99)
_CHUNK = 16
_LOGIT_STD = 4.0
_GAP_K = 0.05      # minimum logit gap between the k-th and (k+1)-th largest values
_GAP_P = 1.0       # minimum logit gap after the last element kept by top-p
_P_MIN_PROB = 1.0e-4   # the last kept element has at least this probability among the candidates


def _row_parameters(n_tokens, vocab, generator, device):
    ids = torch.arange(n_tokens, device=device)
    k_pick = torch.randint(0, len(_K_CHOICES), (n_tokens,), generator=generator, device=device)
    p_pick = torch.randint(0, len(_P_CHOICES), (n_tokens,), generator=generator, device=device)
    k = torch.tensor(_K_CHOICES, dtype=torch.int64, device=device)[k_pick].clamp(max=vocab)
    p = torch.tensor(_P_CHOICES, dtype=torch.float64, device=device)[p_pick]
    kind = ids % 4                      # 0: top-k only, 1: top-p only, 2 and 3: both
    k = torch.where(kind == 1, torch.full_like(k, vocab), k)
    p = torch.where(kind == 0, torch.ones_like(p), p)
    disabled = ids % 16 == 7            # neither filter: the row is unchanged
    k = torch.where(disabled, torch.full_like(k, vocab), k)
    p = torch.where(disabled, torch.ones_like(p), p)
    return k, p


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    vocab = case["axes"]["vocab_size"]
    k, p_nominal = _row_parameters(n_tokens, vocab, generator, device)
    logits = torch.empty(n_tokens, vocab, dtype=torch.float32, device=device)
    p_out = torch.ones(n_tokens, dtype=torch.float64, device=device)
    for start in range(0, n_tokens, _CHUNK):
        end = min(n_tokens, start + _CHUNK)
        x = torch.randn(end - start, vocab, generator=generator, device=device, dtype=torch.float32)
        sv, order = (x * _LOGIT_STD).sort(dim=-1, descending=True)
        kk = k[start:end]
        rows = torch.arange(end - start, device=device)
        position = torch.arange(vocab, device=device)[None, :]
        # Top-k cut: widen the gap after the k-th largest value.
        has_k = kk < vocab
        after_k = (sv[rows, (kk - 1).clamp(max=vocab - 1)]
                   - sv[rows, kk.clamp(max=vocab - 1)]).double()
        shift = torch.where(has_k, (_GAP_K - after_k).clamp(min=0.0), torch.zeros_like(after_k))
        sv = sv - (shift.float()[:, None] * (position >= kk[:, None]))
        # Top-p cut among the k candidates.
        pn = p_nominal[start:end]
        has_p = pn < 1.0
        candidates = position < kk[:, None]
        probs = torch.softmax(torch.where(candidates, sv.double(), -math.inf), dim=-1)
        cum = probs.cumsum(-1)
        c_nominal = (cum < pn[:, None]).sum(-1)
        c_floor = ((probs >= _P_MIN_PROB).sum(-1) - 1).clamp(min=0)
        c = torch.minimum(torch.minimum(c_nominal, c_floor), kk - 1)
        last = c >= kk - 1
        after_p = (sv[rows, c] - sv[rows, (c + 1).clamp(max=vocab - 1)]).double()
        shift = torch.where(has_p & ~last, (_GAP_P - after_p).clamp(min=0.0), torch.zeros_like(after_p))
        sv = sv - (shift.float()[:, None] * (position > c[:, None]))
        probs = torch.softmax(torch.where(candidates, sv.double(), -math.inf), dim=-1)
        excl = probs.cumsum(-1) - probs
        p_mid = excl[rows, c] + probs[rows, c] / 2
        p_out[start:end] = torch.where(has_p, p_mid, torch.ones_like(p_mid))
        out = torch.empty_like(sv)
        out.scatter_(1, order, sv)
        logits[start:end] = out
    return {
        "logits": logits,
        "k": k.to(torch.int32),
        "p": p_out.to(torch.float32),
        "percentile_to_std_table": torch.tensor(_PERCENTILE_TO_STD_TABLE, dtype=torch.float32, device=device),
        "normal_cdf_to_sigma_table": torch.tensor(_NORMAL_CDF_TO_SIGMA_TABLE, dtype=torch.float32,
                                                  device=device),
    }
