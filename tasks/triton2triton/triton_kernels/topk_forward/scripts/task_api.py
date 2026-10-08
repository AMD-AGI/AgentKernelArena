"""Independent fused routing reference and protected Triton launch interface."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_params(p):
    if (set(p) != {"rows", "experts", "topk", "distribution"}
            or p["rows"] <= 0 or p["experts"] not in (64, 128, 256)
            or p["topk"] not in (1, 2, 4, 8)
            or p["distribution"] not in ("normal", "negative", "ties")):
        raise ValueError(f"Unsupported fused TopK parameters: {p}")


def logits(p, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    columns = p["experts"] // 2 if p["distribution"] == "ties" else p["experts"]
    values = torch.randn((p["rows"], columns), generator=gen, device=device,
                         dtype=torch.float32)
    if p["distribution"] == "ties":
        values = values.repeat_interleave(2, dim=1)
    if p["distribution"] == "negative":
        values = -values.abs()
    return values.to(torch.bfloat16)


def make_inputs(p, seed=42, device="cuda"):
    return {"logits": logits(p, seed, device),
            "values": torch.empty((p["rows"], p["topk"]), device=device, dtype=torch.bfloat16),
            "indices": torch.empty((p["rows"], p["topk"]), device=device, dtype=torch.int16),
            "bits": torch.empty((p["experts"] // 32, p["rows"]), device=device, dtype=torch.uint32)}


def readonly(values):
    return {"logits": values["logits"]}


def draw(values, p, seed):
    return {"logits": logits(p, seed, values["logits"].device)}


def reference(values, p):
    x = values["logits"].float()
    # Stable descending order implements the kernel's smaller-index tie rule.
    indices = torch.argsort(x, dim=1, descending=True, stable=True)[:, :p["topk"]]
    selected = x.gather(1, indices)
    probabilities = torch.softmax(selected, dim=1).to(torch.bfloat16)
    words = []
    for word in range(p["experts"] // 32):
        masks = torch.where(indices // 32 == word, 1 << (indices % 32), 0)
        words.append(masks.sum(dim=1))
    bits = torch.stack(words).to(torch.uint32)
    return probabilities, indices.to(torch.int16), bits


def check_output_contract(got, expected):
    if not isinstance(got, tuple) or len(got) != 3:
        raise AssertionError("Expected the (probabilities, indices, bitmatrix) tuple")
    for actual, want in zip(got, expected):
        if (not isinstance(actual, torch.Tensor) or actual.shape != want.shape
                or actual.dtype != want.dtype or actual.device != want.device):
            raise AssertionError("Output shape, dtype or device mismatch")
    if not torch.isfinite(got[0]).all():
        raise AssertionError("Non-finite routing probabilities")


def compare(got, expected, p):
    check_output_contract(got, expected)
    if not torch.equal(got[1], expected[1]):
        raise AssertionError("TopK indices or tie ordering differ from reference")
    if not torch.equal(got[2], expected[2]):
        raise AssertionError("Routing bitmatrix differs from selected expert indices")
    a, b = got[0].float().flatten(), expected[0].float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
    relative = ((a - b).abs() / b.abs().clamp_min(1e-6)).max()
    if not (torch.isfinite(cosine) and torch.isfinite(relative)
            and cosine >= 0.99 and relative <= 1e-2):
        raise AssertionError(f"Routing probability mismatch: cosine={cosine}, max_rel={relative}")


def extra_negative_checks(expected, p):
    for index in (1, 2):
        wrong = tuple(value.clone() for value in expected)
        # uint32 arithmetic is not implemented on every supported runtime.
        value = wrong[index].flatten()[0].item()
        wrong[index].flatten()[0] = int(value) ^ 1
        try:
            compare(wrong, expected, p)
        except AssertionError:
            pass
        else:
            raise AssertionError("Comparator ignored an integer routing output")


def poison_output(output, p):
    output[0].fill_(float("nan"))
    output[1].fill_(-1)
    output[2].fill_(0xFFFFFFFF)


def load_candidate():
    source = ROOT / "source/kernel.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        if any(not (name == "triton" or name.startswith("triton.")) for name in modules):
            raise ValueError("Candidate imports must be limited to Triton")
    spec = importlib.util.spec_from_file_location("routing_candidate", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module._topk_forward

    def invoke(values, p):
        rows, experts, k = p["rows"], p["experts"], p["topk"]
        kernel[((rows + 31) // 32,)](
            values["logits"], experts, (values["values"],), (values["indices"],), k,
            False, (values["bits"],), 1, rows, rows, experts, 0,
            APPLY_SOFTMAX=True, BLOCK_M=32, N_EXPTS_PAD=experts,
            N_EXPTS_ACT=k, BLOCK_N=32,
        )
        return values["values"], values["indices"], values["bits"]

    return invoke
