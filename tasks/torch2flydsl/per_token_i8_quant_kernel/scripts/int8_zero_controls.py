"""Protected correctness-only check of the documented zero-row scale rule."""
import json
from pathlib import Path


CASE = {
    "test_case_id": "control_all_zero_rows",
    "params": {"m": 3, "n": 4096, "input": "all_zero", "dtype": "bfloat16"},
    "checks": ["correctness"],
}


def run(harness, *, device="cuda"):
    import torch

    manifest = json.loads((Path(__file__).resolve().parents[1] / "cases.json").read_text())
    rows = [row for row in manifest["cases"] if row["test_case_id"] == CASE["test_case_id"]]
    if rows != [CASE]:
        raise ValueError("Zero-row manifest control differs from its protected inputs")
    x = torch.zeros((3, 4096), dtype=torch.bfloat16, device=device)
    original = x.clone()
    # Closed-form expected values do not call the model, candidate or AITER.
    expected = (torch.zeros(x.shape, dtype=torch.int8, device=x.device),
                torch.ones((3, 1), dtype=torch.float32, device=x.device))
    model_module = harness._load_module(harness._KERNEL_DIR, harness.MODEL_FILE,
                                        "arena_zero_model")
    if model_module is None:
        raise RuntimeError("Missing provided model for zero-row control")
    try:
        model = model_module.Model(*model_module.get_init_inputs()).to(device)
        with torch.no_grad():
            actual = model(x)
        harness.require_unchanged((x,), (original,))
        harness._compare_quant_outputs(actual, expected, (x,), model_module)
        if not harness.ARENA_PROVIDED_BASELINE:
            candidate = harness._load_module(harness._KERNEL_DIR, harness.KERNEL_FILE,
                                              "arena_zero_candidate")
            entry = getattr(candidate, harness.KERNEL_ENTRY, None)
            if not callable(entry):
                raise RuntimeError("Missing candidate for zero-row control; no baseline fallback")
            actual = entry(x)
            if x.device.type == "cuda":
                torch.cuda.synchronize()
            harness.require_unchanged((x,), (original,))
            harness._compare_quant_outputs(actual, expected, (x,), model_module)
        print("INT8 all-zero-row control: PASS (zero codes and scale one)", flush=True)
    finally:
        x.copy_(original)
    return {"test_case_id": CASE["test_case_id"], "status": "PASS",
            "reference": "independent_zero_codes_scale_ones"}
