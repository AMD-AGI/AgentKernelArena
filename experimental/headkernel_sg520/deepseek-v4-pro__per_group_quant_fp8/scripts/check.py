"""Qualify native baseline and native candidate against one strict oracle."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from oracle import compare, generate, invoke, load_cases, reference
from native import baseline, candidate, configure_workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "cases.json")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    manifest, cases = load_cases(args.cases, args.allow_provisional)
    configure_workspace()
    import torch
    from aiter.jit.utils.chip_info import get_gfx
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        raise RuntimeError("this ABI is bound to the observed SG520 gfx950 runtime")
    production, editable = baseline(), candidate()
    rows = []
    for case in cases:
        for seed in args.seed or [0, 1]:
            x = generate(case, seed, "cuda")
            before = x.clone()
            expected = reference(x)
            # Production correctness is mandatory. No reference-gap waiver and
            # no substituted Triton/PyTorch implementation is a baseline.
            for label, function in (("production_native", production), ("candidate_native", editable)):
                actual = invoke(function, x)
                torch.cuda.synchronize()
                if not torch.equal(x, before):
                    raise AssertionError(label + " modified the input")
                compare(actual, expected, x)
                saved = actual[1][0, 0].clone()
                actual[1][0, 0] = -1.0
                try:
                    compare(actual, expected, x)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("negative control was accepted")
                finally:
                    actual[1][0, 0] = saved
                rows.append({"case_id": case["case_id"], "seed": seed, "leg": label, "correct": True})
    print(json.dumps({"source_run": manifest["source_run"],
                      "confirmed_corrected_1024": manifest["confirmed_corrected_1024"],
                      "complete_case_counts": manifest["complete_case_counts"],
                      "count_scope": manifest.get("count_scope"),
                      "input_kind": "seeded legal values at observed trace dimensions",
                      "timings": None, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
