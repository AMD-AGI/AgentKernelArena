#!/usr/bin/env python3
"""Case construction for the five live dense-BF16 GEMM ledger entries."""

import importlib


def case_map(meta):
    return {case["ledger_id"]: case for case in meta["workload"]["cases"]}


def selected_cases(meta, case_ids):
    by_id = case_map(meta)
    return [by_id[case_id] for case_id in case_ids]


def _torch():
    return importlib.import_module("torch")


def _randn(shape, generator, device):
    torch = _torch()
    value = torch.randn(
        *shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return value.mul_(0.125)


def make_args(case, *, seed=None, rng=None, m=None, weight=None):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if rng is None:
        rng = torch.Generator(device=device).manual_seed(int(seed or 0))
    rows = int(case["m"] if m is None else m)
    a = _randn((rows, int(case["k"])), rng, device)
    b = weight
    if b is None:
        b = _randn((int(case["n"]), int(case["k"])), rng, device)
    return {"A": a, "B": b}


def baseline_call(args):
    torch = _torch()
    return torch.nn.functional.linear(args["A"], args["B"], bias=None)


def candidate_call(args):
    tuned_gemm = importlib.import_module("aiter.tuned_gemm")
    return tuned_gemm.gemm_a16w16(
        args["A"], args["B"], bias=None, otype=_torch().bfloat16
    )


def eager_cases(case):
    rows = []
    for draw in range(2):
        args = make_args(case, seed=1000 + draw)
        rows.append(
            {
                "args": args,
                "ref": baseline_call(args).detach().clone(),
                "sig": f"{case['sig']}:fixed[{draw}]",
                "regime": case["regime"],
            }
        )
    return rows


def random_shapes(case):
    return [
        {
            "sig": case["sig"],
            "make_inputs": lambda rng, case=case: make_args(case, rng=rng),
        }
    ]


def baseline_random_outputs(case, draws, seed=0):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = {}
    for draw in range(max(1, int(draws))):
        rng = torch.Generator(device=device).manual_seed(int(seed) + draw)
        args = make_args(case, rng=rng)
        outputs[f"{case['sig']}|{draw}"] = baseline_call(args).detach().cpu()
        del args
    return outputs


def graph_replay_bundle(case):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_full = int(case["m"])
    k = int(case["k"])
    n = int(case["n"])

    weight_rng = torch.Generator(device=device).manual_seed(2000)
    weight = _randn((n, k), weight_rng, device)
    a_static = torch.zeros((m_full, k), dtype=torch.bfloat16, device=device)
    b_static = weight.detach().clone()
    holder = {}

    def run():
        holder["out"] = candidate_call({"A": a_static, "B": b_static})

    def read_out():
        return holder["out"]

    def fill(replay_case):
        a_static.zero_()
        source = replay_case["args"]["A"]
        a_static[: source.shape[0]].copy_(source)
        b_static.copy_(replay_case["args"]["B"])

    replay_cases = []
    for index in range(2):
        args = make_args(case, seed=2100 + index, m=m_full, weight=weight)
        ref = baseline_call(args).detach().clone()
        replay_cases.append(
            {
                "args": args,
                "ref": ref,
                "sig": f"{case['sig']}:graph_draw[{index}]",
            }
        )

    return {
        "fill": fill,
        "run": run,
        "read_out": read_out,
        "cases": replay_cases,
        "capture_idx": 0,
        "owned": {
            "a_static": a_static,
            "b_static": b_static,
            "holder": holder,
        },
    }


def timing_case(case):
    args = make_args(case, seed=3000)
    return {
        "args": args,
        "sig": case["sig"],
        "regime": case["regime"],
        "m": int(case["m"]),
    }
