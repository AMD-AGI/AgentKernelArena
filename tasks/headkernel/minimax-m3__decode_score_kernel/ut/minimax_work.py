"""Finite MiniMax tensor-work projection; no GPU imports or workload modification."""
import evaluation_contract as cap
import json

PROJECTION_ID = "minimax-block-and-causal-work-v3"
EXTREMA_ID = "minimax-max-total-seqlen-topk-v3"

def _runs(values):
    result = []
    for value in values:
        if result and result[-1][0] == value:
            result[-1][1] += 1
        else:
            result.append([value, 1])
    return result


def _flat(control):
    return [value for value, count in control["runs"] for _ in range(count)]


def structural_work(controls, scalars):
    """Retain loop/causal-tile work variants, separating exact tensor values."""
    block = scalars.get("block_size", scalars.get("block_size_k"))
    cap.require(type(block) is int and block > 0, "missing sparse block width")
    result = json.loads(cap.canonical(controls))
    for name, control in result.items():
        if control is None:
            continue
        if name.endswith(".seq_lens"):
            control["runs"] = _runs([(value + block - 1) // block for value in _flat(control)])
            control["units"] = "sequence_blocks_ceil"
        elif name.endswith(".prefix_lens"):
            control["runs"] = _runs([value // block for value in _flat(control)])
            control["units"] = "causal_prefix_blocks_floor"
    return result


def work_metric(controls):
    lengths = _flat(controls.get("inputs.seq_lens", {"runs": []}))
    counts = _flat(controls.get("inputs.capture_topk_counts", {"runs": []}))
    return (max(lengths, default=0), sum(lengths), sum(counts))



def project_work(scalars, observed, served):
    cap.require("operator_scalars" in scalars, "Missing native scalar controls")
    cap.require("inputs.seq_lens" in observed and observed["inputs.seq_lens"] is not None,
                "MiniMax projection requires actual sequence lengths")
    return structural_work(observed, scalars["operator_scalars"])


def extreme_work(scalars, observed, served):
    return work_metric(observed)
