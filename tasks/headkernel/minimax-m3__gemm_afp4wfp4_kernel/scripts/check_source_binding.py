"""Diagnostic eager/graph probe for one admitted case and submitted source."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
sys.path.insert(0, str(ROOT / "scripts"))
from runtime import FP4Case
from source_guard import validate_sources
from task_runner import load_dataset


def probe(dataset, candidate_workspace, case_id, mode, seed):
    result = {"schema": "minimax-fp4-source-probe-v1", "scoreable": False, "performance_samples": 0,
              "case_id": case_id, "mode": mode, "seed": seed, "reference_calibrated": False,
              "graph_captured": False, "graph_replayed": False, "status": "setup_failure"}
    phase = "setup"
    try:
        manifest = load_dataset(dataset)
        validate_sources(candidate_workspace, ROOT)
        case, = [case for case in manifest["cases"] if case["case_id"] == case_id]
        result["case"] = case
        phase = "reference"
        reference = FP4Case(ROOT, dataset, case, manifest["oracle_policy"], leg="reference")
        if mode == "graph":
            reference.capture_graph()
        reference.check_once(seed)
        result["reference_calibrated"] = True
        del reference
        phase = "candidate_setup"
        candidate = FP4Case(ROOT, dataset, case, manifest["oracle_policy"], candidate_workspace=candidate_workspace,
                            defer_candidate_check=True)
        result["source_proof"] = candidate.proof
        if mode == "graph":
            candidate.capture_graph()
            result["graph_captured"] = True
        phase = mode
        candidate.check_once(seed)
        result.update(status="candidate_accepted", graph_replayed=candidate.proof.get("graph_replayed", False))
        return 0, result
    except BaseException as error:
        result.update(failure_phase=phase, error_type=type(error).__name__, error=str(error))
        if phase == "reference":
            result["status"] = "invalid_reference"
        elif phase in ("eager", "graph") and isinstance(error, AssertionError) and any(
                text in str(error) for text in ("independent packed-value oracle mismatch", "unwritten/nonfinite output or reference")):
            result.update(status="candidate_rejected", graph_replayed=candidate.proof.get("graph_replayed", False))
            return 1, result
        return 2, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-workspace", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    code, result = probe(args.dataset, args.candidate_workspace, args.case_id, args.mode, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
