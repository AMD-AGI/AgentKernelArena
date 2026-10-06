"""Native diagnostic only: real Quark capture, eager/graph and bad-source checks.

Uses explicit synthetic operands, never exports a workload case manifest, and
collects no timing samples. Run only in the pinned image on an allocated GPU.
"""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from admission import case_from_fixture
from evaluation_contract import require, strict_json
from fixture_codec import file_sha
from runtime import FP4Case


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def numerical_rejection(state, seed):
    try:
        state.check_once(seed)
    except AssertionError as error:
        require(any(text in str(error) for text in ("independent packed-value oracle mismatch", "unwritten/nonfinite output or reference")),
                "non-numerical failure cannot qualify as bad-source rejection")
        return {"error_type": type(error).__name__, "message": str(error)}
    raise AssertionError("bad submitted source was accepted")


def run(common, output):
    import torch
    require(torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName,
            "native FP4 smoke requires gfx950")
    pins = strict_json((ROOT / "SOURCE-PROVENANCE.json").read_text())
    adapter = load_module("fp4_capture_adapter", ROOT / "capture/adapter.py")
    cap = load_module("runtime_capture", Path(common) / "runtime_capture.py")
    require(file_sha(cap.__file__) == adapter.COMMON_SHA256, "shared capture version differs")
    modules = json.loads((ROOT / "capture/INTEGRATION.json").read_text())["modules"]
    quark, basic, kernel = [importlib.import_module(modules[key]) for key in ("quark", "wrapper", "kernel")]
    run_id = "fp4-native-smoke-" + uuid.uuid4().hex
    rec = cap.Recorder(output / "capture", {"run_id": run_id, "image": pins["runtime_image"], "tp_rank": 0,
        "synthetic": True, "scoreable": False, "purpose": "native adapter and source-binding diagnostic"},
        cap.Budget(max_snapshot_bytes=64 << 20, max_live_snapshot_bytes=256 << 20, max_artifact_bytes=256 << 20,
                   max_cases=8, max_runtime_case_keys=16), graph_buckets={"smoke"}, snapshot_slots=None, capture_ranks=(0,))
    active = None
    adapter.install(cap, quark, basic, kernel, lambda: active)
    # Deliberately simple diagnostic values make the native result exactly
    # representable. These dimensions are not claimed production shapes.
    m, n, k = 64, 64, 512
    x = torch.full((m, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    w = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda")
    xs = torch.full((m, k // 32), 127, dtype=torch.uint8, device="cuda")
    ws = torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda")
    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    cached_op = torch.ops.sglang.aiter_gemm_afp4wfp4
    cached_op(x, w, xs, ws, y)  # Native compilation before capture.
    served = lambda replay, stage: cap.Served(run_id, replay, stage, 0, 1, m, "synthetic_native_smoke")
    active = {"recorder": rec, "served": served("eager-1", "prefill")}
    cached_op(x, w, xs, ws, y)
    torch.cuda.synchronize()
    require(torch.equal(y.cpu(), torch.full((m, n), k, dtype=torch.bfloat16)), "native eager result is not exact")
    active = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            cached_op(x, w, xs, ws, y)
    torch.cuda.current_stream().wait_stream(stream)
    active = {"recorder": rec, "graph_id": "smoke-graph", "slot_id": "minimax_fp4_gemm:0", "bucket": "smoke"}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cached_op(x, w, xs, ws, y)
    active = None
    for index, packed in enumerate((0x22, 0xAA)):
        x.fill_(packed)
        y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = k if index == 0 else -k
        require(torch.equal(y.cpu(), torch.full((m, n), expected, dtype=torch.bfloat16)),
                "native graph failed to use changed packed inputs")
        rec.after_served_replay("smoke-graph", served("graph-" + str(index), "decode"))
    rec.seal(list(rec.case_counts))
    proof = cap.verify_rank_manifests([output / "capture/manifest.json"], run_id=run_id, required_ranks=(0,),
                                      required_cases_by_rank={0: list(rec.case_counts)}, max_artifact_bytes=256 << 20)
    control_module = load_module("fp4_source_controls", ROOT / "scripts/make_source_controls.py")
    for kind in ("no_op", "wrong_output"):
        source = output / "submitted" / kind / "source/kernel.py"
        source.parent.mkdir(parents=True)
        source.write_text(control_module.control_source(kind))
    records = []
    policy = {"metric": "mixed_rms", "tolerance": 0.02, "basis": "diagnostic only; native ones/sign outputs also require exact equality"}
    for key, ref in proof["fixture_representatives"].items():
        path = Path(ref["path"])
        fixture = strict_json(path.read_text())
        case = case_from_fixture(fixture, {"0": rec.case_counts[key]}, {"path": str(path.relative_to(output)), "sha256": ref["sha256"]})
        for leg in ("reference", "candidate"):
            state = FP4Case(ROOT, output, case, policy, leg=leg, diagnostic=True)
            state.check_once(17)
            state.capture_graph()
            state.check_once(19)
            records.append({"case_key": key, "variant": leg, "accepted": True, "eager_checked": True,
                            **state.proof, "compiled_kernels": state.probe.launches[:1], "runtime_source_launched": True})
            del state
        for kind in ("no_op", "wrong_output"):
            state = FP4Case(ROOT, output, case, policy, candidate_workspace=output / "submitted" / kind,
                            defer_candidate_check=True, diagnostic=True)
            eager = numerical_rejection(state, 17)
            state.capture_graph()
            captured = numerical_rejection(state, 19)
            require(state.proof.get("graph_captured") and state.proof.get("graph_replayed"), "missing actual bad-source graph replay")
            records.append({"case_key": key, "variant": kind, "qualified_numerical_rejection": True,
                            "eager": eager, "graph": captured, **state.proof, "gpu_source_binding_validated": True})
            del state
    return {"schema": "minimax-fp4-native-smoke-v1", "status": "PASS", "scoreable": False,
            "performance_samples": 0, "synthetic": True, "synthetic_dimensions": {"M": m, "N": n, "K_logical": k},
            "image": pins["runtime_image"], "native_eager_exact": True, "native_graph_changed_inputs_exact": True,
            "source_negative_controls_validated": True,
            "capture_verification": proof, "source_controls": records, "framework_task_validator_status": "not_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.common, args.output)
    except BaseException as error:
        (args.output / "SMOKE.json").write_text(json.dumps({"status": "FAIL", "error_type": type(error).__name__,
            "error": str(error), "scoreable": False, "performance_samples": 0, "framework_task_validator_status": "not_run"}, indent=2) + "\n")
        raise
    (args.output / "SMOKE.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
