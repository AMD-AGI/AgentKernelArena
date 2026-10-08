"""Protected portable runner; refuses absent/unadmitted actual FP4 capture."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
sys.path.insert(0, str(ROOT / "scripts"))
from admission import POLICY, oracle_policy
from evaluation_contract import checked_replays, finalize_report, fingerprint, require, strict_json, validate_manifest
from fixture_codec import file_sha, safe_file
from runtime import FP4Case


def load_dataset(root):
    root = Path(root)
    manifest_path = safe_file(root, "cases.json")
    admission = strict_json(safe_file(root, "CAPTURE-ADMISSION.json").read_text())
    require(admission["status"] == "captured_not_qualified" and admission["cases_sha256"] == file_sha(manifest_path),
            "captured case manifest lacks its admission receipt")
    manifest = validate_manifest(strict_json(manifest_path.read_text()))
    pins = strict_json((ROOT / "SOURCE-PROVENANCE.json").read_text())
    require(manifest["runtime_image"] == pins["runtime_image"] and manifest["capture_scope"] == "full_served_workload",
            "wrong image or non-workload cases")
    require(manifest["measurement"] == POLICY, "protected 10/100 fresh replay policy differs")
    oracle_policy(manifest["oracle_policy"])
    return manifest


def package_hash():
    digest = hashlib.sha256()
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        # The framework streams logs while the command runs and gives native
        # extension builds a private cache. These two top-level directories are
        # runtime output, not immutable task inputs. Nested names stay protected.
        if relative.parts[0] in {".validator_audit", ".validator_torch_extensions"}:
            continue
        if any(part in {"build", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        require(not path.is_symlink(), "protected task contains a symlink")
        if path.is_file():
            digest.update(relative.as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def run_source_controls(dataset, manifest, seed):
    from check_source_binding import probe
    from make_source_controls import control_source
    submitted_controls = []
    for kind in ("no_op", "wrong_output"):
        workspace = ROOT / "build" / ("submitted_" + kind)
        (workspace / "source").mkdir(parents=True, exist_ok=True)
        (workspace / "source/kernel.py").write_text(control_source(kind))
        for case in manifest["cases"]:
            for mode in ("eager", "graph"):
                code, evidence = probe(dataset, workspace, case["case_id"], mode, seed)
                require(code == 1 and evidence["status"] == "candidate_rejected"
                        and evidence["reference_calibrated"] and evidence.get("candidate_compiled_and_engaged") is True
                        and (mode != "graph" or evidence.get("graph_captured") is True
                             and evidence.get("graph_replayed") is True),
                        "submitted source control did not reject after calibrated reference and candidate engagement: "
                        + kind + "/" + case["case_id"] + "/" + mode)
                submitted_controls.append({"variant": kind, **evidence})
    return submitted_controls


def run(phase, dataset, request=None):
    manifest = load_dataset(dataset)
    import torch
    source = {"source/kernel.py": file_sha(ROOT / "source/kernel.py")}
    request = request or {"schema_version": 1, "request_id": secrets.token_hex(24), "phase": phase,
        "manifest_sha256": fingerprint(manifest), "package_sha256": package_hash(), "source_sha256": source,
        "challenge_seed": secrets.randbelow(2**30)}
    require(request["phase"] == phase and request["manifest_sha256"] == fingerprint(manifest)
            and request["source_sha256"] == source, "request does not match current source/cases")
    require(torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName,
            "native FP4 runner requires gfx950")
    torch.set_num_threads(16)
    before = package_hash()
    rows, compiled = [], []
    for case in manifest["cases"]:
        state = FP4Case(ROOT, dataset, case, manifest["oracle_policy"])
        compiled.append({"case_id": case["case_id"], **state.proof, "compiled_kernels": state.probe.launches[:1]})
        if phase == "correctness":
            for seed in POLICY["correctness_seeds"]:
                state.check_once(seed)
            state.capture_graph()
            for seed in POLICY["correctness_seeds"]:
                state.check_once(seed)
            controls = {}
            for name in POLICY["negative_controls"]:
                truth = state.reset(request["challenge_seed"])
                state.initialize()
                if name == "wrong_output":
                    state.replay()
                    torch.cuda.synchronize()
                    state.result.fill_(float("nan"))
                try:
                    state.verify(truth)
                except AssertionError:
                    controls[name] = True
                else:
                    raise AssertionError("invalid callback control accepted: " + name)
            rows.append({"case": case, "correct": True, "seeds": POLICY["correctness_seeds"],
                         "negative_controls": controls, "eager_and_graph_checked": True})
        elif phase == "performance":
            state.capture_graph()
            rows.append(checked_replays(case, POLICY, reset_inputs=state.reset, initialize_outputs=state.initialize,
                replay=state.replay, verify=state.verify, measure=state.measure, observe=state.observe,
                seed=request["challenge_seed"]))
        del state
    submitted_controls = []
    if phase == "correctness":
        submitted_controls = run_source_controls(dataset, manifest, request["challenge_seed"])
    require(package_hash() == before, "protected package changed during execution")
    return finalize_report({"schema_version": 1, "status": "ok", "request": request, "compiled": True,
                            "cases": rows, "compiled_kernels": compiled, "oracle": "independent_cpu_packed_fp4",
                            "submitted_source_controls": submitted_controls},
                           manifest, request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("compile", "correctness", "performance"))
    parser.add_argument("--dataset", type=Path, default=ROOT)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    output = ROOT / "build" / (args.phase + "_report.json")
    output.parent.mkdir(exist_ok=True)
    output.unlink(missing_ok=True)
    report = run(args.phase, args.dataset, strict_json(args.request.read_text()) if args.request else None)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.phase + ": PASS")


if __name__ == "__main__":
    main()
