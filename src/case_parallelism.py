"""Framework admission and finalization for opt-in whole-case task workers."""
from pathlib import Path
import os

from .task_contract import strict_json, validate_report
from .tools import case_parallel as protocol


CANONICAL = Path(__file__).resolve().parent / "tools"


def enabled(config):
    descriptor = config.get("trusted_evaluation", {}).get("case_parallelism")
    if descriptor is None:
        return False
    protocol.require(isinstance(descriptor, dict) and set(descriptor) == {"schema_version"}
                     and type(descriptor["schema_version"]) is int and descriptor["schema_version"] == 1,
                     "unsupported task opt-in")
    return True


def verify_helpers(root, config):
    if not enabled(config):
        return
    for name in ("case_parallel.py", "gpu_binding.py"):
        path = protocol._relative(root, "ut/" + name)
        protocol.require(path.read_bytes() == (CANONICAL / name).read_bytes(), "task helper differs from canonical: " + name)


def materialize_helpers(root, config):
    """Install absent helpers before the harness snapshot; reject existing drift."""
    if not enabled(config):
        protocol.require(not os.environ.get(protocol.PLAN_ENV), "selected task did not opt into a GPU group")
        return
    root = Path(root)
    for name in ("case_parallel.py", "gpu_binding.py"):
        target = root / "ut" / name
        protocol.require(target.parent.is_dir() and not target.parent.is_symlink(), "missing regular task UT directory")
        if not target.exists() and not target.is_symlink():
            target.write_bytes((CANONICAL / name).read_bytes())
    verify_helpers(root, config)


def validate_measurement(root, config, manifest, report):
    active = bool(os.environ.get(protocol.PLAN_ENV) or os.environ.get(protocol.HASH_ENV))
    group = bool(report.get("case_parallelism") or report.get("request", {}).get("case_parallelism")
                 or report.get("request", {}).get("gpu_group"))
    request_id = report.get("request", {}).get("request_id")
    if isinstance(request_id, str) and protocol._TOKEN.fullmatch(request_id):
        group = group or (Path(root) / "build/case-parallel" / request_id).exists()
    if not active and not group:
        return
    protocol.require(enabled(config), "group report from a task without opt-in")
    verify_helpers(root, config)
    protocol.validate_group_report(root, manifest, report, required=True)


def validate_framework_reports(root, config):
    """A framework completion needs full compile, correctness and performance."""
    root = Path(root)
    active = bool(os.environ.get(protocol.PLAN_ENV) or os.environ.get(protocol.HASH_ENV))
    performance_path = root / "build/performance_report.json"
    performance = strict_json(protocol._regular(performance_path).read_bytes()) if performance_path.is_file() else {}
    group = bool(performance.get("case_parallelism") or performance.get("request", {}).get("case_parallelism")
                 or performance.get("request", {}).get("gpu_group"))
    request_id = performance.get("request", {}).get("request_id")
    if isinstance(request_id, str) and protocol._TOKEN.fullmatch(request_id):
        group = group or (root / "build/case-parallel" / request_id).exists()
    if not active and not group:
        return
    protocol.require(enabled(config), "framework GPU group requires task opt-in")
    manifest = strict_json(protocol._relative(root, config["trusted_evaluation"].get("case_manifest", "cases.json")).read_bytes())
    group_identity = None
    requests = set()
    for phase in ("compile", "correctness", "performance"):
        report = strict_json(protocol._relative(root, f"build/{phase}_report.json").read_bytes())
        request = report.get("request", {})
        protocol.require(request.get("phase") == phase, "framework phase report differs")
        validate_measurement(root, config, manifest, report)
        protocol.require(all(worker["assignment"]["leg"] == "framework"
                             for worker in report["case_parallelism"]["workers"]),
                         "framework phase contains a foreign trusted source leg")
        validate_report(report, manifest, request)
        identity = (request["case_parallelism"]["plan_sha256"], request["case_parallelism"]["assignment_sha256"],
                    protocol.fingerprint(request["source_sha256"]), request["package_sha256"])
        protocol.require(group_identity is None or identity == group_identity, "framework phase group/source changed")
        protocol.require(request["request_id"] not in requests, "framework phase reused a request")
        requests.add(request["request_id"])
        group_identity = identity
