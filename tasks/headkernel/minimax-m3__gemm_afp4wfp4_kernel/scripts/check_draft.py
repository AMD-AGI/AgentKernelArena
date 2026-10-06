"""CPU source/provenance check; deliberately emits no task result or timings."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from binding import raw_wrapper_tree
from source_guard import validate_sources


def main():
    pins = json.loads((ROOT / "SOURCE-PROVENANCE.json").read_text())
    for relative, record in pins["sources"].items():
        path = ROOT / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("stock draft source changed: " + relative)
    validate_sources(ROOT, ROOT)
    raw_wrapper_tree((ROOT / "ut/native/wrapper.py").read_text())
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps({"status": "CPU_SOURCE_CHECK_PASS", "scoreable": False,
                      "actual_operand_capture": "external_manifest" if (ROOT / "cases.json").exists() else "missing", "gpu_source_binding_validated": False,
                      "framework_task_validator_status": "not_run"}))


if __name__ == "__main__":
    main()
