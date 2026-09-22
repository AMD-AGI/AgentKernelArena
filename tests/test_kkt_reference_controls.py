"""Independent KKT controls must detect a reference that omits decay."""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")

TASK = Path(__file__).resolve().parents[1] / "tasks/triton2flydsl/sglang/chunk_scaled_dot_kkt_fwd"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("ignore_decay", [False, True])
def test_known_answer_controls_detect_missing_decay(monkeypatch, ignore_decay):
    support = load(TASK / "scripts/reference_support.py", "kkt_reference_support_test")
    monkeypatch.setitem(sys.modules, "reference_support", support)
    controls = load(TASK / "scripts/reference_controls.py", "kkt_reference_controls_test")
    if ignore_decay:
        real_references = controls.references

        def without_decay(*args, **kwargs):
            reference = real_references(*args, **kwargs)
            original = reference.reference_kkt
            reference.reference_kkt = lambda inputs: original({**inputs, "g": None})
            return reference

        monkeypatch.setattr(controls, "references", without_decay)
        with pytest.raises(AssertionError, match="decay"):
            controls.run()
    else:
        rows = controls.run()
        assert rows
        assert all(row["known_answer"] == "PASS" and row["negative_output"] == "rejected" for row in rows)
