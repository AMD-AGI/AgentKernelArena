"""target_language: asm — the aka_asm helper, the assembly audit and the candidate contract."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.tools.arena_eval import asm  # noqa: E402
from src.tools.arena_eval.audit import LaunchObserver, PolicyError, audit_python_sources  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FILL_S = """\
.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl fill
.p2align 8
.type fill,@function
fill:
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_load_dword s6, s[0:1], 0x8
  v_lshlrev_b32 v1, 2, v0
  s_waitcnt lgkmcnt(0)
  v_add_u32 v2, s6, v0
  global_store_dword v1, v2, s[4:5]
  s_endpgm
.rodata
.p2align 6
.amdhsa_kernel fill
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 8
  .amdhsa_accum_offset 4
  .amdhsa_kernarg_size 16
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: fill
    .symbol: fill.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 8
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
    .args:
      - { .size: 8, .offset: 0, .value_kind: global_buffer, .address_space: global }
      - { .size: 4, .offset: 8, .value_kind: by_value }
...
.end_amdgpu_metadata
"""

KERNEL_PY = """\
import torch
import aka_asm

FILL = aka_asm.load("fill.s", "fill")


def run(*, x, out) -> None:
    FILL.launch(grid=(1, 1, 1), block=(64, 1, 1), args=[out, aka_asm.u32(x.numel())])
"""


def test_scalars_pack_at_their_natural_alignment_and_the_buffer_pads_to_8_bytes():
    assert asm.pack_arguments([asm.u32(1)]) == struct.pack("<I", 1) + b"\0" * 4
    assert asm.pack_arguments([asm.u32(1), asm.u64(2)]) == struct.pack("<I", 1) + b"\0" * 4 + struct.pack("<Q", 2)
    packed = asm.pack_arguments([asm.i32(-1), asm.f32(1.5), asm.u32(3)])
    assert packed == struct.pack("<ifI", -1, 1.5, 3) + b"\0" * 4
    assert asm.pack_arguments([asm.u64(5), asm.i64(-6)]) == struct.pack("<Qq", 5, -6)


@pytest.mark.parametrize("bad, message", [
    (7, "pass a tensor or a scalar"),
    (1.5, "pass a tensor or a scalar"),
    (torch.zeros(2), "pass GPU tensors"),
])
def test_untyped_scalars_and_host_tensors_are_rejected(bad, message):
    with pytest.raises(asm.AsmError, match=message):
        asm.pack_arguments([bad])


def test_a_scalar_that_does_not_fit_its_type_is_rejected():
    with pytest.raises(asm.AsmError, match="does not fit"):
        asm.pack_arguments([asm.u32(-1)])


@pytest.mark.parametrize("text", ['.incbin "/etc/passwd"\n', '  .INCLUDE "other.s"\n'])
def test_sources_cannot_pull_in_other_files(text):
    with pytest.raises(asm.AsmError, match=".incbin and .include are not allowed"):
        asm.check_source("k.s", FILL_S + text)


def test_a_directive_named_in_a_comment_is_not_a_directive():
    asm.check_source("k.s", FILL_S + "  // .incbin is not used here\n")


def test_an_asm_candidate_passes_the_audit():
    evidence = audit_python_sources({"kernel.py": KERNEL_PY, "fill.s": FILL_S}, "asm")
    assert evidence["language"] == "asm"


@pytest.mark.parametrize("sources, message", [
    ({"kernel.py": KERNEL_PY, "fill.s": FILL_S, "fill.co": "x"}, "only .py and .s sources"),
    ({"kernel.py": KERNEL_PY, "fill.s": '.incbin "x"\n' + FILL_S}, ".incbin and .include"),
    ({"kernel.py": KERNEL_PY.replace("import aka_asm", "import aka_asm\nimport triton"), "fill.s": FILL_S},
     "import triton is not allowed for asm"),
    ({"kernel.py": "import torch\n\n\ndef run(*, x, out) -> None:\n    pass\n", "fill.s": FILL_S},
     "imports no asm kernel language"),
    ({"kernel.py": KERNEL_PY.replace("FILL.launch", "aka_asm.LAUNCHES = []\n    FILL.launch"), "fill.s": FILL_S},
     "cannot mutate an imported module"),
])
def test_asm_candidate_audit_rejections(sources, message):
    with pytest.raises(PolicyError, match=message):
        audit_python_sources(sources, "asm")


def test_other_python_languages_still_reject_non_python_files():
    with pytest.raises(PolicyError, match="contains only .py sources"):
        audit_python_sources({"kernel.py": "import triton\n", "fill.s": FILL_S}, "triton")


def test_the_launch_observer_requires_an_assembly_launch_per_call():
    observer = LaunchObserver("asm", "pkg")
    with observer.observe():
        asm.LAUNCHES.append("fill")
    observer.require("candidate")
    assert observer.launches == ["fill"]
    idle = LaunchObserver("asm", "pkg")
    with idle.observe():
        pass
    with pytest.raises(RuntimeError, match="No asm kernel launch"):
        idle.require("candidate")


def test_asm_is_a_target_language_with_its_own_template_and_contract(tmp_path):
    import shutil

    from src.task_contract import TARGET_LANGUAGES, render_workspace

    assert "asm" in TARGET_LANGUAGES
    task = ROOT / "tasks/triton/elementwise/add/fp32_1d"
    workspace = tmp_path / "ws"
    shutil.copytree(task, workspace)
    render_workspace(task, workspace, task_id="triton/elementwise/add/fp32_1d", target_language="asm")
    template = (workspace / "solution/kernel.py").read_text()
    assert "import aka_asm" in template and "AKA-UNIMPLEMENTED-TEMPLATE" in template
    candidate_doc = (workspace / "_aka_eval/CANDIDATE.md").read_text()
    assert "**asm**" in candidate_doc and "aka_asm.load" in candidate_doc and ".incbin" in candidate_doc
    assert (workspace / "_aka_eval/asm.py").is_file()


def test_the_asm_cheatsheet_is_configured():
    import yaml

    config = yaml.safe_load((ROOT / "src/prompts/cheatsheet/default_cheatsheet.yaml").read_text())
    path = ROOT / config["knowledge"]["asm"]
    assert path.is_file() and "aka_asm.load" in path.read_text()


def test_a_source_assembles_into_a_cached_code_object(tmp_path, monkeypatch):
    try:
        asm._toolchain()
    except asm.AsmError:
        pytest.skip("ROCm assembler not in this environment")
    monkeypatch.setenv("AKA_KERNEL_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "fill.s"
    source.write_text(FILL_S)
    first = asm.assemble(source, "fill.s", "gfx950")
    assert first.is_file() and first.read_bytes()[:4] == b"\x7fELF"
    assert asm.assemble(source, "fill.s", "gfx950") == first
    assert asm.BUILD_EVIDENCE["files"][-1]["source"] == "fill.s"
    source.write_text(FILL_S.replace("v_add_u32 v2, s6, v0", "v_add_u32 v2, s6, s6"))
    assert asm.assemble(source, "fill.s", "gfx950") != first


def test_a_broken_source_reports_the_assembler_error(tmp_path, monkeypatch):
    try:
        asm._toolchain()
    except asm.AsmError:
        pytest.skip("ROCm assembler not in this environment")
    monkeypatch.setenv("AKA_KERNEL_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "bad.s"
    source.write_text(FILL_S.replace("s_endpgm", "s_not_an_instruction"))
    with pytest.raises(asm.AsmError, match="clang failed"):
        asm.assemble(source, "bad.s", "gfx950")


# --- vendored code objects of asm baselines (baseline.provenance.code_objects) ---------------------

def _asm_task(tmp_path, code_objects=None, files=None):
    """A copy of the fp32 add task turned into an asm-baseline package."""
    import hashlib
    import shutil

    import yaml

    task = tmp_path / "task"
    shutil.copytree(ROOT / "tasks/triton/elementwise/add/fp32_1d", task)
    shutil.rmtree(task / "baseline")
    (task / "baseline").mkdir()
    (task / "baseline/kernel.py").write_text(KERNEL_PY.replace("fill.s", "fill.co").replace("x.numel()", "1"))
    for name, data in (files if files is not None else {"baseline/fill.co": b"\x7fELF-fake"}).items():
        (task / name).write_bytes(data)
    raw = yaml.safe_load((task / "task.yaml").read_text())
    raw["baseline"]["language"] = "asm"
    if code_objects is None:
        code_objects = {n: hashlib.sha256((task / n).read_bytes()).hexdigest()
                        for n in (files if files is not None else {"baseline/fill.co": 0})}
    if code_objects:
        raw["baseline"]["provenance"]["code_objects"] = code_objects
    (task / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return task


def test_an_asm_baseline_declares_its_code_objects_by_hash(tmp_path):
    from src.task_contract import load_task_contract

    task = _asm_task(tmp_path)
    contract = load_task_contract(task / "task.yaml", task_id="asm/elementwise/add/fp32_1d")
    digest = contract.code_objects["baseline/fill.co"]
    resolved = contract.resolved_contract("asm")
    assert resolved["code_objects"] == {digest: "baseline/fill.co"}


@pytest.mark.parametrize("mutate, message", [
    (lambda t, d: d.update({"baseline/fill.co": "0" * 64}), "does not match its declared SHA256"),
    (lambda t, d: d.update({"baseline/other.co": "0" * 64}), "missing"),
    (lambda t, d: (t / "baseline/extra.co").write_bytes(b"x"), "undeclared code objects"),
    (lambda t, d: d.update({"baseline/fill.co": "ABC"}), "lowercase hex SHA256"),
    (lambda t, d: d.update({"reference.co": "0" * 64}), "under baseline/"),
])
def test_code_object_declarations_are_enforced(tmp_path, mutate, message):
    import hashlib

    import yaml

    from src.task_contract import TaskConfigError, load_task_contract

    task = _asm_task(tmp_path)
    raw = yaml.safe_load((task / "task.yaml").read_text())
    declared = raw["baseline"]["provenance"]["code_objects"]
    mutate(task, declared)
    (task / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(TaskConfigError, match=message):
        load_task_contract(task / "task.yaml", task_id="asm/elementwise/add/fp32_1d")
    assert hashlib  # imported for parity with _asm_task


def test_only_asm_baselines_may_declare_code_objects(tmp_path):
    import yaml

    from src.task_contract import TaskConfigError, load_task_contract

    task = _asm_task(tmp_path)
    raw = yaml.safe_load((task / "task.yaml").read_text())
    raw["baseline"]["language"] = "triton"
    (task / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(TaskConfigError, match="only for asm baselines"):
        load_task_contract(task / "task.yaml", task_id="asm/elementwise/add/fp32_1d")


def test_aka_asm_loads_only_allowed_code_objects(tmp_path, monkeypatch):
    import hashlib
    import sys
    import types

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "k.co").write_bytes(b"\x7fELF-not-allowed")
    module = types.ModuleType("_aka_impl_test_pkg")
    module.__path__ = [str(package)]
    monkeypatch.setitem(sys.modules, "_aka_impl_test_pkg", module)
    caller = {"__package__": "_aka_impl_test_pkg", "aka_asm": asm}
    monkeypatch.setattr(asm, "ALLOWED_CODE_OBJECTS", frozenset())
    with pytest.raises(asm.AsmError, match="not one of the baseline's vendored code objects"):
        exec("aka_asm.load('k.co', 'kernel')", caller)
    with pytest.raises(asm.AsmError, match="not a file in the implementation directory"):
        exec("aka_asm.load('../outside.co', 'kernel')", caller)
    digest = hashlib.sha256((package / "k.co").read_bytes()).hexdigest()
    monkeypatch.setattr(asm, "ALLOWED_CODE_OBJECTS", frozenset({digest}))
    # An allowed hash passes the policy; loading the fake bytes then fails in HIP or for lack of a GPU.
    with pytest.raises(Exception) as raised:
        exec("aka_asm.load('k.co', 'kernel')", caller)
    assert "vendored code objects" not in str(raised.value)


def test_qa_accepts_declared_code_objects_and_flags_others(tmp_path):
    import shutil

    from agents.qa_auditor.deterministic import check_package

    workspace = tmp_path / "ws"
    shutil.copytree(_asm_task(tmp_path), workspace)
    findings: list = []
    check_package(workspace, findings)
    assert not [f for f in findings if "binary" in str(f)]
    (workspace / "baseline/fill.co").write_bytes(b"changed")
    check_package(workspace, findings)
    assert [f for f in findings if "is a binary" in str(f)]
