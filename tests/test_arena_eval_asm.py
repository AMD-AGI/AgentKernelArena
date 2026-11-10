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
    ({"kernel.py": KERNEL_PY}, "no .s kernel source"),
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
