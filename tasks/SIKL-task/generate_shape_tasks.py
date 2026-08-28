#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Generate one task per workload case, for both SIKL task families.

The GLM-5.2 MXFP4 MoE definition in the workload schema carries six workload
cases that differ only in ``num_tokens``. Arena copies each task directory into
its own workspace, so a task cannot import from a sibling and the harness,
driver and helper modules have to be present in every task. Generating them
from one template per family is what keeps those copies identical: the only
per-task files are ``workload.json`` and ``config.yaml``.

    python3 tasks/SIKL-task/generate_shape_tasks.py [--check]

``--check`` regenerates into a temporary directory and reports any drift
instead of writing, which is what the test uses.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_WORKLOADS = Path(
    "/shared_nfs/jqliu/SIKL-KERNEL/glm5.2-workload-schema/workloads/moe/"
    "aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128.jsonl"
)
DEFINITION = "aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128"

# Per-shape facts measured on MI355X/gfx950 with the image's aiter
# (see the module docstring of each generated workload.json for what they mean).
#
#   relerr  mean relative error of aiter's own output against the task reference
#   ms      the CUDA-graph baseline the task scores against
#   stage1  the tuned stage-1 kernel aiter dispatches to at this shape
#
# The tolerance tiers follow from relerr: every shape whose stage-1 kernel fuses
# the MXFP4 quantization of its output into the epilogue ("kw2_fp4") rounds the
# e8m0 block scale differently from the torch quantizer the reference uses, and
# sits ~20x further from the reference than the other variants do.
MEASUREMENTS = {
    8: {
        "relerr": 0.040629, "ms": 0.047244,
        "stage1": "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w4_kw2_fp4",
        "stage2": "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce",
    },
    16: {
        "relerr": 0.001647, "ms": 0.073987,
        "stage1": "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w2",
        "stage2": "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_bnt2",
    },
    32: {
        "relerr": 0.036040, "ms": 0.091414,
        "stage1": "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4_kw2_fp4",
        "stage2": "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_reduce_persist",
    },
    64: {
        "relerr": 0.001634, "ms": 0.110635,
        "stage1": "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w4",
        "stage2": "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_persist",
    },
    128: {
        "relerr": 0.035199, "ms": 0.116940,
        "stage1": "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_kw2_fp4",
        "stage2": "flydsl_moe2_afp4_wfp4_bf16_t16x128x256_atomic_sbm32",
    },
    256: {
        "relerr": 0.034090, "ms": 0.125229,
        "stage1": "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_kw2_fp4",
        "stage2": "flydsl_moe2_afp4_wfp4_bf16_t16x128x256_atomic_bnt2_persist_sbm32",
    },
}

SEED = 29
EXACT_CONTRACT_GATE = 0.01
FUSED_EPILOGUE_GATE = 0.09

TEMPLATE_SHAPE = 64
REWRITE_TEMPLATE = HERE / f"glm52_moe_mxfp4_per1x32_t{TEMPLATE_SHAPE}"
OPTIMIZE_TEMPLATE = HERE / f"glm52_moe_mxfp4_per1x32_t{TEMPLATE_SHAPE}_flydsl_opt"

# Everything except the two per-task files is copied from the template.
PER_TASK_FILES = ("config.yaml", "workload.json")


def tolerance(num_tokens: int) -> tuple[float, str]:
    """The correctness gate for a shape, and why it is what it is."""
    measured = MEASUREMENTS[num_tokens]
    if "kw2_fp4" in measured["stage1"]:
        return FUSED_EPILOGUE_GATE, (
            "The tuned stage-1 kernel for this shape fuses the MXFP4 "
            "quantization of its output into the epilogue and rounds the e8m0 "
            "block scale differently from the torch quantizer the reference "
            "uses, so aiter's own output sits "
            f"{measured['relerr']:.4f} from the reference. The gate is ~2x that "
            "floor. It is looser than the exact-contract shapes and would not "
            "catch a small accuracy regression; tightening it requires the "
            "reference to reproduce the kernel's epilogue rounding."
        )
    return EXACT_CONTRACT_GATE, (
        "The tuned stage-1 kernel for this shape writes bf16 and the activation "
        "quantization follows the same contract as the reference, so aiter's own "
        f"output sits {measured['relerr']:.4f} from it. The gate is ~6x that floor."
    )


def workload_cases() -> list[dict]:
    """The schema's workload cases, smallest shape first."""
    cases = [
        json.loads(line)
        for line in SCHEMA_WORKLOADS.read_text().splitlines()
        if line.strip()
    ]
    parsed = [
        {
            "num_tokens": int(case["workload"]["axes"]["num_tokens"]),
            "case_uuid": case["workload"]["uuid"],
        }
        for case in cases
    ]
    return sorted(parsed, key=lambda case: case["num_tokens"])


def workload_json(case: dict) -> str:
    num_tokens = case["num_tokens"]
    measured = MEASUREMENTS[num_tokens]
    gate, reason = tolerance(num_tokens)
    payload = {
        "definition": DEFINITION,
        "case_uuid": case["case_uuid"],
        "num_tokens": num_tokens,
        "seed": SEED,
        "max_relerr": gate,
        "tolerance_reason": reason,
        "measured": {
            "baseline_relerr_vs_reference": measured["relerr"],
            "baseline_ms_cuda_graph": measured["ms"],
            "stage1_kernel": measured["stage1"],
            "stage2_kernel": measured["stage2"],
            "platform": "MI355X/gfx950",
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def rewrite_config(case: dict) -> str:
    num_tokens = case["num_tokens"]
    measured = MEASUREMENTS[num_tokens]
    gate, _ = tolerance(num_tokens)
    return f"""\
task_type: rewrite_by_flydsl

# Generated by tasks/SIKL-task/generate_shape_tasks.py -- edit the template task
# (glm52_moe_mxfp4_per1x32_t{TEMPLATE_SHAPE}) or the generator, then regenerate.

# The agent-editable file: the FlyDSL port lands here. Everything else in the
# task (harness, driver, reference, baseline, input construction) is protected.
source_file_path:
  - kernel.py
target_kernel_functions:
  - build_glm52_mxfp4_moe_2stage_module

# ---------------------------------------------------------------------------
# Rewrite pipeline inputs (consumed by agents/forge_rewrite/launch_agent.py).
#
# port_source is the baseline implementation's entry file. It is read-only
# reference material for the PORT phase; the driver owns the interface and the
# measurement. The rewrite prompt embeds only the first 16000 characters of it,
# so scripts/forge_driver.py carries the full implementation chain as pointers.
# ---------------------------------------------------------------------------
rewrite:
  port_source: /sgl-workspace/aiter/aiter/fused_moe.py
  port_source_entry: fused_moe
  port_target: kernel.py
  logical_operator: glm52_mxfp4_moe_2stage
  source_owner: aiter
  snr_threshold: 30.0
  max_port_attempts: 5

# The numeric contract (axes, dtypes, seed, tolerance, benchmark parameters)
# lives in workload.json and scripts/task_inputs.py as the single source of
# truth; duplicating it here would let the two drift.
workload:
  definition: {DEFINITION}
  case_uuid: {case['case_uuid']}
  num_tokens: {num_tokens}

compile_command:
  - python3 test_kernel_harness.py --compile
correctness_command:
  - python3 test_kernel_harness.py --correctness
performance_command:
  - python3 test_kernel_harness.py --full-benchmark

compile_timeout: 1800
correctness_timeout: 1800
performance_timeout: 1800
task_result_template: null

platform_support:
  required_arch: gfx950
  status: active
  skip_reason: null

prompt:
  source_code: null
  cheatsheet: null
  instructions: >-
    Implement the GLM-5.2 routed-expert MoE layer in FlyDSL for MI355X/gfx950,
    replacing the production implementation the task measures as its baseline.

    Per-rank (TP=8) configuration: model_dim=6144, inter_dim=256, 257 experts
    (256 routed + 1 fused shared), topk=9, MXFP4 group_size=32
    (QuantType.per_1x32), SiLU, bf16 in and out. This is the afp4_wfp4 path:
    BOTH operands are MXFP4. The weights arrive pre-quantized and preshuffled
    (shuffle_weight(w, (16,16)) on the packed weights, e8m0_shuffle on the
    scales); the activation must be quantized to MXFP4 twice, once into stage 1
    and again on the stage-1 output going into stage 2.

    This task scores the num_tokens={num_tokens} workload case. At that shape the
    production implementation measures {measured['ms']:.4f} ms under CUDA-graph
    replay, dispatching to {measured['stage1']}
    and {measured['stage2']}.

    The interface you must expose, the exact tensor layouts, and the file paths
    of the baseline implementation (aiter's entry, its FlyDSL stage kernels and
    its HIP quant/sort/reduce kernels) are all documented at the top of
    scripts/forge_driver.py. Read it first, then read the baseline sources it
    points at.

    Write the implementation in FlyDSL only. Do not call aiter, torch or any
    other GPU library to compute the result -- that defeats the rewrite.
    Correctness is measured against scripts/task_reference.py with a mean
    relative error gate of {gate}; the production baseline measures
    {measured['relerr']:.6f}. Edit only kernel.py; the harness, driver,
    reference, baseline and input construction are protected.
"""


def optimize_config(case: dict) -> str:
    num_tokens = case["num_tokens"]
    measured = MEASUREMENTS[num_tokens]
    gate, _ = tolerance(num_tokens)
    return f"""\
task_type: image_kernel

# Generated by tasks/SIKL-task/generate_shape_tasks.py -- edit the template task
# (glm52_moe_mxfp4_per1x32_t{TEMPLATE_SHAPE}_flydsl_opt) or the generator, then
# regenerate.
#
# The operator ships inside aiter, so the editable source is the aiter tree the
# image provides, seeded into the workspace per run. The sibling task
# glm52_moe_mxfp4_per1x32_t{num_tokens} measures the same operator with the same
# inputs, reference and baseline, but rewrites it in FlyDSL from scratch; this
# one optimizes the FlyDSL implementation aiter already ships, so the two scores
# answer the same question from opposite directions.
image_repo_path: /sgl-workspace/aiter
repo_subdir: aiter
# Paths are matched REPO-RELATIVE and exactly, so the build scratch is
# `aiter/jit/build`, not `jit/build`. Getting it wrong seeds an extra ~2.9 GB
# into every run workspace; the prebuilt .so that matter live in aiter/jit
# itself and are kept, which is also what keeps the FlyDSL cache warm.
image_repo_exclude:
  - __pycache__
  - aiter/jit/build

# Selects the FlyDSL cheatsheet and the Forge fellow.
repository_language: flydsl

# ALL PATHS ARE REPO-ROOT RELATIVE. `image_repo_path` is the aiter REPO ROOT, so
# the Python package sits one level down and a package file is `aiter/<...>.py`.
# Package-relative paths only resolve through a whole-tree rglob fallback, which
# is a scan of a multi-GB tree per entry and a hard error the moment two files
# share a suffix.
#
# ENTRY 0 IS THE FORGE ANCHOR: it is passed as `--kernel` and forge-loop's task
# statement is literally "optimize the kernel at <that file>", so it must hold
# the compute core rather than the dispatcher. mfma_moe1_silu_mul_* is defined
# at :313 and mfma_moe2_*_cshuffle at :7432 of the anchor file.
source_file_path:
  - aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
  - aiter/ops/flydsl/kernels/moe_gemm_2stage.py
  - aiter/ops/flydsl/moe_kernels.py
  - aiter/fused_moe.py

# The rest of the FlyDSL MoE surface, plus the tuned table: picking a
# better-suited kernel variant for this shape is a legitimate optimization.
#
# What is deliberately NOT editable: the HIP quant/sort kernels under csrc/
# (this task is same-language FlyDSL optimization), and aiter/ops/quant.py,
# which provides the MXFP4 quantizer the correctness reference uses -- an edit
# there would move the operator and its oracle together.
editable_sources:
  - aiter/ops/flydsl/kernels/moe_sorting_kernel.py
  - aiter/ops/flydsl/kernels/moe_gather_reduce.py
  - aiter/ops/flydsl/kernels/moe_m_tile_map.py
  - aiter/ops/flydsl/kernels/moe_route_maps.py
  - aiter/ops/flydsl/mxfp4_kname.py
  - aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv

target_kernel_functions:
  - fused_moe
  - _flydsl_stage1_wrapper
  - _flydsl_stage2_wrapper

kernel_identity:
  logical_operator: mxfp4_moe_2stage
  kernel_kind: flydsl
  source_owner: aiter
  workload:
    source: workload.json
    primary_case: num_tokens_{num_tokens}

compile_command:
  - python3 test_kernel_harness.py --compile
correctness_command:
  - python3 test_kernel_harness.py --correctness
performance_command:
  - python3 test_kernel_harness.py --full-benchmark

compile_timeout: 1800
correctness_timeout: 1800
performance_timeout: 1800
task_result_template: null

platform_support:
  required_arch: gfx950
  status: active
  skip_reason: null

prompt:
  source_code: null
  cheatsheet: null
  instructions: >-
    Optimize the GLM-5.2 routed-expert MoE on MI355X/gfx950 by improving the
    FlyDSL kernels aiter already ships. This is same-language optimization: keep
    the implementation in FlyDSL, do not port it to HIP, Triton or CUDA, and do
    not replace it with a torch or library call.

    Per-rank (TP=8) configuration: model_dim=6144, inter_dim=256, 257 experts
    (256 routed + 1 fused shared), topk=9, MXFP4 group_size=32
    (QuantType.per_1x32), SiLU, bf16 in and out. This is the afp4_wfp4 path:
    BOTH operands are MXFP4. The weights arrive pre-quantized and preshuffled
    (shuffle_weight(w, (16,16)) plus e8m0_shuffle on the scales); the activation
    is quantized to MXFP4 twice, once into stage 1 and again on the stage-1
    output going into stage 2.

    This task scores the num_tokens={num_tokens} workload case, where the shipped
    implementation measures {measured['ms']:.4f} ms under CUDA-graph replay. The
    tuned table dispatches this shape to {measured['stage1']}
    and {measured['stage2']}.
    M is small: {num_tokens} tokens x topk 9 = {num_tokens * 9} rows spread over
    257 experts, so occupancy and per-expert tile efficiency dominate rather
    than raw FLOPs.

    Useful levers: the MFMA pipeline and tile/block loop structure in
    mixed_moe_gemm_2stage.py, stage-2's reduction strategy (atomic vs reduce vs
    persist), the FlyDSL reduction kernel, the per-M-bucket kernel choice in
    configs/model_configs/glm5_fp4_tuned_fmoe.csv, and the Python dispatch in
    fused_moe.py. Preserve the dispatch contract: block_m / ksplit semantics,
    moe_sorting behaviour, the separated gate/up layout implied by
    shuffle_weight((16,16)), and doweight_stage1=False so the routing weights
    are applied in the stage-2 reduction.

    Correctness is measured against scripts/task_reference.py with a mean
    relative error gate of {gate}; the shipped kernels measure
    {measured['relerr']:.6f}. Do not edit the harness, the driver, or anything
    under scripts/.
"""


FAMILIES = (
    {
        "template": REWRITE_TEMPLATE,
        "suffix": "",
        "config": rewrite_config,
    },
    {
        "template": OPTIMIZE_TEMPLATE,
        "suffix": "_flydsl_opt",
        "config": optimize_config,
    },
)


def task_dir(root: Path, family: dict, num_tokens: int) -> Path:
    return root / f"glm52_moe_mxfp4_per1x32_t{num_tokens}{family['suffix']}"


def template_files(template: Path) -> list[Path]:
    return sorted(
        path for path in template.rglob("*")
        if path.is_file()
        and path.name not in PER_TASK_FILES
        and "__pycache__" not in path.parts
        and path.name != "_aka_benchmark.py"
    )


def generate(root: Path) -> list[Path]:
    written: list[Path] = []
    for case in workload_cases():
        if case["num_tokens"] not in MEASUREMENTS:
            raise SystemExit(
                f"no measurements recorded for num_tokens={case['num_tokens']}; "
                "measure the shape before generating a task for it"
            )
        for family in FAMILIES:
            destination = task_dir(root, family, case["num_tokens"])
            destination.mkdir(parents=True, exist_ok=True)
            for source in template_files(family["template"]):
                relative = source.relative_to(family["template"])
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # Regenerating in place makes the template its own destination.
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
                written.append(target)
            (destination / "workload.json").write_text(workload_json(case))
            (destination / "config.yaml").write_text(family["config"](case))
            written.extend([destination / name for name in PER_TASK_FILES])
    return written


def check() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        # The templates are read from the repository, the output goes to scratch.
        generate(root)
        drift: list[str] = []
        for case in workload_cases():
            for family in FAMILIES:
                expected = task_dir(root, family, case["num_tokens"])
                actual = task_dir(HERE, family, case["num_tokens"])
                if not actual.is_dir():
                    drift.append(f"missing task: {actual.name}")
                    continue
                for path in sorted(expected.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(expected)
                    other = actual / relative
                    if not other.is_file():
                        drift.append(f"{actual.name}/{relative}: missing")
                    elif not filecmp.cmp(path, other, shallow=False):
                        drift.append(f"{actual.name}/{relative}: differs")
    if drift:
        print("generated tasks are out of date:")
        for item in drift:
            print(f"  {item}")
        return 1
    print("generated tasks are up to date")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report drift instead of writing")
    args = parser.parse_args()
    if args.check:
        return check()
    written = generate(HERE)
    print(f"generated {len(written)} files across "
          f"{len(workload_cases()) * len(FAMILIES)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
