#!/usr/bin/env python3
"""Generate the headkernel_ut_0914_full benchmark suite from tools/manifest.json.

Every task gets the same layout and a byte-identical runner:

    tasks/headkernel/<task>/
      config.yaml            arena schema + a `headkernel:` provenance block
      README.md              what the kernel is, where it came from, how to run it
      scripts/task_runner.py compile | correctness | performance   (shared template)
      scripts/_bench.py      native 10 warmup / 100 measured timing (shared template)
      source/                THE EDITABLE KERNEL - the only thing an optimizer changes
      ut/                    the frozen GEAK op package (oracle, harness, overlays)

``ut/kernel_src/<f>`` is a relative symlink to ``../../source/<f>``, so the arena
view and the GEAK harness view are the same bytes and the task stays
self-contained. Oracle blobs are hardlinked (same NFS export) so the suite costs
no extra disk.

Idempotent: re-running rebuilds each task from scratch.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)
TASKS = os.path.join(SUITE, "tasks", "headkernel")
TEMPLATES = os.path.join(HERE, "templates")

# Regenerable build/run output - the arena validator flags these as shipped junk.
SKIP_JUNK = {"__pycache__", ".pytest_cache", "build", ".rocprofv3", ".git",
             "_cand_overlay", "_baseline_random.pt", "unittest_smoke.log",
             "heartbeat.json", "specialist_done.json", "specialist_done.partial.json"}
# Prior optimization results. A benchmark that ships the answer measures nothing,
# so these are stripped from the task and preserved under _prior_solutions/.
SKIP_SOLUTION = {"_candidate_best", "accepted_overlay", "seam_patch", "_evidence",
                 "_provenance", "final.patch", "overlay.tar.gz", "recipe.json",
                 "report.md", "_cycle0_frozen", "RUN_0911"}
# Capture-time scaffolding: not needed to run the task, and the sweeps are partial answers.
SKIP_SCAFFOLD = {"_capture", "_capture_out", "_capture_out2", "_capture_overlay",
                 "_capture_overlay1", "_derive_overlay", "_probe_overlay",
                 "_s1k_overlay", "_stage1_overlay",
                 # 2026-09-13 callable-UT generation. Each of these was checked
                 # against every .py and meta.json in all five packages and is
                 # referenced by none of them:
                 #   bench/               a 111 MB serving transcript
                 #   selection_evidence/  capture-time scratch
                 #   capture.log, capture_{started,finished}.json  bookkeeping
                 # NOT in this list, and they must not be: capture_telemetry.json
                 # (opened by _verify_provenance in the MoE and paged-attention
                 # UTs) and attempts/ (fused_add_rmsnorm's meta.json points
                 # shape_validation.path at attempts/001_.../meta.json and its
                 # _verify_provenance opens it). Dropping either turns a passing
                 # correctness leg into FileNotFoundError.
                 "bench", "selection_evidence",
                 "capture.log", "capture_started.json", "capture_finished.json"}
SKIP_NAMES = SKIP_JUNK | SKIP_SOLUTION | SKIP_SCAFFOLD
# .log: every *.log in these packages is a stale per-case transcript. One of them
# (unittest.hk06.log) records a FAIL that the final result.json contradicts, so
# shipping them actively misleads an auditor. No task reads one.
SKIP_SUFFIX = (".pyc", ".pyo", ".so", ".o", ".log", ".ninja_log", ".ninja_deps",
               ".patch", ".tar.gz")
SKIP_PREFIX = ("_sweep_", "_diag_", "selection_trace.")
# Above this size we hardlink instead of copying (oracles are up to 7 GB).
HARDLINK_OVER = 8 * 1024 * 1024

# Target symbols per task, verified present in the package's kernel_src.
TARGETS = {
    "deepseek-v4-pro__dsa_sparse_mla_attn": [
        "dpsk_v4_fp8_attention_fwd", "sparse_mla_fwd_decode_partial",
        "sparse_mla_fwd_decode_combine"],
    "deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl": ["flydsl_moe_stage1"],
    "deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4": ["opus_a8w4_stage2_wrapper"],
    "qwen3.8-2.4t__fused_moe_2stage_mxfp4": ["fused_moe"],
    "kimi-k3__fwd_grouped_kernel_stage1": ["_fwd_grouped_kernel_stage1_tm", "make_launcher"],
    "kimi-k3__moe_gemm1_stage1": ["flydsl_moe_stage1"],
    "kimi-k3__moe_gemm2_stage2": ["flydsl_moe_stage2"],
    "glm-5.3-flash__fused_moe_kernel": ["fused_experts_impl", "fused_experts"],
    "glm-5.3-flash__elementwise_copy_cluster": ["materialize_bpreshuffle_fp8_scale"],
    "minimax-m3__decode_score_kernel": ["_decode_score_kernel", "flash_decode_with_topk_idx"],
    "minimax-m3__gqa_share_sparse_decode_kernel": [
        "_gqa_share_sparse_decode_kernel", "flash_decode_with_gqa_share_sparse"],
    "minimax-m3__gqa_share_sparse_fwd_kernel": [
        "_gqa_share_sparse_fwd_kernel", "flash_prefill_with_gqa_share_sparse"],
    "glm-5.2-mxfp4__dsa_mla_core_prefill": ["tilelang_sparse_fwd", "sparse_attention_fwd_kernel_v1"],
    "glm-5.2-mxfp4__dsa_decode": ["tilelang_sparse_fwd", "sparse_mla_fwd_decode_partial"],
    "glm-5.2-mxfp4__fused_moe_mxfp4_flydsl": ["fused_moe_"],
}


def load_manifest():
    with open(os.path.join(HERE, "manifest.json")) as fh:
        man = json.load(fh)
    for key, path in man["roots"].items():
        if not os.path.isdir(path):
            raise SystemExit(f"manifest root {key!r} -> {path} does not exist or is not readable")
    return man


def abspkg(roots, ref):
    root, _, rest = ref.partition("/")
    return os.path.join(roots[root], rest)


def yaml_list(items, indent="  "):
    return "\n".join(f"{indent}- {i}" for i in items)


def yaml_str(value):
    if value is None:
        return "null"
    s = str(value)
    if s == "" or any(c in s for c in ":#{}[],&*?|-<>=!%@`\"'\n"):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'
    return s


# --------------------------------------------------------------------------- copying
def excluded(name):
    return (name in SKIP_NAMES or name.endswith(SKIP_SUFFIX)
            or name.startswith(SKIP_PREFIX))


def copy_package(src_pkg, dst_ut, stats):
    """Copy the GEAK package, hardlinking the bulk and dropping junk and prior answers."""
    for root, dirs, files in os.walk(src_pkg):
        dirs[:] = [d for d in dirs if not excluded(d)]
        rel = os.path.relpath(root, src_pkg)
        if rel == ".":
            rel = ""
        if rel.split(os.sep)[0] == "kernel_src":
            continue                       # handled separately -> source/
        out_dir = os.path.join(dst_ut, rel) if rel else dst_ut
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            if excluded(name):
                continue
            s, d = os.path.join(root, name), os.path.join(out_dir, name)
            size = os.path.getsize(s)
            if size > HARDLINK_OVER:
                try:
                    os.link(s, d)
                    stats["hardlinked"] += 1
                    stats["hardlinked_bytes"] += size
                    continue
                except OSError:
                    # fs.protected_hardlinks blocks linking a file you neither own
                    # nor can write. Copying is correct but costs real disk, and
                    # silently spending gigabytes on a 95%-full export is not ok.
                    stats["link_failed"] += 1
                    stats["link_failed_bytes"] += size
            shutil.copy2(s, d)
            stats["copied"] += 1
            stats["copied_bytes"] += size


def stock_counterpart(src_pkg, rel):
    """The pristine pre-optimization version of a kernel_src file, if the package kept one.

    Six of the fifteen packages ship a kernel_src that a previous GEAK run already
    tuned. Seeding a benchmark from those means the task starts from someone else's
    answer and its speedup is not comparable to any other task in the suite, so the
    stock file wins whenever it exists.
    """
    cand = os.path.join(src_pkg, "baseline_ref", os.path.basename(rel) + ".orig")
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(src_pkg, "baseline_src", rel)
    if os.path.isfile(cand):
        return cand
    return None


def renamed_stock(src_pkg, rel, stock_ref):
    """The stock file a manifest row says this kernel_src file was derived from.

    The 2026-09-13 callable-UT packages rename the editable copy
    (``kernel_src/moe_candidate.py`` from ``baseline_ref/fused_moe.py.orig``) and
    add a handful of harness lines to it, so ``stock_counterpart``'s
    ``<basename>.orig`` convention misses and the file would be mislabelled a
    prior candidate. The manifest names the pairing explicitly.

    The shipped file stays the seed even so: the ``.orig`` lacks the UT's
    ``baseline_callable`` shim, and seeding from it would break the two-leg UT.
    This is used only to tell the truth about how far from stock it is.
    """
    ref = (stock_ref or {}).get(rel)
    if not ref:
        return None
    path = os.path.join(src_pkg, ref)
    return path if os.path.isfile(path) else None


def install_source(src_pkg, task_dir, suite_prior, stats, stock_ref=None):
    """kernel_src/ becomes source/ (real, writable) and ut/kernel_src/ symlinks back.

    source/ is seeded from the STOCK baseline wherever the package kept one; the
    tuned file the package shipped is preserved under <suite>/_prior_solutions/
    so it stays available to humans without being visible inside the task.
    """
    src_ks = os.path.join(src_pkg, "kernel_src")
    dst_src = os.path.join(task_dir, "source")
    dst_ks = os.path.join(task_dir, "ut", "kernel_src")
    os.makedirs(dst_src, exist_ok=True)
    os.makedirs(dst_ks, exist_ok=True)

    rels, seeds = [], {}
    for root, dirs, files in os.walk(src_ks):
        dirs[:] = [d for d in dirs if not excluded(d)]
        for name in files:
            if excluded(name):
                continue
            rel = os.path.relpath(os.path.join(root, name), src_ks)
            rels.append(rel)
            shipped = os.path.join(src_ks, rel)
            renamed = renamed_stock(src_pkg, rel, stock_ref)
            if renamed:
                # Stock upstream code plus the UT's own harness shim. Seed from
                # the shipped file (the shim is load-bearing) and record how many
                # lines separate it from the pristine runtime-image copy.
                import difflib
                a = open(renamed, encoding="utf-8", errors="ignore").read().splitlines()
                b = open(shipped, encoding="utf-8", errors="ignore").read().splitlines()
                delta = sum(1 for ln in difflib.unified_diff(a, b, n=0)
                            if ln[:1] in "+-" and not ln.startswith(("+++", "---")))
                seeds[rel] = "stock" if delta == 0 else "stock+harness-shim"
                stats["shim_lines"] += delta
                d = os.path.join(dst_src, rel)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(shipped, d)
                stats["copied"] += 1
                stats["copied_bytes"] += os.path.getsize(shipped)
                link = os.path.join(dst_ks, rel)
                os.makedirs(os.path.dirname(link), exist_ok=True)
                if os.path.lexists(link):
                    os.unlink(link)
                os.symlink(os.path.relpath(d, os.path.dirname(link)), link)
                continue

            stock = stock_counterpart(src_pkg, rel)

            if stock and open(stock, "rb").read() != open(shipped, "rb").read():
                seed, kind = stock, "stock"
                prior = os.path.join(suite_prior, rel)
                os.makedirs(os.path.dirname(prior), exist_ok=True)
                shutil.copy2(shipped, prior)
                stats["reset_to_stock"] += 1
            elif stock:
                seed, kind = stock, "stock"
            else:
                seed, kind = shipped, "candidate"
                stats["no_stock"] += 1
            seeds[rel] = kind

            d = os.path.join(dst_src, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(seed, d)
            stats["copied"] += 1
            stats["copied_bytes"] += os.path.getsize(seed)

            link = os.path.join(dst_ks, rel)
            os.makedirs(os.path.dirname(link), exist_ok=True)
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink(os.path.relpath(d, os.path.dirname(link)), link)
    return sorted(rels), seeds


def py_sources(task_dir, rels):
    """The source files the arena should show the optimizer: Python only, biggest first."""
    out = []
    for rel in rels:
        if rel.endswith(".py"):
            out.append((os.path.getsize(os.path.join(task_dir, "source", rel)), rel))
    out.sort(reverse=True)
    return [rel for _, rel in out]


def locate_targets(task_dir, rels, targets):
    """Map each target symbol to the source file that defines it."""
    where = {}
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        path = os.path.join(task_dir, "source", rel)
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in targets and node.name not in where:
                    where[node.name] = rel
    return where


RUN_ROOT_ESCAPE = "RUN_ROOT = os.path.dirname(os.path.dirname(HERE))"
RUN_ROOT_MARK = "# PATCHED BY tools/build_suite.py: RUN_ROOT was two levels above HERE,"
RUN_ROOT_CONTAINED = (
    RUN_ROOT_MARK + "\n"
    "# which is right in the delivery layout (HERE = <delivery>/tasks/<task>) but\n"
    "# escapes the task here (HERE = <task>/ut), dropping reports/ledger/<case>.json\n"
    "# into tasks/headkernel/ next to the sibling tasks on every run.\n"
    "RUN_ROOT = os.environ.get(\"HK_RUN_ROOT\", HERE)")


def contain_ut(task_dir, stats):
    """Keep the copied UT's writes inside the task directory.

    The arena scans tasks/headkernel/ for task directories and this suite's own
    README promises nothing points outside a task. The 2026-09-13 callable UTs
    compute their ledger destination two levels above themselves, which lands in
    the tasks root once the package is re-rooted at <task>/ut. Every other write
    they do is already relative to HERE.
    """
    ut_py = os.path.join(task_dir, "ut", "unittest.py")
    if not os.path.isfile(ut_py):
        return
    text = open(ut_py, encoding="utf-8").read()
    if RUN_ROOT_MARK in text:
        return                                   # already contained; idempotent
    if RUN_ROOT_ESCAPE not in text:
        # Either the package never escaped, or upstream respelled the line. The
        # second case would silently reintroduce writes into the tasks root, so
        # distinguish them rather than shrugging.
        if "RUN_ROOT" in text and "dirname" in text.split("RUN_ROOT", 1)[1][:120]:
            raise SystemExit(
                f"{os.path.basename(task_dir)}: ut/unittest.py computes RUN_ROOT in a spelling "
                f"build_suite.py does not recognise. Check whether it still escapes the task "
                f"directory and update RUN_ROOT_ESCAPE.")
        return
    open(ut_py, "w", encoding="utf-8").write(
        text.replace(RUN_ROOT_ESCAPE, RUN_ROOT_CONTAINED, 1))
    stats["contained"] += 1


def install_extra_files(task_dir, extra, roots, meta_for_hash, stats):
    """Pull in a file the UT needs that its package did not ship.

    The three MiniMax packages declare ``geometry_file: timing_geometry.pt`` in
    meta.json, record its sha256, and ``torch.load`` it unconditionally -- but
    none of them actually contains it. The authentic blob is in the original GEAK
    capture directory the package's own README names. Where meta.json records a
    hash, it is verified here, so a wrong or drifted file can never be silently
    grafted in.
    """
    import hashlib
    for name, ref in (extra or {}).items():
        src = abspkg(roots, ref)
        if not os.path.isfile(src):
            raise SystemExit(f"{os.path.basename(task_dir)}: extra_files names {ref} -> {src}, "
                             f"which does not exist")
        want = (meta_for_hash or {}).get(f"{os.path.splitext(name)[0]}_sha256")
        if want:
            got = hashlib.sha256(open(src, "rb").read()).hexdigest()
            if got != want:
                raise SystemExit(
                    f"{os.path.basename(task_dir)}: {name} sha256 mismatch.\n"
                    f"  ut/meta.json expects {want}\n  {src} is {got}\n"
                    f"Refusing to graft in a file the package did not describe.")
        dst = os.path.join(task_dir, "ut", name)
        size = os.path.getsize(src)
        if size > HARDLINK_OVER:
            try:
                os.link(src, dst)
                stats["hardlinked"] += 1
                stats["hardlinked_bytes"] += size
                stats["extra_files"] += 1
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
        stats["copied"] += 1
        stats["copied_bytes"] += size
        stats["extra_files"] += 1


def apply_ut_patches(task_dir, patches, stats):
    """Apply recorded, auditable fixes to the copied UT.

    ``ut/`` is frozen on principle -- it is the oracle -- so a patch here needs a
    higher bar than a code change anywhere else in this suite. Each one is
    declared in the manifest with an exact find/replace and a ``why``, is applied
    to the COPY only (the upstream package is never touched), is stamped into the
    file as a comment, and fails the build loudly if the text it expects is not
    there. Nothing is fuzzy-matched.
    """
    for p in patches or []:
        path = os.path.join(task_dir, "ut", p["file"])
        if not os.path.isfile(path):
            raise SystemExit(f"{os.path.basename(task_dir)}: ut_patches names {p['file']}, "
                             f"which does not exist in the package")
        text = open(path, encoding="utf-8").read()
        if p["replace"] in text and p["find"] not in text:
            continue                                   # already applied; idempotent
        if text.count(p["find"]) != 1:
            raise SystemExit(
                f"{os.path.basename(task_dir)}: ut_patches expects exactly one occurrence of\n"
                f"  {p['find']!r}\nin ut/{p['file']}, found {text.count(p['find'])}. The upstream "
                f"package changed - re-verify the patch before rebuilding.")
        # Pure replacement -- no inline stamp. These edits land inside expressions
        # (a dict literal in a list comprehension, in the one case that exists),
        # where an injected comment would be legal Python but unreadable. The
        # audit trail lives in config.yaml's headkernel.ut_patches block, in the
        # task README, and in MANIFEST.tsv.
        text = text.replace(p["find"], p["replace"], 1)
        open(path, "w", encoding="utf-8").write(text)
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise SystemExit(f"{os.path.basename(task_dir)}: ut_patches on {p['file']} produced "
                             f"invalid Python at line {exc.lineno}: {exc.msg}")
        stats["ut_patched"] += 1


def relativize_meta(task_dir):
    """Drop absolute pointers back at the delivery directory from ut/meta.json.

    ``baseline_overlay`` is recorded as an absolute path into someone else's
    read-only tree. Nothing dereferences it (harness_lib always recomputes
    ``<task>/baseline_overlay``), but a task that ships a stale pointer at
    /shared_nfs/<someone-else> is not self-contained in any honest sense.
    """
    path = os.path.join(task_dir, "ut", "meta.json")
    if not os.path.isfile(path):
        return 0
    try:
        meta = json.load(open(path))
    except Exception:
        return 0
    fixed = 0
    for key, val in list(meta.items()):
        if not isinstance(val, str):
            continue
        # An absolute pointer back into the delivery directory this package was
        # copied out of, e.g. "baseline_overlay": "/shared_nfs/<someone>/.../baseline_overlay".
        if val.startswith("/") and os.path.basename(val) == key:
            meta[key] = key
            fixed += 1
        # The UT's own output destination, spelled relative to the delivery root,
        # e.g. "result_path": "../../reports/ledger/hk11.json". contain_ut() fixes
        # the same escape in the Python; this is the copy recorded in metadata.
        # Narrow on purpose: other "../" values are cross-references to a sibling
        # package (kimi-k3 stage 1 points at stage 2's oracle) and rewriting those
        # would turn a documented pointer into a dangling one.
        elif val.startswith("../") and "/reports/" in val:
            meta[key] = val.replace("../", "")
            fixed += 1
    if fixed:
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
    return fixed


# --------------------------------------------------------------------------- emit
def oracle_kind(task_dir, meta):
    """How this task decides whether a candidate is correct.

    Three kinds exist in the suite and they are not interchangeable, so neither
    config.yaml nor the README may describe one as another:
      * a frozen live-capture tensor blob (ut/reference_io.pt),
      * a deterministic baseline regenerated at run time from live shape/stride/
        dispatch evidence - value-independent ops (dense GEMM, RMSNorm) use this
        instead of freezing hundreds of MB,
      * whatever the package's own hand-written unittest does.
    """
    if os.path.isfile(os.path.join(task_dir, "ut", "reference_io.pt")):
        return "frozen live-capture ut/reference_io.pt"
    if meta.get("oracle_policy") and not meta.get("reference_io_sha256"):
        return "runtime frozen baseline (no persistent tensor oracle) + live shape/dispatch evidence"
    return "GEAK unittest built-in"


def seed_summary(seeds):
    kinds = set(seeds.values())
    if len(kinds) == 1:
        only = kinds.pop()
        return {"stock": "stock", "candidate": "prior-candidate",
                "stock+harness-shim": "stock+harness-shim"}[only]
    return "mixed"


def write_config(task_dir, task, rows, model_meta, pkg_ref, src_files, targets, where,
                 meta, seeds, stats):
    head = rows[0]
    tol = meta.get("tol")
    instructions = build_instructions(task, rows, model_meta, meta, targets, where)

    lines = []
    lines.append(f"# {task}")
    lines.append(f"# Head kernel from the 2026-09-14 info table, rows: "
                 f"{', '.join(r['row'] for r in rows)}")
    lines.append("")
    lines.append("source_file_path:")
    lines.append(yaml_list([f"source/{f}" for f in src_files]))
    lines.append("")
    lines.append("target_kernel_functions:")
    lines.append(yaml_list(targets))
    lines.append("")
    lines.append("compile_command:")
    lines.append("  - python3 scripts/task_runner.py compile")
    lines.append("")
    lines.append("correctness_command:")
    lines.append("  - python3 scripts/task_runner.py correctness")
    lines.append("")
    lines.append("performance_command:")
    lines.append("  - python3 scripts/task_runner.py performance")
    lines.append("")
    lines.append("task_type: triton2triton")
    lines.append("")
    lines.append("task_result_template: null")
    lines.append("")
    lines.append("prompt:")
    lines.append("  source_code: null")
    lines.append("  instructions: |")
    for ln in instructions.splitlines():
        lines.append(f"    {ln}" if ln else "")
    lines.append("  cheatsheet: null")
    lines.append("")
    lines.append("# ---------------------------------------------------------------------------")
    lines.append("# Provenance. Not read by the arena schema check; read by scripts/task_runner.py")
    lines.append("# and by anyone auditing where these numbers came from.")
    lines.append("headkernel:")
    lines.append(f"  model: {yaml_str(head['model'])}")
    lines.append(f"  owner: {yaml_str(model_meta.get('owner'))}")
    # The image is a property of the PACKAGE, not the model: the two Qwen3.8
    # deliveries were captured under different sglang builds, and run_on_gpu.sh
    # reads this line per task.
    lines.append(f"  docker: {yaml_str(head.get('docker') or model_meta.get('docker'))}")
    if head.get("ut_patches"):
        lines.append("  # Recorded fixes applied to the COPY of the frozen UT at build time.")
        lines.append("  # ut/ is the oracle and is frozen on principle, so each of these is an")
        lines.append("  # exact find/replace with a stated reason, and the build fails loudly if")
        lines.append("  # the upstream text it expects is not found. The upstream package itself")
        lines.append("  # is never modified. Report these to the package owner.")
        lines.append("  ut_patches:")
        for p in head["ut_patches"]:
            lines.append(f"    - file: {yaml_str('ut/' + p['file'])}")
            lines.append(f"      from: {yaml_str(p['find'])}")
            lines.append(f"      to: {yaml_str(p['replace'])}")
            lines.append(f"      why: {yaml_str(p['why'])}")
    if head.get("pre_run_patch"):
        lines.append("  # A framework patch the stock image does not carry. tools/run_on_gpu.sh")
        lines.append("  # applies it to /sgl-workspace/sglang inside the container before the run.")
        lines.append(f"  pre_run_patch: {yaml_str(head['pre_run_patch'])}")
    lines.append(f"  serving: {yaml_str(model_meta.get('serving'))}")
    lines.append(f"  backend: {yaml_str(head.get('backend'))}")
    lines.append(f"  regime: {yaml_str(head.get('regime'))}")
    lines.append(f"  device_symbol: {yaml_str(head.get('device_symbol'))}")
    lines.append(f"  target_callable: {yaml_str(head.get('target_callable'))}")
    lines.append(f"  tol: {yaml_str(tol)}")
    lines.append(f"  oracle: {yaml_str(oracle_kind(task_dir, meta))}")
    lines.append(f"  source_package: {yaml_str(pkg_ref)}")
    lines.append(f"  source_seed: {yaml_str(seed_summary(seeds))}")
    if head.get("preserve_symbols"):
        lines.append("  # Symbols the UT resolves its frozen baseline through. They are not")
        lines.append("  # optimization targets, but deleting one turns a working kernel into an")
        lines.append("  # opaque correctness FAIL, so scripts/task_runner.py compile checks them.")
        lines.append("  preserve_symbols:")
        lines.append(yaml_list(head["preserve_symbols"], indent="    "))
    if head.get("preserve_text"):
        lines.append("  # Load-bearing lines that are NOT a def or a class, so the AST symbol check")
        lines.append("  # cannot see them. Checked as literal substrings by task_runner.py compile.")
        lines.append("  preserve_text:")
        lines.append(yaml_list(head["preserve_text"], indent="    "))
    if stats["shim_lines"]:
        lines.append(f"  # source/ is the stock runtime-image file plus {stats['shim_lines']} line(s) of "
                     f"UT harness shim (the two-leg")
        lines.append("  # UT resolves its frozen baseline through them). Nothing has been pre-optimized.")
    if stats["reset_to_stock"]:
        lines.append(f"  # {stats['reset_to_stock']} file(s) shipped pre-tuned by a previous GEAK run "
                     f"and were reset to baseline_ref/*.orig; the tuned version is kept at")
        lines.append(f"  # _prior_solutions/{task}/")
    if stats["no_stock"]:
        lines.append(f"  # {stats['no_stock']} file(s) have no stock counterpart - they ARE the "
                     f"candidate seam (a new module the seam rebinds to), so the")
        lines.append("  # starting point is a prior candidate, not upstream code. Baseline speedup "
                     "is measured against the live stack, not against this file.")
    lines.append("  info_rows:")
    for r in rows:
        lines.append(f"    - row: {yaml_str(r['row'])}")
        lines.append(f"      info_kernel: {yaml_str(r['info_kernel'])}")
        lines.append(f"      gpu_pct: {yaml_str(r.get('gpu_pct'))}")
        lines.append(f"      empirical_roofline: {yaml_str(r.get('empirical_roofline'))}")
        lines.append(f"      optimized_roofline: {yaml_str(r.get('optimized_roofline'))}")
        lines.append(f"      e2e_uplift: {yaml_str(r.get('e2e_uplift'))}")
    lines.append("  symbol_locations:")
    for sym in targets:
        lines.append(f"    {sym}: {yaml_str('source/' + where.get(sym, '?'))}")
    lines.append("")

    with open(os.path.join(task_dir, "config.yaml"), "w") as fh:
        fh.write("\n".join(lines))


def build_instructions(task, rows, model_meta, meta, targets, where):
    head = rows[0]
    files = sorted({where[s] for s in targets if s in where})
    body = []
    body.append(f"Optimize the {head['model']} head kernel "
                f"`{head['info_kernel']}` ({head.get('backend')}) for maximum GPU")
    body.append("throughput on MI355X (gfx950) while keeping it numerically correct.")
    body.append("")
    body.append(f"It is {head.get('gpu_pct')}% of de-inflated GPU time in the serving profile")
    body.append(f"({model_meta.get('serving')}), currently at "
                f"{head.get('empirical_roofline')} of its empirical roofline.")
    if len(rows) > 1:
        body.append("")
        body.append("The profiler resolves several device symbols to this one seam, so a change here")
        body.append("moves all of them:")
        for r in rows:
            body.append(f"  - {r['info_kernel']}  ({r.get('gpu_pct')}% GPU, "
                        f"roofline {r.get('empirical_roofline')})")
    body.append("")
    body.append("Edit only:")
    for f in files:
        body.append(f"  source/{f}")
    body.append("")
    body.append("Entry points that must keep their signatures:")
    for s in targets:
        body.append(f"  {s}")
    body.append("")
    if head.get("preserve_symbols"):
        body.append("Do NOT remove these - they are not optimization targets, but the two-leg unit")
        body.append("test resolves its frozen baseline through them, and deleting one turns a")
        body.append("working kernel into an unexplained correctness failure:")
        for s in head["preserve_symbols"]:
            body.append(f"  {s}")
        body.append("")
    body.append(f"The production seam is `{head.get('target_callable')}`; correctness is judged")
    if meta.get("reference_io_sha256") or meta.get("reference_io"):
        body.append("against a frozen live-server capture, not a re-derived reference, so the kernel")
    else:
        body.append("against a deterministic baseline regenerated from the live shapes, strides and")
        body.append("dispatch selections that were actually observed in the serving profile, so the kernel")
    body.append("must stay faithful to the deployed contract (dtypes, layouts, quantization).")
    if head.get("note"):
        body.append("")
        body.append(f"Note: {head['note']}")
    return "\n".join(body)


def write_task_readme(task_dir, task, rows, model_meta, pkg_ref, pkg_abs, src_files,
                      targets, where, meta, seeds, stats):
    head = rows[0]
    has_oracle = os.path.isfile(os.path.join(task_dir, "ut", "reference_io.pt"))
    L = []
    L.append(f"# {task}")
    L.append("")
    L.append(f"**{head['model']}** head kernel - `{head['info_kernel']}` "
             f"({head.get('backend')}, {head.get('regime')}).")
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    L.append(f"| GPU time share | {head.get('gpu_pct')}% |")
    L.append(f"| empirical roofline | {head.get('empirical_roofline')} |")
    L.append(f"| optimized roofline | {head.get('optimized_roofline') or '-'} |")
    L.append(f"| e2e uplift measured | {head.get('e2e_uplift') or '-'} |")
    L.append(f"| device symbol | `{head.get('device_symbol')}` |")
    L.append(f"| production seam | `{head.get('target_callable')}` |")
    L.append(f"| serving contract | {model_meta.get('serving')} |")
    L.append(f"| image | `{head.get('docker') or model_meta.get('docker')}` |")
    L.append(f"| owner | {model_meta.get('owner')} |")
    L.append(f"| info rows | {', '.join(r['row'] for r in rows)} |")
    L.append("")
    if len(rows) > 1:
        L.append("## One seam, several device symbols")
        L.append("")
        L.append("The info table lists these separately, but the profiler resolves all of them to")
        L.append(f"`{head.get('target_callable')}`. They are one optimization target, not several:")
        L.append("")
        for r in rows:
            L.append(f"- **{r['info_kernel']}** - {r.get('gpu_pct')}% GPU, "
                     f"roofline {r.get('empirical_roofline')}"
                     + (f" -> {r['optimized_roofline']}" if r.get("optimized_roofline") else ""))
        L.append("")
    L.append("## Layout")
    L.append("")
    L.append("```")
    L.append("config.yaml              arena task schema + a headkernel: provenance block")
    L.append("scripts/task_runner.py   compile | correctness | performance")
    L.append("scripts/_bench.py        native 10 warmup / 100 measured timing")
    L.append("source/                  THE EDITABLE KERNEL - change only this")
    L.append("ut/                      frozen GEAK op package (oracle, harness, overlays)")
    L.append("ut/kernel_src/           symlinks back into source/ - same bytes, two views")
    L.append("```")
    L.append("")
    L.append("Edit targets:")
    L.append("")
    for sym in targets:
        L.append(f"- `{sym}` in `source/{where.get(sym, '?')}`")
    L.append("")
    L.append("## Running it")
    L.append("")
    L.append("On one GPU from the optimization pool (never the serving set), inside the image above:")
    L.append("")
    L.append("```bash")
    L.append("cd <task>")
    L.append("python3 scripts/task_runner.py compile")
    L.append("python3 scripts/task_runner.py correctness")
    L.append("python3 scripts/task_runner.py performance")
    L.append("```")
    L.append("")
    L.append("- **compile** AST-parses `source/` and asserts every target symbol is defined there.")
    L.append("  No GPU needed.")
    if has_oracle:
        L.append("- **correctness** runs `ut/unittest.py`: the frozen live-capture oracle")
        L.append(f"  (`ut/reference_io.pt`) plus random-value parity against the live baseline leg,")
        L.append(f"  at tol `{meta.get('tol')}`. Exit 0 pass, 1 correctness fail, 2 environment,")
        L.append("  3 harness incomplete.")
    elif meta.get("oracle_policy"):
        L.append("- **correctness** runs `ut/unittest.py`. This op is value-independent, so instead")
        L.append("  of freezing hundreds of MB of tensors the UT regenerates a deterministic")
        L.append(f"  baseline at run time at tol `{meta.get('tol')}`, over the live shapes, strides")
        L.append("  and dispatch selections recorded in `ut/meta.json`. Its own gates are a")
        L.append("  deliberate output corruption that must be rejected (`ut/negative_check.json`)")
        L.append("  and an identity check that the two legs resolve to different code")
        L.append("  (`ut/selection_validation.json`).")
    else:
        L.append("- **correctness** runs `ut/unittest.py`. This package has no frozen")
        L.append("  `reference_io.pt`; read its own README for what the oracle actually is.")
    if has_oracle:
        L.append("- **performance** replays the captured argument records from `ut/reference_io.pt`")
        L.append("  with 10 warmup + 100 measured iterations and reports the mean cuda-event device")
        L.append("  time. If a record cannot be rebuilt it falls back to the GEAK interleaved")
        L.append("  median-of-3 legs and says so in `build/performance_report.json`.")
    else:
        L.append("- **performance** builds this op's live geometries through the package's own")
        L.append("  `ut/cases.py` - there is no frozen blob to replay - and times them with the same")
        L.append("  10 warmup + 100 measured cuda-event methodology. It falls back to the GEAK")
        L.append("  interleaved median-of-3 legs only if that entry point is missing, and says which")
        L.append("  it used in `build/performance_report.json`.")
    L.append("  A run whose unit test did not pass reports no cases at all.")
    L.append("")
    if head.get("preserve_symbols"):
        L.append("Must survive in `source/` (not optimization targets - the two-leg UT resolves its")
        L.append("frozen baseline through them, and `task_runner.py compile` fails if one is gone):")
        L.append("")
        for sym in head["preserve_symbols"]:
            L.append(f"- `{sym}`")
        L.append("")
    L.append("## Starting point")
    L.append("")
    if stats["shim_lines"]:
        L.append("`source/` is the **stock** file from the pinned runtime image plus "
                 f"{stats['shim_lines']} line(s)")
        L.append("of unit-test harness shim (an `importlib` import and a `baseline_callable()`")
        L.append("accessor, so the candidate overlay can still reach the unshadowed production")
        L.append("function). Nothing has been pre-optimized - the first measured speedup on this")
        L.append("task is by construction a null run, and small deviations from 1.00x are")
        L.append("timing-slot bias rather than optimization.")
    elif stats["reset_to_stock"]:
        L.append(f"`source/` is seeded from the **stock** pre-optimization code")
        L.append(f"(`baseline_ref/*.orig` in the upstream package). The package itself shipped a")
        L.append(f"kernel_src that a previous GEAK run had already tuned "
                 f"({stats['reset_to_stock']} file(s)); that version is kept out of the task at")
        L.append(f"`_prior_solutions/{task}/` so this benchmark starts where every other task in")
        L.append("the suite starts.")
    elif stats["no_stock"]:
        L.append("This op is optimized by **rebinding the seam to a new module**, so the file in")
        L.append("`source/` has no upstream counterpart - it is itself a candidate implementation")
        L.append("rather than stock library code. Speedup is still measured against the live")
        L.append("stack (the baseline leg resolves outside the task dir), but be aware the")
        L.append("starting point already encodes design choices from the capture.")
    else:
        L.append("`source/` is the **stock** upstream code, byte-identical to")
        L.append("`ut/baseline_ref/*.orig`. Nothing has been pre-optimized.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"Copied {os.path.basename(pkg_abs)} from `{os.path.dirname(pkg_abs)}` on 2026-09-14.")
    L.append("Prior optimization results (`_candidate_best/`, `accepted_overlay/`, tuning sweeps,")
    L.append("patches) were deliberately **not** copied - a benchmark that ships the answer")
    L.append("measures nothing. They remain in the upstream package.")
    if stats["hardlinked"]:
        L.append(f"Oracle blobs are hardlinked, not duplicated ({stats['hardlinked']} file(s), "
                 f"{stats['hardlinked_bytes'] / 1e9:.2f} GB shared with the source package).")
    if stats["link_failed"]:
        L.append(f"**{stats['link_failed']} oracle blob(s) ({stats['link_failed_bytes'] / 1e9:.2f} GB) "
                 f"are real copies, not hardlinks** - the upstream file belongs to another user and")
        L.append("`fs.protected_hardlinks` forbids linking it. Run `tools/relink_oracles.py` as root")
        L.append("to convert them back and reclaim the space.")
    if not (stats["hardlinked"] or stats["link_failed"]):
        L.append("This package has no oracle blob large enough to hardlink; everything was copied.")
    if os.path.isfile(os.path.join(task_dir, "ut", "README.md")):
        L.append("The original package README is preserved at `ut/README.md` and is the authority on")
        L.append("this op's measurement caveats - read it before trusting a speedup.")
    else:
        L.append("This package shipped no README. `ut/meta.json` (oracle policy, tolerance, case")
        L.append("geometries), `ut/selection_validation.json` (proof the two legs resolve to")
        L.append("different code) and `ut/negative_check.json` (proof a corrupted output is")
        L.append("rejected) are what it records instead - read those before trusting a speedup.")
    if head.get("note"):
        L.append("")
        L.append(f"**Note.** {head['note']}")
    L.append("")
    with open(os.path.join(task_dir, "README.md"), "w") as fh:
        fh.write("\n".join(L))


def write_notbuilt(task_dir, task, rows, model_meta):
    head = rows[0]
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, "NOT_BUILT"), "w") as fh:
        fh.write(head.get("gap_reason", "") + "\n")
    L = []
    L.append(f"# {task} - NOT_BUILT")
    L.append("")
    L.append(f"**{head['model']}** - `{head['info_kernel']}` ({head.get('backend')}, "
             f"{head.get('gpu_pct')}% GPU).")
    L.append("")
    L.append("This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena")
    L.append("task scanner will not pick it up and it cannot be run or scored.")
    L.append("")
    L.append("## Why")
    L.append("")
    L.append(head.get("gap_reason", "(no reason recorded)"))
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    L.append(f"| GPU time share | {head.get('gpu_pct')}% |")
    L.append(f"| empirical roofline | {head.get('empirical_roofline')} |")
    L.append(f"| optimized roofline | {head.get('optimized_roofline') or '-'} |")
    L.append(f"| e2e uplift measured | {head.get('e2e_uplift') or '-'} |")
    L.append(f"| device symbol | `{head.get('device_symbol')}` |")
    L.append(f"| production seam | `{head.get('target_callable') or '-'}` |")
    L.append(f"| info rows | {', '.join(r['row'] for r in rows)} |")
    if head.get("pkg"):
        L.append(f"| upstream UT package | `{head['pkg']}` |")
    if head.get("seed_pkg"):
        L.append(f"| editable seed available in | `{head['seed_pkg']}` |")
    L.append("")
    L.append("## To promote it into the suite")
    L.append("")
    L.append("1. Get an editable implementation of the seam into `source/`. Read the Why above")
    L.append("   first - it says whether the source has to be written (the profiled symbol is a")
    L.append("   prebuilt vendor artifact with no Python behind it) or merely vendored (the")
    L.append("   package shipped an empty `kernel_src/` but the Triton source exists upstream,")
    L.append("   and for the `aiter.tuned_gemm` rows a stock copy already ships in this suite at")
    L.append("   `tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/source/`).")
    L.append("2. Add `candidate_bind` to the package `meta.json` so the candidate leg actually")
    L.append("   shadows the production callable - without it both legs resolve to the same code")
    L.append("   and any measured speedup is noise. For the `aiter.tuned_gemm` rows note that a")
    L.append("   bare `setattr` on the module is a DEAD rebind: `solMap` is built at import time")
    L.append("   holding direct function objects, so the dispatcher keeps calling the original.")
    L.append("3. Re-capture parity against the live server and confirm")
    L.append("   `selection_validation.ok == true`.")
    L.append("4. Re-run `tools/build_suite.py`; flip `built` to true in `tools/manifest.json`.")
    L.append("")
    with open(os.path.join(task_dir, "README.md"), "w") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- driver
def build_task(task, rows, man, force):
    head = rows[0]
    task_dir = os.path.join(TASKS, task)
    built = head.get("built", head.get("tier") == 1)

    if os.path.isdir(task_dir):
        if not force:
            print(f"  skip (exists): {task}")
            return None
        shutil.rmtree(task_dir)

    model_meta = man["models"][head["model"]]

    if not built:
        write_notbuilt(task_dir, task, rows, model_meta)
        print(f"  NOT_BUILT   {task}")
        return {"task": task, "built": False, "rows": [r["row"] for r in rows]}

    pkg_abs = abspkg(man["roots"], head["pkg"])
    if not os.path.isdir(pkg_abs):
        raise SystemExit(f"{task}: source package {head['pkg']} -> {pkg_abs} does not exist")
    stats = {"copied": 0, "copied_bytes": 0, "hardlinked": 0, "hardlinked_bytes": 0,
             "reset_to_stock": 0, "no_stock": 0, "shim_lines": 0,
             "link_failed": 0, "link_failed_bytes": 0, "contained": 0, "ut_patched": 0, "extra_files": 0}
    suite_prior = os.path.join(SUITE, "_prior_solutions", task)

    os.makedirs(os.path.join(task_dir, "scripts"), exist_ok=True)
    copy_package(pkg_abs, os.path.join(task_dir, "ut"), stats)
    rels, seeds = install_source(pkg_abs, task_dir, suite_prior, stats,
                                 head.get("stock_ref"))
    if not rels:
        raise SystemExit(f"{task}: {head['pkg']}/kernel_src is empty - there is nothing "
                         f"for an optimizer to edit, so this cannot be a task")
    contain_ut(task_dir, stats)
    _m = os.path.join(task_dir, "ut", "meta.json")
    install_extra_files(task_dir, head.get("extra_files"), man["roots"],
                        json.load(open(_m)) if os.path.isfile(_m) else {}, stats)
    apply_ut_patches(task_dir, head.get("ut_patches"), stats)
    stats["meta_relativized"] = relativize_meta(task_dir)

    for tpl in ("task_runner.py", "_bench.py"):
        shutil.copy2(os.path.join(TEMPLATES, tpl),
                     os.path.join(task_dir, "scripts", tpl))

    meta_path = os.path.join(task_dir, "ut", "meta.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}

    targets = head.get("targets") or TARGETS.get(task)
    if not targets:
        raise SystemExit(f"{task}: no target symbols - add a 'targets' list to the "
                         f"manifest row or an entry to TARGETS in build_suite.py")
    where = locate_targets(task_dir, rels, set(targets))
    missing = [t for t in targets if t not in where]
    if missing:
        raise SystemExit(f"{task}: target symbols not found in source/: {missing}")

    src_files = py_sources(task_dir, rels)
    # Keep the declared source list focused on the files that define the targets,
    # plus anything else large enough to matter, capped so config stays readable.
    primary = sorted({where[t] for t in targets})
    rest = [f for f in src_files if f not in primary][:8]
    declared = primary + rest

    write_config(task_dir, task, rows, model_meta, head["pkg"], declared, targets, where,
                 meta, seeds, stats)
    write_task_readme(task_dir, task, rows, model_meta, head["pkg"], pkg_abs,
                      declared, targets, where, meta, seeds, stats)

    note = ""
    if stats["reset_to_stock"]:
        note = f", {stats['reset_to_stock']} file(s) reset to stock"
    if stats["no_stock"]:
        note += f", {stats['no_stock']} file(s) have no stock counterpart"
    if stats["shim_lines"]:
        note += f", stock + {stats['shim_lines']} harness-shim line(s)"
    if stats["contained"]:
        note += ", ut/unittest.py RUN_ROOT contained"
    if stats["ut_patched"]:
        note += f", {stats['ut_patched']} recorded ut patch(es) applied"
    if stats["extra_files"]:
        note += f", {stats['extra_files']} sha-verified extra file(s) vendored in"
    print(f"  built       {task}  "
          f"(source {len(rels)} file(s), copied {stats['copied_bytes']/1e6:.1f} MB, "
          f"hardlinked {stats['hardlinked_bytes']/1e9:.2f} GB{note})")
    if stats["link_failed"]:
        print(f"    WARNING: {stats['link_failed']} oracle blob(s) "
              f"({stats['link_failed_bytes']/1e9:.2f} GB) were COPIED, not hardlinked - "
              f"you do not own them and fs.protected_hardlinks is on.")
        print(f"             Recover the space with: sudo python3 tools/relink_oracles.py")
    return {"task": task, "built": True, "rows": [r["row"] for r in rows], **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild tasks that already exist")
    ap.add_argument("--only", help="build just this task id")
    args = ap.parse_args()

    man = load_manifest()
    os.makedirs(TASKS, exist_ok=True)

    grouped = {}
    for row in man["rows"]:
        if not row.get("task"):
            continue
        grouped.setdefault(row["task"], []).append(row)

    print(f"{len(man['rows'])} info rows -> {len(grouped)} task directories")
    results = []
    for task, rows in sorted(grouped.items()):
        if args.only and task != args.only:
            continue
        r = build_task(task, rows, man, args.force)
        if r:
            results.append(r)

    built = [r for r in results if r["built"]]
    print(f"\n{len(built)} built, {len(results) - len(built)} NOT_BUILT")


if __name__ == "__main__":
    main()
