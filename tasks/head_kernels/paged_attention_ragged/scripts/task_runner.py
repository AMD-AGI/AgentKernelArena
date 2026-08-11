#!/usr/bin/env python3
"""Auto-generated task runner for aiter_paged_attention_ragged (AITER cpp_itfs/jinja).

The kernel source is a jinja-rendered HIP file under
``src/aiter_meta/csrc/cpp_itfs/csrc/cpp_itfs/pa``.  A Python entry function
in the same directory calls ``compile_template_op`` (from cpp_itfs/utils.py)
which renders the template, builds a shared library, and dispatches via
ctypes. Compilation results are cached under ``$AITER_ROOT_DIR/build``
(default ``$HOME/.aiter/build``).

We make the in-task ``aiter_meta/`` importable as the ``csrc.cpp_itfs.*``
namespace by inserting it on sys.path before importing the entry function.
No prebuilt .so is shipped; first ``compile`` invocation pays the build
cost.
"""
import sys, os, json, argparse, importlib, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _runtime as rt

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "aiter_paged_attention_ragged"
PY_FN_NAME = "paged_attention_ragged"            # entry function (e.g. paged_attention_ragged)
FC_NAME = "paged_attention_ragged"                  # mirrors PY_FN_NAME for cpp_itfs
MD_NAME = "pa_ragged"              # MD_NAME constant inside the cpp_itfs .py
PY_MODULE = "csrc.cpp_itfs.pa.pa_ragged"         # e.g. csrc.cpp_itfs.pa.pa_ragged
REF_SOURCE = "aiter"
TEST_CASES = os.path.join(TASK_DIR, "test_cases.json")

SRC_AITER_META = os.path.join(TASK_DIR, "src", "aiter_meta")
CK_INCLUDE = os.path.join(SRC_AITER_META, "3rdparty", "composable_kernel")
# Direct cpp_itfs/utils.py at the in-task copies of headers and includes.
os.environ.setdefault("CK_DIR", CK_INCLUDE)
# Keep the JIT build cache inside the task (isolated per-task, cleared by
# ``make clean``) instead of the shared global ``$HOME/.aiter``.
os.environ.setdefault("AITER_ROOT_DIR", os.path.join(TASK_DIR, "build", ".aiter"))

# Files whose contents the rendered kernel depends on. The cpp_itfs cache key
# (get_default_func_name) only hashes launch kwargs, NOT source contents, and
# not_built() only checks that lib.so exists -- so an edited kernel would
# silently reuse a stale binary. Drop any cached build older than these.
_KERNEL_SRC_GLOBS = [
    os.path.join(SRC_AITER_META, "csrc", "cpp_itfs", "pa", "*"),
    os.path.join(SRC_AITER_META, "csrc", "cpp_itfs", "utils.h"),
]
def _newest_src_mtime():
    newest = 0.0
    for pattern in _KERNEL_SRC_GLOBS:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                newest = max(newest, os.path.getmtime(path))
    return newest


def _invalidate_stale_cache():
    build_dir = os.path.join(os.environ["AITER_ROOT_DIR"], "build")
    if not os.path.isdir(build_dir):
        return
    src_mtime = _newest_src_mtime()
    for folder in glob.glob(os.path.join(build_dir, f"{MD_NAME}_*")):
        lib = os.path.join(folder, "lib.so")
        if not os.path.exists(lib) or os.path.getmtime(lib) < src_mtime:
            shutil.rmtree(folder, ignore_errors=True)


def _prepare_paths():
    # Make ``csrc.cpp_itfs.*`` resolve from src/aiter_meta/.
    if SRC_AITER_META not in sys.path:
        sys.path.insert(0, SRC_AITER_META)
    # Some cpp_itfs modules import from ``aiter`` for type aliases (rare). Keep
    # the installed aiter on sys.path if present (the env we captured ran with
    # aiter installed). The task does NOT rebuild aiter — we only consume its
    # cpp_itfs subtree, which we already shipped.


def _resolve_fn():
    _prepare_paths()
    _invalidate_stale_cache()
    mod = importlib.import_module(PY_MODULE)
    return getattr(mod, PY_FN_NAME)


def _test_cases():
    if not os.path.isfile(TEST_CASES):
        return []
    with open(TEST_CASES) as f:
        return json.load(f)


def _prep_inputs(tc, seed=42):
    """Build + normalize inputs, then apply op-specific consistency repair so
    the kernel receives a well-formed launch (e.g. valid ragged paged-KV
    bookkeeping). Deterministic for a fixed seed."""
    args, kwargs = rt.build_inputs(tc, seed=seed)
    args, kwargs = rt.normalize_aiter_call(PY_FN_NAME, FC_NAME, args, kwargs)
    if PY_FN_NAME == "paged_attention_ragged" and hasattr(rt, "make_consistent_paged_attention_ragged"):
        rt.make_consistent_paged_attention_ragged(args, kwargs)
    return args, kwargs


def run_compile():
    try:
        fn = _resolve_fn()
    except Exception as e:
        return False, f"import {PY_MODULE}.{PY_FN_NAME}: {e}"
    cases = [c for c in _test_cases() if c.get("args_sig") or c.get("kwargs_sig")]
    if not cases:
        return True, "no recorded launch signatures (compile-only check)"
    # Trigger the on-demand jinja render + build by invoking once.
    tc = cases[0]
    try:
        args, kwargs = _prep_inputs(tc)
        fn(*args, **kwargs)
    except Exception as e:
        return False, f"first-call compile failed: {e}"
    return True, None


def run_correctness():
    import torch
    try:
        fn = _resolve_fn()
    except Exception as e:
        return False, str(e)
    # Skip perf_only cases: they are large HBM-streaming workloads for roofline,
    # and the pure-PyTorch ragged reference loops per-sequence in Python —
    # unusably slow at S=512/ctx=4096. Correctness stays on the captured cases.
    cases = [c for c in _test_cases()
             if (c.get("args_sig") or c.get("kwargs_sig")) and not c.get("perf_only")]
    if not cases:
        return True, "no recorded launch signatures (compile-only check)"
    ref = rt.reference_for(PY_FN_NAME, REF_SOURCE)
    for tc in cases:
        try:
            args1, kwargs1 = _prep_inputs(tc)
            pre = rt.snapshot(args1)
            ret1 = fn(*args1, **kwargs1)
            out1 = rt.detect_output(pre, args1, ret1)
            out1 = rt.normalize_aiter_output(PY_FN_NAME, out1)
            if out1 is None:
                continue
            if ref is not None:
                args_r, kwargs_r = _prep_inputs(tc)
                expected = ref(args_r, kwargs_r)
                err = rt.compare(out1, expected)
                if err:
                    return False, f"{tc['test_case_id']}: vs reference: {err}"
            else:
                args2, kwargs2 = _prep_inputs(tc)
                pre2 = rt.snapshot(args2)
                ret2 = fn(*args2, **kwargs2)
                out2 = rt.detect_output(pre2, args2, ret2)
                out2 = rt.normalize_aiter_output(PY_FN_NAME, out2)
                err = rt.compare(out1, out2)
                if err:
                    return False, f"{tc['test_case_id']}: non-deterministic: {err}"
        except Exception as e:
            return False, f"{tc['test_case_id']}: kernel raised {e}"
    return True, None


def run_performance():
    import torch
    try:
        fn = _resolve_fn()
    except Exception:
        return []
    # Default runs ALL cases (captured + perf_only). GEAK_PERF_LARGE=1 narrows
    # to ONLY perf_only cases for a clean single large-case timing run.
    _v = os.environ.get("GEAK_PERF_LARGE", "")
    _only_perf = _v not in ("", "0", "false", "False", "no")
    _allcases = [c for c in _test_cases() if c.get("args_sig") or c.get("kwargs_sig")]
    _cases = [c for c in _allcases if c.get("perf_only")] if _only_perf else _allcases
    out = []
    for tc in _cases:
        try:
            args, kwargs = _prep_inputs(tc)
            for _ in range(10):
                fn(*args, **kwargs)
            torch.cuda.synchronize()
            n_iter = 100
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
            for j in range(n_iter):
                starts[j].record(); fn(*args, **kwargs); ends[j].record()
            torch.cuda.synchronize()
            avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
            out.append({"test_case_id": tc["test_case_id"], "execution_time_ms": avg, "params": tc.get("params_repr", {})})
        except Exception as e:
            out.append({"test_case_id": tc["test_case_id"], "execution_time_ms": -1.0, "params": {"error": str(e)[:120]}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = ap.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err}, open(os.path.join(build_dir, "compile_report.json"), "w"))
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print("Error:", err)
        sys.exit(0 if ok else 1)
    if args.mode == "correctness":
        ok, err = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err}, open(os.path.join(build_dir, "correctness_report.json"), "w"))
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print("Error:", err)
        sys.exit(0 if ok else 1)
    cases = run_performance()
    json.dump({"test_cases": cases}, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
    for c in cases:
        print(f"Performance: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
