# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
#
# LAZY variant. Everything below installs through a single `sys.meta_path` finder instead of running
# at interpreter startup. Two concrete failures forced this:
#   * EAGER MODULE INJECTION deadlocks on circular imports: exec'ing the patched
#     `...triton_utils.fused_moe` before `sglang` is imported publishes a HALF-BUILT module in
#     sys.modules, so the first `from ...fused_moe import fused_experts` performed during that exec
#     raises ImportError and the whole injection is silently skipped (the candidate leg then resolves
#     to the pristine install and `assert_legs_differ` fails).
#   * EAGER CAPTURE/MARKER INSTALL force-imports `sglang` in EVERY python process on this PYTHONPATH,
#     including the `rocm_agent_enumerator` helper that importing sglang itself spawns. That recursion
#     made TP worker startup ~40x slower and blew the 601s torch.distributed rendezvous.
# Deferring to real import time fixes both and is strictly more correct: the patched module is built
# in the natural import order, and hooks land at the END of the owning module's exec -- before any
# other module can `from <mod> import <attr>` an unwrapped alias.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
try:
    with open(_MAN) as _fh:
        _m = json.load(_fh)
except Exception as _e:
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}

# (a) patched submodules: serve the dotted name from the overlay's file when it is first imported.
_MODULES = {}
for _e in _m.get("modules", []):
    _MODULES[_e["module"]] = os.path.join(_HERE, _e["file"])

# (c)+(d) capture hooks then marker hooks, keyed by the module that owns the target. Captures go on
# FIRST so the capture wrapper is the innermost stand-in.
_PENDING = {}
for _e in _m.get("captures", []):
    _PENDING.setdefault(_e["target"].split(":")[0], {}).setdefault("captures", []).append(_e)
for _e in _m.get("markers", []):
    _PENDING.setdefault(_e["target"].split(":")[0], {}).setdefault("markers", []).append(_e)


def _apply_hooks(modname):
    todo = _PENDING.pop(modname, None)
    if not todo:
        return
    for _e in todo.get("captures", []):
        try:
            import capture_shapes
            capture_shapes.install(_e["target"], _e["out"], int(_e.get("max", 5)))
        except Exception as _ex:
            sys.stderr.write("[overlay] capture install FAILED %r: %r\n" % (_e, _ex))
    for _e in todo.get("markers", []):
        try:
            import seam_trace
            seam_trace.install(_e["target"])
        except Exception as _ex:
            sys.stderr.write("[overlay] seam marker install FAILED %r: %r\n" % (_e, _ex))


class _OverlayFinder(object):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _MODULES:
            spec = importlib.util.spec_from_file_location(fullname, _MODULES[fullname])
            if spec is None or spec.loader is None:
                return None
            sys.stderr.write("[overlay] injecting module %s <- %s\n" % (fullname, _MODULES[fullname]))
            return self._wrap(spec, fullname)
        if fullname in _PENDING:
            spec = self._delegate(fullname, path, target)
            if spec is None or getattr(spec, "loader", None) is None:
                return None
            return self._wrap(spec, fullname)
        return None

    def _delegate(self, fullname, path, target):
        """Let the NORMAL finders locate the real module; we only wrap its loader."""
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                return spec
        return None

    def _wrap(self, spec, fullname):
        loader = spec.loader
        if not hasattr(loader, "exec_module"):
            return None
        _real = loader.exec_module

        def exec_module(module, _real=_real, _name=fullname):
            _real(module)
            _apply_hooks(_name)

        try:
            loader.exec_module = exec_module
        except Exception:
            return None
        return spec

    def invalidate_caches(self):
        return None


if _MODULES or _PENDING:
    sys.meta_path.insert(0, _OverlayFinder())
    # Already-imported targets (rare) cannot go through the finder -- hook them now.
    for _name in list(_PENDING):
        if _name in sys.modules:
            _apply_hooks(_name)

# (b) rebind single attributes (monkeypatch). Deferred the same way when the owner is not loaded yet
# would need a third table; rebinds are used for tiny leaf helpers, so keep them eager but tolerant.
for _e in _m.get("rebinds", []):
    try:
        _modname, _attr = _e["target"].split(":")
        _t = importlib.import_module(_modname)
        _impl = importlib.import_module(_e["impl_module"])
        setattr(_t, _attr, getattr(_impl, _e["impl_attr"]))
        sys.stderr.write("[overlay] rebound %s -> %s.%s\n" % (_e["target"], _e["impl_module"], _e["impl_attr"]))
    except Exception as _ex:
        sys.stderr.write("[overlay] rebind FAILED %r: %r\n" % (_e, _ex))
