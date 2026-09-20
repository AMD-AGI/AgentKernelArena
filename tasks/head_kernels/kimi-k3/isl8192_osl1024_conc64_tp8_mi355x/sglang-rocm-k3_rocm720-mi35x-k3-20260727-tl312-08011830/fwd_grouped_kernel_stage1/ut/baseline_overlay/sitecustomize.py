# Auto-generated reversible overlay (e2e_workflow) -- LAZY variant.
# Drop this dir from PYTHONPATH to revert.
#
# WHY LAZY: PYTHONPATH is inherited by every python subprocess the server spawns.
# The stock EAGER sitecustomize imported the patched sglang module (and therefore
# torch + aiter + module_aiter_core.so) at interpreter startup in ALL of them.
# That deadlocked /usr/bin/rocm_agent_enumerator (itself a #!/usr/bin/env python3
# script), which is what flydsl's get_rocm_arch() shells out to; flydsl swallowed
# the timeout and silently fell back to the literal "gfx942", whose LDS cap
# (65536 vs gfx950's 163840) makes selection_filter() REJECT the shipped
# kimik3 row (64,7168,1536 -> flydsl t32x64x128 STAGES=4, 98304 B) and raise
# ValueError inside tgemm.mm -> TP worker death -> server abort.
# Observed twice: tuning/ab/post1/replica_001/attempt_{1,2}/server.log.
# A meta_path finder defers ALL work until something genuinely imports the
# target module, so an unrelated subprocess pays nothing.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")
try:
    with open(_MAN) as _fh:
        _m = json.load(_fh)
except Exception:
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}

_MODMAP = {}
for _e in _m.get("modules", []):
    try:
        _MODMAP[_e["module"]] = os.path.join(_HERE, _e["file"])
    except Exception:
        pass


class _OverlayFinder(object):
    """meta_path finder: serve the patched file for exactly the mapped dotted names."""

    def find_spec(self, fullname, path=None, target=None):
        _f = _MODMAP.get(fullname)
        if _f is None:
            return None
        try:
            _spec = importlib.util.spec_from_file_location(fullname, _f)
        except Exception as _ex:
            sys.stderr.write("[overlay] spec FAILED %s: %r\n" % (fullname, _ex))
            return None
        sys.stderr.write("[overlay] injected module %s <- %s\n" % (fullname, _f))
        return _spec

    # py3.4-style API kept for tools that still probe it
    def find_module(self, fullname, path=None):
        return None


if _MODMAP:
    sys.meta_path.insert(0, _OverlayFinder())
    sys.stderr.write("[overlay] lazy finder armed for %s\n" % ",".join(sorted(_MODMAP)))

# (b) attribute rebinds and (c) capture/marker hooks are still eager, because they
# monkeypatch an already-importable target. The manifest for this overlay has none;
# if any are added, revisit -- they will reintroduce the subprocess blast radius.
for _e in _m.get("rebinds", []) + _m.get("markers", []) + _m.get("captures", []):
    sys.stderr.write("[overlay] WARNING eager entry present, not handled by lazy sitecustomize: %r\n" % (_e,))
