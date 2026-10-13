"""Task-local legacy API fallback; never modifies the installed FlyDSL package."""
try:
    import flydsl.expr.buffer_ops as buffer_ops
except ModuleNotFoundError as exc:
    if exc.name != "flydsl.expr.buffer_ops":
        raise
    from . import buffer_ops
try:
    import flydsl.expr.vector as vector
except ModuleNotFoundError as exc:
    if exc.name != "flydsl.expr.vector":
        raise
    from . import vector
