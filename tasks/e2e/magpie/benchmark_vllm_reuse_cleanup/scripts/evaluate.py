"""The Docker runner prepares this recipe before any task action executes."""
raise RuntimeError(
    "This serving recipe has no prepared runtime. Select it with the normal Docker "
    "runner and declare its immutable binding in serving_runtime.bindings."
)
