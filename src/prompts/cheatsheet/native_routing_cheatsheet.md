# Native routing policies

The optimization surface is a literal mapping in a declared Python function.
The protected task runner parses the policy as data and resolves each choice to
an existing implementation and configuration in the pinned runtime image.

Keep every declared case and choose only routes admitted by the task contract.
Preserve source provenance, tensor geometry, dtype, strides, scalar controls,
input generation, reference computation, tolerances, and negative controls.
Do not add imports, calls, expressions, decorators, or helper functions to the
policy. Do not edit the native implementations or evaluation harness.

Measure the selected implementations with the configured device timing method,
warmups, and complete ordered sample count. Preserve state restoration and
output checks around every measured invocation. An unsupported route must fail;
it must not silently fall back or remove a case. A policy spelling change that
resolves to the same native configuration is an unchanged-source control.
