# Python task interfaces

Python may describe an existing implementation, a production-library wrapper,
or an independent reference. Read the task's candidate declaration to distinguish
the initial language from the required final language. A Python starting point
does not change a Triton, HIP, or FlyDSL implementation requirement.

## Preserve the interface

Follow the declared entrypoint and parameter names. Keep keyword arguments,
scalar parameters, optional inputs, and return containers consistent with the
task contract. For multiple outputs, preserve names and order; do not omit an
auxiliary output merely because the main tensor appears correct. Some interfaces
write into supplied output buffers instead of allocating a return value.

Tensor shape, dtype, device, strides, storage encoding, and aliasing are part of
the interface. Packed values and quantization scales must be interpreted using
the declared layout. Do not reinterpret raw storage as a logical tensor without
the specified conversion. Allocate scratch and outputs on the input device and
preserve the caller's stream semantics. Avoid modifying inputs unless the task
explicitly defines them as mutable state or destination buffers.

## Keep implementation and validation separate

The baseline measures the starting implementation; the reference defines the
expected numerical behavior. A baseline can fail its independent reference.
Keep that failure visible and investigate it without changing the reference,
comparison thresholds, input distribution, or case set.

Only edit declared candidate paths and symbols. References, initializers,
comparators, workloads, and harness files remain protected. Imports permitted in
a production baseline are not automatically permitted in the final candidate.
Follow the task's dependency policy and implement its requested GPU computation;
do not return reference results or delegate to a prohibited baseline operator.

## Check runtime behavior

Syntax checks and successful imports do not establish GPU correctness. Lazy
compilation must be exercised on the declared cases. Use the task-owned commands
for compilation, correctness, and performance, and preserve their failure exits.
Do not suppress exceptions or turn missing dependencies into successful results.

Runtime tensor values can affect behavior even when shapes stay fixed. Respect
valid lengths, padding, masks, indices, and changing state on every invocation.
Cache only computations the contract permits; never reuse an output from a
previous input. Scalar extraction, host copies, and Python dispatch can change
the measured execution boundary, so keep equivalent work for both roles and use
the task's supplied timing and replay checks. The shared evaluator owns scores
and final validation reports.
