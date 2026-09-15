# matrix_multiplication: native HIP task

Optimize only `source/kernel.hpp`. All task inputs, references,
CPU comparisons, wrappers, bindings, compiler flags, launch boundaries, and
benchmark helpers are protected. The initial implementation is present and
written in HIP. Arena freezes that initial candidate in a separate workspace
for every baseline action; final candidate actions always build and execute the
submitted implementation, with no baseline fallback.

The complete independent manifest in `workload.json` includes **5 cases**
from the original 5 shapes, including every measured operation/layout.
The original harness remains `scripts/task_runner.py`; its seed schedule,
numerical tolerances and output-contract checks are retained. Its independent numerical reference is implemented in the protected C++ host
program; the native benchmark also validates graph replay. The Python reference
check module performs structural checks only. Do not reduce these
cases or alter expected outputs to improve a score.

Compilation uses the actual original HIP compiler/extension build, including
both native verification and benchmark binaries where applicable. The original
warmup (10), sample count (100), CUDA-graph/event fallback policy, state reset,
and allocation/timing boundaries remain in the protected harness. Runtime
requirements are the selected ROCm image, a compatible GPU and HIP compiler,
and PyTorch for extension tasks. Generated performance helpers are supplied by
Arena and must not be edited.

Call `python3 scripts/evaluate.py` followed by `validate-task`, or by
`baseline|candidate` and `compile|correctness|performance`. Each action emits one
`ARENA_EVAL_RESULT=` envelope. Missing cases, compiler errors, numerical errors,
and unavailable runtime dependencies are failures, never implicit skips.

The GPU implementation was extracted verbatim into `source/kernel.hpp`. The
original C++ host harness remains protected and includes that header. Its launch
interface and constants are part of the fixed task contract; a Python symbol
scope is not used to protect C++ code.

## Previous task notes

# HIP-Basic Matrix Multiplication Example

## Description

This example showcases the multiplication of two dynamically sized two-dimensional matrices on the GPU ($\mathrm{A \cdot B=C}$). The sizes of the matrices can be provided on the command line, however the sizes must be multiples of the hard-coded block size, which is 16x16. This implementation is not aimed at best performance or best generality, although some optimizations, such as the utilization of shared memory, are in place.

### Application flow

1. Default values for dimensions of matrix $\mathrm{A}$ and the number of columns of matrix $\mathrm{B}$ are set.
2. Command line arguments are parsed (if any) and the matrix dimensions are updated. If the command line arguments do not match the specification, an error message is printed to the standard output and the program terminates with a non-zero exit code.
3. Host memory is allocated for the matrices $\mathrm{A}$, $\mathrm{B}$ and $\mathrm{C}$ (using `std::vector<float>`) and the elements of both $\mathrm{A}$ and $\mathrm{B}$ are set to two different constant values.
4. Device memory is allocated for all matrices and the elements of $\mathrm{A}$ and $\mathrm{B}$ are copied to the device.
5. The dimensions of the kernel grid is calculated based on the matrix dimensions. The matrix multiplication kernel is queued to the default stream.
6. The elements of the resulting matrix $\mathrm{C}$ are copied to the host and all device memory is freed.
7. The elements of $\mathrm{C}$ are compared with the expected result. The result of the comparison is printed to the standard output.

### Command line interface

- If no command line argument is provided, the default matrix sizes are used.

- Otherwise, exactly 3 arguments must be provided. All must be positive integers which are multiples of the block size (16). The order of the arguments is the following: rows of $\mathrm{A}$, columns of $\mathrm{A}$, columns of $\mathrm{B}$. Notice that rows of $\mathrm{B}$ cannot be specified, as it must match the columns of $\mathrm{A}$.

## Key APIs and Concepts

- The kernel implemented in this example performs a matrix multiplication over dynamically sized matrices. The value of $\mathrm{C}$ at row $i$ and column $j$ is calculated with the following formula (where $N$ equals to the columns of $\mathrm{A}$ and rows of $\mathrm{B}$):

$$c_{ij}=\sum_{k=1}^{N}a_{ik}b_{kj}$$

- The kernel is launched in a two-dimensional grid in which each thread is responsible for calculating a single element of the resulting matrix. The threads are organized into 16x16 blocks. Since each block is executed on a single compute unit of the GPU hardware, data can be exchanged between these threads via shared memory.

- The matrix multiplication is conducted in multiple steps, each step calculating the partial results of a submatrix of size 16x16 (the block size). The number of steps is the columns of $\mathrm{A}$ divided by the block size.

- For improved performance, in each step the threads first load the corresponding submatrices from both $\mathrm{A}$ and $\mathrm{B}$ to the shared memory. Thereby each thread has to perform only one global memory fetch instead of loading the full 16 item row from each submatrix.

- Between loading and using values to/from shared memory, a call to `__syncthreads` has to be invoked. This is to ensure that all threads have finished writing to the shared memory before other threads might use the same memory locations.

  - The reason behind this is that it is not guaranteed that all threads in the block execute concurrently. Indeed, the compute unit schedules the threads to execute in so called "wavefronts". While one wavefront is waiting for memory operations to complete, another one might get scheduled to execute. The call to `__syncthreads` ensures that all threads in the block finish the pending memory operations and the loaded memory can safely be used from any other thread.

## Used API surface

### HIP runtime

#### Device symbols

- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- `__shared__`
- `__syncthreads`

#### Host symbols

- `hipMalloc`
- `hipMemcpy`
- `hipGetLastError`
- `hipFree`

The original constant-input correctness gate remains unchanged. Correctness also
runs the benchmark's original nonuniform inputs against a full CPU product for
all five shapes. The same full-output reference checks the timed replay after
poisoning output storage. Its original scaled tolerance is unchanged (0.002 times
max(1, abs(expected))); CPU reference construction and validation are outside the
timed samples. Repeated CPU dot products are reused only after comparing actual
rows/columns for equality. Warmups, sample counts and GPU timing are unchanged.
