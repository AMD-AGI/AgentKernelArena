#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t error = (expr);                                                  \
    if (error != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(error)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

__global__ void write_indices(int *data) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  data[index] = index;
}
int main(int argc, char **argv) {
  const bool out_of_bounds = argc > 1 && std::strcmp(argv[1], "oob") == 0;
  constexpr int allocation_elements = 128;
  const int launched_threads = out_of_bounds ? 256 : allocation_elements;
  int *device_data = nullptr;
  HIP_CHECK(hipMalloc(&device_data, allocation_elements * sizeof(int)));
  hipLaunchKernelGGL(write_indices, dim3(1), dim3(launched_threads), 0, nullptr, device_data);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(device_data));
  std::puts(out_of_bounds ? "OOB_RUN_COMPLETED" : "SAFE_RUN_COMPLETED");
  return 0;
}
