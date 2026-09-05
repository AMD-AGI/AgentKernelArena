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

template <bool Safe> __global__ void lds_exchange(int *output) {
  __shared__ int tile[128];
  const int lane = static_cast<int>(threadIdx.x);
  tile[lane] = lane;
  if constexpr (Safe)
    __syncthreads();
  output[lane] = tile[127 - lane];
}
int main(int argc, char **argv) {
  const bool racy = argc > 1 && std::strcmp(argv[1], "racy") == 0;
  int *output = nullptr;
  HIP_CHECK(hipMalloc(&output, 128 * sizeof(int)));
  if (racy)
    hipLaunchKernelGGL((lds_exchange<false>), dim3(1), dim3(128), 0, nullptr, output);
  else
    hipLaunchKernelGGL((lds_exchange<true>), dim3(1), dim3(128), 0, nullptr, output);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(output));
  std::puts(racy ? "RACY_RUN_COMPLETED" : "SAFE_RUN_COMPLETED");
  return 0;
}
