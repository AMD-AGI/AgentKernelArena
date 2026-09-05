#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t error = (expr);                                                  \
    if (error != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(error)); \
      return 2;                                                                 \
    }                                                                           \
  } while (0)

#ifdef AKA_CONSAN_RACY
constexpr int kProbeWidth = 128;
#else
constexpr int kProbeWidth = 64;
#endif

extern "C" __global__ void consan_probe_kernel(int *output) {
  volatile __shared__ int tile[kProbeWidth];
  const int lane = static_cast<int>(threadIdx.x);
  tile[lane] = lane;
#ifdef AKA_CONSAN_RACY
  output[lane] = tile[kProbeWidth - 1 - lane];
#else
  output[lane] = tile[lane];
#endif
}

int main() {
  int *device = nullptr;
  int host[kProbeWidth] = {};
  HIP_CHECK(hipMalloc(&device, sizeof(host)));
  hipLaunchKernelGGL(
      consan_probe_kernel, dim3(1), dim3(kProbeWidth), 0, nullptr, device);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device));
#ifndef AKA_CONSAN_RACY
  for (int lane = 0; lane < kProbeWidth; ++lane) {
    if (host[lane] != lane) {
      std::fprintf(stderr, "safe control mismatch at lane %d\n", lane);
      return 3;
    }
  }
  std::puts("CONSAN_SAFE_ORACLE_PASS");
#else
  std::puts("CONSAN_RACY_RUN_COMPLETED");
#endif
  return 0;
}
