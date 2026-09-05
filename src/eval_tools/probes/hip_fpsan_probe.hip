#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <fpsan/fpsan.hpp>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t error = (expr);                                                  \
    if (error != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(error)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

using FpValue = fpsan::Value<float, fpsan::Semantics::Triton,
                             fpsan::Conversions::Implicit>;

__global__ void compare_expressions(unsigned *payloads, bool wrong) {
  FpValue a{1.25f};
  FpValue b{-4.5f};
  FpValue c{3.75f};
  const FpValue reference = (a + b) + c;
  const FpValue candidate = wrong ? ((a + b) + (c + FpValue{1.0f})) : (a + (b + c));
  payloads[0] = reference.fpsan_payload();
  payloads[1] = candidate.fpsan_payload();
}
int main(int argc, char **argv) {
  const bool wrong = argc > 1 && std::strcmp(argv[1], "wrong") == 0;
  unsigned *device_payloads = nullptr;
  unsigned host_payloads[2] = {};
  HIP_CHECK(hipMalloc(&device_payloads, sizeof(host_payloads)));
  hipLaunchKernelGGL(compare_expressions, dim3(1), dim3(1), 0, nullptr,
                     device_payloads, wrong);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host_payloads, device_payloads, sizeof(host_payloads),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_payloads));
  std::printf("AKA_FPSAN_RESULT {\"instrumented\":true,"
              "\"reference_digest\":\"%08x\","
              "\"candidate_digest\":\"%08x\"}\n",
              host_payloads[0], host_payloads[1]);
  return 0;
}
