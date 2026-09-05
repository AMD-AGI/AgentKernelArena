#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

extern char **environ;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t error = (expr);                                                  \
    if (error != hipSuccess) {                                                  \
      std::fprintf(stderr, "%s failed: %s\n", #expr, hipGetErrorString(error)); \
      return 2;                                                                 \
    }                                                                           \
  } while (0)

bool has_environment_prefix(const char *prefix) {
  const std::size_t length = std::strlen(prefix);
  for (char **entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    if (std::strncmp(*entry, prefix, length) == 0) {
      return true;
    }
  }
  return false;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s CODE_OBJECT THREADS instrumented|oracle\n",
                 argv[0]);
    return 2;
  }
  const int threads = std::atoi(argv[2]);
  if (threads != 64 && threads != 128) {
    std::fputs("thread count must be 64 or 128\n", stderr);
    return 2;
  }
  const bool oracle = std::strcmp(argv[3], "oracle") == 0;
  const bool instrumented = std::strcmp(argv[3], "instrumented") == 0;
  if (!oracle && !instrumented) {
    std::fputs("mode must be instrumented or oracle\n", stderr);
    return 2;
  }
  if (oracle &&
      (std::getenv("HSA_TOOLS_LIB") != nullptr ||
       std::getenv("HSA_TOOLS_DISABLE_REGISTER") != nullptr ||
       has_environment_prefix("RJ_CONSAN_"))) {
    std::fputs("oracle inherited ConSan instrumentation environment\n", stderr);
    return 3;
  }
  if (instrumented &&
      (std::getenv("HSA_TOOLS_LIB") == nullptr ||
       std::getenv("RJ_CONSAN_MODE") == nullptr)) {
    std::fputs("instrumented command is missing ConSan environment\n", stderr);
    return 3;
  }

  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  int *device = nullptr;
  int host[128] = {};
  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "consan_probe_kernel"));
  HIP_CHECK(hipMalloc(&device, static_cast<std::size_t>(threads) * sizeof(int)));
  HIP_CHECK(hipMemset(device, 0, static_cast<std::size_t>(threads) * sizeof(int)));
  void *arguments[] = {&device};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, threads, 1, 1, 0, nullptr,
                                  arguments, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(host, device,
                      static_cast<std::size_t>(threads) * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device));
  HIP_CHECK(hipModuleUnload(module));

  if (threads == 64) {
    for (int lane = 0; lane < threads; ++lane) {
      if (host[lane] != lane) {
        std::fprintf(stderr, "safe output mismatch at lane %d\n", lane);
        return 4;
      }
    }
  }
  if (oracle) {
    std::puts("CONSAN_ORACLE_ENV_CLEAN");
  } else {
    std::puts("CONSAN_INSTRUMENTED_LAUNCH_COMPLETED");
  }
  return 0;
}
