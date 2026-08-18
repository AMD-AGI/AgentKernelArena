#include <hip/hip_runtime.h>

extern "C" __global__ void waitcheck_probe_kernel() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
#ifdef AKA_WAITCHECK_HAZARD
               "s_mov_b32 s5, s4"
#else
               "s_waitcnt lgkmcnt(0)\n\t"
               "s_mov_b32 s5, s4"
#endif
               :
               :
               : "s4", "s5", "memory");
}
