ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena ROCm GPU ASan tool runtime" \
      org.opencontainers.image.base.digest="sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78" \
      org.opencontainers.image.version="rocm-7.2.0-asan"

COPY docker/eval-tools/gpu-asan/packages.sha256 /tmp/packages.sha256

RUN mkdir -p /tmp/gpu-asan-debs \
    && cd /tmp/gpu-asan-debs \
    && apt-get update \
    && apt-get download \
        rocm-core-asan=7.2.0.70200-43~22.04 \
        comgr-asan=3.0.0.70200-43~22.04 \
        hsa-rocr-asan=1.18.0.70200-43~22.04 \
        hip-runtime-amd-asan=7.2.26015.70200-43~22.04 \
    && sha256sum --check /tmp/packages.sha256 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ./*.deb \
    && rm -rf /tmp/gpu-asan-debs /tmp/packages.sha256 /var/lib/apt/lists/*

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    AKA_GPU_ASAN_RUNTIME_DIR=/opt/rocm-7.2.0/lib/asan \
    AKA_GPU_ASAN_HIP_RUNTIME=/opt/rocm-7.2.0/lib/asan/libamdhip64.so \
    AKA_GPU_ASAN_HOST_PRELOAD=/opt/rocm-7.2.0/lib/llvm/lib/clang/22/lib/linux/libclang_rt.asan-x86_64.so \
    AKA_GPU_ASAN_HOST_LIB_DIR=/opt/rocm-7.2.0/lib/llvm/lib/clang/22/lib/linux \
    AKA_GPU_ASAN_NORMAL_ROCM_LIB_DIR=/opt/rocm-7.2.0/lib
