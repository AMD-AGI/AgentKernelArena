ARG BASE_IMAGE=lmsysorg/sglang:v0.5.20-rocm10-mi35x@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena ROCm GPU ASan tool runtime" \
      org.opencontainers.image.base.digest="sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69" \
      org.opencontainers.image.version="rocm-10.0.0-asan"

COPY docker/eval-tools/gpu-asan/packages.sha256 /tmp/packages.sha256

# Unpack checksum-locked ROCm 10 ASan packages into their own versioned prefix.
# Installing their metapackage would also mutate the base image's SDK selection.
RUN mkdir -p /tmp/gpu-asan-debs /tmp/gpu-asan-root \
    && cd /tmp/gpu-asan-debs \
    && while read -r checksum package; do \
        curl --fail --location --retry 3 -o "$package" \
          "https://stable.repo.amd.com/rocm/core/packages-asan/ubuntu2404/pool/main/$package"; \
       done < /tmp/packages.sha256 \
    && sha256sum --check /tmp/packages.sha256 \
    && for package in *.deb; do dpkg-deb --extract "$package" /tmp/gpu-asan-root; done \
    && mv /tmp/gpu-asan-root/opt/rocm/core-asan-10.0 /opt/rocm-asan-10.0 \
    && test -L /opt/rocm \
    && test -x /opt/rocm/bin/hipcc \
    && rm -rf /tmp/gpu-asan-debs /tmp/gpu-asan-root /tmp/packages.sha256

# PyTorch preloads these SDK libraries by absolute path. Point those paths at
# the instrumented copies too; LD_LIBRARY_PATH alone would load two LLVMs.
RUN /opt/venv/bin/python - <<'PY'
from pathlib import Path
import rocm_sdk

runtime = Path('/opt/rocm-asan-10.0/lib')
for name in ('amd_comgr', 'amdhip64'):
    target = runtime / f'lib{name}.so'
    assert target.is_file(), target
    for path in rocm_sdk.find_libraries(name):
        path.unlink()
        path.symlink_to(target)
PY

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    AKA_GPU_ASAN_RUNTIME_DIR=/opt/rocm-asan-10.0/lib \
    AKA_GPU_ASAN_HIP_RUNTIME=/opt/rocm-asan-10.0/lib/libamdhip64.so \
    AKA_GPU_ASAN_HOST_PRELOAD=/opt/rocm-asan-10.0/lib/llvm/lib/clang/23/lib/linux/libclang_rt.asan-x86_64.so \
    AKA_GPU_ASAN_HOST_LIB_DIR=/opt/rocm-asan-10.0/lib/llvm/lib/clang/23/lib/linux \
    AKA_GPU_ASAN_NORMAL_ROCM_LIB_DIR=/opt/rocm/lib
