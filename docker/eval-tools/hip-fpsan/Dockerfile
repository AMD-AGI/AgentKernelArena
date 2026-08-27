ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78
FROM ${BASE_IMAGE}

ARG HIP_FPSAN_COMMIT=0ac9be8a1539a473ba21dfa686564c3be33c890e

LABEL org.opencontainers.image.title="AgentKernelArena HIP-FpSan tool runtime" \
      org.opencontainers.image.base.digest="sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78" \
      org.opencontainers.image.revision="0ac9be8a1539a473ba21dfa686564c3be33c890e"

COPY src/eval_tools/probes/hip_fpsan_probe.hip /tmp/hip_fpsan_probe.hip

RUN git init /opt/hip-fpsan \
    && git -C /opt/hip-fpsan remote add origin https://github.com/ROCm/hip-fpsan.git \
    && git -C /opt/hip-fpsan fetch --depth 1 origin ${HIP_FPSAN_COMMIT} \
    && git -C /opt/hip-fpsan checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/hip-fpsan rev-parse HEAD)" = "${HIP_FPSAN_COMMIT}" \
    && mkdir -p /opt/eval-tools/probes \
    && /opt/rocm/bin/hipcc -O2 --offload-arch=gfx950 \
        -I/opt/hip-fpsan/include \
        /tmp/hip_fpsan_probe.hip \
        -o /opt/eval-tools/probes/hip_fpsan_probe \
    && rm -f /tmp/hip_fpsan_probe.hip

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    HIP_FPSAN_ROOT=/opt/hip-fpsan \
    AKA_HIP_FPSAN_INCLUDE_DIR=/opt/hip-fpsan/include \
    AKA_HIP_FPSAN_PROBE=/opt/eval-tools/probes/hip_fpsan_probe \
    AKA_HIP_FPSAN_COMMIT=0ac9be8a1539a473ba21dfa686564c3be33c890e
