ARG BASE_IMAGE=lmsysorg/sglang:v0.5.20-rocm10-mi35x@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena Triton FpSan tool runtime" \
      org.opencontainers.image.base.digest="sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69" \
      org.opencontainers.image.version="triton-3.8.0+git4cff872c.rocm10.0.0"

# The digest-pinned ROCm 10 base already supplies FpSan-enabled Triton.
# Do not replace it with a ROCm 7.2 wheel or infer support from import success.
RUN python -c "import importlib.metadata as m; assert m.version('triton') == '3.8.0+git4cff872c.rocm10.0.0'"

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    AKA_TRITON_FPSAN_VERSION=3.8.0+git4cff872c.rocm10.0.0
