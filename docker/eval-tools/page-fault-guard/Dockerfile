ARG BASE_IMAGE=lmsysorg/sglang:v0.5.20-rocm10-mi35x@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena page-fault guard tool runtime" \
      org.opencontainers.image.base.digest="sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69"

# The guard needs nothing beyond the scoring base: it reruns correctness with
# PyTorch's caching allocator and ROCr's fragment allocator turned off.
COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools
