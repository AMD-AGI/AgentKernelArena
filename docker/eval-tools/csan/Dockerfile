ARG BASE_IMAGE=lmsysorg/sglang:v0.5.20-rocm10-mi35x@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena PyTorch CUDA Sanitizer tool runtime" \
      org.opencontainers.image.base.digest="sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69"

# CSAN ships with the base image's PyTorch (torch.cuda._sanitizer).
RUN python -c "import torch.cuda._sanitizer as s; assert callable(s.enable_cuda_sanitizer)"

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools
