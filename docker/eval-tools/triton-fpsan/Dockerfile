ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena Triton FpSan tool runtime" \
      org.opencontainers.image.base.digest="sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78" \
      org.opencontainers.image.version="triton-3.7.0+amd.rocm7.2.0.gitd0d77a509"

COPY docker/eval-tools/triton-fpsan/requirements.lock /tmp/eval-tool-requirements.lock

RUN python -m pip uninstall -y \
        triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton triton-kernels \
    && python -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /tmp/eval-tool-requirements.lock \
    && python -c "import importlib.metadata as m, triton, triton_kernels; assert m.version('triton') == '3.7.0+amd.rocm7.2.0.gitd0d77a509'; assert m.version('triton-kernels') == '1.0.0+amd.rocm7.2.0.gitd0d77a509'" \
    && rm -f /tmp/eval-tool-requirements.lock

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    AKA_TRITON_FPSAN_VERSION=3.7.0+amd.rocm7.2.0.gitd0d77a509
