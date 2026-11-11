ARG BASE_IMAGE=lmsysorg/sglang:v0.5.20-rocm10-mi35x@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena TileLens tool runtime" \
      org.opencontainers.image.base.digest="sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69" \
      org.opencontainers.image.version="tilelens-3.1.1"

# TileLens lives beside, not inside, the base environment: only the sanitizer
# lane puts /opt/tilelens/site-packages on its import path.
COPY docker/eval-tools/tilelens/requirements.lock /opt/tilelens/requirements.lock
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps --require-hashes --only-binary=:all: \
        --target /opt/tilelens/site-packages -r /opt/tilelens/requirements.lock \
    && PYTHONPATH=/opt/tilelens/site-packages /opt/venv/bin/python -c \
        "import tilelens, z3; from tilelens.clients import Sanitizer; print('tilelens', z3.get_version_string())"

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools
# The startup control exercises the lane's concrete bounds check, which
# confirms a TileLens finding before it counts.
COPY src/tools/arena_eval/sanitize.py /opt/aka-eval-tools/src/tools/arena_eval/sanitize.py

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    AKA_TILELENS_ROOT=/opt/tilelens/site-packages \
    AKA_TILELENS_VERSION=3.1.1
