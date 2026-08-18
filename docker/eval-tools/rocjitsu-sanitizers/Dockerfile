ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78
FROM ${BASE_IMAGE} AS builder

ARG ROCJITSU_SANITIZERS_COMMIT=ed35c0b54547c98bab359c8732529d9f5e8fd1ae
ARG GOOGLETEST_COMMIT=b514bdc898e2951020cbdca1304b75f5950d1f59
ARG FLATBUFFERS_COMMIT=595bf0007ab1929570c7671f091313c8fc20644e
ARG ZSTD_VERSION=1.5.7
ARG ZSTD_SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git software-properties-common zlib1g-dev \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc-13=13.4.0-6ubuntu1~22~ppa2 \
        g++-13=13.4.0-6ubuntu1~22~ppa2 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git init /src/rocm-systems \
    && git -C /src/rocm-systems remote add origin https://github.com/ROCm/rocm-systems.git \
    && git -C /src/rocm-systems fetch --depth 1 origin ${ROCJITSU_SANITIZERS_COMMIT} \
    && git -C /src/rocm-systems checkout --detach FETCH_HEAD \
    && test "$(git -C /src/rocm-systems rev-parse HEAD)" = "${ROCJITSU_SANITIZERS_COMMIT}" \
    && git init /src/googletest \
    && git -C /src/googletest remote add origin https://github.com/google/googletest.git \
    && git -C /src/googletest fetch --depth 1 origin ${GOOGLETEST_COMMIT} \
    && git -C /src/googletest checkout --detach FETCH_HEAD \
    && git init /src/flatbuffers \
    && git -C /src/flatbuffers remote add origin https://github.com/google/flatbuffers.git \
    && git -C /src/flatbuffers fetch --depth 1 origin ${FLATBUFFERS_COMMIT} \
    && git -C /src/flatbuffers checkout --detach FETCH_HEAD

RUN curl -sSfL -o /tmp/zstd.tar.gz \
        "https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz" \
    && printf '%s  %s\n' "${ZSTD_SHA256}" /tmp/zstd.tar.gz | sha256sum --check --quiet \
    && mkdir -p /src/zstd \
    && tar -xzf /tmp/zstd.tar.gz -C /src/zstd --strip-components=1 \
    && cmake -S /src/zstd/build/cmake -B /build/zstd -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/zstd \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DZSTD_BUILD_SHARED=OFF \
        -DZSTD_BUILD_STATIC=ON \
        -DZSTD_BUILD_PROGRAMS=OFF \
        -DZSTD_BUILD_TESTS=OFF \
    && cmake --build /build/zstd --parallel \
    && cmake --install /build/zstd \
    && rm -f /tmp/zstd.tar.gz

RUN cmake -S /src/rocm-systems/emulation/rocjitsu -B /build/rocjitsu -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc-13 \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DCMAKE_INSTALL_PREFIX=/opt/rocjitsu \
        -Dzstd_DIR=/opt/zstd/lib/cmake/zstd \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/src/googletest \
        -DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=/src/flatbuffers \
        -DBUILD_TESTING=OFF \
    && cmake --build /build/rocjitsu --parallel \
        --target rj_waitcheck rocjitsu_waitcheck rocjitsu_dbi_hooks \
    && mkdir -p /opt/rocjitsu/include /opt/rocjitsu/lib \
    && cp -a /src/rocm-systems/emulation/rocjitsu/lib/rocjitsu/include/rocjitsu \
        /opt/rocjitsu/include/ \
    && cp -a /build/rocjitsu/librocjitsu_waitcheck.so* /opt/rocjitsu/lib/ \
    && install -D -m 0755 \
        /build/rocjitsu/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so \
        /opt/rocjitsu/lib/librocjitsu_dbi_hooks.so \
    && install -D -m 0755 /build/rocjitsu/tools/rj_waitcheck /opt/rocjitsu/bin/rj_waitcheck \
    && mkdir -p /opt/rocjitsu/runtime \
    && cp -a /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /opt/rocjitsu/runtime/ \
    && cp -a /lib/x86_64-linux-gnu/libgcc_s.so.1 /opt/rocjitsu/runtime/

COPY docker/eval-tools/rocjitsu-sanitizers/waitcheck-capi.cpp /src/waitcheck-capi.cpp
RUN g++-13 -std=c++17 -O2 -Wall -Wextra -Werror \
        -I/opt/rocjitsu/include /src/waitcheck-capi.cpp \
        -L/opt/rocjitsu/lib -Wl,-rpath,/opt/rocjitsu/lib \
        -lrocjitsu_waitcheck -o /opt/rocjitsu/bin/aka-waitcheck-capi \
    && /opt/rocjitsu/bin/aka-waitcheck-capi --help >/dev/null

FROM ${BASE_IMAGE} AS runtime-common

LABEL org.opencontainers.image.base.digest="sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78" \
      org.opencontainers.image.revision="ed35c0b54547c98bab359c8732529d9f5e8fd1ae"

COPY --from=builder /opt/rocjitsu /opt/rocjitsu
COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    LD_LIBRARY_PATH=/opt/rocjitsu/lib:/opt/rocjitsu/runtime \
    PATH=/opt/rocjitsu/bin:${PATH} \
    AKA_ROCJITSU_SANITIZERS_COMMIT=ed35c0b54547c98bab359c8732529d9f5e8fd1ae

FROM runtime-common AS rocjitsu-waitcheck-runtime

LABEL org.opencontainers.image.title="AgentKernelArena rocJITsu Waitcheck runtime"

ENV AKA_WAITCHECK_BINARY=/opt/rocjitsu/bin/rj_waitcheck \
    AKA_WAITCHECK_CAPI_WRAPPER=/opt/rocjitsu/bin/aka-waitcheck-capi

RUN /opt/rocjitsu/bin/rj_waitcheck --help >/dev/null \
    && /opt/venv/bin/python -I \
        /opt/aka-eval-tools/src/eval_tools/adapters/waitcheck_entrypoint.py \
        --help >/dev/null

FROM runtime-common AS rocjitsu-consan-runtime

LABEL org.opencontainers.image.title="AgentKernelArena rocJITsu ConSan runtime"

ENV AKA_CONSAN_HOOK=/opt/rocjitsu/lib/librocjitsu_dbi_hooks.so

RUN test -f /opt/rocjitsu/lib/librocjitsu_dbi_hooks.so \
    && /opt/venv/bin/python -I \
        /opt/aka-eval-tools/src/eval_tools/adapters/consan_entrypoint.py \
        --help >/dev/null
