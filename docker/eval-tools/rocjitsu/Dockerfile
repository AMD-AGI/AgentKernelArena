ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78
FROM ${BASE_IMAGE} AS builder

ARG ROCJITSU_COMMIT=0bf561a0d8a4a6b88954f2c46bd3a50871cda140
ARG GOOGLETEST_COMMIT=b514bdc898e2951020cbdca1304b75f5950d1f59
ARG FLATBUFFERS_COMMIT=595bf0007ab1929570c7671f091313c8fc20644e

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates git software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc-13=13.4.0-6ubuntu1~22~ppa2 \
        g++-13=13.4.0-6ubuntu1~22~ppa2 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git init /src/rocm-systems \
    && git -C /src/rocm-systems remote add origin https://github.com/ROCm/rocm-systems.git \
    && git -C /src/rocm-systems fetch --depth 1 origin ${ROCJITSU_COMMIT} \
    && git -C /src/rocm-systems checkout --detach FETCH_HEAD \
    && test "$(git -C /src/rocm-systems rev-parse HEAD)" = "${ROCJITSU_COMMIT}" \
    && git init /src/googletest \
    && git -C /src/googletest remote add origin https://github.com/google/googletest.git \
    && git -C /src/googletest fetch --depth 1 origin ${GOOGLETEST_COMMIT} \
    && git -C /src/googletest checkout --detach FETCH_HEAD \
    && git init /src/flatbuffers \
    && git -C /src/flatbuffers remote add origin https://github.com/google/flatbuffers.git \
    && git -C /src/flatbuffers fetch --depth 1 origin ${FLATBUFFERS_COMMIT} \
    && git -C /src/flatbuffers checkout --detach FETCH_HEAD

RUN cmake -S /src/rocm-systems/emulation/rocjitsu -B /build/rocjitsu -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc-13 \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DCMAKE_INSTALL_PREFIX=/opt/rocjitsu \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/src/googletest \
        -DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=/src/flatbuffers \
        -DCMAKE_HIP_ARCHITECTURES=gfx950 \
        -DBUILD_TESTING=ON \
        -DRJ_INSTALL_TESTS=OFF \
    && cmake --build /build/rocjitsu --parallel \
    && cmake --install /build/rocjitsu \
    && mkdir -p /opt/rocjitsu/runtime \
    && cp -a /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /opt/rocjitsu/runtime/ \
    && cp -a /lib/x86_64-linux-gnu/libgcc_s.so.1 /opt/rocjitsu/runtime/

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="AgentKernelArena rocJITsu tool runtime" \
      org.opencontainers.image.base.digest="sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78" \
      org.opencontainers.image.revision="0bf561a0d8a4a6b88954f2c46bd3a50871cda140"

COPY --from=builder /opt/rocjitsu /opt/rocjitsu
COPY docker/eval-tools/rocjitsu/rocjitsu-wrapper /usr/local/bin/rocjitsu
RUN chmod 0755 /usr/local/bin/rocjitsu \
    && /usr/local/bin/rocjitsu --help >/dev/null

COPY src/eval_tools /opt/aka-eval-tools/src/eval_tools
RUN /opt/venv/bin/python -I \
        /opt/aka-eval-tools/src/eval_tools/adapters/rocjitsu_replay_entrypoint.py \
        --help >/dev/null

ENV AKA_EVAL_TOOL_FRAMEWORK_ROOT=/opt/aka-eval-tools \
    PYTHONPATH=/opt/aka-eval-tools \
    PATH=/opt/rocjitsu/bin:${PATH} \
    AKA_ROCJITSU_BINARY=/usr/local/bin/rocjitsu \
    AKA_ROCJITSU_CONFIG=/opt/rocjitsu/share/rocjitsu/configs/gfx950_cdna4.json \
    AKA_ROCJITSU_COMMIT=0bf561a0d8a4a6b88954f2c46bd3a50871cda140
