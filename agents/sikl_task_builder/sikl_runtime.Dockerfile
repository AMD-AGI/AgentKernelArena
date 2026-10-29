# Runtime of SIKL v5 tasks: SIKL CI's image and pins, with the SIKL checkout at /opt/sikl owned by root.
# Build context: a clean SIKL clone (full .git, no alternates, submodules initialized). BASE is ci.yaml's
# SIKL_CI_IMAGE with its digest; SIKL_COMMIT is that clone's HEAD.
ARG BASE
FROM ${BASE}
ARG SIKL_COMMIT
LABEL sikl.commit=${SIKL_COMMIT}
COPY --chown=root:root . /opt/sikl
# ci.yaml setup_sikl: install_triton.sh, then its pinned Triton; setup.py's FlyDSL pin. No editable install:
# task runners import aiter and sikl from a private checkout of the pinned commit.
RUN chmod -R a+rX,go-w /opt/sikl \
 && bash /opt/sikl/.github/scripts/install_triton.sh \
 && python3 -m pip install --no-deps --force-reinstall \
      "$(grep -oE -- "--force-reinstall 'triton==[0-9.]+'" /opt/sikl/.github/workflows/ci.yaml | grep -oE 'triton==[0-9.]+')" \
 && python3 -m pip install "$(sed -nE 's/^FLYDSL_VERSION = "(flydsl==[^"]+)"$/\1/p' /opt/sikl/setup.py)" \
 && rm -rf /tmp/aiter_configs && mkdir -m 1777 /tmp/aiter_configs
# aiter merges its tuned configs into the fixed /tmp/aiter_configs; tasks run as a non-root user, and the
# base image's root-owned merges come from its own aiter, not from the pinned checkout's.
ENV AKA_SIKL_ROOT=/opt/sikl PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1
