# Extend an architecture-qualified Arena scoring image without replacing its
# Python, ROCm or benchmark dependencies. Pass the base by immutable digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*
