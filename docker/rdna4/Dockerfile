# rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
# Keep the GPU stack fixed; add only the missing report dependencies below.
FROM rocm/vllm@sha256:b8a082f346d069376d35784250e38b23a043efe979408ae3a33d7c6b62ee3276

USER root
COPY prepare_runtime.py /tmp/aka-prepare-runtime.py
RUN python /tmp/aka-prepare-runtime.py && rm /tmp/aka-prepare-runtime.py

COPY requirements.lock /tmp/aka-rdna4-requirements.lock
RUN PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
    /opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
    --only-binary=:all: --require-hashes --index-url https://pypi.org/simple \
    -r /tmp/aka-rdna4-requirements.lock && rm /tmp/aka-rdna4-requirements.lock

# Verify the same public paths the runner uses, without a GPU or host mounts.
# /root remains private; neither execution nor extension builds should need it.
RUN setpriv --reuid=65534 --regid=65534 --clear-groups env HOME=/tmp \
    /opt/venv/bin/python -c \
    "import os, torch, triton, pytest, yaml, numpy, matplotlib.pyplot; assert not os.access('/root', os.X_OK); assert os.access('/opt/rocm/bin/hipcc', os.X_OK); assert os.access('/opt/rocm/bin/rocprofv3', os.X_OK)"
