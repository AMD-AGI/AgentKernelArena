#!/usr/bin/env bash
# Sets up the container environment for hip2hip/aiter L1 + L2 cases.
# Must be run inside the aiter test container (requires /workspace/).
#
# Usage:
#   bash tasks/hip2hip/aiter/setup_aiter_env.sh
#   AITER_SRC_DIR=/your/path bash tasks/hip2hip/aiter/setup_aiter_env.sh
#   AITER_COMMIT=<sha>      bash tasks/hip2hip/aiter/setup_aiter_env.sh

set -euo pipefail

AITER_SRC_DIR="${AITER_SRC_DIR:-/workspace/aiter-src}"
AITER_COMMIT="${AITER_COMMIT:-d098ae5}"
AITER_REPO_URL="https://github.com/ROCm/aiter.git"

echo "=========================================================="
echo "  AITER_SRC_DIR = $AITER_SRC_DIR"
echo "  AITER_COMMIT  = $AITER_COMMIT"
echo "=========================================================="

if [ -d "$AITER_SRC_DIR/.git" ]; then
    echo "[clone] Found existing aiter at $AITER_SRC_DIR, skipping clone."
    echo "        -> To force re-clone: rm -rf $AITER_SRC_DIR"
else
    echo "[clone] Cloning $AITER_REPO_URL -> $AITER_SRC_DIR ..."
    git clone "$AITER_REPO_URL" "$AITER_SRC_DIR"
    (
        cd "$AITER_SRC_DIR"
        echo "[clone] Checking out commit $AITER_COMMIT ..."
        git checkout "$AITER_COMMIT"
        echo "[clone] Initializing submodules (depth=1) ..."
        git submodule update --init --depth 1
    )
fi

cd "$AITER_SRC_DIR"
echo "[pip] Installing aiter ..."
pip install . --no-build-isolation

echo "[pip] Installing tabulate ..."
pip install tabulate

AITER_META_DIR="$(python3 -c 'import aiter_meta, os; print(os.path.dirname(aiter_meta.__file__))')"
mkdir -p "$AITER_META_DIR/3rdparty"
ln -sfn "$AITER_SRC_DIR/3rdparty/composable_kernel" \
        "$AITER_META_DIR/3rdparty/composable_kernel"
echo "[link] $AITER_META_DIR/3rdparty/composable_kernel -> $AITER_SRC_DIR/3rdparty/composable_kernel"

echo "[check] Verifying imports ..."
python3 -c "import aiter, aiter_meta, tabulate; print('All imports OK')"

echo ""
echo "=========================================================="
echo "  Environment ready."
echo "=========================================================="
echo "Run a smoke test to verify the GPU stack:"
echo "  cd tasks/hip2hip/aiter/level1/unary_operator"
echo "  python3 scripts/task_runner.py correctness"
echo ""
