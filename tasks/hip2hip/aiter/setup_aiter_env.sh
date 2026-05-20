#!/usr/bin/env bash
# setup_aiter_env.sh
#
# 一键配置 hip2hip/aiter L1 + L2 case 所需的容器环境。
# 必须在 aiter 测试容器内运行（容器有 /workspace/）。
# 宿主机执行会因为路径不存在 / 权限不足直接报错。
#
# 用法:
#   bash tasks/hip2hip/aiter/setup_aiter_env.sh
#   AITER_SRC_DIR=/your/path bash tasks/hip2hip/aiter/setup_aiter_env.sh
#   AITER_COMMIT=<sha>      bash tasks/hip2hip/aiter/setup_aiter_env.sh
#
# 详细设计与决策记录见:
#   work-docs/20260520-xxy/plan.md

set -euo pipefail

# ---------------------------------------------------------------------------
# [Step 1] 解析环境变量
# ---------------------------------------------------------------------------
AITER_SRC_DIR="${AITER_SRC_DIR:-/workspace/aiter-src}"
AITER_COMMIT="${AITER_COMMIT:-d098ae5}"
AITER_REPO_URL="https://github.com/ROCm/aiter.git"

echo "=========================================================="
echo "  AITER_SRC_DIR = $AITER_SRC_DIR"
echo "  AITER_COMMIT  = $AITER_COMMIT"
echo "=========================================================="

# ---------------------------------------------------------------------------
# [Step 2] Clone aiter (幂等)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# [Step 3] 装 aiter (单步非 editable)
#
# 历史踩坑记录: aiter-kernel-test-env-setup.md 旧 SOP 要求"两步 install:
# 先 pip install . 再 pip install -e ."。 实测在 rocm7.0.2 / Python 3.12.3 /
# pip 25.2 / aiter d098ae5 组合下, 第二步 editable 安装会:
#   1) 先 uninstall 第一步装好的 amd-aiter (含 aiter_meta/ 实体);
#   2) 再装一个仅含 .pth 的 editable shim;
#   3) 源码目录里 setup.py 末尾主动删掉源码树里的 aiter_meta/.
# 结果: import aiter_meta 直接 ModuleNotFoundError, CK 软链那步立刻挂.
# 由于 L1/L2 case 都不改 aiter 源码, editable 没有实际价值, 改为单步.
# ---------------------------------------------------------------------------
cd "$AITER_SRC_DIR"

echo "[pip ] pip install . --no-build-isolation  (生成 aiter_meta 子包)"
pip install . --no-build-isolation

# ---------------------------------------------------------------------------
# [Step 4] 装 tabulate (L1 部分 case 渲染汇总表依赖)
# ---------------------------------------------------------------------------
echo "[pip ] pip install tabulate"
pip install tabulate

# ---------------------------------------------------------------------------
# [Step 5] CK 软链 (L2 attention / attention_ragged 编译需要)
# ---------------------------------------------------------------------------
AITER_META_DIR="$(python3 -c 'import aiter_meta, os; print(os.path.dirname(aiter_meta.__file__))')"
echo "[link] aiter_meta dir = $AITER_META_DIR"
mkdir -p "$AITER_META_DIR/3rdparty"
ln -sfn "$AITER_SRC_DIR/3rdparty/composable_kernel" \
        "$AITER_META_DIR/3rdparty/composable_kernel"
echo "[link] $AITER_META_DIR/3rdparty/composable_kernel -> $AITER_SRC_DIR/3rdparty/composable_kernel"

# ---------------------------------------------------------------------------
# [Step 6] import 自检
# ---------------------------------------------------------------------------
echo "[check] Verifying imports ..."
python3 -c "import aiter, aiter_meta, tabulate; print('All imports OK')"

# ---------------------------------------------------------------------------
# [Step 7] 完成提示 + 冒烟测试建议
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "  Environment ready."
echo "=========================================================="
echo "建议手动跑一次冒烟测试确认 GPU 链路:"
echo "  cd tasks/hip2hip/aiter/level1/unary_operator"
echo "  python3 scripts/task_runner.py correctness"
echo ""
