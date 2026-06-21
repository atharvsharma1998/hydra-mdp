#!/bin/bash
# Rebuild the unified Python env on a fresh pod, matching the known-good local
# stack: Python 3.8 + torch 1.10.2+cu113 + mmcv-full 1.4.0 + spconv-cu114 2.3.6.
#
# Prereqs (do these FIRST, see notes at bottom if `python -m venv` fails):
#   - a Python 3.8 interpreter with pip
#   - cloned repos: hydra-mdp (this) + bev-inference (the BEVFusion/mmdet3d fork)
#
# Usage:
#   export NAVSIM_REPO=/workspace/hydra-mdp
#   export BEVFUSION_ROOT=/workspace/bev-inference/CUDA-BEVFusion/bevfusion
#   export VENV=/workspace/venv            # where to create the env
#   bash cloud/setup_env.sh
set -euo pipefail

NAVSIM_REPO="${NAVSIM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BEVFUSION_ROOT="${BEVFUSION_ROOT:-/workspace/bev-inference/CUDA-BEVFusion/bevfusion}"
VENV="${VENV:-/workspace/venv}"
NUPLAN_SRC="${NUPLAN_SRC:-/workspace/nuplan-devkit}"
REQ="$NAVSIM_REPO/cloud/requirements-frozen.txt"

echo "=== creating venv at $VENV (python3.8) ==="
python3.8 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

echo "=== 1. torch 1.10.2 + cu113 (matched trio) ==="
"$PY" -m pip install \
  torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "=== 2. mmcv-full 1.4.0 (prebuilt for torch1.10/cu113) ==="
"$PY" -m pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

echo "=== 3. pinned third-party deps (frozen, minus torch/mmcv/editables/trt) ==="
"$PY" -m pip install -r "$REQ"

echo "=== 4. nuplan-devkit (editable) ==="
if [ ! -d "$NUPLAN_SRC" ]; then
  git clone https://github.com/motional/nuplan-devkit.git "$NUPLAN_SRC"
  git -C "$NUPLAN_SRC" checkout e9241677997dd86bfc0bcd44817ab04fe631405b || true
fi
"$PY" -m pip install -e "$NUPLAN_SRC" --no-deps

echo "=== 5. mmdet3d fork (bev-inference) + navsim (hydra-mdp), editable ==="
"$PY" -m pip install -e "$BEVFUSION_ROOT" --no-deps
"$PY" -m pip install -e "$NAVSIM_REPO" --no-deps

echo "=== 6. sanity import check ==="
"$PY" - <<'PY'
import torch, mmcv, mmdet, spconv, numpy, einops, navsim, nuplan
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
print("mmcv", mmcv.__version__, "| spconv", spconv.__version__, "| numpy", numpy.__version__)
print("env OK")
PY

echo
echo "=== DONE. Set PYTHON=$PY in cloud/env.sh (or: export PYTHON=$PY) ==="
echo "NOTE: 'tensorrt' is intentionally skipped (export-only, local wheel)."
echo
echo "If 'python3.8 -m venv' failed earlier ('ensurepip not available'):"
echo "  A) conda (recommended on RunPod): conda create -y -n hydra python=3.8 && conda activate hydra,"
echo "     then re-run with VENV unused (PY=\$(which python)); or"
echo "  B) apt-get update && apt-get install -y python3.8-venv python3-pip"
