#!/bin/bash
# Central configuration for the GTRS-BEVFusion cloud run.
# Source this from the other cloud/*.sh scripts:  source "$(dirname "$0")/env.sh"
#
# Override any of these by exporting them before sourcing, e.g.:
#   export NAVSIM_WS=/runpod-volume/navsim_workspace
#   export PYTHON=/runpod-volume/venv/bin/python

# --- repo + workspace --------------------------------------------------------
# NAVSIM_REPO = parent of this cloud/ dir (wherever you launched cloud/*.sh from).
# Always resolve from this file's location — do NOT honor a stale NAVSIM_REPO=/opt/...
# left in the shell from an earlier session (that silently runs the image-baked
# code without FATAL-DIAG / NaN abort). Override only via NAVSIM_REPO_OVERRIDE.
_SCRIPT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NAVSIM_REPO="${NAVSIM_REPO_OVERRIDE:-$_SCRIPT_REPO}"

# NAVSIM_WS = workspace dir on the persistent network volume (logs, sensors,
# maps, caches, checkpoints all live here so they survive pod restarts).
export NAVSIM_WS="${NAVSIM_WS:-/workspace/navsim_workspace}"

# CUDA-BEVFusion mmdet3d fork that provides ConvFuser / DepthLSSTransform /
# SparseEncoder. Must be importable; backbone reads BEVFUSION_ROOT.
# RunPod layout: /workspace/bev-inference/CUDA-BEVFusion/bevfusion
# (NOT /workspace/bevfusion — that empty stub dir is a common footgun).
export BEVFUSION_ROOT="${BEVFUSION_ROOT:-/workspace/bev-inference/CUDA-BEVFusion/bevfusion}"

# nuplan-devkit source (editable). On the Debian system Python the `setup.py
# develop` link doesn't always land on sys.path, so we add it to PYTHONPATH.
export NUPLAN_SRC="${NUPLAN_SRC:-/workspace/nuplan-devkit}"

# Python interpreter from the unified env (torch 1.10 + mmdet3d/spconv + navsim).
export PYTHON="${PYTHON:-python}"

# --- derived env that navsim / nuplan expect --------------------------------
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NAVSIM_WS/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_WS}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_WS}"
export PYTHONPATH="$NAVSIM_REPO:$BEVFUSION_ROOT:$NUPLAN_SRC:${PYTHONPATH:-}"

# data layout (NAVSIM standard)
export TRAINVAL_LOGS="$NAVSIM_WS/trainval_navsim_logs/trainval"
export TRAINVAL_SENSORS="$NAVSIM_WS/trainval_sensor_blobs/trainval"
export TEACHER_CACHE="$NAVSIM_WS/teacher_scores_cache"
# GTRS released teacher scores: one big pickle {token: {metric: (8192,)}}.
# Loaded once in the trainer main process (matches GTRS); no per-token convert.
export TEACHER_PKL="${TEACHER_PKL:-$NAVSIM_WS/navtrain_8192.pkl}"
export CKPT_DIR="$NAVSIM_WS/checkpoints"

echo "[env] NAVSIM_REPO     = $NAVSIM_REPO"
echo "[env] NAVSIM_WS       = $NAVSIM_WS"
echo "[env] BEVFUSION_ROOT  = $BEVFUSION_ROOT"
echo "[env] PYTHON          = $PYTHON ($($PYTHON --version 2>&1))"
echo "[env] NUPLAN_MAPS_ROOT= $NUPLAN_MAPS_ROOT"
