#!/bin/bash
# Precompute Hydra-MDP PDM teacher scores (the distillation targets) over the
# trainval logs into $TEACHER_CACHE. CPU/RAM heavy (PDM sim per scene); the
# script already multiprocesses and skips tokens whose .pkl already exists, so
# it is safe to re-run / resume. Run this to completion BEFORE training, else
# loss_distill is silently inactive for uncached tokens.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

mkdir -p "$TEACHER_CACHE"
cd "$NAVSIM_REPO"

echo "=== precomputing teacher scores -> $TEACHER_CACHE ==="
"$PYTHON" scripts/training/precompute_teacher_scores.py \
  --workspace "$NAVSIM_WS" \
  --maps-path "$NUPLAN_MAPS_ROOT" \
  --vocab-path "$NAVSIM_REPO/traj_final/8192.npy" \
  --output-path "$TEACHER_CACHE"

echo "=== teacher cache now holds $(ls "$TEACHER_CACHE" | wc -l) pkl files ==="
