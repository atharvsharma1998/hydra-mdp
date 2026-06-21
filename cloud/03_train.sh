#!/bin/bash
# Launch the full from-scratch GTRS-BEVFusion training on navtrain.
# Runs under nohup, logs to $NAVSIM_WS/logs, and auto-resumes from the latest
# checkpoint if present (so a pod restart continues instead of restarting).
#
# Tunables (override via env): EPOCHS, BATCH, LR, WORKERS, RUN_NAME, AMP.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RUN_NAME="${RUN_NAME:-gtrs_bevfusion_navtrain}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-4}"            # tune to GPU memory (BEVFusion is heavy)
LR="${LR:-2e-4}"
WORKERS="${WORKERS:-8}"
AMP="${AMP:-1}"               # 1=mixed precision (set 0 if you see NaNs)

mkdir -p "$NAVSIM_WS/logs" "$CKPT_DIR"
cd "$NAVSIM_REPO"

LATEST="$CKPT_DIR/${RUN_NAME}_latest.pth"
RESUME_ARG=""
[ -f "$LATEST" ] && RESUME_ARG="--resume $LATEST" && echo "auto-resume from $LATEST"
AMP_ARG=""; [ "$AMP" = "1" ] && AMP_ARG="--amp"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$NAVSIM_WS/logs/${RUN_NAME}_${STAMP}.log"
echo "logging to $LOG"

nohup "$PYTHON" scripts/training/train_gtrs_bevfusion.py \
  --workspace "$NAVSIM_WS" \
  --maps-path "$NUPLAN_MAPS_ROOT" \
  --sensor-blobs-path "$TRAINVAL_SENSORS" \
  --teacher-cache-path "$TEACHER_CACHE" \
  --num-scenes 0 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --weight-decay 1e-2 \
  --lr-schedule cosine --warmup-epochs 1 \
  --grad-clip 35 \
  --num-workers "$WORKERS" \
  --save-every 1 \
  --run-name "$RUN_NAME" \
  --log-file "$LOG" \
  $AMP_ARG $RESUME_ARG \
  > "$NAVSIM_WS/logs/${RUN_NAME}_${STAMP}.out" 2>&1 &

echo "launched PID $! (run_name=$RUN_NAME epochs=$EPOCHS batch=$BATCH lr=$LR amp=$AMP)"
echo "tail -f $LOG"
