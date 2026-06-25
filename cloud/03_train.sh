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
BATCH="${BATCH:-4}"            # PER-GPU batch (tune to GPU memory; BEVFusion is heavy)
LR="${LR:-2e-4}"
WORKERS="${WORKERS:-10}"      # per-GPU DataLoader workers
PREFETCH="${PREFETCH:-4}"     # batches prefetched per worker (deeper queue vs stalls)
AMP="${AMP:-1}"               # 1=mixed precision (set 0 if you see NaNs)
# NGPU: number of GPUs to train on. >1 => DistributedDataParallel via torchrun.
NGPU="${NGPU:-$($PYTHON -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

# Dataloader parallelism comes from many workers (one __getitem__ each), NOT from
# OMP threads inside each worker. Keep OMP small to avoid thread oversubscription
# (num_workers x OMP x ngpu must stay near the core count). A modest bump from
# torchrun's default of 1 helps the small numpy/map matmuls without thrashing.
CORES="$(nproc 2>/dev/null || echo 8)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS | cores=$CORES | workers/gpu=$WORKERS | ngpu=$NGPU"
echo "  (total dataloader threads ~= workers*omp*ngpu = $(( WORKERS * OMP_NUM_THREADS * NGPU )); keep <= ~$CORES)"

mkdir -p "$NAVSIM_WS/logs" "$CKPT_DIR"
cd "$NAVSIM_REPO"

LATEST="$CKPT_DIR/${RUN_NAME}_latest.pth"
RESUME_ARG=""
[ -f "$LATEST" ] && RESUME_ARG="--resume $LATEST" && echo "auto-resume from $LATEST"
AMP_ARG=""; [ "$AMP" = "1" ] && AMP_ARG="--amp"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$NAVSIM_WS/logs/${RUN_NAME}_${STAMP}.log"
echo "logging to $LOG"

[ -f "$TEACHER_PKL" ] || { echo "ERROR: teacher pkl not found: $TEACHER_PKL"; exit 1; }
echo "teacher scores (big pkl): $TEACHER_PKL"

# DDP RAM warning: each rank loads its own copy of TEACHER_PKL.
if [ "$NGPU" -gt 1 ]; then
  echo "multi-GPU: $NGPU GPUs via torchrun (each loads its own teacher pkl ~30GB; "
  echo "  ensure system RAM >= ${NGPU} x pkl size). effective batch = $NGPU x $BATCH."
  LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU")
else
  echo "single-GPU run"
  LAUNCH=("$PYTHON")
fi

TRAIN_ARGS=(scripts/training/train_gtrs_bevfusion.py
  --workspace "$NAVSIM_WS"
  --maps-path "$NUPLAN_MAPS_ROOT"
  --sensor-blobs-path "$TRAINVAL_SENSORS"
  --teacher-pkl "$TEACHER_PKL"
  --num-scenes 0
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --lr "$LR"
  --weight-decay 1e-2
  --lr-schedule cosine --warmup-epochs 1
  --grad-clip 35
  --num-workers "$WORKERS"
  --prefetch-factor "$PREFETCH"
  --save-every 1
  --run-name "$RUN_NAME"
  --log-file "$LOG"
  $AMP_ARG $RESUME_ARG)

nohup "${LAUNCH[@]}" "${TRAIN_ARGS[@]}" \
  > "$NAVSIM_WS/logs/${RUN_NAME}_${STAMP}.out" 2>&1 &

echo "launched PID $! (run_name=$RUN_NAME ngpu=$NGPU per_gpu_batch=$BATCH lr=$LR amp=$AMP)"
echo "tail -f $LOG"
