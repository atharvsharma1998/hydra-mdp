#!/bin/bash
# Fail-fast checks before spending GPU hours: env imports, GPU, vocab, paths.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

fail=0
note() { printf "  %-22s %s\n" "$1" "$2"; }

echo "=== 1. Python / CUDA / core libs ==="
"$PYTHON" - <<'PY' || fail=1
import importlib, torch
print("  torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
for m in ["spconv", "mmcv", "mmdet3d", "einops", "navsim", "nuplan"]:
    try:
        importlib.import_module(m); print(f"  ok   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}"); raise SystemExit(1)
PY

echo "=== 2. BEVFusion fork importable (ConvFuser/SparseEncoder) ==="
"$PYTHON" - <<'PY' || fail=1
import os, sys
sys.path.insert(0, os.environ["BEVFUSION_ROOT"])
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_backbone import BEVFusionBackbone
bb = BEVFusionBackbone(checkpoint_path=None, geometry=GTRSBevfusionConfig())
n = sum(p.numel() for p in bb.parameters())
print(f"  backbone built from scratch ok, params={n/1e6:.1f}M")
PY

echo "=== 3. Trajectory vocab present ==="
if [ -f "$NAVSIM_REPO/traj_final/8192.npy" ]; then
  note "vocab" "OK ($NAVSIM_REPO/traj_final/8192.npy)"
else
  note "vocab" "MISSING $NAVSIM_REPO/traj_final/8192.npy"; fail=1
fi

echo "=== 4. Data paths ==="
for p in "$NUPLAN_MAPS_ROOT" "$TRAINVAL_LOGS" "$TRAINVAL_SENSORS"; do
  if [ -d "$p" ]; then note "$(basename "$p")" "OK ($p)"; else note "$(basename "$p")" "MISSING $p (run 01_download_data.sh)"; fi
done
nlogs=$(ls "$TRAINVAL_LOGS" 2>/dev/null | wc -l)
nsens=$(ls "$TRAINVAL_SENSORS" 2>/dev/null | wc -l)
ncache=$(ls "$TEACHER_CACHE" 2>/dev/null | wc -l)
note "trainval logs" "$nlogs files"
note "sensor log dirs" "$nsens dirs"
note "teacher cache" "$ncache pkl (run 02_precompute_teacher_scores.sh if low)"

echo
if [ "$fail" -ne 0 ]; then echo "PREFLIGHT FAILED"; exit 1; else echo "PREFLIGHT OK"; fi
