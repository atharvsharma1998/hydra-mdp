# SOPHI — Reproducibility Cookbook

Companion to the JMLR MLOSS submission. Commands match validated local
(RTX 3060) and cloud (A100) runs. Adjust absolute paths for your machine.

**Related pages:** [`INSTALL.md`](../INSTALL.md) · [`MODELS.md`](MODELS.md) · [`DEPLOY.md`](DEPLOY.md)

**Repo:** https://github.com/atharvsharma1998/hydra-mdp  
**Tag:** `v0.1.0-mloss`  
**License:** Apache-2.0

## Hardware / software assumptions

| Component | Tested version |
|-----------|----------------|
| GPU | NVIDIA RTX 3060 12GB (deploy) / A100 80GB (train + PDM) |
| CUDA | 11.3 (PyTorch) / driver supporting TensorRT 8.6 |
| TensorRT | 8.6.1.6 (`/usr/local/TensorRT-8.6.1.6`) |
| PyTorch | 1.10.2+cu113 |
| Python | 3.8 |
| `mmcv-full` | 1.4.0 (cu113 / torch1.10 wheel) |
| `spconv` | 2.x with `libspconv.so` |
| CUDA arch | sm_86 (Ampere consumer); set `CUDASM` / `CUMM_CUDA_ARCH_LIST` for yours |

## Environment variables

```bash
export NAVSIM_REPO=/path/to/hydra-mdp          # this repository root
export WORKSPACE=/path/to/navsim_workspace     # data + checkpoints + onnx
export PYTHON=/path/to/venv/bin/python         # e.g. Lidar_AI_Solution/venv
export BEVFUSION_ROOT=/path/to/CUDA-BEVFusion/bevfusion
export NUPLAN_MAPS_ROOT=$WORKSPACE/maps
export PYTHONPATH=$NAVSIM_REPO:$NAVSIM_REPO/scripts/training:$PYTHONPATH
export TensorRT_Root=/usr/local/TensorRT-8.6.1.6
export SPCONV_ROOT=/path/to/spconv
export LD_LIBRARY_PATH=$TensorRT_Root/lib:/usr/local/cuda/lib64:\
$SPCONV_ROOT/example/libspconv/build/spconv/src:\
$(dirname $BEVFUSION_ROOT)/build:$LD_LIBRARY_PATH
```

Install the missing OpenMMLab piece if needed:

```bash
$PYTHON -m pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

Build `libbevfusion_core.so` once in the CUDA-BEVFusion fork (see its `tool/run.sh`).

---

## 1. Data + metric cache

Download NAVSIM splits per upstream scripts under `download/`. Layout expected:

```
$WORKSPACE/
  mini_navsim_logs/mini/   (or trainval / test logs)
  mini_sensor_blobs/mini/  (or full sensor blobs)
  maps/
  checkpoints/
  onnx/
  metric_cache/            # output of caching
```

Precompute the PDM world cache (CPU-bound; use a **process** pool):

```bash
cd $NAVSIM_REPO
$PYTHON navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  worker=single_machine_thread_pool \
  worker.use_process_pool=True \
  worker.max_workers=$(nproc) \
  force_feature_computation=False \
  cache.cache_path=$WORKSPACE/metric_cache
```

On resume after interrupt: keep `force_feature_computation=False` and delete the
single most-recently-modified cache shard if Ctrl-C mid-write.

---

## 2. Training

```bash
cd $NAVSIM_REPO
# See cloud/03_train.sh for multi-GPU DDP; local smoke:
$PYTHON scripts/training/train_gtrs_bevfusion.py \
  --workspace $WORKSPACE \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  # ... additional flags as in cloud/env.sh / cloud/03_train.sh
```

Checkpoint format: `{"state_dict": <GTRSBevfusionModel weights>}`. Keys look like
`backbone.*`, `planning_head.*` (no `agent.` / `_model.` prefix required).

Released eval checkpoint:

`$WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth`

---

## 3. Qualitative Python inference

```bash
cd $NAVSIM_REPO
$PYTHON scripts/training/viz_gtrs_bevfusion_seg.py \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --out $WORKSPACE/navtrain_v1_best_dl_viz.png \
  --eval-mode --show-gt-boxes --num-frames 6
```

Expect: `loaded checkpoint ... (missing=0 unexpected=0)` and PNGs + GIF under `$WORKSPACE`.

---

## 4. Official PDM scoring (navtest)

```bash
cd $NAVSIM_REPO
source cloud/env.sh   # if using the cloud layout; else export vars above
export CKPT_DIR=$WORKSPACE/checkpoints
export NAVSIM_EXP_ROOT=$WORKSPACE
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PYTHONWARNINGS=ignore $PYTHON navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=gtrs_bevfusion_agent \
  agent.checkpoint_path=$CKPT_DIR/gtrs_bevfusion_navtrain_v1_best.pth \
  worker=single_machine_thread_pool \
  worker.use_process_pool=True \
  worker.max_workers=10 \
  experiment_name=gtrs_bevfusion_navtest \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
  2>&1 | grep -v "invalid value encountered in cast" | tee /tmp/pdm.log
```

Notes:
- Hydra agent config: `navsim/planning/script/config/common/agent/gtrs_bevfusion_agent.yaml`
- ~7.5 GB VRAM per process (CUDA context overhead) → ~10 workers on 80GB A100
- Aggregate row token: `extended_pdm_score_combined` → **score ≈ 0.7925** on full navtest

Pretty-print:

```bash
CSV=$(find $NAVSIM_EXP_ROOT -name "*.csv" -newermt "-1 day" | head -1)
$PYTHON - "$CSV" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(df.tail(1).T)
PY
```

---

## 5–7. ONNX export, TensorRT, C++ inference, parity

Full step-by-step commands (including stale-`.plan` caveats and C++ argv):
**[`DEPLOY.md`](DEPLOY.md)**.

Download published checkpoint / ONNX: **[`MODELS.md`](MODELS.md)**.

Minimal C++ run after engines exist:

```bash
cd $NAVSIM_REPO/deploy
./build/gtrs_bevfusion parity-data gtrs_bevfusion fp16 parity-data/cpp
```

---

## Licensing boundaries (honest)

| Component | License / status |
|-----------|------------------|
| This repo (`navsim/`, `deploy/`, scripts) | Apache-2.0 |
| Open `spconv` SCN runtime | Open source (spconv project) |
| CUDA-BEVFusion camera/BEVPool core | NVIDIA Apache-2.0 template; closed `libspconv` **replaced** |
| CUDA / TensorRT | Proprietary NVIDIA runtime (required for documented C++ path) |
| NAVSIM / nuPlan data | Separate dataset licenses — download yourself |

Do not claim a fully proprietary-free stack. The claim is: **open LiDAR SCN inference + full planning/seg deploy path** on top of the Apache-2.0 BEVFusion template.

## Extending the stack (develop your own)

1. Swap or retrain heads under `navsim/agents/gtrs_bevfusion/`.
2. Keep the ONNX split contract (camera / vtransform / SCN / fuser / three heads).
3. Re-export ONNX, wipe `.plan` files, rebuild engines, re-run parity.
4. Re-score with `run_pdm_score.py` against the shared metric cache.

Questions / issues: open a GitHub issue on the release tag.
