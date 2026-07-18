# Quickstart: `.pth` → Python → ONNX → TensorRT → C++

Stay on `main`. Two smoke tests:

| Path | Needs full NAVSIM? | Entry point |
|------|--------------------|-------------|
| **Python** (`.pth`) | OpenScene-**mini** logs + sensors + maps | `scripts/training/viz_gtrs_bevfusion_seg.py` |
| **C++** (TensorRT) | No — use shipped `deploy/example-data/` | `./build/gtrs_bevfusion …` |

```bash
git clone https://github.com/atharvsharma1998/hydra-mdp.git
cd hydra-mdp
```

Full training / `navtest` PDM: [`docs/TRAINING.md`](docs/TRAINING.md).

---

## 0. One-time environment

You need:

- Python 3.8 + PyTorch 1.10+cu113 (Python inference + ONNX export)
- TensorRT 8.6 (`trtexec`) — C++ path only
- Built `libbevfusion_core.so` from a CUDA-BEVFusion fork — C++ path only
- Built open `libspconv.so` — C++ path only

```bash
export REPO=$PWD
export WORKSPACE=$HOME/navsim_workspace          # wherever you keep weights
export BEVFUSION_ROOT=/path/to/CUDA-BEVFusion/bevfusion
export TensorRT_Root=/usr/local/TensorRT-8.6.1.6
export SPCONV_ROOT=/path/to/spconv
export PYTHON=/path/to/venv/bin/python           # venv with torch + mmcv-full

export PYTHONPATH=$REPO:$REPO/scripts/training:$PYTHONPATH
export LD_LIBRARY_PATH=$TensorRT_Root/lib:/usr/local/cuda/lib64:\
$SPCONV_ROOT/example/libspconv/build/spconv/src:\
$(dirname $BEVFUSION_ROOT)/build:$LD_LIBRARY_PATH

mkdir -p $WORKSPACE/checkpoints $WORKSPACE/onnx
```

Install `mmcv-full` if missing:

```bash
$PYTHON -m pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

---

## 1. Download the PyTorch checkpoint

Get `gtrs_bevfusion_navtrain_v1_best.pth` from Google Drive
(link in [`docs/MODELS.md`](docs/MODELS.md)) and place it at:

```bash
# example after download:
mv ~/Downloads/gtrs_bevfusion_navtrain_v1_best.pth $WORKSPACE/checkpoints/
```

**Optional shortcut:** if you download the pre-exported ONNX zip from Drive
into `$WORKSPACE/onnx/`, skip steps 2–3 and go straight to TensorRT (step 4).

---

## 2. Python inference (PyTorch `.pth`)

Runs the full model in PyTorch and writes a multi-panel PNG (cameras, dets,
BEV seg, planned trajectory). Needs OpenScene-**mini** under `$WORKSPACE`
(not the full navtest set):

```
$WORKSPACE/
  maps/
  mini_navsim_logs/mini/
  mini_sensor_blobs/mini/
  checkpoints/gtrs_bevfusion_navtrain_v1_best.pth
```

```bash
cd $REPO
$PYTHON scripts/training/viz_gtrs_bevfusion_seg.py \
  --workspace $WORKSPACE \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --token 8bc34517e08758ff \
  --eval-mode \
  --out $WORKSPACE/python_inference.png
```

- `--eval-mode` — full vocab + running BN (use this for real trajectories)
- omit `--token` to take the first available mini scene
- demo GIF in the README was produced this way

If you only care about C++ and already have the Drive ONNX zip, skip this step.

---

## 3. Convert `.pth` → ONNX

```bash
cd $REPO
$PYTHON scripts/export/export_gtrs_bevfusion_onnx.py \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --out-dir $WORKSPACE/onnx \
  --lidar-scn
```

### Why `--sensor-blobs-path`?

ONNX export **traces** the network on one real batch to freeze shapes and
validate outputs. The script loads a single mini-split frame for that.
It does **not** mean you need the full navtest dataset.

- For **C++ inference only**, you do **not** need sensor blobs: use the
  shipped `deploy/example-data/` (one frame) after TensorRT engines exist.
- For **export from `.pth`**, you need either:
  - OpenScene-mini sensor blobs under `$WORKSPACE/mini_sensor_blobs/mini`, or
  - skip export and download the ONNX zip from Drive ([`docs/MODELS.md`](docs/MODELS.md)).

Writes:

```
$WORKSPACE/onnx/
  camera.backbone.onnx
  camera.vtransform.onnx
  fuser.onnx
  planning_head.onnx
  det_head.onnx
  seg_head.onnx
  lidar.backbone.onnx
```

---

## 4. Convert ONNX → TensorRT (fp16)

Delete any old engines first (the build script skips existing `.plan` files):

```bash
rm -f $REPO/deploy/model/gtrs_bevfusion/build/*.plan \
      $REPO/deploy/model/gtrs_bevfusion/*.onnx

cd $REPO
GTRS_ONNX_DIR=$WORKSPACE/onnx \
TensorRT_Root=$TensorRT_Root \
  bash deploy/tool/build_gtrs_engines.sh fp16
```

---

## 5. Run C++ inference (one-frame sample, no full dataset)

`deploy/example-data/` is included in the repo (~11 MB): cameras, LiDAR points,
calibration, ego status — same role as CUDA-BEVFusion’s `example-data/`.

```bash
cd $REPO/deploy

# build binary once (if not already built)
mkdir -p build && cd build
cmake .. -DBEVFUSION_ROOT=$(dirname $BEVFUSION_ROOT) \
         -DTensorRT_Root=$TensorRT_Root \
         -DCUDA_HOME=/usr/local/cuda -DCUDASM=86
make gtrs_bevfusion -j
cd ..

# run on the shipped sample
./build/gtrs_bevfusion example-data gtrs_bevfusion fp16
```

You should see detections + a planned trajectory, plus:

- `build/gtrs_cpp_viz.jpg`
- `build/gtrs_trajectory.txt`

Optional parity dump:

```bash
./build/gtrs_bevfusion example-data gtrs_bevfusion fp16 example-data/cpp
# then: $PYTHON scripts/export/compare_cpp_parity.py --ref ... --cpp ...
```

Or use the helper script:

```bash
cd $REPO
BEVFUSION_ROOT=$(dirname $BEVFUSION_ROOT) \
GTRS_ONNX_DIR=$WORKSPACE/onnx \
GTRS_DATA=$REPO/deploy/example-data \
  bash deploy/tool/run_gtrs.sh fp16
```

---

## Checklist

1. [ ] Env vars set (`PYTHON`; plus `BEVFUSION_ROOT` / `TensorRT_Root` / `SPCONV_ROOT` for C++)
2. [ ] Checkpoint in `$WORKSPACE/checkpoints/` (or ONNX already in `$WORKSPACE/onnx/`)
3. [ ] (optional) Python viz PNG from step 2
4. [ ] ONNX graphs present
5. [ ] TensorRT `.plan` engines built
6. [ ] `./build/gtrs_bevfusion example-data gtrs_bevfusion fp16` runs
