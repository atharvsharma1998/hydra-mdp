# Deploy: ONNX export → TensorRT → C++ inference

Commands for the reviewed SOPHI checkpoint
(`gtrs_bevfusion_navtrain_v1_best.pth`). Paths assume the env from
[`INSTALL.md`](../INSTALL.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Pretrained weights and exported ONNX graphs: see [`MODELS.md`](MODELS.md).

---

## 1. Export ONNX from a PyTorch checkpoint

Exports six dense graphs + the LiDAR SCN ONNX:

| Graph | File |
|-------|------|
| Camera backbone | `camera.backbone.onnx` |
| Camera view transform | `camera.vtransform.onnx` |
| Fuser → \(\mathbf{F}_{\mathrm{env}}\) | `fuser.onnx` |
| Planning head | `planning_head.onnx` |
| Detection head | `det_head.onnx` |
| Segmentation head | `seg_head.onnx` |
| LiDAR SCN | `lidar.backbone.onnx` |

```bash
cd $NAVSIM_REPO
$PYTHON scripts/export/export_gtrs_bevfusion_onnx.py \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --out-dir $WORKSPACE/onnx \
  --lidar-scn
```

Refresh C++ sample tensors + PyTorch per-stage references (same mini token):

```bash
$PYTHON scripts/export/export_gtrs_bevfusion_onnx.py \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --dump-cpp-inputs $NAVSIM_REPO/deploy/parity-data
```

Do **not** pass `--dump-cpp-inputs` in the same invocation as the full ONNX export
(dump mode returns early and skips writing the graphs).

---

## 2. Build TensorRT engines (fp16)

`build_gtrs_engines.sh` **skips** any existing `*.plan`. After a new ONNX export,
delete stale engines first:

```bash
rm -f $NAVSIM_REPO/deploy/model/gtrs_bevfusion/build/*.plan \
      $NAVSIM_REPO/deploy/model/gtrs_bevfusion/*.onnx

cd $NAVSIM_REPO
GTRS_ONNX_DIR=$WORKSPACE/onnx \
TensorRT_Root=${TensorRT_Root:-/usr/local/TensorRT-8.6.1.6} \
  bash deploy/tool/build_gtrs_engines.sh fp16
```

Engines land in `deploy/model/gtrs_bevfusion/build/*.plan`.
The SCN graph stays as ONNX and is run by the open `spconv` parser at runtime
(`deploy/model/gtrs_bevfusion/lidar.backbone.xyz.onnx`).

---

## 3. Build and run the C++ binary

One-shot (engines + compile + run):

```bash
cd $NAVSIM_REPO
BEVFUSION_ROOT=$(dirname $BEVFUSION_ROOT) \
GTRS_ONNX_DIR=$WORKSPACE/onnx \
GTRS_DATA=$NAVSIM_REPO/deploy/parity-data \
  bash deploy/tool/run_gtrs.sh fp16
```

Or run an already-built binary with per-stage dumps:

```bash
cd $NAVSIM_REPO/deploy
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${TensorRT_Root}/lib:\
${SPCONV_ROOT}/example/libspconv/build/spconv/src:\
$(dirname $BEVFUSION_ROOT)/build:${LD_LIBRARY_PATH}"

./build/gtrs_bevfusion parity-data gtrs_bevfusion fp16 parity-data/cpp
```

Arguments: `<data_dir> <model_tag> <precision> [parity_dump_dir]`.

Outputs:

- console detections + planned trajectory
- `build/gtrs_cpp_viz.jpg` — 6-cam + BEV viz
- `build/gtrs_trajectory.txt`
- `parity-data/cpp/*.tensor` — per-stage dumps when the 4th arg is set

---

## 4. Parity check (Python vs C++)

```bash
cd $NAVSIM_REPO
$PYTHON scripts/export/compare_cpp_parity.py \
  --ref deploy/parity-data/ref \
  --cpp deploy/parity-data/cpp
```

Validated on token `8bc34517e08758ff` with the `navtrain_v1_best` weights:
trajectory cosine **1.000**, max |Δ| **0.0017 m**.

---

## 5. Qualitative Python viz (optional)

```bash
$PYTHON scripts/training/viz_gtrs_bevfusion_seg.py \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --out $WORKSPACE/sophi_viz.png \
  --eval-mode --show-gt-boxes --num-frames 6
```

Demo GIF committed in-repo: [`assets/demo/sophi_navtrain_v1.gif`](../assets/demo/sophi_navtrain_v1.gif).
