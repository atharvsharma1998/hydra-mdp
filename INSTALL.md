# Install SOPHI

Short path for new users. Full train / PDM / ONNX / C++ commands live in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 1. Clone the reviewed release

```bash
git clone https://github.com/atharvsharma1998/hydra-mdp.git
cd hydra-mdp
git checkout v0.1.0-mloss
```

## 2. Python environment (tested)

| Package | Version |
|---------|---------|
| Python | 3.8 |
| PyTorch | 1.10.2+cu113 |
| `mmcv-full` | 1.4.0 (cu113 / torch1.10 wheel) |
| `spconv` | 2.x (+ built `libspconv.so`) |
| TensorRT | 8.6.1.6 (C++ deploy) |
| CUDA | 11.3+ (matching PyTorch / TRT) |

Example:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install -r requirements.txt
pip install -e .
```

Also required:

- A CUDA-BEVFusion / `mmdet3d` fork checkout with `bevfusion/` configs  
  → set `BEVFUSION_ROOT=/path/to/CUDA-BEVFusion/bevfusion`
- Built `libbevfusion_core.so` in that fork (see its `tool/run.sh`)
- Open `spconv` with `libspconv.so` on `LD_LIBRARY_PATH`

## 3. Environment variables

```bash
export NAVSIM_REPO=$PWD
export WORKSPACE=/path/to/navsim_workspace   # data + checkpoints + onnx
export BEVFUSION_ROOT=/path/to/CUDA-BEVFusion/bevfusion
export NUPLAN_MAPS_ROOT=$WORKSPACE/maps
export PYTHONPATH=$NAVSIM_REPO:$NAVSIM_REPO/scripts/training:$PYTHONPATH
export TensorRT_Root=/usr/local/TensorRT-8.6.1.6
export SPCONV_ROOT=/path/to/spconv
export LD_LIBRARY_PATH=$TensorRT_Root/lib:/usr/local/cuda/lib64:\
$SPCONV_ROOT/example/libspconv/build/spconv/src:\
$(dirname $BEVFUSION_ROOT)/build:$LD_LIBRARY_PATH
```

## 4. Maps and NAVSIM data

Follow [`docs/install.md`](docs/install.md) for OpenScene / nuPlan map downloads
(respect those dataset licenses). Typical workspace layout:

```
$WORKSPACE/
  maps/
  mini_navsim_logs/mini/          # or trainval / test logs
  mini_sensor_blobs/mini/         # or full sensor blobs
  checkpoints/
  onnx/
  metric_cache/
```

## 5. Smoke test

```bash
python scripts/training/viz_gtrs_bevfusion_seg.py \
  --sensor-blobs-path $WORKSPACE/mini_sensor_blobs/mini \
  --checkpoint $WORKSPACE/checkpoints/gtrs_bevfusion_navtrain_v1_best.pth \
  --out /tmp/sophi_viz.png \
  --eval-mode --show-gt-boxes --num-frames 1
```

Expect: `loaded checkpoint ... (missing=0 unexpected=0)`.

## Next steps

| Goal | Doc |
|------|-----|
| Download checkpoint / ONNX | [`docs/MODELS.md`](docs/MODELS.md) |
| Export → TensorRT → C++ inference | [`docs/DEPLOY.md`](docs/DEPLOY.md) |
| Train / metric cache / PDM scoring | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| SCN ONNX details | [`docs/LIDAR_ONNX_CPP_GUIDE.md`](docs/LIDAR_ONNX_CPP_GUIDE.md) |
| Upstream NAVSIM concepts | [`docs/agents.md`](docs/agents.md), [`docs/metrics.md`](docs/metrics.md) |
| License boundaries | [`NOTICE`](NOTICE) |
