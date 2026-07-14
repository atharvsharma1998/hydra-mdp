# GTRS-BEVFusion C++ deployment

End-to-end C++/TensorRT inference for the GTRS-BEVFusion model (fused camera+LiDAR
`F_env` → detection + BEV segmentation + HydraMDP planning), published alongside
the NAVSIM training/export code.

## Layout

```
deploy/
├─ src/gtrs/              # authored here
│  ├─ camera_frontend.*   # roiconvert: raw frames -> normalized CHW fp16 (6 cams)
│  ├─ lidar_frontend.*    # 5-ch range crop before voxelization
│  ├─ gtrs_heads.*        # decoders: DETR det (+NMS), seg argmax, planner
│  ├─ trt_module.hpp      # name-agnostic TensorRT wrapper (fuser + 3 heads)
│  ├─ gtrs_pipeline.*     # orchestrator (SCN + camera LSS + fuser + heads)
│  └─ gtrs_main.cpp       # entry point + cuOSD BEV render
├─ tool/build_gtrs_engines.sh   # trtexec: ONNX -> .plan (6 graphs)
├─ tool/run_gtrs.sh             # build engines + compile + run
└─ CMakeLists.txt
```

## Dependencies

The heavy BEVFusion **core** (camera/lidar/fuser CUDA modules, the custom
spconv-2.x ONNX parser + open-source `libspconv`, the TensorRT wrapper,
`nv::Tensor`, cuOSD, roiconvert) is reused from the CUDA-BEVFusion fork via
`BEVFUSION_ROOT`. We link its prebuilt `libbevfusion_core.so` instead of
vendoring it.

* CUDA 11.x, TensorRT 8.6 (`trtexec` at `/usr/local/TensorRT-8.6.1.6/bin`)
* `libspconv.so` (`SPCONV_ROOT`, default `/home/atharv/spconv`)
* `libbevfusion_core.so` — build once in the fork

## Pipeline graphs

| ONNX (from `scripts/export/export_gtrs_bevfusion_onnx.py`) | runtime |
|---|---|
| camera.backbone / camera.vtransform / fuser / planning_head / det_head / seg_head | TensorRT engines |
| lidar.backbone.xyz.onnx | custom spconv parser + open-source `libspconv` |

## Build & run

```bash
# 1) build the core once in the fork
(cd "$BEVFUSION_ROOT" && bash tool/run.sh)   # produces build/libbevfusion_core.so

# 2) build engines + the deploy binary + run
BEVFUSION_ROOT=/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion \
GTRS_ONNX_DIR=/home/atharv/Downloads/hydramdp/navsim_workspace/onnx \
GTRS_DATA=/path/to/example-data \
bash deploy/tool/run_gtrs.sh fp16
```

`example-data/` must contain (same format as the fork): `camera2lidar.tensor`,
`camera_intrinsics.tensor`, `lidar2image.tensor`, `img_aug_matrix.tensor`,
`points.tensor`, `0-FRONT.jpg … 5-BACK_RIGHT.jpg`, and optionally `status.tensor`
(ego status `[24]`). Outputs: console detections/plan, `build/gtrs-bev.jpg`,
`build/gtrs_seg_256x256_u8.bin`, `build/gtrs_trajectory.txt`.

## Full cookbook (train → PDM → ONNX → TRT → parity)

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) and
[`../../jmlr/REPRODUCIBILITY.md`](../../jmlr/REPRODUCIBILITY.md) for the JMLR MLOSS
end-to-end recipe. After re-exporting ONNX from a new checkpoint, **delete stale
`.plan` engines** before `build_gtrs_engines.sh` (the script skips existing plans).

**How the LiDAR ONNX runs in C++** (custom parser vs ONNX Runtime / TensorRT):
[`../docs/LIDAR_ONNX_CPP_GUIDE.md`](../docs/LIDAR_ONNX_CPP_GUIDE.md).

## Licensing

Deploy sources in this tree are Apache-2.0 (see repo `LICENSE` / `jmlr/NOTICE`).
CUDA and TensorRT are required at runtime; the LiDAR SCN path uses open
`spconv` 2.x rather than NVIDIA's closed `libspconv`.

## Calibration notes (verify on first real run)

* Camera resize convention (roiconvert stretch vs. aspect-preserving crop) must
  match the training feature builder / `img_aug_matrix`.
* `geometry_dim` / BEVPool grid (200×200, C=80) and the BEV render
  pixels-per-meter are set from the exported shapes — confirm against a sample.
