# Models (Google Drive)

Large weights are hosted on Google Drive (not in git).

| Artifact | File | Google Drive |
|----------|------|--------------|
| PyTorch checkpoint (navtrain → navtest PDM 0.7925) | `gtrs_bevfusion_navtrain_v1_best.pth` | **[PASTE_DRIVE_LINK_PTH]** |
| ONNX graphs (optional shortcut; skip `.pth`→ONNX export) | `gtrs_bevfusion_onnx_navtrain_v1.tar.gz` | **[PASTE_DRIVE_LINK_ONNX]** |

> Replace the two placeholders above with your shareable Drive links before publishing.

## After download

```bash
mkdir -p $WORKSPACE/checkpoints $WORKSPACE/onnx

# checkpoint
mv /path/to/gtrs_bevfusion_navtrain_v1_best.pth $WORKSPACE/checkpoints/

# optional: pre-exported ONNX (skip export step in QUICKSTART.md)
tar -xzf /path/to/gtrs_bevfusion_onnx_navtrain_v1.tar.gz -C $WORKSPACE/onnx
```

ONNX zip should contain:

```
camera.backbone.onnx
camera.vtransform.onnx
fuser.onnx
planning_head.onnx
det_head.onnx
seg_head.onnx
lidar.backbone.onnx
```

Next: [`QUICKSTART.md`](../QUICKSTART.md) (TensorRT → C++).

## What is in the repo vs Drive

| Item | In git | On Drive |
|------|--------|----------|
| Source + docs | yes | — |
| `deploy/example-data/` (1 frame) | yes (~11 MB) | — |
| Demo GIF | yes | — |
| `.pth` checkpoint | no | yes |
| ONNX graphs | no | yes (optional) |
| TensorRT `.plan` | no | no (rebuild per GPU) |
