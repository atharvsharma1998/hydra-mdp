# Models and ONNX artifacts

Large weights are **not** shipped inside the JMLR source tarball.
Publish them on the GitHub Release for tag `v0.1.0-mloss` (or a mirror you control),
then keep this page as the download index.

Reviewed checkpoint: **`gtrs_bevfusion_navtrain_v1_best.pth`**
(navtrain → navtest PDM **0.7925**).

## Recommended Release layout

Attach to https://github.com/atharvsharma1998/hydra-mdp/releases/tag/v0.1.0-mloss :

```
gtrs_bevfusion_navtrain_v1_best.pth          # PyTorch checkpoint (~488 MB)
sophi_onnx_navtrain_v1.tar.gz                # exported ONNX graphs (~175 MB)
```

Pack ONNX locally before upload:

```bash
cd $WORKSPACE/onnx
tar -czf /tmp/sophi_onnx_navtrain_v1.tar.gz \
  camera.backbone.onnx camera.vtransform.onnx fuser.onnx \
  planning_head.onnx det_head.onnx seg_head.onnx lidar.backbone.onnx
```

After upload, set the URLs below to the release asset links.

## Downloads (fill after publishing Release assets)

| Artifact | Size (approx.) | SHA256 | URL |
|----------|----------------|--------|-----|
| PyTorch checkpoint | 488 MB | `4b1a919b76cab0d6737ea6c68eb22dc3b10b0c5ff318d2161339657179db0df2` | _add Release URL_ |
| ONNX bundle | ~175 MB | see per-file hashes below | _add Release URL_ |

### Per-file ONNX SHA256 (navtrain_v1 export, 2026-07-12)

| File | SHA256 |
|------|--------|
| `camera.backbone.onnx` | `360f1786fb8845a0ceb6e246ef88c47cc1f7a4cbcc3a7d0fa49e9cbcc7fa2964` |
| `camera.vtransform.onnx` | `c80abf5f98974c7c6d844ae548df7321c2f989d7608d5233aa4c23df0e15df7e` |
| `fuser.onnx` | `0de8e0732fc759d2d8a101f6fb1e3f180a2d8ffed02100d507a017eed1b99352` |
| `planning_head.onnx` | `15cccba6641dd7a1987ed0d37cd71fe8d99aada59d332a290bd759125df12f19` |
| `det_head.onnx` | `ddc15606be65303a3b01325f207d96bcd0aef5cd08ac1e2e82053697d4210b45` |
| `seg_head.onnx` | `7fae5306167f04359eb61e193da944268a6dc50e9496a6d3e7adec258953a4da` |
| `lidar.backbone.onnx` | `0236921aef4c68ccf5cf7a9b66f80a1b06a3e0cb2e07123d020c541ae5fb428b` |

Verify after download:

```bash
sha256sum -c <<'EOF'
4b1a919b76cab0d6737ea6c68eb22dc3b10b0c5ff318d2161339657179db0df2  gtrs_bevfusion_navtrain_v1_best.pth
EOF
```

## Install into the workspace layout

```bash
mkdir -p $WORKSPACE/checkpoints $WORKSPACE/onnx
# after downloading the Release assets:
mv gtrs_bevfusion_navtrain_v1_best.pth $WORKSPACE/checkpoints/
tar -xzf sophi_onnx_navtrain_v1.tar.gz -C $WORKSPACE/onnx
```

Then follow [`DEPLOY.md`](DEPLOY.md) for TensorRT engines and C++ inference,
or re-export ONNX yourself from the `.pth` if you prefer a from-scratch path.

## What is / is not redistributed

| Item | In git repo | In JMLR code tarball | On GitHub Release |
|------|-------------|----------------------|-------------------|
| Source + docs | yes | yes | n/a |
| Demo GIF (`assets/demo/`) | yes (~1.7 MB) | yes | n/a |
| PyTorch `.pth` | no | no | **yes (recommended)** |
| ONNX graphs | no | no | **yes (recommended)** |
| TensorRT `.plan` | no | no | optional (GPU-specific; prefer rebuild) |
| NAVSIM sensor data | no | no | no (benchmark license) |

TensorRT engines are tied to GPU / TRT version; users should rebuild with
`deploy/tool/build_gtrs_engines.sh fp16` from the published ONNX.
