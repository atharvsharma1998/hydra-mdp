# example-data (one frame)

Single preprocessed sample for C++ inference — same idea as CUDA-BEVFusion’s
`example-data/`. You do **not** need to download the full NAVSIM dataset to try
the deploy path.

Contains:

- `camera.tensor` / `camera_rgb.tensor` — 6 surround cameras  
- `points.tensor` — LiDAR  
- calibration matrices (`camera2lidar`, `lidar2image`, …)  
- `status.tensor` — ego status  
- `gt_detections.txt` / `gt_trajectory.txt` — for viz overlays  
- `token.txt` — scene token `8bc34517e08758ff`

Run (after TensorRT engines are built):

```bash
cd deploy
./build/gtrs_bevfusion example-data gtrs_bevfusion fp16
```

See [`QUICKSTART.md`](../../QUICKSTART.md).
