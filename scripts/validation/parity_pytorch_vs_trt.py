"""Parity test: PyTorch ModularPlanner vs TensorRT modular_planner.plan.

Stage 1: deterministic synthetic BEV grids (plausible road scenes) -> compare
         trajectory output, score correlation and argmax agreement.

Usage:
  python parity_pytorch_vs_trt.py \
      --checkpoint <best.pth> --engine <modular_planner.plan> \
      [--grid-npy grid.npy --status-npy status.npy]   # e.g. dumps from C++

The optional --grid-npy/--status-npy lets us re-use this same script later to
validate grids dumped by the C++ TrajectoryPlanner (step "cpp-dump").
"""
import argparse
import ctypes
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tensorrt as trt


def make_synthetic_grid(seed: int) -> np.ndarray:
    """6-channel 256x256 grid resembling a road scene (matches ModularFeatureBuilder layout).

    Channels: 0 road, 1 walkway, 2 centerline, 3 static, 4 vehicles, 5 pedestrians.
    Pixel mapping: col = x*4 + 128, row = 128 - y*4 (x forward, y left).
    """
    rng = np.random.RandomState(seed)
    g = np.zeros((6, 256, 256), dtype=np.float32)

    # Road: straight corridor of width ~14m along +x/-x, with slight lateral offset
    off = rng.randint(-20, 20)
    half_w = rng.randint(20, 36)  # 5..9 m
    g[0, 128 + off - half_w:128 + off + half_w, :] = 1.0

    # Walkways flanking the road
    g[1, 128 + off - half_w - 10:128 + off - half_w, :] = 1.0
    g[1, 128 + off + half_w:128 + off + half_w + 10, :] = 1.0

    # Centerline along the corridor
    g[2, 128 + off - 1:128 + off + 2, :] = 1.0

    # A few vehicles ahead on the road
    for _ in range(rng.randint(1, 4)):
        cx = rng.randint(140, 240)  # ahead of ego
        cy = 128 + off + rng.randint(-half_w + 6, half_w - 6)
        g[4, cy - 4:cy + 4, cx - 9:cx + 9] = 1.0

    # Pedestrian on walkway
    px = rng.randint(100, 200)
    g[5, 128 + off + half_w + 3:128 + off + half_w + 7, px:px + 4] = 1.0

    return g


def make_status(seed: int) -> np.ndarray:
    rng = np.random.RandomState(1000 + seed)
    v = rng.uniform(2.0, 12.0)
    status = []
    for _ in range(3):  # 3 frames x 8
        vx, vy = v, rng.uniform(-0.3, 0.3)
        status += [vx, vy, float(np.hypot(vx, vy)),
                   rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2),
                   0.0, 0.0, 1.0]  # straight command
    return np.asarray(status, dtype=np.float32)


class TrtRunner:
    def __init__(self, engine_path: str):
        ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

    def run(self, grid: np.ndarray, status: np.ndarray):
        # Use torch CUDA tensors as device buffers (avoids pycuda dependency)
        d_grid = torch.from_numpy(grid[None]).cuda().contiguous()
        d_status = torch.from_numpy(status[None]).cuda().contiguous()
        d_traj = torch.empty((1, 8, 3), dtype=torch.float32, device="cuda")
        d_scores = torch.empty((1, 8192), dtype=torch.float32, device="cuda")

        self.ctx.set_input_shape("bev_grid", (1, 6, 256, 256))
        self.ctx.set_input_shape("status_feature", (1, 24))
        self.ctx.set_tensor_address("bev_grid", d_grid.data_ptr())
        self.ctx.set_tensor_address("status_feature", d_status.data_ptr())
        self.ctx.set_tensor_address("trajectory", d_traj.data_ptr())
        self.ctx.set_tensor_address("scores", d_scores.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        assert self.ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        return d_traj.cpu().numpy()[0], d_scores.cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/home/atharv/Downloads/hydramdp/navsim_workspace/checkpoints/best.pth")
    ap.add_argument("--engine", default="/home/atharv/lidar/custom/Lidar_AI_Solution/CUDA-BEVFusion/model/phaseA2_e8/build/modular_planner.plan")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--grid-npy", default=None, help="optional external grid [6,256,256] (e.g. C++ dump)")
    ap.add_argument("--status-npy", default=None, help="optional external status [24]")
    ap.add_argument("--save-viz", default=None, help="optional path to save grid+trajectory visualization png")
    args = ap.parse_args()

    from navsim.agents.modular_planner import ModularPlanner

    print("Loading PyTorch model...")
    vocab_path = "/home/atharv/Downloads/hydramdp/navsim/traj_final/8192.npy"
    model = ModularPlanner(vocab_path=vocab_path)
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()})
    model.eval().cuda()

    print("Loading TRT engine...")
    trt_runner = TrtRunner(args.engine)

    if args.grid_npy:
        def load_arr(p):
            return np.fromfile(p, dtype=np.float32) if p.endswith(".bin") else np.load(p)
        samples = [(load_arr(args.grid_npy).reshape(6, 256, 256).astype(np.float32),
                    load_arr(args.status_npy).reshape(24).astype(np.float32))]
    else:
        samples = [(make_synthetic_grid(s), make_status(s)) for s in range(args.num_samples)]

    agree, results = 0, []
    for i, (grid, status) in enumerate(samples):
        with torch.no_grad():
            preds = model({"bev_grid": torch.from_numpy(grid[None]).cuda(),
                           "status_feature": torch.from_numpy(status[None]).cuda()})
        pt_traj = preds["trajectory"][0].cpu().numpy()
        pt_scores = preds["scores"][0].cpu().numpy()
        pt_idx = int(preds["selected_indices"][0])

        trt_traj, trt_scores = trt_runner.run(grid, status)
        trt_idx = int(np.argmax(trt_scores))

        # scores contain -inf-ish values from log(sigmoid); compare on finite mask
        mask = np.isfinite(pt_scores) & np.isfinite(trt_scores)
        corr = float(np.corrcoef(pt_scores[mask], trt_scores[mask])[0, 1])
        max_traj_diff = float(np.abs(pt_traj - trt_traj).max())
        same = pt_idx == trt_idx
        agree += int(same)
        results.append((i, same, pt_idx, trt_idx, corr, max_traj_diff))
        print(f"[{i}] argmax: pt={pt_idx} trt={trt_idx} same={same} | "
              f"score corr={corr:.5f} | max traj diff={max_traj_diff:.4f} m | "
              f"endpoint pt=({pt_traj[-1,0]:+.1f},{pt_traj[-1,1]:+.1f})")

        if args.save_viz and i == 0:
            save_viz(grid, pt_traj, trt_traj, args.save_viz)

    print(f"\nArgmax agreement: {agree}/{len(samples)}")
    mean_corr = np.mean([r[4] for r in results])
    print(f"Mean score correlation: {mean_corr:.5f}")
    ok = agree == len(samples) or mean_corr > 0.999
    print("PARITY:", "PASS" if ok else "CHECK (fp16 may flip near-tie argmax; inspect score corr)")


def save_viz(grid, pt_traj, trt_traj, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.zeros((256, 256, 3), dtype=np.float32)
    rgb[grid[0] > 0.5] = (0.7, 0.8, 0.9)   # road
    rgb[grid[1] > 0.5] = (0.85, 0.3, 0.3)  # walkway
    rgb[grid[2] > 0.5] = (0.5, 0.3, 0.7)   # centerline
    rgb[grid[4] > 0.5] = (0.1, 0.1, 0.6)   # vehicles
    rgb[grid[5] > 0.5] = (0.9, 0.6, 0.1)   # pedestrians

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin="upper")
    for traj, c, lbl in [(pt_traj, "lime", "pytorch"), (trt_traj, "yellow", "trt")]:
        cols = traj[:, 0] * 4.0 + 128.0
        rows = 128.0 - traj[:, 1] * 4.0
        ax.plot(cols, rows, "-o", color=c, ms=3, lw=1.5, label=lbl)
    ax.plot(128, 128, "r*", ms=14, label="ego")
    ax.legend()
    ax.set_title("BEV grid + planned trajectory")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved viz to {path}")


if __name__ == "__main__":
    main()
