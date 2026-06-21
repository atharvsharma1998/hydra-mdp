"""Visualize GTRS-BEVFusion seg head vs GT to calibrate F_env orientation.

Renders a 4-panel PNG for one frame:
    [front camera | GT bev_semantic | predicted bev_semantic | LiDAR occupancy]
so we can confirm the F_env -> NAVSIM seg-frame mapping (forward-half crop/flip).

Example:
    python scripts/training/viz_gtrs_bevfusion_seg.py \
        --workspace /home/atharv/Downloads/hydramdp/navsim_workspace \
        --sensor-blobs-path /home/atharv/Downloads/hydramdp/navsim_workspace/openscene-v1.1/sensor_blobs/mini \
        --checkpoint /home/atharv/Downloads/hydramdp/navsim_workspace/checkpoints/gtrs_bevfusion_latest.pth \
        --out /home/atharv/Downloads/hydramdp/navsim_workspace/seg_calib.png
"""
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--workspace", type=str, default="/home/atharv/Downloads/hydramdp/navsim_workspace")
parser.add_argument("--maps-path", type=str, default=None)
_a, _ = parser.parse_known_args()
if "NUPLAN_MAPS_ROOT" not in os.environ:
    os.environ["NUPLAN_MAPS_ROOT"] = str(Path(_a.maps_path) if _a.maps_path else Path(_a.workspace) / "maps")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from navsim.common.dataclasses import SensorConfig, SceneFilter
from navsim.common.enums import LidarIndex
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_features import BEVFusionFeatureBuilder
from navsim.agents.gtrs_bevfusion.bevfusion_model import GTRSBevfusionModel
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from train_modular_planner import LazySceneLoader

# 7-class discrete colormap (0=bg ... 6)
_CMAP = ListedColormap(
    ["#000000", "#404040", "#00a0ff", "#ffd000", "#ff5050", "#00ff00", "#ff00ff"]
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=str, default="/home/atharv/Downloads/hydramdp/navsim_workspace")
    p.add_argument("--maps-path", type=str, default=None)
    p.add_argument("--sensor-blobs-path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--out", type=str, default="seg_calib.png")
    args = p.parse_args()

    ws = Path(args.workspace)
    sensor_blobs_path = Path(args.sensor_blobs_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GTRSBevfusionConfig()
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)

    log_paths = [p for p in [ws / "mini_navsim_logs" / "mini"] if p.exists()]
    loader = LazySceneLoader(
        original_sensor_path=sensor_blobs_path,
        data_paths=log_paths,
        scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True),
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )
    tokens = [
        t for t in loader.tokens
        if (sensor_blobs_path / loader.token_to_slice[t][0].name.replace(".pkl", "") / "MergedPointCloud").is_dir()
        and (sensor_blobs_path / loader.token_to_slice[t][0].name.replace(".pkl", "") / "CAM_F0").is_dir()
    ]
    assert len(tokens) > 0, "no tokens with sensors"
    token = tokens[0]
    print("viz token:", token)

    scene = loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()
    feats = BEVFusionFeatureBuilder(config).compute_features(agent_input)
    tgts = TransfuserTargetBuilder(trajectory_sampling=ts, config=TransfuserConfig()).compute_targets(scene)

    model = GTRSBevfusionModel(config, num_poses=ts.num_poses).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict(sd)
        print("loaded checkpoint", args.checkpoint)
    # NOTE: small-batch/few-step overfit checkpoints have unreliable BatchNorm
    # running stats; use batch stats (train mode) for a faithful viz. Real
    # training (large batch, many steps) makes eval-mode stats valid.
    model.train()

    batched = {k: ([v.to(device)] if k == "lidar" else v.unsqueeze(0).to(device)) for k, v in feats.items()}
    with torch.no_grad():
        out = model(batched)
    pred = out["bev_semantic_map"][0].argmax(0).cpu().numpy()
    gt = tgts["bev_semantic_map"].cpu().numpy().astype(int)

    # lidar occupancy in seg frame (same mapping as TransfuserTargetBuilder._coords_to_pixel)
    H, W = config.bev_seg_frame
    pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T  # (P,3)
    occ = np.zeros((H, W), dtype=np.float32)
    px = (pc[:, 0] / 0.25).astype(int)            # x (forward) -> row
    py = (pc[:, 1] / 0.25 + W / 2).astype(int)    # y (lateral) -> col
    m = (px >= 0) & (px < H) & (py >= 0) & (py < W)
    occ[px[m], py[m]] = 1.0
    occ = np.rot90(occ)[::-1]  # match transfuser frame ops

    front = agent_input.cameras[-1].cam_f0.image

    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    ax[0].imshow(front); ax[0].set_title("front camera (CAM_F0)")
    ax[1].imshow(gt, cmap=_CMAP, vmin=0, vmax=6); ax[1].set_title("GT bev_semantic")
    ax[2].imshow(pred, cmap=_CMAP, vmin=0, vmax=6); ax[2].set_title("pred bev_semantic")
    ax[3].imshow(occ, cmap="gray"); ax[3].set_title("LiDAR occupancy (seg frame)")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
