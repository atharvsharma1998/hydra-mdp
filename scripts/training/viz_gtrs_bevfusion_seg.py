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
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_features import BEVFusionFeatureBuilder
from navsim.agents.gtrs_bevfusion.bevfusion_model import GTRSBevfusionModel
from navsim.agents.gtrs_bevfusion.bevfusion_target import BEVFusionTargetBuilder
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
    tgts = BEVFusionTargetBuilder(trajectory_sampling=ts).compute_targets(scene)

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

    # detection boxes: GT (from targets) and predicted (label prob > 0.5)
    gt_states = tgts["agent_states"].cpu().numpy()
    gt_mask = tgts["agent_labels"].cpu().numpy().astype(bool)
    pred_states = pred_mask = None
    if "agent_states" in out:
        pred_states = out["agent_states"][0].cpu().numpy()
        pred_mask = (out["agent_labels"][0].sigmoid().cpu().numpy() > 0.5)

    def _draw_boxes(ax, states, mask, color):
        if states is None:
            return
        for i in range(states.shape[0]):
            if not mask[i]:
                continue
            x, y, hd, length, width = states[i, :5]
            corners = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                                [-length / 2, -width / 2], [-length / 2, width / 2]])
            rot = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
            world = corners @ rot.T + np.array([x, y])  # (4,2) in ego (x fwd, y left)
            # plot as (y, x): horizontal=lateral, vertical=forward
            ax.add_patch(plt.Polygon(np.stack([world[:, 1], world[:, 0]], axis=1),
                                     closed=True, fill=False, edgecolor=color, lw=1.5))

    # 3x3 grid: row0/1 = 6 cameras, row2 = GT seg | pred seg | detection BEV
    cams = agent_input.cameras[-1]
    cam_order = ["cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_b0", "cam_r1"]
    fig, ax = plt.subplots(3, 3, figsize=(18, 16))
    for i, name in enumerate(cam_order):
        a = ax[i // 3, i % 3]
        a.imshow(getattr(cams, name).image)
        a.set_title(name.upper()); a.axis("off")

    ax[2, 0].imshow(gt, cmap=_CMAP, vmin=0, vmax=6); ax[2, 0].set_title("GT bev_semantic (360)")
    ax[2, 1].imshow(pred, cmap=_CMAP, vmin=0, vmax=6); ax[2, 1].set_title("pred bev_semantic (360)")
    ax[2, 0].axis("off"); ax[2, 1].axis("off")

    bx = ax[2, 2]
    bx.set_title("detection: GT=green pred=red")
    bx.set_xlim(32, -32); bx.set_ylim(-32, 32)  # left(+y) on left, forward(+x) up
    bx.set_aspect("equal"); bx.grid(True, alpha=0.2)
    bx.plot(0, 0, "k*", ms=10)  # ego
    _draw_boxes(bx, gt_states, gt_mask, "green")
    _draw_boxes(bx, pred_states, pred_mask, "red")

    plt.tight_layout()
    plt.savefig(args.out, dpi=100, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
