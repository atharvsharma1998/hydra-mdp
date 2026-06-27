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
from navsim.agents.gtrs_bevfusion.bevfusion_target import BEVFusionTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from train_modular_planner import LazySceneLoader

# 7-class discrete colormap (0=bg ... 6)
_CMAP = ListedColormap(
    ["#000000", "#404040", "#00a0ff", "#ffd000", "#ff5050", "#00ff00", "#ff00ff"]
)

# 12 edges of a 3D box given 8 corners (0-3 bottom, 4-7 top)
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _cam_calib(camera):
    """Original-resolution intrinsics (3x3) + lidar->camera (4x4) for a Camera."""
    K = np.asarray(camera.intrinsics, dtype=np.float64)
    c2l = np.eye(4, dtype=np.float64)
    c2l[:3, :3] = np.asarray(camera.sensor2lidar_rotation, dtype=np.float64)
    c2l[:3, 3] = np.asarray(camera.sensor2lidar_translation, dtype=np.float64)
    return K, np.linalg.inv(c2l)


def _project(pts3d, l2c, K):
    """LiDAR/ego 3D points (N,3) -> image px (N,2) + camera-frame depth (N,)."""
    ph = np.c_[pts3d, np.ones(len(pts3d))]
    cam = (l2c @ ph.T).T[:, :3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-3, None)
    return uv, z


def _box3d_corners(state, z0, height):
    """BEV box (x,y,heading,length,width) -> 8 corners on ground plane z0."""
    x, y, hd, length, width = state[:5]
    c = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                  [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
    base = c @ rot.T + np.array([x, y])
    return np.vstack([np.c_[base, np.full(4, z0)], np.c_[base, np.full(4, z0 + height)]])


def _draw_boxes_on_img(ax, states, mask, classes, colors, l2c, K, z0, height, imw, imh, ls="-"):
    """Project 3D boxes onto a camera image (CUDA-BEVFusion style), per-class color."""
    if states is None:
        return
    for i in range(states.shape[0]):
        if not mask[i]:
            continue
        uv, z = _project(_box3d_corners(states[i], z0, height), l2c, K)
        if (z < 0.5).any():  # any corner behind the camera -> skip whole box
            continue
        if uv[:, 0].max() < 0 or uv[:, 0].min() > imw or uv[:, 1].max() < 0 or uv[:, 1].min() > imh:
            continue  # fully outside frame
        color = colors[int(classes[i]) % len(colors)]
        for a, b in _BOX_EDGES:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=color, lw=1.2, ls=ls)


def _draw_traj_on_img(ax, traj, color, l2c, K, z0):
    """Project an ego-frame trajectory (T,>=2) onto a camera as a ground polyline."""
    pts = np.c_[traj[:, 0], traj[:, 1], np.full(len(traj), z0)]
    uv, z = _project(pts, l2c, K)
    uv = uv[z > 0.5]
    if len(uv) >= 2:
        ax.plot(uv[:, 0], uv[:, 1], "-", color=color, lw=2.5)
        ax.scatter(uv[:, 0], uv[:, 1], s=10, color=color, zorder=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=str, default="/home/atharv/Downloads/hydramdp/navsim_workspace")
    p.add_argument("--maps-path", type=str, default=None)
    p.add_argument("--sensor-blobs-path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--out", type=str, default="seg_calib.png")
    p.add_argument("--scene-index", type=int, default=0,
                   help="start token (by index) among scenes that have sensors")
    p.add_argument("--token", type=str, default=None, help="explicit start token (overrides --scene-index)")
    p.add_argument("--num-frames", type=int, default=1,
                   help="render this many CONSECUTIVE scenes from the start index (+ a GIF)")
    p.add_argument("--all", action="store_true",
                   help="render EVERY sensor token in the split (overrides --num-frames/--scene-index)")
    p.add_argument("--show-gt-boxes", action="store_true",
                   help="also draw ground-truth detection boxes (default: predicted only)")
    p.add_argument("--score-thresh", type=float, default=0.2,
                   help="min foreground (non-background) class prob to draw a predicted box")
    p.add_argument("--nms-dist", type=float, default=2.0,
                   help="suppress lower-score predicted boxes within this many meters (center NMS)")
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
    if args.all:
        start, n_frames = 0, len(tokens)
    elif args.token is not None:
        assert args.token in tokens, f"token {args.token} not among {len(tokens)} sensor tokens"
        start = tokens.index(args.token)
        n_frames = max(1, args.num_frames)
    else:
        start = args.scene_index % len(tokens)
        n_frames = max(1, args.num_frames)
    sel_tokens = [tokens[(start + k) % len(tokens)] for k in range(n_frames)]
    print(f"rendering {n_frames} consecutive frame(s) from index {start} / {len(tokens)} sensor tokens")

    feature_builder = BEVFusionFeatureBuilder(config)
    target_builder = BEVFusionTargetBuilder(trajectory_sampling=ts)

    model = GTRSBevfusionModel(config, num_poses=ts.num_poses).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded checkpoint {args.checkpoint} (missing={len(missing)} unexpected={len(unexpected)})")
    # NOTE: small-batch/few-step overfit checkpoints have unreliable BatchNorm
    # running stats; use batch stats (train mode) for a faithful viz. Real
    # training (large batch, many steps) makes eval-mode stats valid.
    model.train()

    det_classes = list(config.detection_class_names)
    num_det = len(det_classes)
    # per-class colors (extend if you add classes)
    _CLS_COLORS = ["#00ff00", "#ff00ff", "#ffa500", "#ffff00", "#00bfff",
                   "#ff7fbf", "#7fff7f", "#bf7fff"]

    def _draw_boxes_bev(ax, states, mask, classes, ls="-"):
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
            color = _CLS_COLORS[int(classes[i]) % len(_CLS_COLORS)]
            ax.add_patch(plt.Polygon(np.stack([world[:, 1], world[:, 0]], axis=1),
                                     closed=True, fill=False, edgecolor=color, lw=2.0, ls=ls))

    def render_frame(token, frame_label):
        scene = loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        feats = feature_builder.compute_features(agent_input)
        tgts = target_builder.compute_targets(scene)

        batched = {k: ([v.to(device)] if k == "lidar" else v.unsqueeze(0).to(device)) for k, v in feats.items()}
        with torch.no_grad():
            out = model(batched)
        pred = out["bev_semantic_map"][0].argmax(0).cpu().numpy()
        gt = tgts["bev_semantic_map"].cpu().numpy().astype(int)

        # predicted detection boxes (per class). GT boxes only if requested.
        pred_states = pred_mask = pred_classes = None
        if "agent_class_logits" in out:  # multi-class head
            probs = out["agent_class_logits"][0].softmax(-1).cpu().numpy()  # [N, K+1]
            # Decode by FOREGROUND score (DETR-style): pick the best non-background
            # class and keep it if its prob clears the threshold. Using argmax over
            # all K+1 would always lose to the dominant background class.
            fg = probs[:, :num_det]                 # [N, K] foreground probs
            pred_classes = fg.argmax(-1)
            pred_scores = fg.max(-1)
            pred_mask = pred_scores > args.score_thresh
            pred_states = out["agent_states"][0].cpu().numpy()
            # greedy center-distance NMS: DETR-style heads emit duplicate queries
            # per object (no NMS built in), which look like phantom deviation.
            order = np.argsort(-pred_scores)
            for ii in order:
                if not pred_mask[ii]:
                    continue
                for jj in order:
                    if jj == ii or not pred_mask[jj]:
                        continue
                    if pred_scores[jj] <= pred_scores[ii] and \
                       np.linalg.norm(pred_states[ii, :2] - pred_states[jj, :2]) < args.nms_dist:
                        pred_mask[jj] = False
        elif "agent_states" in out:  # legacy binary head
            pred_states = out["agent_states"][0].cpu().numpy()
            pred_mask = (out["agent_labels"][0].sigmoid().cpu().numpy() > 0.5)
            pred_classes = np.zeros(len(pred_states), dtype=int)

        gt_states = gt_mask = gt_classes = None
        if args.show_gt_boxes:
            gt_states = tgts["agent_states"].cpu().numpy()
            gt_mask = tgts["agent_labels"].cpu().numpy().astype(bool)
            gt_classes = (tgts["agent_classes"].cpu().numpy() if "agent_classes" in tgts
                          else np.zeros(len(gt_states), dtype=int))

        pred_traj = out["trajectory"][0].cpu().numpy() if "trajectory" in out else None
        gt_traj = tgts["trajectory"].cpu().numpy() if "trajectory" in tgts else None
        if gt_traj is not None and gt_traj.ndim == 1:
            gt_traj = gt_traj.reshape(-1, 3)

        # ground plane for lifting boxes/traj into the cameras (low-pct LiDAR z)
        pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T  # (P,3)
        ground_z = float(np.percentile(pc[:, 2], 5))
        box_h = 1.6

        cams = agent_input.cameras[-1]
        cam_order = ["cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_b0", "cam_r1"]
        fig, ax = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(frame_label, fontsize=12)
        for i, name in enumerate(cam_order):
            a = ax[i // 3, i % 3]
            cam = getattr(cams, name)
            img = cam.image
            a.imshow(img)
            imh, imw = img.shape[:2]
            K, l2c = _cam_calib(cam)
            if args.show_gt_boxes:
                _draw_boxes_on_img(a, gt_states, gt_mask, gt_classes, _CLS_COLORS, l2c, K, ground_z, box_h, imw, imh, ls="-")
            _draw_boxes_on_img(a, pred_states, pred_mask, pred_classes, _CLS_COLORS, l2c, K, ground_z, box_h, imw, imh, ls="--")
            if gt_traj is not None:
                _draw_traj_on_img(a, gt_traj, "white", l2c, K, ground_z)
            if pred_traj is not None:
                _draw_traj_on_img(a, pred_traj, "red", l2c, K, ground_z)
            a.set_xlim(0, imw); a.set_ylim(imh, 0)
            a.set_title(name.upper()); a.axis("off")

        seg_h, seg_w = gt.shape
        ax[2, 0].imshow(gt, cmap=_CMAP, vmin=0, vmax=6); ax[2, 0].set_title("GT bev_semantic (360)")
        ax[2, 1].imshow(pred, cmap=_CMAP, vmin=0, vmax=6); ax[2, 1].set_title("pred bev_semantic (360)")
        for a in (ax[2, 0], ax[2, 1]):
            a.plot(seg_w / 2, seg_h / 2, "w+", ms=14, mew=2)  # ego
            a.axis("off")

        bx = ax[2, 2]
        gt_note = "GT=solid pred=dashed" if args.show_gt_boxes else "pred=dashed"
        bx.set_title(f"BEV: boxes {gt_note} (color=class) | traj GT=white pred=red")
        bx.set_xlim(32, -32); bx.set_ylim(-32, 32)  # left(+y) on left, forward(+x) up
        bx.set_aspect("equal"); bx.grid(True, alpha=0.2)
        ego_l, ego_w = 4.6, 2.0
        ego_c = np.array([[ego_l / 2, ego_w / 2], [ego_l / 2, -ego_w / 2],
                          [-ego_l / 2, -ego_w / 2], [-ego_l / 2, ego_w / 2]])
        bx.add_patch(plt.Polygon(np.stack([ego_c[:, 1], ego_c[:, 0]], axis=1), closed=True,
                                 facecolor="0.6", edgecolor="black", lw=1.5, zorder=4))
        bx.arrow(0, 0, 0, 4.0, color="black", head_width=1.0, length_includes_head=True, zorder=4)
        if args.show_gt_boxes:
            _draw_boxes_bev(bx, gt_states, gt_mask, gt_classes, ls="-")
        _draw_boxes_bev(bx, pred_states, pred_mask, pred_classes, ls="--")
        if gt_traj is not None:
            bx.plot(gt_traj[:, 1], gt_traj[:, 0], "-o", color="white", lw=3.5, ms=4, zorder=5)
        if pred_traj is not None:
            bx.plot(pred_traj[:, 1], pred_traj[:, 0], "--o", color="red", lw=2, ms=3, zorder=6)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=_CLS_COLORS[i % len(_CLS_COLORS)], lw=3, label=n)
                   for i, n in enumerate(det_classes)]
        handles += [Line2D([0], [0], color="white", marker="o", lw=2, label="GT traj"),
                    Line2D([0], [0], color="red", marker="o", lw=2, ls="--", label="pred traj")]
        bx.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.5)
        plt.tight_layout()
        return fig

    out_path = Path(args.out)
    saved = []
    for k, token in enumerate(sel_tokens):
        label = f"frame {k+1}/{n_frames}  idx={start + k}  token={token}"
        fig = render_frame(token, label)
        p = out_path if n_frames == 1 else out_path.with_name(f"{out_path.stem}_f{k:03d}{out_path.suffix}")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print("saved", p)

    if n_frames > 1:
        try:
            from PIL import Image
            frames = [Image.open(p).convert("RGB") for p in saved]
            gif_path = out_path.with_suffix(".gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
            print("saved", gif_path)
        except Exception as e:  # pragma: no cover - optional dep
            print("gif assembly skipped:", e)


if __name__ == "__main__":
    main()
