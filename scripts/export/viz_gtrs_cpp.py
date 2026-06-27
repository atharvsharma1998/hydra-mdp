"""Visualize the C++ (deployed TensorRT) GTRS-BEVFusion outputs on 6 cameras.

Same multi-panel layout as scripts/training/viz_gtrs_bevfusion_seg.py (the .pth
visualizer), but instead of running the PyTorch model it overlays the outputs the
C++ binary wrote (detections / trajectory / seg). Use it to eyeball that the
deployed pipeline produces the same scene understanding as the training model.

Pipeline:
  1) dump a token's inputs + record token:
       python scripts/export/export_gtrs_bevfusion_onnx.py --checkpoint ... \
           --sensor-blobs-path ... --dump-cpp-inputs deploy/parity-data
  2) run the C++ binary on that token (writes build/gtrs_{detections,trajectory}.txt
     and build/gtrs_seg_256x256_u8.bin):
       ./build/gtrs_bevfusion parity-data gtrs_bevfusion fp16
  3) render:
       python scripts/export/viz_gtrs_cpp.py \
           --sensor-blobs-path .../mini_sensor_blobs/mini \
           --cpp-dir deploy/build --token-file deploy/parity-data/token.txt \
           --out deploy/build/gtrs_cpp_viz.png
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--workspace", type=str, default="/home/atharv/Downloads/hydramdp/navsim_workspace")
parser.add_argument("--maps-path", type=str, default=None)
_a, _ = parser.parse_known_args()
if "NUPLAN_MAPS_ROOT" not in os.environ:
    os.environ["NUPLAN_MAPS_ROOT"] = str(Path(_a.maps_path) if _a.maps_path else Path(_a.workspace) / "maps")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from navsim.common.dataclasses import SensorConfig, SceneFilter
from navsim.common.enums import LidarIndex
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.gtrs_bevfusion.bevfusion_target import BEVFusionTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from train_gtrs_bevfusion import LazySceneLoader

_CMAP = ListedColormap(["#000000", "#404040", "#00a0ff", "#ffd000", "#ff5050", "#00ff00", "#ff00ff"])
_CLS_COLORS = ["#00ff00", "#ff00ff", "#ffa500", "#ffff00", "#00bfff", "#ff7fbf", "#7fff7f", "#bf7fff"]
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _cam_calib(camera):
    K = np.asarray(camera.intrinsics, dtype=np.float64)
    c2l = np.eye(4, dtype=np.float64)
    c2l[:3, :3] = np.asarray(camera.sensor2lidar_rotation, dtype=np.float64)
    c2l[:3, 3] = np.asarray(camera.sensor2lidar_translation, dtype=np.float64)
    return K, np.linalg.inv(c2l)


def _project(pts3d, l2c, K):
    ph = np.c_[pts3d, np.ones(len(pts3d))]
    cam = (l2c @ ph.T).T[:, :3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-3, None)
    return uv, z


def _box3d_corners(state, z0, height):
    x, y, hd, length, width = state[:5]
    c = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                  [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
    base = c @ rot.T + np.array([x, y])
    return np.vstack([np.c_[base, np.full(4, z0)], np.c_[base, np.full(4, z0 + height)]])


def _draw_boxes_on_img(ax, states, classes, l2c, K, z0, height, imw, imh, ls="--"):
    if states is None:
        return
    for i in range(states.shape[0]):
        uv, z = _project(_box3d_corners(states[i], z0, height), l2c, K)
        if (z < 0.5).any():
            continue
        if uv[:, 0].max() < 0 or uv[:, 0].min() > imw or uv[:, 1].max() < 0 or uv[:, 1].min() > imh:
            continue
        color = _CLS_COLORS[int(classes[i]) % len(_CLS_COLORS)]
        for a, b in _BOX_EDGES:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=color, lw=1.2, ls=ls)


def _draw_traj_on_img(ax, traj, color, l2c, K, z0):
    pts = np.c_[traj[:, 0], traj[:, 1], np.full(len(traj), z0)]
    uv, z = _project(pts, l2c, K)
    uv = uv[z > 0.5]
    if len(uv) >= 2:
        ax.plot(uv[:, 0], uv[:, 1], "-", color=color, lw=2.5)
        ax.scatter(uv[:, 0], uv[:, 1], s=10, color=color, zorder=5)


def _draw_boxes_bev(ax, states, classes, ls="--"):
    if states is None:
        return
    for i in range(states.shape[0]):
        x, y, hd, length, width = states[i, :5]
        corners = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                            [-length / 2, -width / 2], [-length / 2, width / 2]])
        rot = np.array([[np.cos(hd), -np.sin(hd)], [np.sin(hd), np.cos(hd)]])
        world = corners @ rot.T + np.array([x, y])
        color = _CLS_COLORS[int(classes[i]) % len(_CLS_COLORS)]
        ax.add_patch(plt.Polygon(np.stack([world[:, 1], world[:, 0]], axis=1),
                                 closed=True, fill=False, edgecolor=color, lw=2.0, ls=ls))


def _load_detections(path):
    """build/gtrs_detections.txt -> (states[N,5], classes[N], scores[N])."""
    if not os.path.exists(path):
        return None, None, None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    if not rows:
        return np.zeros((0, 5)), np.zeros(0, int), np.zeros(0)
    arr = np.array(rows)  # cls x y hd len wid score
    classes = arr[:, 0].astype(int)
    states = arr[:, 1:6]  # x y heading length width
    scores = arr[:, 6]
    return states, classes, scores


def _load_traj(path):
    if not os.path.exists(path):
        return None
    t = np.loadtxt(path)
    return t.reshape(-1, 3) if t.size else None


def _load_seg(path, h, w):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.uint8).reshape(h, w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=str, default="/home/atharv/Downloads/hydramdp/navsim_workspace")
    p.add_argument("--maps-path", type=str, default=None)
    p.add_argument("--sensor-blobs-path", type=str, required=True)
    p.add_argument("--cpp-dir", type=str, default="/home/atharv/Downloads/hydramdp/navsim/deploy/build",
                   help="dir with gtrs_detections.txt / gtrs_trajectory.txt / gtrs_seg_256x256_u8.bin")
    p.add_argument("--token", type=str, default=None)
    p.add_argument("--token-file", type=str, default=None,
                   help="read the token from this file (token.txt written by --dump-cpp-inputs)")
    p.add_argument("--out", type=str, default="gtrs_cpp_viz.png")
    p.add_argument("--show-gt", action="store_true", help="also draw GT boxes/traj for comparison")
    args = p.parse_args()

    if args.token is None and args.token_file:
        args.token = Path(args.token_file).read_text().strip()
    assert args.token, "provide --token or --token-file"

    ws = Path(args.workspace)
    sb = Path(args.sensor_blobs_path)
    config = GTRSBevfusionConfig()
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)

    loader = LazySceneLoader(
        original_sensor_path=sb,
        data_paths=[ws / "mini_navsim_logs" / "mini"],
        scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True),
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )
    assert args.token in loader.tokens, f"token {args.token} not found among {len(loader.tokens)} tokens"
    scene = loader.get_scene_from_token(args.token)
    agent_input = scene.get_agent_input()
    target_builder = BEVFusionTargetBuilder(trajectory_sampling=ts)
    tgts = target_builder.compute_targets(scene)

    # ---- C++ outputs ----
    seg_h, seg_w = config.bev_seg_frame
    pred_states, pred_classes, pred_scores = _load_detections(os.path.join(args.cpp_dir, "gtrs_detections.txt"))
    pred_traj = _load_traj(os.path.join(args.cpp_dir, "gtrs_trajectory.txt"))
    pred_seg = _load_seg(os.path.join(args.cpp_dir, "gtrs_seg_256x256_u8.bin"), seg_h, seg_w)
    print(f"C++ outputs: {0 if pred_states is None else len(pred_states)} boxes, "
          f"traj={'yes' if pred_traj is not None else 'no'}, seg={'yes' if pred_seg is not None else 'no'}")

    gt = tgts["bev_semantic_map"].cpu().numpy().astype(int)
    gt_traj = tgts["trajectory"].cpu().numpy() if "trajectory" in tgts else None
    if gt_traj is not None and gt_traj.ndim == 1:
        gt_traj = gt_traj.reshape(-1, 3)
    gt_states = gt_classes = None
    if args.show_gt:
        gt_states = tgts["agent_states"].cpu().numpy()
        gt_mask = tgts["agent_labels"].cpu().numpy().astype(bool)
        gt_states = gt_states[gt_mask]
        gt_classes = (tgts["agent_classes"].cpu().numpy()[gt_mask] if "agent_classes" in tgts
                      else np.zeros(len(gt_states), int))

    pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T
    ground_z = float(np.percentile(pc[:, 2], 5))
    box_h = 1.6

    cams = agent_input.cameras[-1]
    cam_order = ["cam_l0", "cam_f0", "cam_r0", "cam_l1", "cam_b0", "cam_r1"]
    fig, ax = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f"C++ (TensorRT) inference  token={args.token}", fontsize=12)
    for i, name in enumerate(cam_order):
        a = ax[i // 3, i % 3]
        cam = getattr(cams, name)
        img = cam.image
        a.imshow(img)
        imh, imw = img.shape[:2]
        K, l2c = _cam_calib(cam)
        if args.show_gt and gt_states is not None:
            _draw_boxes_on_img(a, gt_states, gt_classes, l2c, K, ground_z, box_h, imw, imh, ls="-")
        _draw_boxes_on_img(a, pred_states, pred_classes, l2c, K, ground_z, box_h, imw, imh, ls="--")
        if args.show_gt and gt_traj is not None:
            _draw_traj_on_img(a, gt_traj, "white", l2c, K, ground_z)
        if pred_traj is not None:
            _draw_traj_on_img(a, pred_traj, "red", l2c, K, ground_z)
        a.set_xlim(0, imw); a.set_ylim(imh, 0)
        a.set_title(name.upper()); a.axis("off")

    ax[2, 0].imshow(gt, cmap=_CMAP, vmin=0, vmax=6); ax[2, 0].set_title("GT bev_semantic (360)")
    if pred_seg is not None:
        ax[2, 1].imshow(pred_seg, cmap=_CMAP, vmin=0, vmax=6)
    ax[2, 1].set_title("C++ pred bev_semantic (360)")
    for a in (ax[2, 0], ax[2, 1]):
        a.plot(seg_w / 2, seg_h / 2, "w+", ms=14, mew=2)
        a.axis("off")

    bx = ax[2, 2]
    note = "GT=solid C++=dashed" if args.show_gt else "C++=dashed"
    bx.set_title(f"BEV: boxes {note} (color=class) | traj GT=white C++=red")
    bx.set_xlim(32, -32); bx.set_ylim(-32, 32)
    bx.set_aspect("equal"); bx.grid(True, alpha=0.2)
    ego_l, ego_w = 4.6, 2.0
    ego_c = np.array([[ego_l / 2, ego_w / 2], [ego_l / 2, -ego_w / 2],
                      [-ego_l / 2, -ego_w / 2], [-ego_l / 2, ego_w / 2]])
    bx.add_patch(plt.Polygon(np.stack([ego_c[:, 1], ego_c[:, 0]], axis=1), closed=True,
                             facecolor="0.6", edgecolor="black", lw=1.5, zorder=4))
    bx.arrow(0, 0, 0, 4.0, color="black", head_width=1.0, length_includes_head=True, zorder=4)
    if args.show_gt and gt_states is not None:
        _draw_boxes_bev(bx, gt_states, gt_classes, ls="-")
    _draw_boxes_bev(bx, pred_states, pred_classes, ls="--")
    if args.show_gt and gt_traj is not None:
        bx.plot(gt_traj[:, 1], gt_traj[:, 0], "-o", color="white", lw=3.5, ms=4, zorder=5)
    if pred_traj is not None:
        bx.plot(pred_traj[:, 1], pred_traj[:, 0], "--o", color="red", lw=2, ms=3, zorder=6)
    handles = [Line2D([0], [0], color=_CLS_COLORS[i % len(_CLS_COLORS)], lw=3, label=n)
               for i, n in enumerate(config.detection_class_names)]
    handles += [Line2D([0], [0], color="red", marker="o", lw=2, ls="--", label="C++ traj")]
    if args.show_gt:
        handles += [Line2D([0], [0], color="white", marker="o", lw=2, label="GT traj")]
    bx.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.5)

    plt.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    main()
