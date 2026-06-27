# SPDX-License-Identifier: Apache-2.0
"""Config for the GTRS-BEVFusion agent.

This agent replaces ``ModularPlanner``'s placeholder perception (a ResNet18
over a GT-rasterized ``bev_grid``) with a real BEVFusion-style fused
camera+LiDAR BEV backbone that emits ``F_env`` features. Those feed:

  * the existing HydraMDP ``PlanningHead``  -> planning (PDM distill + imitation)
  * a detection head (Transfuser-style)     -> agent_states / agent_labels
  * a BEV segmentation head (cross-entropy)  -> map (viz / debug)

Geometry is tuned to NAVSIM ego-frame ranges (not nuScenes). The fused-BEV
perception modules come from the CUDA-BEVFusion mmdet3d fork, which is on the
PYTHONPATH in the unified environment.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GTRSBevfusionConfig:
    # ---------------- planner head (HydraMDP decoder; mirrors ModularPlanner) -------------
    vocab_path: str = "traj_final/8192.npy"
    d_model: int = 256
    d_ffn: int = 512
    nhead: int = 4
    nlayers: int = 2
    num_poses: int = 40
    num_ego_status: int = 3            # status_feature = num_ego_status * 8
    env_token_grid: Tuple[int, int] = (32, 32)   # PlanningHead bev_spatial_shape (1024 tokens)
    vocab_dropout_size: int = 2048

    # ---------------- which auxiliary heads are active -----------------
    use_detection_head: bool = True
    use_bev_seg_head: bool = True

    # ---------------- camera inputs (per-camera, for LSS view transform) ---------------
    # 6-camera surround: 3 front (l0,f0,r0) + 2 side (l1,r1) + 1 back (b0).
    # The feature builder is camera-agnostic (loops this list, builds per-camera
    # calibration); the agent already loads all 8 sensors, so this is the only
    # switch needed. Detection + planning become full-surround; BEV-seg stays
    # forward-only until a 360-deg seg GT is generated (see BEVSegHead).
    camera_names: Tuple[str, ...] = (
        "cam_l0", "cam_f0", "cam_r0",  # front trio
        "cam_l1", "cam_r1",            # left / right sides
        "cam_b0",                       # back
    )
    image_size: Tuple[int, int] = (256, 704)     # (H, W) fed to image backbone
    img_norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ---------------- LiDAR / voxelization (ego frame) -----------------
    # Geometry is internally consistent so the camera-LSS BEV and the LiDAR
    # SparseEncoder BEV land on the SAME grid (required by ConvFuser concat):
    #   lidar BEV = (2*R)/voxel/8   = 64/0.08/8 = 100
    #   cam   BEV = (2*R)/xbound/ds = 64/0.32/2 = 100
    #   sparse_shape_xy = (2*R)/voxel = 800 ; z = (z_ext/0.2)+1 = 41
    # => fused/F_env spatial = 100x100, channels 512.
    point_cloud_range: Tuple[float, ...] = (-32.0, -32.0, -3.0, 32.0, 32.0, 5.0)
    voxel_size: Tuple[float, float, float] = (0.08, 0.08, 0.2)
    sparse_shape: Tuple[int, int, int] = (800, 800, 41)
    max_points_per_voxel: int = 10
    max_voxels: Tuple[int, int] = (90000, 120000)
    lidar_in_channels: int = 5        # x, y, z, intensity, t

    # ---------------- camera BEV (LSS) grid; XY matches point_cloud_range ---------------
    lss_xbound: Tuple[float, float, float] = (-32.0, 32.0, 0.32)
    lss_ybound: Tuple[float, float, float] = (-32.0, 32.0, 0.32)
    lss_zbound: Tuple[float, float, float] = (-10.0, 10.0, 20.0)
    lss_dbound: Tuple[float, float, float] = (1.0, 60.0, 0.5)
    camera_out_channels: int = 80
    lss_downsample: int = 2

    # ---------------- fuser / shared BEV decoder -----------------
    fuser_out_channels: int = 256
    fused_bev_channels: int = 512     # SECONDFPN concat -> F_env channels
    fused_bev_size: int = 100         # F_env spatial H=W (informational)

    # ---------------- detection head -----------------
    num_bounding_boxes: int = 30
    det_range: float = 32.0           # +/- meters used to filter/scale boxes
    latent: bool = False              # consumed by reused transfuser _agent_loss
    # Multi-class detection (CUDA-BEVFusion style). NAVSIM gt_names available:
    # vehicle, pedestrian, bicycle, traffic_cone, barrier, czone_sign,
    # generic_object. We keep the dynamic + safety-relevant classes and drop the
    # noisy `generic_object` catch-all + rare `czone_sign` (add them here to train
    # on them). Order defines the class id (0..K-1); a background class (id K) is
    # added internally by the head/loss. Single-entry list == legacy vehicle-only.
    detection_class_names: Tuple[str, ...] = (
        "vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier",
    )

    # DETR "eos_coef": weight of the background/no-object class in the detection
    # classification CE. <1 down-weights background so the head doesn't collapse
    # to predicting "nothing" (30 queries >> #objects). 0.1 is the DETR default.
    detection_bg_weight: float = 0.1

    @property
    def num_detection_classes(self) -> int:
        return len(self.detection_class_names)

    # ---------------- bev segmentation head (NAVSIM single-label CE) ---------------
    num_bev_classes: int = 7          # 0=background + 6 NAVSIM classes
    # Full-surround (360 deg) ego-centered BEV semantic frame: x,y in [-32,32] m
    # at 4 px/m => (256, 256). Built by BEVFusionTargetBuilder (Seg360Config).
    # (The old forward-only frame was (128, 256).)
    bev_seg_frame: Tuple[int, int] = (256, 256)   # (H, W) of bev_semantic_map GT
    # Per-class CE weights (idx 0=bg ... 6); rare classes (centerline, peds,
    # static) up-weighted. Calibrate against real class frequencies after the
    # first data pass.
    bev_class_weights: Tuple[float, ...] = (1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0)
    lovasz_loss_weight: float = 0.5

    # ---------------- loss weights -----------------
    trajectory_imi_weight: float = 1.0
    trajectory_distill_weight: float = 1.0
    agent_class_weight: float = 10.0
    # Box L1 supervises x,y position + size + heading. Kept too low (1.0) it gets
    # drowned by class/seg (both 10x) and position stays loose even after overfit
    # (DETR uses L1=5 vs class=1). Raised so position actually converges.
    agent_box_weight: float = 5.0
    bev_semantic_weight: float = 10.0

    def __post_init__(self):
        if self.fused_bev_channels <= 0:
            raise ValueError("fused_bev_channels must be > 0")
