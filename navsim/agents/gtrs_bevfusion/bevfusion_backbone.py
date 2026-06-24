# SPDX-License-Identifier: Apache-2.0
"""F_env backbone: the trained CUDA-BEVFusion perception net, heads removed.

This wraps the *exact* fused camera+LiDAR BEVFusion model the user already
trained on nuScenes (mmdet3d fork in ``CUDA-BEVFusion/bevfusion``) and reuses it
as an ``F_env`` extractor for the NAVSIM HydraMDP-style agent:

    camera (ResNet50 -> GeneralizedLSSFPN -> DepthLSSTransform)  ->  cam BEV  [B, 80,  H, W]
    lidar  (Voxelization -> SparseEncoder)                       ->  lid BEV  [B, 256, H, W]
    ConvFuser([cam, lid]) -> SECOND (decoder.backbone) -> SECONDFPN (decoder.neck)
                                                                ->  F_env    [B, 512, H, W]

The nuScenes detection/map heads are dropped; NAVSIM planning/detection/seg
heads attach downstream off ``F_env``. Geometry stays nuScenes-native (so the
trained weights and grid buffers load cleanly); NAVSIM sensors are mapped into
that frame by the feature builder. The camera branch is view-count agnostic
(applied per-view via a ``B*N`` reshape), so NAVSIM's camera set is fine.

Warm-start source (local): runs/mapfix_posw/epoch_7.pth (best joint det+seg).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# The mmdet3d *fork* (with ConvFuser/DepthLSSTransform/SparseEncoder + open spconv
# runtime) lives in the CUDA-BEVFusion tree, not in any installed mmdet3d.
_BEVFUSION_ROOT = os.environ.get(
    "BEVFUSION_ROOT",
    "/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion",
)

_DEFAULT_CONFIG = os.path.join(
    _BEVFUSION_ROOT,
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50_v0p075/convfuser_mapfix_posw.yaml",
)
_DEFAULT_CHECKPOINT = (
    "/home/atharv/planner_release/workspace/runs/mapfix_posw/epoch_7.pth"
)


def _ensure_bevfusion_on_path() -> None:
    if _BEVFUSION_ROOT not in sys.path:
        sys.path.insert(0, _BEVFUSION_ROOT)


def _neck_first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def _patch_geometry(cfg, g) -> None:
    """Override the loaded nuScenes geometry with NAVSIM ranges from ``g``.

    Keeps the camera-LSS BEV and LiDAR SparseEncoder BEV on the same grid so the
    ConvFuser concat is valid. ``g`` is a GTRSBevfusionConfig-like object.
    """
    pcr = list(g.point_cloud_range)
    vox = list(g.voxel_size)
    enc = cfg.model["encoders"]

    # camera view transform (DepthLSSTransform)
    vt = enc["camera"]["vtransform"]
    vt["xbound"] = list(g.lss_xbound)
    vt["ybound"] = list(g.lss_ybound)
    vt["zbound"] = list(g.lss_zbound)
    vt["dbound"] = list(g.lss_dbound)
    vt["downsample"] = g.lss_downsample
    vt["image_size"] = list(g.image_size)
    vt["feature_size"] = [g.image_size[0] // 8, g.image_size[1] // 8]

    # lidar voxelization + sparse encoder
    vox_cfg = enc["lidar"]["voxelize"]
    vox_cfg["point_cloud_range"] = pcr
    vox_cfg["voxel_size"] = vox
    vox_cfg["max_voxels"] = list(g.max_voxels)
    vox_cfg["max_num_points"] = g.max_points_per_voxel
    enc["lidar"]["backbone"]["sparse_shape"] = list(g.sparse_shape)
    enc["lidar"]["backbone"]["in_channels"] = g.lidar_in_channels


class BEVFusionBackbone(nn.Module):
    """Trained BEVFusion (encoders + fuser + decoder) exposed as an F_env extractor.

    forward(...) consumes the same tensors ``BEVFusion.forward_single`` expects and
    returns the fused BEV feature ``F_env`` of shape ``[B, out_channels, H, W]``.
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        checkpoint_path: Optional[str] = _DEFAULT_CHECKPOINT,
        out_channels: int = 512,
        freeze: bool = False,
        geometry: Optional["object"] = None,
    ) -> None:
        super().__init__()
        _ensure_bevfusion_on_path()

        # Imports are deferred until after sys.path is patched so we pick up the fork.
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model
        from mmdet3d.utils import recursive_eval
        from torchpack.utils.config import configs

        try:
            configs.clear()  # torchpack global config is dict-like
        except Exception:
            pass
        configs.load(config_path, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_path)

        if geometry is not None:
            _patch_geometry(cfg, geometry)

        # ``test_cfg`` lives under ``model`` in this fork; build_model wants train+test.
        model = build_model(
            cfg.model,
            train_cfg=cfg.model.get("train_cfg"),
            test_cfg=cfg.model.get("test_cfg"),
        )

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            # strict=False: heads may be pruned / NAVSIM-side changes are fine.
            load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False)
            self._warm_started_from = checkpoint_path
        else:
            self._warm_started_from = None

        # Drop the nuScenes heads + their loss machinery; we only need F_env.
        if hasattr(model, "heads"):
            del model.heads
        model.heads = nn.ModuleDict()

        self.bevfusion = model
        self.out_channels = out_channels

        if freeze:
            for p in self.bevfusion.parameters():
                p.requires_grad_(False)
            self.bevfusion.eval()
        self._frozen = freeze

    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            # Keep BN/dropout in eval when the perception net is frozen.
            self.bevfusion.eval()
        return self

    @property
    def warm_started_from(self) -> Optional[str]:
        return self._warm_started_from

    # ------------------------------------------------------------------ #
    def forward(
        self,
        img: torch.Tensor,
        points: List[torch.Tensor],
        camera2ego: torch.Tensor,
        lidar2ego: torch.Tensor,
        lidar2camera: torch.Tensor,
        lidar2image: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2lidar: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        lidar_aug_matrix: torch.Tensor,
        metas: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """Return F_env BEV feature [B, out_channels, H, W].

        ``img``    : [B, N, C, H, W]   (N = number of cameras)
        ``points`` : list of length B, each [P_i, 5]  (x, y, z, intensity, dt)
        transform tensors follow mmdet3d BEVFusion conventions.
        """
        m = self.bevfusion
        if metas is None:
            metas = [{} for _ in range(img.shape[0])]

        # BEVFusion's LSS view-transform clamps depth to 1e5, which overflows
        # fp16 (max ~65504). It was never written for autocast, so run the whole
        # F_env extractor in fp32 regardless of any outer AMP context. The
        # planning / detection / segmentation heads still benefit from AMP.
        with torch.cuda.amp.autocast(enabled=False):
            img = img.float()
            camera2ego = camera2ego.float()
            lidar2ego = lidar2ego.float()
            lidar2camera = lidar2camera.float()
            lidar2image = lidar2image.float()
            camera_intrinsics = camera_intrinsics.float()
            camera2lidar = camera2lidar.float()
            img_aug_matrix = img_aug_matrix.float()
            lidar_aug_matrix = lidar_aug_matrix.float()
            points = [p.float() for p in points]

            features = []
            # Match training order (camera then lidar) so ConvFuser sees [cam, lid].
            for sensor in m.encoders:
                if sensor == "camera":
                    feat = m.extract_camera_features(
                        img,
                        points,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas,
                    )
                elif sensor == "lidar":
                    feat = m.extract_lidar_features(points)
                else:
                    raise ValueError(f"unsupported sensor: {sensor}")
                features.append(feat)

            if m.fuser is not None:
                x = m.fuser(features)
            else:
                assert len(features) == 1
                x = features[0]

            x = m.decoder["backbone"](x)
            x = m.decoder["neck"](x)
            return _neck_first(x)
