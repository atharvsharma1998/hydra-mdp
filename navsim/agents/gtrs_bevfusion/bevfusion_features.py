# SPDX-License-Identifier: Apache-2.0
"""Feature builder for the GTRS-BEVFusion agent.

Turns a NAVSIM ``AgentInput`` into the tensors a BEVFusion (mmdet3d) backbone
needs to produce ``F_env``:

  * ``img``                : [N, 3, H, W]  normalized multi-view images
  * ``camera_intrinsics``  : [N, 4, 4]
  * ``camera2lidar``       : [N, 4, 4]
  * ``lidar2camera``       : [N, 4, 4]
  * ``lidar2image``        : [N, 4, 4]   (intrinsics @ lidar2camera)
  * ``camera2ego``         : [N, 4, 4]   (== camera2lidar; NAVSIM lidar2ego = I)
  * ``img_aug_matrix``     : [N, 4, 4]   (anisotropic resize orig->(H,W))
  * ``lidar2ego``          : [4, 4]      (identity in NAVSIM)
  * ``lidar_aug_matrix``   : [4, 4]      (identity; no train-time 3D aug here)
  * ``lidar``              : [P, 5]      (x, y, z, intensity, t=0)
  * ``status_feature``     : [8]         (driving_command[4] + vel[2] + accel[2])

Calibration follows mmdet3d BEVFusion conventions so the trained/define-from-
scratch DepthLSSTransform + SparseEncoder consume them directly. The default
camera set is the 3 forward cameras (cam_l0, cam_f0, cam_r0), matching the
NAVSIM Transfuser/HydraMDP frontal setup; extendable to all 8 via config.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.common.dataclasses import AgentInput, Camera
from navsim.common.enums import LidarIndex
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


def _camera2lidar_4x4(camera: Camera) -> np.ndarray:
    """Build a 4x4 camera->lidar transform from NAVSIM sensor2lidar calib."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.asarray(camera.sensor2lidar_rotation, dtype=np.float32)
    mat[:3, 3] = np.asarray(camera.sensor2lidar_translation, dtype=np.float32)
    return mat


def _intrinsics_4x4(camera: Camera) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.asarray(camera.intrinsics, dtype=np.float32)
    return mat


class BEVFusionFeatureBuilder(AbstractFeatureBuilder):
    """Builds multi-view + LiDAR + status features for the BEVFusion backbone."""

    def __init__(self, config: GTRSBevfusionConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "gtrs_bevfusion_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        cameras = agent_input.cameras[-1]
        target_h, target_w = self._config.image_size

        imgs: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        camera2lidar: List[np.ndarray] = []
        lidar2camera: List[np.ndarray] = []
        lidar2image: List[np.ndarray] = []
        img_aug: List[np.ndarray] = []

        mean = np.array(self._config.img_norm_mean, dtype=np.float32)
        std = np.array(self._config.img_norm_std, dtype=np.float32)

        for cam_name in self._config.camera_names:
            camera: Camera = getattr(cameras, cam_name)
            image = camera.image
            orig_h, orig_w = image.shape[:2]

            # Anisotropic resize to the backbone input size; record the exact
            # scale so the LSS projection stays consistent (img_aug_matrix).
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized = resized.astype(np.float32) / 255.0
            resized = (resized - mean) / std
            imgs.append(resized.transpose(2, 0, 1))  # HWC -> CHW

            sw = target_w / float(orig_w)
            sh = target_h / float(orig_h)
            aug = np.eye(4, dtype=np.float32)
            aug[0, 0] = sw
            aug[1, 1] = sh

            cam2lid = _camera2lidar_4x4(camera)
            lid2cam = np.linalg.inv(cam2lid).astype(np.float32)
            intrin = _intrinsics_4x4(camera)
            # Apply the image-resize to intrinsics so lidar2image lands in the
            # resized image frame (consistent with img_aug_matrix bookkeeping).
            lid2img = (aug @ intrin @ lid2cam).astype(np.float32)

            intrinsics.append(intrin)
            camera2lidar.append(cam2lid)
            lidar2camera.append(lid2cam)
            lidar2image.append(lid2img)
            img_aug.append(aug)

        # NAVSIM: lidar frame == ego frame (lidar2ego is identity).
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar_aug = np.eye(4, dtype=np.float32)
        camera2ego = np.stack(camera2lidar, axis=0)  # == camera2lidar

        features: Dict[str, torch.Tensor] = {
            "img": torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32),
            "camera_intrinsics": torch.tensor(np.stack(intrinsics, axis=0)),
            "camera2lidar": torch.tensor(np.stack(camera2lidar, axis=0)),
            "lidar2camera": torch.tensor(np.stack(lidar2camera, axis=0)),
            "lidar2image": torch.tensor(np.stack(lidar2image, axis=0)),
            "camera2ego": torch.tensor(camera2ego),
            "img_aug_matrix": torch.tensor(np.stack(img_aug, axis=0)),
            "lidar2ego": torch.tensor(lidar2ego),
            "lidar_aug_matrix": torch.tensor(lidar_aug),
            "lidar": self._get_lidar(agent_input),
            "status_feature": self._get_status(agent_input),
        }
        return features

    def _get_lidar(self, agent_input: AgentInput) -> torch.Tensor:
        """Raw points as [P, 5] = (x, y, z, intensity, t=0) for the voxelizer."""
        pc = agent_input.lidars[-1].lidar_pc  # (6, P)
        xyz = pc[LidarIndex.POSITION].T  # (P, 3)
        intensity = pc[LidarIndex.INTENSITY][:, None]  # (P, 1)
        time = np.zeros_like(intensity)  # single sweep
        points = np.concatenate([xyz, intensity, time], axis=1).astype(np.float32)

        # Range-crop to the configured BEV extent (drops far returns early).
        x_min, y_min, z_min, x_max, y_max, z_max = self._config.point_cloud_range
        m = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return torch.tensor(points[m])

    def _get_status(self, agent_input: AgentInput) -> torch.Tensor:
        """Concatenate the last ``num_ego_status`` frames -> [num_ego_status*8]."""
        n = self._config.num_ego_status
        statuses = agent_input.ego_statuses[-n:]
        # pad (repeat earliest) if fewer history frames are available
        while len(statuses) < n:
            statuses = [statuses[0]] + statuses
        per_frame = [
            torch.cat(
                [
                    torch.tensor(s.driving_command, dtype=torch.float32),
                    torch.tensor(s.ego_velocity, dtype=torch.float32),
                    torch.tensor(s.ego_acceleration, dtype=torch.float32),
                ]
            )
            for s in statuses
        ]
        return torch.cat(per_frame)
