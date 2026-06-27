# SPDX-License-Identifier: Apache-2.0
"""Full-surround (360 deg) target builder for the GTRS-BEVFusion agent.

The stock ``TransfuserTargetBuilder`` rasterizes a *forward-only* BEV semantic
frame: ``bev_pixel_height = lidar_resolution_height // 2`` with the ego pinned to
the rear edge (``pixel_center = [0, W/2]`` -> "remove half in backward
direction"). That throws away everything behind the ego, so the rear/side
cameras can't contribute to the map.

Here we keep the *exact* same rasterization + rot90/flip pipeline (so the
row=x(forward) / col=y(left) orientation convention is preserved and stays
aligned with F_env), but:

  * use a full ``(256, 256)`` square frame covering x,y in [-32, 32] m, and
  * center the ego (``pixel_center = [H/2, W/2]``) so x<0 (behind) is kept.

Because only the frame size + center change, the forward half of this 360 GT is
identical to the old forward-only GT (handy for orientation verification).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserTargetBuilder, BoundingBox2DIndex
from navsim.common.enums import BoundingBoxIndex

# Default multi-class detection set (CUDA-BEVFusion style; see GTRSBevfusionConfig).
DEFAULT_DET_CLASSES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier")


@dataclass
class Seg360Config(TransfuserConfig):
    """TransfuserConfig with a full-square (ego-centered) BEV semantic frame.

    Only ``bev_pixel_height`` changes (``//2`` removed) so the frame becomes
    ``(256, 256)`` instead of the forward-only ``(128, 256)``. Everything else
    (classes, pixel size, lidar extent, detection range) is inherited.
    """

    bev_pixel_height: int = TransfuserConfig.lidar_resolution_height  # 256 (full, no //2)


class BEVFusionTargetBuilder(TransfuserTargetBuilder):
    """Transfuser targets, but with a 360 deg ego-centered BEV semantic map and
    multi-class detection targets (adds ``agent_classes``)."""

    def __init__(self, trajectory_sampling, config: TransfuserConfig = None,
                 detection_class_names=DEFAULT_DET_CLASSES):
        super().__init__(trajectory_sampling=trajectory_sampling, config=config or Seg360Config())
        self._det_classes = tuple(detection_class_names)
        self._det_class_to_id = {n: i for i, n in enumerate(self._det_classes)}

    def get_unique_name(self) -> str:
        # distinct from "transfuser_target" so cached forward-only targets are
        # never silently reused for the 360 frame. The detection-class set is in
        # the name so a different class set never reuses a stale cache.
        return "gtrs_bevfusion_target_360_det" + str(len(self._det_classes))

    def compute_targets(self, scene):
        """Base targets (traj + seg + binary boxes), then overwrite the detection
        targets with the multi-class version (adds ``agent_classes``)."""
        targets = super().compute_targets(scene)
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        states, valid, classes = self._compute_agent_targets_multiclass(annotations)
        targets["agent_states"] = states
        targets["agent_labels"] = valid
        targets["agent_classes"] = classes
        return targets

    def _compute_agent_targets_multiclass(self, annotations):
        """Like ``TransfuserTargetBuilder._compute_agent_targets`` but keeps all
        configured classes and records a per-box class id (nearest-K by range)."""
        cfg = self._config
        max_agents = cfg.num_bounding_boxes

        def _in_lidar(x, y):
            return (cfg.lidar_min_x <= x <= cfg.lidar_max_x) and (cfg.lidar_min_y <= y <= cfg.lidar_max_y)

        states_list, cls_list = [], []
        for box, name in zip(annotations.boxes, annotations.names):
            if name not in self._det_class_to_id:
                continue
            x, y = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]
            if not _in_lidar(x, y):
                continue
            states_list.append(np.array(
                [x, y, box[BoundingBoxIndex.HEADING], box[BoundingBoxIndex.LENGTH],
                 box[BoundingBoxIndex.WIDTH]], dtype=np.float32))
            cls_list.append(self._det_class_to_id[name])

        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        agent_classes = np.zeros(max_agents, dtype=np.int64)
        if states_list:
            arr = np.asarray(states_list, dtype=np.float32)
            cls = np.asarray(cls_list, dtype=np.int64)
            order = np.argsort(np.linalg.norm(arr[..., BoundingBox2DIndex.POINT], axis=-1))[:max_agents]
            arr, cls = arr[order], cls[order]
            n = len(arr)
            agent_states[:n] = arr
            agent_labels[:n] = True
            agent_classes[:n] = cls
        return torch.tensor(agent_states), torch.tensor(agent_labels), torch.tensor(agent_classes)

    def _coords_to_pixel(self, coords):
        """Local (x fwd, y left) metres -> pixel idcs, with the ego CENTERED.

        Mirrors ``TransfuserTargetBuilder._coords_to_pixel`` but offsets x by
        ``H/2`` (instead of 0) so the rear half (x<0) lands inside the frame.
        """
        pixel_center = np.array([[self._config.bev_pixel_height / 2.0,
                                  self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)
