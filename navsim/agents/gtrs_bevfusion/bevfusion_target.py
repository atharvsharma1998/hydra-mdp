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

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserTargetBuilder


@dataclass
class Seg360Config(TransfuserConfig):
    """TransfuserConfig with a full-square (ego-centered) BEV semantic frame.

    Only ``bev_pixel_height`` changes (``//2`` removed) so the frame becomes
    ``(256, 256)`` instead of the forward-only ``(128, 256)``. Everything else
    (classes, pixel size, lidar extent, detection range) is inherited.
    """

    bev_pixel_height: int = TransfuserConfig.lidar_resolution_height  # 256 (full, no //2)


class BEVFusionTargetBuilder(TransfuserTargetBuilder):
    """Transfuser targets, but with a 360 deg ego-centered BEV semantic map."""

    def __init__(self, trajectory_sampling, config: TransfuserConfig = None):
        super().__init__(trajectory_sampling=trajectory_sampling, config=config or Seg360Config())

    def get_unique_name(self) -> str:
        # distinct from "transfuser_target" so cached forward-only targets are
        # never silently reused for the 360 frame.
        return "gtrs_bevfusion_target_360"

    def _coords_to_pixel(self, coords):
        """Local (x fwd, y left) metres -> pixel idcs, with the ego CENTERED.

        Mirrors ``TransfuserTargetBuilder._coords_to_pixel`` but offsets x by
        ``H/2`` (instead of 0) so the rear half (x<0) lands inside the frame.
        """
        pixel_center = np.array([[self._config.bev_pixel_height / 2.0,
                                  self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)
