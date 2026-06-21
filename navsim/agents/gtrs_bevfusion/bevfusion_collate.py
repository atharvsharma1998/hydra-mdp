# SPDX-License-Identifier: Apache-2.0
"""Custom collate for GTRS-BEVFusion.

LiDAR point clouds are variable-length per sample, so they cannot be stacked.
Everything else (images, calibration matrices, status, targets) stacks normally.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def bevfusion_collate(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
):
    """Collate a list of (features, targets) into batched dicts.

    ``features['lidar']`` is kept as a list of [P_i, 5] tensors (for the mmdet3d
    voxelizer); all other feature/target tensors are stacked along a new batch dim.
    """
    features_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]

    features: Dict[str, object] = {}
    for key in features_list[0].keys():
        if key == "lidar":
            features[key] = [f[key] for f in features_list]
        else:
            features[key] = torch.stack([f[key] for f in features_list], dim=0)

    targets: Dict[str, object] = {}
    for key in targets_list[0].keys():
        if key == "gt_scores":
            # nested dict of per-metric score tensors (PDM distillation)
            gt_scores: Dict[str, torch.Tensor] = {}
            for sk in targets_list[0]["gt_scores"].keys():
                gt_scores[sk] = torch.stack([t["gt_scores"][sk] for t in targets_list], dim=0)
            targets[key] = gt_scores
        else:
            targets[key] = torch.stack([t[key] for t in targets_list], dim=0)

    return features, targets
