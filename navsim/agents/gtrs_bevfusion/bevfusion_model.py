# SPDX-License-Identifier: Apache-2.0
"""Unified GTRS-BEVFusion model: F_env -> {planning, detection, BEV-seg}.

    sensors --(BEVFusionBackbone)--> F_env [B, 512, 100, 100]
        |-- PlanningHead(in=512)           -> HydraMDP trajectory scoring
        |-- det decoder + AgentHead        -> agent_states / agent_labels
        |-- BEVSegHead                     -> bev_semantic_map  [B, C, 128, 256]

Geometry (option A): F_env is a NAVSIM-aligned +/-32 m square BEV (see
GTRSBevfusionConfig). The detection head predicts metric coordinates directly
(extent-agnostic) and the planner pools tokens (extent-agnostic). Only the seg
head is extent-sensitive; its mapping from the +/-32 square to NAVSIM's
forward-only (128, 256) frame is marked for one-time orientation calibration
against a real sample (see ``BEVSegHead``).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.gtrs_bevfusion.bevfusion_backbone import BEVFusionBackbone
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.agents.modular_planner import PlanningHead
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


class MultiClassAgentHead(nn.Module):
    """DETR-style multi-class box head: regresses 2D boxes and predicts a class
    distribution over ``num_classes`` + 1 (background) per query.

    Output keys:
      * ``agent_states``       [B, N, 5]   (x, y, heading, length, width)
      * ``agent_class_logits`` [B, N, K+1] (last index = background/no-object)
      * ``agent_labels``       [B, N]      objectness logit (= -background logit),
                                           kept for back-compat / thresholded viz.
    """

    def __init__(self, num_agents: int, num_classes: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_classes = num_classes
        self._mlp_states = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, BoundingBox2DIndex.size()),
        )
        self._mlp_label = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, num_classes + 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        states = self._mlp_states(agent_queries)
        states[..., BoundingBox2DIndex.POINT] = states[..., BoundingBox2DIndex.POINT].tanh() * 32
        states[..., BoundingBox2DIndex.HEADING] = states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        class_logits = self._mlp_label(agent_queries)  # [B, N, K+1]
        agent_labels = -class_logits[..., -1]  # objectness = logit(not background)
        return {"agent_states": states, "agent_class_logits": class_logits, "agent_labels": agent_labels}


class BEVSegHead(nn.Module):
    """Conv classifier over F_env producing NAVSIM single-label BEV logits.

    360 deg (full-surround) map: F_env already covers the full +/-32 m square, so
    we classify the WHOLE feature (no forward-half crop) and resample to the
    ego-centered (256, 256) GT frame produced by ``BEVFusionTargetBuilder``.

    Orientation (CALIBRATED): an F_env axis probe + single-scene overfit confirmed
    F_env rows = x (forward -> high rows), cols = y (left -> high cols), matching
    NAVSIM's GT seg frame after its rot90/flip. The 360 GT reuses that exact
    rasterization (just ego-centered + full square), so the convention is
    preserved end to end -> no transpose/flip needed (kept as toggles for re-cal).
    GroupNorm (not BatchNorm) keeps train/eval identical and small-batch robust.
    """

    def __init__(self, in_channels: int, num_classes: int, out_hw, mid_channels: int = 128):
        super().__init__()
        self.out_hw = tuple(out_hw)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )
        self.transpose_xy = False  # orientation re-cal toggle (see docstring)
        self.flip_rows = False     # set True only if viz shows forward at low rows

    def forward(self, fenv: torch.Tensor) -> torch.Tensor:
        x = fenv
        if self.transpose_xy:
            x = x.transpose(-1, -2)
        if self.flip_rows:
            x = torch.flip(x, dims=[-2])
        # full-surround: classify the whole F_env (no forward-half crop)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=self.out_hw, mode="bilinear", align_corners=False)
        return logits


class GTRSBevfusionModel(nn.Module):
    """End-to-end perception+planning model emitting F_env-driven heads."""

    def __init__(
        self,
        config: GTRSBevfusionConfig,
        num_poses: int,
        backbone_checkpoint=None,
        det_token_grid=(16, 16),
    ):
        super().__init__()
        self._config = config
        self._det_token_grid = det_token_grid

        self.backbone = BEVFusionBackbone(
            checkpoint_path=backbone_checkpoint,
            out_channels=config.fused_bev_channels,
            geometry=config,
        )
        c = config.fused_bev_channels

        # ---- planning head (HydraMDP decoder) ----
        self.planning_head = PlanningHead(
            in_channels=c,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            nhead=config.nhead,
            nlayers=config.nlayers,
            num_poses=config.num_poses,
            vocab_path=config.vocab_path,
            num_ego_status=config.num_ego_status,
            bev_spatial_shape=config.env_token_grid,
            vocab_dropout_size=config.vocab_dropout_size,
        )

        # ---- detection head (DETR-style decoder over pooled F_env) ----
        if config.use_detection_head:
            d = config.d_model
            n_tokens = det_token_grid[0] * det_token_grid[1]
            self.det_pool = nn.AdaptiveAvgPool2d(det_token_grid)
            self.det_downscale = nn.Conv2d(c, d, kernel_size=1)
            self.det_status_encoding = nn.Linear(config.num_ego_status * 8, d)
            self.det_keyval_embedding = nn.Embedding(n_tokens + 1, d)
            self.det_query_embedding = nn.Embedding(config.num_bounding_boxes, d)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d, nhead=config.nhead, dim_feedforward=config.d_ffn,
                dropout=0.0, batch_first=True,
            )
            self.det_decoder = nn.TransformerDecoder(dec_layer, num_layers=config.nlayers)
            # multi-class (CUDA-BEVFusion style) when >1 class is configured;
            # otherwise the legacy binary (vehicle-only) transfuser head.
            if config.num_detection_classes > 1:
                self.agent_head = MultiClassAgentHead(
                    num_agents=config.num_bounding_boxes,
                    num_classes=config.num_detection_classes,
                    d_ffn=config.d_ffn, d_model=d,
                )
            else:
                self.agent_head = AgentHead(
                    num_agents=config.num_bounding_boxes, d_ffn=config.d_ffn, d_model=d,
                )

        # ---- BEV segmentation head ----
        if config.use_bev_seg_head:
            self.seg_head = BEVSegHead(
                in_channels=c, num_classes=config.num_bev_classes, out_hw=config.bev_seg_frame,
            )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        status_feature = features["status_feature"].to(torch.float32)
        points: List[torch.Tensor] = features["lidar"]  # list of [P,5] per sample

        fenv = self.backbone(
            img=features["img"],
            points=points,
            camera2ego=features["camera2ego"],
            lidar2ego=features["lidar2ego"],
            lidar2camera=features["lidar2camera"],
            lidar2image=features["lidar2image"],
            camera_intrinsics=features["camera_intrinsics"],
            camera2lidar=features["camera2lidar"],
            img_aug_matrix=features["img_aug_matrix"],
            lidar_aug_matrix=features["lidar_aug_matrix"],
        )

        output: Dict[str, torch.Tensor] = {}

        # planning
        plan = self.planning_head(fenv, status_feature)
        selected_trajectory = plan["selected_trajectory"]
        if selected_trajectory.shape[1] == 40:
            selected_trajectory = selected_trajectory[:, 4::5]
        output["trajectory"] = selected_trajectory
        for k in [
            "selected_indices", "scores", "vocab", "imi", "indices",
            "no_at_fault_collisions", "drivable_area_compliance",
            "time_to_collision_within_bound", "ego_progress",
            "driving_direction_compliance", "lane_keeping", "traffic_light_compliance",
        ]:
            if k in plan:
                output[k] = plan[k]

        # detection
        if self._config.use_detection_head:
            B = fenv.shape[0]
            tokens = self.det_downscale(self.det_pool(fenv)).flatten(-2, -1).permute(0, 2, 1)
            status_enc = self.det_status_encoding(status_feature)
            keyval = torch.cat([tokens, status_enc[:, None]], dim=1)
            keyval = keyval + self.det_keyval_embedding.weight[None]
            query = self.det_query_embedding.weight[None].repeat(B, 1, 1)
            agent_q = self.det_decoder(query, keyval)
            output.update(self.agent_head(agent_q))

        # bev segmentation
        if self._config.use_bev_seg_head:
            output["bev_semantic_map"] = self.seg_head(fenv)

        return output
