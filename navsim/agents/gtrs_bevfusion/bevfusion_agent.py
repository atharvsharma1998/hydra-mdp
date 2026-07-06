# SPDX-License-Identifier: Apache-2.0
"""GTRS-BEVFusion agent: HydraMDP-style end-to-end model on NAVSIM.

Perception (fused camera+LiDAR BEVFusion) emits F_env, which drives a HydraMDP
trajectory decoder plus auxiliary detection and BEV-segmentation heads. Trained
from scratch on NAVSIM.

NOTE: LiDAR point clouds are variable-length, so training uses the custom
collate in ``bevfusion_collate.py`` (keeps ``lidar`` as a list). PDM ``gt_scores``
for distillation are injected by the training dataset (as in ModularPlanner);
without them the planner trains imitation-only.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.gtrs_bevfusion.bevfusion_features import BEVFusionFeatureBuilder
from navsim.agents.gtrs_bevfusion.bevfusion_loss import gtrs_bevfusion_loss
from navsim.agents.gtrs_bevfusion.bevfusion_model import GTRSBevfusionModel
from navsim.agents.gtrs_bevfusion.bevfusion_target import BEVFusionTargetBuilder
from navsim.agents.gtrs_bevfusion.config import GTRSBevfusionConfig
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class GTRSBevfusionAgent(AbstractAgent):
    """End-to-end BEVFusion + HydraMDP planning agent."""

    def __init__(
        self,
        config: Optional[GTRSBevfusionConfig] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        lr: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._config = config or GTRSBevfusionConfig()
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        self._model = GTRSBevfusionModel(
            config=self._config,
            num_poses=trajectory_sampling.num_poses,
            backbone_checkpoint=backbone_checkpoint,
        )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path is not None and os.path.exists(self._checkpoint_path):
            state_dict = torch.load(self._checkpoint_path, map_location="cpu")["state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        self._model.eval()

    def get_sensor_config(self) -> SensorConfig:
        # cameras + lidar at the current frame only (idx = num_history_frames-1).
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [BEVFusionTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [BEVFusionFeatureBuilder(self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = gtrs_bevfusion_loss(targets, predictions, self._config, self._model.planning_head)
        return losses["loss_total"]

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self._model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[Any]:
        return []

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene = None) -> Trajectory:
        self.eval()
        builder = self.get_feature_builders()[0]
        features = builder.compute_features(agent_input)

        device = next(self._model.parameters()).device
        batched: Dict[str, Any] = {}
        for k, v in features.items():
            if k == "lidar":
                batched[k] = [v.to(device)]
            else:
                batched[k] = v.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = self.forward(batched)
            poses = preds["trajectory"].squeeze(0).cpu().numpy()
        return Trajectory(poses, self._trajectory_sampling)
