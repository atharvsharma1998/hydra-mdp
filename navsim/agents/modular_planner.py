import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Any, Dict, List, Optional, Union
from torch.optim import Optimizer
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Scene, Trajectory, AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.modular_features import ModularFeatureBuilder, ModularTargetBuilder
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario


class PlanningHead(nn.Module):
    """Transformer Decoder based Planning Head (Hydra-MDP Decoder replica)."""

    def __init__(
        self,
        in_channels=256,
        d_model=256,
        d_ffn=512,
        nhead=4,
        nlayers=2,
        num_poses=40,
        vocab_path="traj_final/8192.npy",
        num_ego_status=3,
        bev_spatial_shape=(32, 32),
        vocab_dropout_size=2048,
        scorer_w_imi=0.05,
        scorer_w_nc=0.5,
        scorer_w_dac=0.5,
        scorer_w_ddc=0.5,
        scorer_w_tlc=0.5,
        scorer_w_progress=5.0,
        scorer_w_lk=5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_poses = num_poses
        self.vocab_path = vocab_path
        self.bev_spatial_shape = bev_spatial_shape
        self.vocab_dropout_size = vocab_dropout_size
        # Hydra-MDP inference cost weights (Eq. 11). Paper grid-search ranges:
        #   w1 (imitation) in [0.01, 0.1];  w2 (NC), w3 (DAC) in [0.1, 1];
        #   w4 (progress bundle) in [1, 10]  -- "prioritize rule-based costs over imitation".
        # Extension over Hydra-MDP: we additionally score driving_direction_compliance
        # (DDC), which the paper neglected only because of the NAVSIM metric bug
        # (issue #14, now fixed). w_ddc shares the compliance range [0.1, 1].
        self.scorer_w_imi = scorer_w_imi
        self.scorer_w_nc = scorer_w_nc
        self.scorer_w_dac = scorer_w_dac
        self.scorer_w_ddc = scorer_w_ddc
        self.scorer_w_tlc = scorer_w_tlc
        self.scorer_w_progress = scorer_w_progress
        self.scorer_w_lk = scorer_w_lk

        self.bev_pool = nn.AdaptiveAvgPool2d(bev_spatial_shape)
        if in_channels != d_model:
            self.bev_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        else:
            self.bev_proj = nn.Identity()

        # Load the trajectory vocabulary parameters
        # Shape: [k, 40, 3] -> [8192, 40, 3]
        if os.path.exists(vocab_path):
            vocab_np = np.load(vocab_path)
        else:
            fallback = os.path.join(os.path.dirname(__file__), "../../../", vocab_path)
            vocab_np = np.load(fallback)
            
        self.vocab = nn.Parameter(
            torch.from_numpy(vocab_np).float(),
            requires_grad=False
        )
        self.vocab_size = self.vocab.shape[0]

        # MLPs for embedding coordinates and ego status
        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_model),
        )
        
        self.status_embed = nn.Sequential(
            nn.Linear(num_ego_status * 8, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

        # 8 prediction heads
        self.heads = nn.ModuleDict({
            'no_at_fault_collisions': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'drivable_area_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'time_to_collision_within_bound': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'ego_progress': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'driving_direction_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'lane_keeping': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'traffic_light_compliance': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            ),
            'imi': nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(d_ffn, 1),
            )
        })

    def forward(self, bev_feature, status_feature, **kwargs):
        if isinstance(bev_feature, (list, tuple)):
            bev_feature = bev_feature[0]
        B = bev_feature.shape[0]

        # 1. Pool and project BEV feature map
        bev_feature = self.bev_pool(bev_feature)
        bev_feature = self.bev_proj(bev_feature)
        bev_tokens = bev_feature.flatten(2).permute(0, 2, 1)

        # 2. Vocabulary Selection (Dropout during training to save memory)
        vocab = self.vocab
        vocab_size = self.vocab_size
        indices = None
        if self.training and self.vocab_dropout_size is not None and self.vocab_dropout_size < self.vocab_size:
            indices = torch.randperm(self.vocab_size, device=self.vocab.device)[:self.vocab_dropout_size]
            vocab = self.vocab[indices]
            vocab_size = self.vocab_dropout_size

        vocab_flat = vocab.view(vocab_size, -1)
        embedded_vocab = self.pos_embed(vocab_flat)[None].repeat(B, 1, 1)

        # 3. Cross-Attention
        tr_out = self.transformer(embedded_vocab, bev_tokens)

        # 4. Status feature fusion
        status_encoding = self.status_embed(status_feature)
        dist_status = tr_out + status_encoding.unsqueeze(1)

        # 5. Prediction Heads
        result = {}
        for name, head in self.heads.items():
            result[name] = head(dist_status).squeeze(-1)

        # 6. Trajectory selection scorer (Hydra-MDP Eq. 11 + EPDMS-style penalties):
        #   f_tilde = -( w_imi logS_im + w_NC logS_NC + w_DAC logS_DAC + w_DDC logS_DDC
        #               + w_TLC logS_TLC + w_prog log(5 S_TTC + 5 S_EP) )  ; select argmax(score)
        # Penalty terms (NC, DAC, DDC, TLC) follow the EPDMS multiplicative penalties
        # (here additive in log-sigmoid space). Deviations from the base paper, by design:
        #   * Comfort (S_C) head is not available here -> bundle is 5 S_TTC + 5 S_EP.
        #   * DDC + TLC are ADDED back as penalties (Hydra-MDP dropped DDC only due to the
        #     NAVSIM metric bug #14, now fixed; TLC follows Hydra-MDP++ EPDMS). Set the
        #     corresponding scorer_w_*=0 to recover the exact base paper scorer.
        #   * lane_keeping (LK) is added INSIDE the weighted bundle (additive, EPDMS-style),
        #     NOT as a penalty: a single lane touch should trade off against progress, not
        #     zero the trajectory. Set scorer_w_lk=0 to drop it.
        # log() args are clamped to a small eps: a saturated sigmoid (~0) or an
        # underflowed softmax prob would give log(0) = -inf, making scores non-finite.
        _eps = 1e-6
        scores = (
            self.scorer_w_imi * result['imi'].softmax(-1).clamp_min(_eps).log() +
            self.scorer_w_nc * result['no_at_fault_collisions'].sigmoid().clamp_min(_eps).log() +
            self.scorer_w_dac * result['drivable_area_compliance'].sigmoid().clamp_min(_eps).log() +
            self.scorer_w_ddc * result['driving_direction_compliance'].sigmoid().clamp_min(_eps).log() +
            self.scorer_w_tlc * result['traffic_light_compliance'].sigmoid().clamp_min(_eps).log() +
            self.scorer_w_progress * (
                5.0 * result['time_to_collision_within_bound'].sigmoid() +
                5.0 * result['ego_progress'].sigmoid() +
                self.scorer_w_lk * result['lane_keeping'].sigmoid()
            ).clamp_min(_eps).log()
        )

        selected_indices = scores.argmax(dim=1)
        selected_trajectories = vocab[selected_indices]

        result["scores"] = scores
        result["selected_indices"] = selected_indices
        result["selected_trajectory"] = selected_trajectories
        result["vocab"] = vocab
        if indices is not None:
            result["indices"] = indices

        return result

    def loss(self, preds, targets):
        losses = {}
        B = preds['selected_indices'].shape[0]

        # 1. Imitation Loss (Soft cross-entropy)
        expert_traj = targets['trajectory'] # [B, 40, 3]
        vocab = preds['vocab']
        if expert_traj.shape[1] != vocab.shape[1]:
            # Interpolate from 8 to 40 poses
            expert_traj = F.interpolate(
                expert_traj.permute(0, 2, 1),
                size=vocab.shape[1],
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)
        dist_sq = torch.sum((expert_traj.unsqueeze(1) - vocab.unsqueeze(0)) ** 2, dim=(2, 3))
        imi_targets = F.softmax(-dist_sq, dim=1)
        
        pred_imi_log = F.log_softmax(preds['imi'], dim=1)
        losses['loss_imitation'] = -torch.sum(imi_targets * pred_imi_log) / B

        # 2. Distillation Loss (BCE)
        if 'gt_scores' in targets:
            losses_kd = []
            for name, head_preds in preds.items():
                if name in ['selected_indices', 'selected_trajectory', 'scores', 'vocab', 'imi', 'indices']:
                    continue
                if name in targets['gt_scores']:
                    gt = targets['gt_scores'][name]
                    if 'indices' in preds and preds['indices'] is not None:
                        gt = gt[:, preds['indices']]
                    loss_head = F.binary_cross_entropy_with_logits(head_preds, gt)
                    losses_kd.append(loss_head)
                    losses[f'loss_{name}'] = loss_head
            
            if len(losses_kd) > 0:
                losses['loss_distill'] = sum(losses_kd)

        losses['loss_total'] = losses['loss_imitation'] + losses.get('loss_distill', 0.0)
        return losses


class ModularPlanner(AbstractAgent):
    """Modular Planning Agent: ResNet-based Grid Encoder + PlanningHead."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        lr: float = 1e-4,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        vocab_path: str = "traj_final/8192.npy",
    ):
        super().__init__(trajectory_sampling, requires_scene=True)
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.vocab_path = vocab_path
        self._feature_builder = ModularFeatureBuilder()

        # 1. ResNet18 Grid Encoder
        # input: [B, 6, 256, 256] -> outputs after layer2: [B, 128, 32, 32]
        self.encoder_resnet = models.resnet18(pretrained=False)
        self.encoder_resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Grid feature projection to d_model channels
        self.feature_proj = nn.Conv2d(128, 256, kernel_size=1)

        # 2. Planning Head
        self.planning_head = PlanningHead(
            in_channels=256,
            d_model=256,
            d_ffn=512,
            nhead=4,
            nlayers=2,
            num_poses=40,
            vocab_path=self.vocab_path,
            num_ego_status=3,
            bev_spatial_shape=(32, 32),
            vocab_dropout_size=2048,
        )

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
            # Remove lighting module wrapper prefixes if present
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        self.eval()
        
        # 1. Compute status_feature
        status_feature = self._feature_builder._get_status_feature(agent_input)
        
        # 2. Compute bev_grid
        maps_root = os.environ.get("NUPLAN_MAPS_ROOT")
        if maps_root is None:
            raise ValueError(
                "Environment variable 'NUPLAN_MAPS_ROOT' must be set to run ModularPlanner "
                "in order to reconstruct the BEV grid map."
            )
        scenario = NavSimScenario(scene, map_root=maps_root, map_version="nuplan-maps-v1.0")
        ego_pose = scenario.initial_ego_state.rear_axle
        map_api = scenario.map_api
        annotations = scene.frames[-1].annotations
        
        bev_grid = self._feature_builder.compute_bev_grid(annotations, map_api, ego_pose)
        
        # Add batch dimension
        features = {
            "bev_grid": bev_grid.unsqueeze(0),
            "status_feature": status_feature.unsqueeze(0),
        }
        
        # Move features to device
        device = next(self.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()
            
        return Trajectory(poses, self._trajectory_sampling)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [ModularTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ModularFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        :param features: Dict containing:
                         - "bev_grid": [B, 6, 256, 256]
                         - "status_feature": [B, 24]
        """
        bev_grid = features["bev_grid"].to(torch.float32)
        status_feature = features["status_feature"].to(torch.float32)

        # ResNet forward pass up to layer2
        x = self.encoder_resnet.conv1(bev_grid)
        x = self.encoder_resnet.bn1(x)
        x = self.encoder_resnet.relu(x)
        x = self.encoder_resnet.maxpool(x)

        x = self.encoder_resnet.layer1(x)
        x = self.encoder_resnet.layer2(x) # shape: [B, 128, 32, 32]
        
        # Project to d_model=256 channels
        bev_feature = self.feature_proj(x) # shape: [B, 256, 32, 32]

        # Planning head forward pass
        preds = self.planning_head(bev_feature, status_feature)
        
        # Wrap outputs
        selected_trajectory = preds["selected_trajectory"]
        if selected_trajectory.shape[1] == 40:
            selected_trajectory = selected_trajectory[:, 4::5]
        predictions = {
            "trajectory": selected_trajectory,
            "selected_indices": preds["selected_indices"],
            "scores": preds["scores"],
            "vocab": preds["vocab"],
            "imi": preds["imi"],
        }
        if "indices" in preds:
            predictions["indices"] = preds["indices"]

        # Keep individual heads for loss computation
        for k in ['no_at_fault_collisions', 'drivable_area_compliance', 
                  'time_to_collision_within_bound', 'ego_progress', 
                  'driving_direction_compliance', 'lane_keeping', 
                  'traffic_light_compliance']:
            if k in preds:
                predictions[k] = preds[k]

        return predictions

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = self.planning_head.loss(predictions, targets)
        return losses["loss_total"]

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_training_callbacks(self) -> List[Any]:
        return []
