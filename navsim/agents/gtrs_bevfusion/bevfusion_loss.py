# SPDX-License-Identifier: Apache-2.0
"""Multi-task loss for GTRS-BEVFusion.

  * planning   : HydraMDP imitation (soft-CE) + PDM distillation (BCE)
                 -> reuses ``PlanningHead.loss``
  * detection  : Hungarian matching L1 + BCE  -> reuses transfuser ``_agent_loss``
  * map (seg)  : weighted cross-entropy + multiclass Lovasz-softmax
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from navsim.agents.transfuser.transfuser_loss import _agent_loss


# --------------------------------------------------------------------------- #
# Lovasz-softmax (Berman et al., 2018) — multiclass, single-label.
# --------------------------------------------------------------------------- #
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None) -> torch.Tensor:
    """probas: [B, C, H, W] softmax probs; labels: [B, H, W] int."""
    B, C, H, W = probas.shape
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
    labels = labels.reshape(-1)  # [N]

    if ignore is not None:
        valid = labels != ignore
        probas = probas[valid]
        labels = labels[valid]

    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def _seg_loss(targets, predictions, config) -> torch.Tensor:
    logits = predictions["bev_semantic_map"]  # [B, C, H, W]
    gt = targets["bev_semantic_map"].long()  # [B, H, W]
    weight = torch.tensor(config.bev_class_weights, dtype=logits.dtype, device=logits.device)
    ce = F.cross_entropy(logits, gt, weight=weight)
    lov = lovasz_softmax(logits.softmax(dim=1), gt)
    return ce + config.lovasz_loss_weight * lov


# --------------------------------------------------------------------------- #
def gtrs_bevfusion_loss(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    config,
    planning_head,
) -> Dict[str, torch.Tensor]:
    """Returns dict of individual + total losses."""
    losses: Dict[str, torch.Tensor] = {}

    # planning (imitation + distillation)
    plan_losses = planning_head.loss(predictions, targets)
    losses["loss_imitation"] = plan_losses["loss_imitation"]
    if "loss_distill" in plan_losses:
        losses["loss_distill"] = plan_losses["loss_distill"]
    plan_total = (
        config.trajectory_imi_weight * plan_losses["loss_imitation"]
        + config.trajectory_distill_weight * plan_losses.get("loss_distill", 0.0)
    )

    total = plan_total

    # detection
    if config.use_detection_head and "agent_states" in predictions:
        agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
        losses["loss_agent_class"] = agent_class_loss
        losses["loss_agent_box"] = agent_box_loss
        total = total + (
            config.agent_class_weight * agent_class_loss
            + config.agent_box_weight * agent_box_loss
        )

    # segmentation
    if config.use_bev_seg_head and "bev_semantic_map" in predictions:
        seg_loss = _seg_loss(targets, predictions, config)
        losses["loss_bev_semantic"] = seg_loss
        total = total + config.bev_semantic_weight * seg_loss

    losses["loss_total"] = total
    return losses
