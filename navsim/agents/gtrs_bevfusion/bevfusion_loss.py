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
from scipy.optimize import linear_sum_assignment

from navsim.agents.transfuser.transfuser_loss import _agent_loss, _get_l1_cost, _get_src_permutation_idx
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


# --------------------------------------------------------------------------- #
# Multi-class (CUDA-BEVFusion style) Hungarian detection loss.
# Mirrors transfuser._agent_loss (square N-pred <-> N-gt assignment) but the
# classifier is multi-class: K classes + a background class at index K. Unmatched
# / padding gt slots get the background target, giving negative supervision for
# free. Box L1 (xy+heading+size) is applied only on valid (non-padding) matches.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _mc_class_cost(pred_logits: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
    """cost[b, i_pred, j_gt] = -P(class = target_cls[j_gt]); shape [B, M, N]."""
    probs = pred_logits.softmax(-1)  # [B, M, K+1]
    idx = target_cls[:, None, :].expand(-1, probs.shape[1], -1)  # [B, M, N]
    return -probs.gather(2, idx)


def _agent_loss_multiclass(targets, predictions, config):
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    gt_classes = targets["agent_classes"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_class_logits"]
    bg = config.num_detection_classes  # background class id

    if config.latent:
        rad_to_ego = torch.arctan2(gt_states[..., BoundingBox2DIndex.Y], gt_states[..., BoundingBox2DIndex.X])
        in_latent = torch.logical_and(-config.latent_rad_thresh <= rad_to_ego, rad_to_ego <= config.latent_rad_thresh)
        gt_valid = torch.logical_and(in_latent, gt_valid)

    batch_dim = pred_states.shape[0]
    num_gt = gt_valid.sum()
    num_gt = num_gt if num_gt > 0 else num_gt + 1

    # per gt slot target class: valid -> its class, padding/invalid -> background
    target_cls = torch.where(gt_valid, gt_classes, torch.full_like(gt_classes, bg))  # [B, N]

    cost = config.agent_class_weight * _mc_class_cost(pred_logits, target_cls) \
        + config.agent_box_weight * _get_l1_cost(gt_states, pred_states, gt_valid)
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for c in cost]
    matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)
    pred_logits_idx = pred_logits[idx]  # [total, K+1]
    target_cls_idx = torch.cat([t[i] for t, (_, i) in zip(target_cls, indices)], dim=0)  # [total]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none").sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt

    # class CE with DETR-style background down-weighting (eos_coef): foreground
    # classes weight 1.0, background (last) weight = config.detection_bg_weight.
    cls_weight = torch.ones(bg + 1, dtype=pred_logits.dtype, device=pred_logits.device)
    cls_weight[bg] = getattr(config, "detection_bg_weight", 0.1)
    ce_loss = F.cross_entropy(pred_logits_idx, target_cls_idx, weight=cls_weight, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()
    return ce_loss, l1_loss


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

    # detection (multi-class when the head emits class logits, else binary)
    if config.use_detection_head and "agent_states" in predictions:
        if "agent_class_logits" in predictions:
            agent_class_loss, agent_box_loss = _agent_loss_multiclass(targets, predictions, config)
        else:
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
