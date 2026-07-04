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

    # Box L1 over (x, y, heading, length, width). All components matter equally
    # for perception, so comp_w defaults to all-ones (agent_box_xy_weight=1.0).
    # The knob is kept only as a re-cal lever; localization is handled by decoding
    # the full 32x32 F_env grid, not by reweighting this loss.
    l1_elt = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")  # [total, 5]
    xy_w = getattr(config, "agent_box_xy_weight", 1.0)
    comp_w = torch.ones(l1_elt.shape[-1], dtype=l1_elt.dtype, device=l1_elt.device)
    comp_w[BoundingBox2DIndex.POINT] = xy_w  # x, y (1.0 == equal with heading/size)
    l1_loss = (l1_elt * comp_w).sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt

    # class CE with DETR-style background down-weighting (eos_coef): foreground
    # classes weight 1.0, background (last) weight = config.detection_bg_weight.
    cls_weight = torch.ones(bg + 1, dtype=pred_logits.dtype, device=pred_logits.device)
    cls_weight[bg] = getattr(config, "detection_bg_weight", 0.1)
    ce_loss = F.cross_entropy(pred_logits_idx, target_cls_idx, weight=cls_weight, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()
    return ce_loss, l1_loss


# --------------------------------------------------------------------------- #
# CenterPoint-style dense detection loss (CUDA-BEVFusion TransFusion heatmap).
# Dense per-class gaussian-focal heatmap (CenterNet) + L1 regression of
# (sub-cell offset, log size, sin/cos heading) at the GT cell. Targets are
# rendered on the fly from agent_states/agent_classes/agent_labels -- no change
# to the target builder. Grid is the native F_env 100x100 (+/-32 m, rows=x).
# --------------------------------------------------------------------------- #
def _gaussian_radius(height: float, width: float, min_overlap: float) -> float:
    """CenterNet gaussian radius so a box keeps >= min_overlap IoU within it."""
    import math

    a1, b1 = 1.0, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / (2 * a1)
    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / (2 * a2)
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def _draw_gaussian(heatmap: torch.Tensor, r: int, c: int, radius: int) -> None:
    """Splat a 2D gaussian peak (in-place max) at cell (r, c) of [H, W]."""
    radius = int(max(radius, 0))
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ax = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=heatmap.dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))

    H, W = heatmap.shape
    top, bottom = min(r, radius), min(H - r, radius + 1)
    left, right = min(c, radius), min(W - c, radius + 1)
    if bottom <= -top or right <= -left:
        return
    masked = heatmap[r - top:r + bottom, c - left:c + right]
    mg = g[radius - top:radius + bottom, radius - left:radius + right]
    torch.maximum(masked, mg, out=masked)


def _centerpoint_loss(targets, predictions, config):
    """Returns (heatmap_focal_loss, box_reg_l1_loss)."""
    hm = predictions["det_heatmap"]                  # [B, K, H, W]
    reg = torch.cat([predictions["det_offset"], predictions["det_size"],
                     predictions["det_heading"]], dim=1)  # [B, 6, H, W]
    B, K, H, W = hm.shape
    device, dtype = hm.device, hm.dtype

    gt_states = targets["agent_states"]              # [B, N, 5] (x,y,heading,l,w)
    gt_valid = targets["agent_labels"].bool()        # [B, N]
    gt_classes = targets["agent_classes"].long()     # [B, N]

    x_min, y_min = float(config.point_cloud_range[0]), float(config.point_cloud_range[1])
    x_max, y_max = float(config.point_cloud_range[3]), float(config.point_cloud_range[4])
    cell_x = (x_max - x_min) / H                      # metres per row (x)
    cell_y = (y_max - y_min) / W                      # metres per col (y)
    min_overlap = config.det_gaussian_overlap
    min_radius = config.det_gaussian_min_radius

    gt_hm = torch.zeros(B, K, H, W, device=device, dtype=dtype)
    reg_tgt = torch.zeros(B, 6, H, W, device=device, dtype=dtype)
    reg_mask = torch.zeros(B, H, W, device=device, dtype=dtype)

    for b in range(B):
        idx = gt_valid[b].nonzero(as_tuple=True)[0]
        for n in idx.tolist():
            x, y, heading, length, width = gt_states[b, n].tolist()
            cls = int(gt_classes[b, n])
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            r = min(int((x - x_min) / cell_x), H - 1)
            c = min(int((y - y_min) / cell_y), W - 1)
            radius = max(int(_gaussian_radius(length / cell_x, width / cell_y, min_overlap)), min_radius)
            _draw_gaussian(gt_hm[b, cls], r, c, radius)
            cell_cx = x_min + (r + 0.5) * cell_x
            cell_cy = y_min + (c + 0.5) * cell_y
            reg_tgt[b, 0, r, c] = x - cell_cx                      # offset dx (m)
            reg_tgt[b, 1, r, c] = y - cell_cy                      # offset dy (m)
            reg_tgt[b, 2, r, c] = float(torch.log(torch.tensor(max(length, 1e-3))))  # log L
            reg_tgt[b, 3, r, c] = float(torch.log(torch.tensor(max(width, 1e-3))))   # log W
            reg_tgt[b, 4, r, c] = torch.sin(torch.tensor(heading))                   # sin
            reg_tgt[b, 5, r, c] = torch.cos(torch.tensor(heading))                   # cos
            reg_mask[b, r, c] = 1.0

    num_pos = reg_mask.sum().clamp(min=1.0)

    # CenterNet gaussian-focal loss on the dense heatmap. MUST be computed in fp32:
    # under AMP the sigmoid is fp16, where clamp(1e-4, 1-1e-4) collapses the upper
    # bound to 1.0 (fp16 resolution near 1.0 is ~5e-4), so log(1 - pred) = log(0)
    # = -inf -> NaN. This was the source of the epoch-66/late-run divergence that
    # grad-clip could not fix. Upcasting keeps the log terms finite.
    pred = hm.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    gt_hm_f = gt_hm.float()
    pos = gt_hm_f.eq(1.0).float()
    neg = 1.0 - pos
    neg_weight = (1.0 - gt_hm_f).pow(4)
    pos_loss = torch.log(pred) * (1 - pred).pow(2) * pos
    neg_loss = torch.log(1 - pred) * pred.pow(2) * neg_weight * neg
    heatmap_loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos.float()

    # L1 box regression only at the GT cells (offset + log-size + sin/cos heading)
    l1 = F.l1_loss(reg, reg_tgt, reduction="none") * reg_mask[:, None]
    reg_loss = l1.sum() / num_pos
    return heatmap_loss, reg_loss


# --------------------------------------------------------------------------- #
# Lovasz-softmax (Berman et al., 2018) — multiclass, single-label.
# --------------------------------------------------------------------------- #
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    # clamp: empty prefix unions are 0 and would yield NaN in intersection/union
    union = (gts + (1 - gt_sorted).float().cumsum(0)).clamp(min=1e-6)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None) -> torch.Tensor:
    """probas: [B, C, H, W] softmax probs; labels: [B, H, W] int."""
    # Always fp32 — AMP forward leaves probs in fp16 where sort/cumsum is fragile.
    probas = probas.float().clamp(0.0, 1.0)
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
        return probas.new_zeros(())
    out = torch.stack(losses).mean()
    # Defensive: never let a single Lovasz edge case poison the whole step.
    if not torch.isfinite(out):
        return probas.new_zeros(())
    return out


def _seg_loss(targets, predictions, config) -> torch.Tensor:
    # MUST be fp32: AMP forward leaves seg logits in fp16. Softmax + Lovasz on
    # extreme fp16 values produced NaN at epoch 8 (only loss_bev_semantic BAD;
    # det/planning OK; weights still finite). Seen on tokens cafed437eec155a1 etc.
    logits = predictions["bev_semantic_map"].float()  # [B, C, H, W]
    gt = targets["bev_semantic_map"].long()  # [B, H, W]
    weight = torch.tensor(config.bev_class_weights, dtype=torch.float32, device=logits.device)
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

    # detection: CenterPoint dense head (heatmap focal + reg L1) when present,
    # else the legacy DETR Hungarian head (multi-class or binary).
    if config.use_detection_head and "det_heatmap" in predictions:
        agent_class_loss, agent_box_loss = _centerpoint_loss(targets, predictions, config)
        losses["loss_agent_class"] = agent_class_loss   # heatmap focal
        losses["loss_agent_box"] = agent_box_loss       # offset+size+heading L1
        total = total + (
            config.det_heatmap_weight * agent_class_loss
            + config.det_reg_weight * agent_box_loss
        )
    elif config.use_detection_head and "agent_states" in predictions:
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
