from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftRecallLoss(nn.Module):
    """Recall-oriented penalty in a small neighborhood around the centerline."""

    def __init__(self, weight: float = 1.0, pool_size: int = 5):
        super().__init__()
        self.weight = weight
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, pred_prob: torch.Tensor, center_mask: torch.Tensor) -> torch.Tensor:
        support = self.pool(center_mask.float())
        miss = support * (1.0 - pred_prob)
        return self.weight * miss.sum() / support.sum().clamp(min=1.0)


class BranchHeatmapLoss(nn.Module):
    """Heatmap shape + centerline BCE + soft recall + DRIU-style side losses."""

    def __init__(
        self,
        side_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
        recall_weight: float = 1.0,
        main_weight: float = 1.0,
        center_weight: float = 1.0,
        s1_center_weight: float = 0.2,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.side_weights = side_weights
        self.recall = SoftRecallLoss(weight=recall_weight)
        self.main_weight = main_weight
        self.center_weight = center_weight
        self.s1_center_weight = s1_center_weight
        self.pos_weight = pos_weight

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gaussian_target: torch.Tensor,
        center_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused, s1, s2, s3, s4 = preds
        sides = (s1, s2, s3, s4)
        center_mask = center_mask.float()
        gaussian_target = gaussian_target.float()

        total = fused.new_tensor(0.0)
        for side_logit, weight in zip(sides, self.side_weights):
            total = total + weight * F.mse_loss(
                torch.sigmoid(side_logit), gaussian_target
            )

        pred_fused = torch.sigmoid(fused)
        main_mse = F.mse_loss(pred_fused, gaussian_target)
        bce_weight = 1.0 + self.pos_weight * center_mask
        center_bce = F.binary_cross_entropy_with_logits(
            fused, center_mask, weight=bce_weight
        )
        s1_center_bce = F.binary_cross_entropy_with_logits(
            s1, center_mask, weight=bce_weight
        )
        recall = self.recall(pred_fused, center_mask)

        return (
            total
            + self.main_weight * main_mse
            + self.center_weight * center_bce
            + self.s1_center_weight * s1_center_bce
            + recall
        )
