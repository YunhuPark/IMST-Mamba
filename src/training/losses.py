"""
Loss functions for IMST-Mamba training.

Primary: Focal Loss for sepsis prediction (handles class imbalance)
Multi-task: combined loss with auxiliary mortality and SOFA predictions
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy negatives to focus on hard examples.
    Lin et al. (2017), RetinaNet.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        gamma: focusing parameter (default 2.0)
        alpha: weight for positive class (default 0.75)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logit: torch.Tensor,   # any shape
        target: torch.Tensor,  # same shape, float {0, 1}
        mask: torch.Tensor | None = None,  # optional boolean mask
    ) -> torch.Tensor:
        # Replace NaN targets with 0 and exclude from mask
        nan_mask = torch.isnan(target)
        target = target.clone()
        target[nan_mask] = 0.0
        if mask is not None:
            mask = mask & ~nan_mask
        else:
            mask = ~nan_mask

        p = torch.sigmoid(logit)
        # Weight: α for positive, (1-α) for negative
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        # Focal factor
        p_t = p * target + (1 - p) * (1 - target)
        focal = (1 - p_t) ** self.gamma
        # BCE
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        loss = alpha_t * focal * bce

        if mask is not None:
            loss = loss * mask.float()
            if self.reduction == "mean":
                return loss.sum() / (mask.float().sum() + 1e-8)
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for IMST-Mamba:
      L = L_sepsis + λ_mort · L_mortality + λ_sofa · L_sofa

    Args:
        gamma:           focal loss gamma
        alpha:           focal loss alpha (positive class weight)
        lambda_mortality: weight for mortality auxiliary task
        lambda_sofa:      weight for SOFA auxiliary task
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        lambda_mortality: float = 0.3,
        lambda_sofa: float = 0.1,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.lambda_mortality = lambda_mortality
        self.lambda_sofa = lambda_sofa

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: model output dict (logit_sepsis, logit_mortality, pred_sofa)
            batch:   data batch dict  (y, sofa, attention_mask, ...)

        Returns:
            dict with 'total', 'sepsis', 'mortality', 'sofa' loss values
        """
        attn_mask = batch["attention_mask"]   # (B, T)
        y = batch["y"]                         # (B, T)

        # Primary: sepsis focal loss (step-wise, only on valid positions)
        logit_sep = outputs["logit_sepsis"].squeeze(-1)   # (B, T)
        loss_sepsis = self.focal(logit_sep, y, mask=attn_mask)

        total = loss_sepsis
        losses = {"total": total, "sepsis": loss_sepsis.detach()}

        # Auxiliary: mortality (patient-level)
        if "logit_mortality" in outputs and self.lambda_mortality > 0:
            logit_mort = outputs["logit_mortality"].squeeze(-1)   # (B,)
            # Mortality label: 1 if patient died during hospitalization
            # Use max of sepsis labels as proxy if mortality not available
            y_mort = batch.get("mortality", y.max(dim=1).values)
            loss_mort = F.binary_cross_entropy_with_logits(logit_mort, y_mort)
            total = total + self.lambda_mortality * loss_mort
            losses["mortality"] = loss_mort.detach()

        # Auxiliary: SOFA regression
        if "pred_sofa" in outputs and self.lambda_sofa > 0:
            pred_sofa = outputs["pred_sofa"].squeeze(-1)   # (B, T)
            sofa_true = batch["sofa"]                       # (B, T)
            # Only compute where SOFA is not NaN
            valid = attn_mask & ~torch.isnan(sofa_true)
            if valid.any():
                loss_sofa = F.mse_loss(
                    pred_sofa[valid],
                    sofa_true[valid].clamp(0, 24),  # normalize SOFA to [0,24]
                )
                total = total + self.lambda_sofa * loss_sofa
                losses["sofa"] = loss_sofa.detach()

        losses["total"] = total
        return losses
