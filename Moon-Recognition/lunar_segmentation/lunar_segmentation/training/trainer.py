import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Binary focal loss for multi-label segmentation.

    gamma=0 reduces to plain BCE.  alpha weights the positive class;
    (1-alpha) weights the negative class.

    p is computed once from sigmoid and reused for both p_t and BCE,
    avoiding a redundant torch.exp(-bce) kernel on MPS/CUDA.
    """
    p       = torch.sigmoid(logits)
    bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t     = p * targets + (1.0 - p) * (1.0 - targets)      # confidence on correct label
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def weighted_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss with optional per-class weighting.

    class_weights shape: (C,) — set higher values for rare classes.
    Sums over batch and spatial dims, giving one Dice score per class.
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(0, 2, 3))          # (C,)
    den = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice_per_class = 1.0 - (num + eps) / (den + eps)           # (C,)
    if class_weights is not None:
        dice_per_class = dice_per_class * class_weights.to(logits.device)
    return dice_per_class.mean()


# Backward-compat alias used by old checkpoints / notebooks
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return weighted_dice_loss(logits, targets, eps=eps)


class FocalDiceLoss(nn.Module):
    """Focal loss + weighted soft Dice loss.

    Focal loss handles extreme foreground/background imbalance.
    Per-class Dice weights let you emphasise rare classes
    (e.g. pass higher weights for pit_skylight, wrinkle_ridge, etc.).

    Args:
        gamma: focal modulating exponent (2.0 is the standard value).
        alpha: foreground balance factor for the focal term.
        class_weights: 1-D tensor of length num_classes. Rare classes
            should receive a weight > 1.0. Defaults to uniform weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            focal_loss(logits, targets, self.gamma, self.alpha)
            + weighted_dice_loss(logits, targets, self.class_weights)
        )


class BCEDiceLoss(nn.Module):
    """Kept for backward compatibility. Prefer FocalDiceLoss for new training."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets) + dice_loss(logits, targets)


def multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
):
    """Compute per-class precision, recall, F1, IoU and Average Precision.

    Expects the FULL validation set (not per-batch) so that AP is computed
    over all pixels at once.  Average Precision is threshold-free and is
    the preferred metric when comparing models with different loss functions
    (e.g. BCEDice vs FocalDice) whose output probabilities live on different
    scales.

    Memory: sigmoid/threshold are computed one class at a time instead of
    materialising full (N, C, H, W) prob/pred tensors — for 3 186 tiles of
    256x256x7 that saves ~12 GB of peak RAM with identical results.

    A class with zero positive pixels in the set gets AP/precision/recall
    = NaN ('support' column gives the positive-pixel count): AP is undefined
    without positives, and counting it as 0 would penalise the model for the
    data split rather than its predictions.  pandas .mean() skips NaN, so
    macro averages are taken over the classes actually present.
    """
    from ..data.preprocessing import CLASS_NAMES

    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        # Flatten spatial dims → (N*H*W,), one class at a time
        prob_i = torch.sigmoid(logits[:, i]).reshape(-1).cpu().numpy()
        t      = targets[:, i].reshape(-1).cpu().numpy().astype(np.float32)
        p      = (prob_i > threshold).astype(np.float32)

        tp = float((p * t).sum())
        fp = float((p * (1 - t)).sum())
        fn = float(((1 - p) * t).sum())
        support = int(t.sum())

        if support > 0:
            precision = tp / (tp + fp + eps)
            recall    = tp / (tp + fn + eps)
            f1        = 2 * precision * recall / (precision + recall + eps)
            iou       = tp / (tp + fp + fn + eps)
            ap        = float(average_precision_score(t, prob_i))
        else:
            precision = recall = f1 = iou = ap = float('nan')

        per_class.append({
            'class': name, 'precision': precision, 'recall': recall,
            'f1': f1, 'iou': iou, 'ap': ap, 'support': support,
        })
        del prob_i, t, p
    return pd.DataFrame(per_class)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler=None,
        grad_clip: float = None,
    ):
        """
        scheduler: any torch.optim.lr_scheduler instance, stepped once per epoch.
            ReduceLROnPlateau is handled automatically (receives the epoch loss).
        grad_clip: max gradient norm. None (default) disables clipping — avoid on
            MPS as clip_grad_norm_ forces a CPU sync on every batch.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.model.to(self.device)

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if losses else float('nan')

        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(mean_loss)
            else:
                self.scheduler.step()

        return mean_loss

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def evaluate(self, loader, criterion=None):  # criterion kept for backward compat
        self.model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                all_logits.append(self.model(x).cpu())
                # bool storage: targets are binary masks, 4x smaller than float32
                all_targets.append(y.bool())

        if not all_logits:
            return None

        return multilabel_metrics(
            torch.cat(all_logits),
            torch.cat(all_targets),
        )
