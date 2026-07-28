"""Mathematically correct PyTorch implementations of FixMatch and FlexMatch loss functions,
training step entrypoints, and cross-domain evaluation utilities.

Formulations:
- FixMatch: Sohn et al., "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence", NeurIPS 2020.
- FlexMatch: Zhang et al., "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling", NeurIPS 2020/2021.
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_fixmatch_loss(
    labeled_logits: torch.Tensor,
    labels: torch.Tensor,
    unlabeled_weak_logits: torch.Tensor,
    unlabeled_strong_logits: torch.Tensor,
    tau: float = 0.95,
    lambda_u: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute mathematically correct FixMatch loss.

    Equations:
        1. Supervised Loss:
           L_s = (1 / B) * sum_{b=1}^B CE(y_b, z_{l,b})
        2. Weak view Pseudo-label & Confidence (detached):
           q_b = softmax(z_{u,w,b}).detach()
           hat_q_b = argmax_c(q_{b,c}),  p_b^max = max_c(q_{b,c})
        3. Fixed Confidence Mask:
           m_b = I(p_b^max >= tau)
        4. Unsupervised Loss:
           L_u = (1 / N_u) * sum_{b=1}^{N_u} m_b * CE(hat_q_b, z_{u,s,b})
        5. Total Loss:
           L_total = L_s + lambda_u * L_u

    Tensor Shapes:
        - labeled_logits: (B, num_classes)
        - labels: (B,) [class indices] or (B, num_classes) [one-hot]
        - unlabeled_weak_logits: (N_u, num_classes)
        - unlabeled_strong_logits: (N_u, num_classes)
        - Returns dict with total_loss, loss_s, loss_u, mask_ratio, pseudo_labels, max_probs, mask
    """
    device = labeled_logits.device
    num_classes = labeled_logits.shape[1]

    # 1. Supervised Loss
    if labels.ndim > 1 and labels.shape[1] == num_classes:
        # One-hot targets
        loss_s = -torch.sum(F.log_softmax(labeled_logits, dim=-1) * labels, dim=-1).mean()
    else:
        loss_s = F.cross_entropy(labeled_logits, labels.long())

    # 2. Pseudo-labeling from detached weak logits
    with torch.no_grad():
        probs_w = torch.softmax(unlabeled_weak_logits.detach(), dim=-1)
        max_probs, pseudo_labels = torch.max(probs_w, dim=-1)
        mask = (max_probs >= tau).float()
        mask_ratio = mask.mean().item()

    # 3. Unsupervised Loss on strongly augmented logits
    if mask.sum() > 0:
        ce_u = F.cross_entropy(unlabeled_strong_logits, pseudo_labels, reduction="none")
        loss_u = (ce_u * mask).mean()
    else:
        # Safe zero loss keeping autograd graph alive for strong logits
        loss_u = 0.0 * unlabeled_strong_logits.sum()

    total_loss = loss_s + lambda_u * loss_u

    return {
        "total_loss": total_loss,
        "loss_s": loss_s,
        "loss_u": loss_u,
        "mask_ratio": mask_ratio,
        "pseudo_labels": pseudo_labels,
        "max_probs": max_probs,
        "mask": mask,
    }


class FlexMatchLoss(nn.Module):
    """
    Stateful PyTorch module for FlexMatch Curriculum Pseudo-Labeling (CPL) loss.

    Equations:
        1. Class Counter Update (without gradients):
           Delta sigma_t(c) = sum_{b=1}^{N_u} I(hat_q_b = c) * I(p_b^max >= tau)
           sigma_{t+1}(c) = sigma_t(c) + Delta sigma_t(c)
        2. Normalized Class Learning Status:
           beta_t(c) = sigma_t(c) / max(max_{c'} sigma_t(c'), 1.0)
        3. Convex Mapping Function M(x):
           M(beta_t(c)) = beta_t(c) / (2 - beta_t(c))
        4. Class Adaptive Threshold:
           tau_t(c) = M(beta_t(c)) * tau
        5. Adaptive Confidence Mask:
           m_{b, flex} = I(p_b^max >= tau_t(hat_q_b))
        6. Loss Calculation:
           L_s = CE(y, z_l)
           L_u = (1 / N_u) * sum_{b=1}^{N_u} m_{b, flex} * CE(hat_q_b, z_{u,s,b})
           L_total = L_s + lambda_u * L_u
    """

    def __init__(
        self,
        num_classes: int,
        tau: float = 0.95,
        lambda_u: float = 1.0,
        mapping: str = "convex",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.lambda_u = lambda_u
        self.mapping = mapping

        # Per-class pseudo-label learning counter (persistent buffer)
        self.register_buffer(
            "class_counts",
            torch.zeros(num_classes, dtype=torch.float32, device=device),
        )

    def compute_class_thresholds(self) -> torch.Tensor:
        """Calculate class-specific adaptive thresholds tau_t(c)."""
        max_count = torch.max(self.class_counts)
        if max_count == 0:
            # Unlearned state: return base threshold scaled or zeroed according to normalized status
            beta = torch.zeros_like(self.class_counts)
        else:
            beta = self.class_counts / torch.clamp(max_count, min=1.0)

        if self.mapping == "convex":
            mapping_val = beta / (2.0 - beta + 1e-12)
        elif self.mapping == "concave":
            mapping_val = torch.sin(0.5 * torch.pi * beta)
        elif self.mapping == "linear":
            mapping_val = beta
        else:
            raise ValueError(f"Unknown mapping function: {self.mapping!r}")

        class_thresholds = mapping_val * self.tau
        return class_thresholds

    def update_class_counts(
        self,
        pseudo_labels: torch.Tensor,
        max_probs: torch.Tensor,
    ) -> None:
        """Update per-class selection statistics for valid high-confidence pseudo labels."""
        with torch.no_grad():
            valid_mask = (max_probs >= self.tau)
            if valid_mask.any():
                valid_labels = pseudo_labels[valid_mask]
                counts = torch.bincount(valid_labels, minlength=self.num_classes).float()
                self.class_counts.add_(counts)

    def forward(
        self,
        labeled_logits: torch.Tensor,
        labels: torch.Tensor,
        unlabeled_weak_logits: torch.Tensor,
        unlabeled_strong_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute FlexMatch loss and update learning statistics.
        """
        num_classes = labeled_logits.shape[1]

        # 1. Supervised loss
        if labels.ndim > 1 and labels.shape[1] == num_classes:
            loss_s = -torch.sum(F.log_softmax(labeled_logits, dim=-1) * labels, dim=-1).mean()
        else:
            loss_s = F.cross_entropy(labeled_logits, labels.long())

        # 2. Compute detached probabilities from weak logits
        with torch.no_grad():
            probs_w = torch.softmax(unlabeled_weak_logits.detach(), dim=-1)
            max_probs, pseudo_labels = torch.max(probs_w, dim=-1)

            # Update learning counts using base threshold tau
            self.update_class_counts(pseudo_labels, max_probs)

            # Compute current class adaptive thresholds
            class_thresholds = self.compute_class_thresholds()

            # Per-sample threshold according to predicted class
            sample_thresholds = class_thresholds[pseudo_labels]
            mask = (max_probs >= sample_thresholds).float()
            mask_ratio = mask.mean().item()

        # 3. Unsupervised loss
        if mask.sum() > 0:
            ce_u = F.cross_entropy(unlabeled_strong_logits, pseudo_labels, reduction="none")
            loss_u = (ce_u * mask).mean()
        else:
            loss_u = 0.0 * unlabeled_strong_logits.sum()

        total_loss = loss_s + self.lambda_u * loss_u

        return {
            "total_loss": total_loss,
            "loss_s": loss_s,
            "loss_u": loss_u,
            "mask_ratio": mask_ratio,
            "pseudo_labels": pseudo_labels,
            "max_probs": max_probs,
            "mask": mask,
            "class_thresholds": class_thresholds,
            "class_counts": self.class_counts.clone(),
        }


def train_fixmatch_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_x: torch.Tensor,
    labels: torch.Tensor,
    unlabeled_w: torch.Tensor,
    unlabeled_s: torch.Tensor,
    tau: float = 0.95,
    lambda_u: float = 1.0,
) -> Dict[str, Any]:
    """
    Execute one FixMatch training step.
    Gradients flow through labeled model outputs and strongly augmented unlabeled outputs.
    No gradients flow through weakly augmented unlabeled outputs.
    """
    model.train()
    optimizer.zero_grad()

    labeled_logits = model(labeled_x)
    
    with torch.no_grad():
        unlabeled_weak_logits = model(unlabeled_w)

    unlabeled_strong_logits = model(unlabeled_s)

    results = compute_fixmatch_loss(
        labeled_logits=labeled_logits,
        labels=labels,
        unlabeled_weak_logits=unlabeled_weak_logits,
        unlabeled_strong_logits=unlabeled_strong_logits,
        tau=tau,
        lambda_u=lambda_u,
    )

    results["total_loss"].backward()
    optimizer.step()

    # Convert scalar tensors for logging
    return {
        "total_loss": results["total_loss"].item(),
        "loss_s": results["loss_s"].item(),
        "loss_u": results["loss_u"].item(),
        "mask_ratio": results["mask_ratio"],
    }


def train_flexmatch_step(
    model: nn.Module,
    flexmatch_loss_module: FlexMatchLoss,
    optimizer: torch.optim.Optimizer,
    labeled_x: torch.Tensor,
    labels: torch.Tensor,
    unlabeled_w: torch.Tensor,
    unlabeled_s: torch.Tensor,
) -> Dict[str, Any]:
    """
    Execute one FlexMatch training step using stateful FlexMatchLoss module.
    """
    model.train()
    optimizer.zero_grad()

    labeled_logits = model(labeled_x)

    with torch.no_grad():
        unlabeled_weak_logits = model(unlabeled_w)

    unlabeled_strong_logits = model(unlabeled_s)

    results = flexmatch_loss_module(
        labeled_logits=labeled_logits,
        labels=labels,
        unlabeled_weak_logits=unlabeled_weak_logits,
        unlabeled_strong_logits=unlabeled_strong_logits,
    )

    results["total_loss"].backward()
    optimizer.step()

    return {
        "total_loss": results["total_loss"].item(),
        "loss_s": results["loss_s"].item(),
        "loss_u": results["loss_u"].item(),
        "mask_ratio": results["mask_ratio"],
        "class_thresholds": results["class_thresholds"].detach().cpu().numpy().tolist(),
        "class_counts": results["class_counts"].detach().cpu().numpy().tolist(),
    }


def evaluate_domain_performance(
    model: nn.Module,
    source_loader: Any,
    target_loader: Any,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate accuracy and loss of model on both Source (supervised domain)
    and Target (pseudo-unsupervised / test domain) datasets.
    """
    model.eval()

    def _eval_loader(loader: Any) -> Tuple[float, float]:
        total_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y.long(), reduction="sum")
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total_samples += y.size(0)
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = correct / max(total_samples, 1)
        return avg_loss, accuracy

    source_loss, source_acc = _eval_loader(source_loader)
    target_loss, target_acc = _eval_loader(target_loader)

    return {
        "source_loss": source_loss,
        "source_accuracy": source_acc,
        "target_loss": target_loss,
        "target_accuracy": target_acc,
    }


__all__ = [
    "compute_fixmatch_loss",
    "FlexMatchLoss",
    "train_fixmatch_step",
    "train_flexmatch_step",
    "evaluate_domain_performance",
]
