"""Unit tests for PyTorch FixMatch and FlexMatch loss functions, gradient flows, and training entries."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from wingbeat_ml.training.ssl_losses import (
    FlexMatchLoss,
    compute_fixmatch_loss,
    evaluate_domain_performance,
    train_fixmatch_step,
    train_flexmatch_step,
)


@pytest.fixture
def dummy_setup():
    torch.manual_seed(42)
    num_classes = 3
    batch_size = 4
    num_unlabeled = 6

    model = nn.Linear(8, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    labeled_x = torch.randn(batch_size, 8)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    unlabeled_w = torch.randn(num_unlabeled, 8)
    unlabeled_s = torch.randn(num_unlabeled, 8)

    return {
        "num_classes": num_classes,
        "batch_size": batch_size,
        "num_unlabeled": num_unlabeled,
        "model": model,
        "optimizer": optimizer,
        "labeled_x": labeled_x,
        "labels": labels,
        "unlabeled_w": unlabeled_w,
        "unlabeled_s": unlabeled_s,
    }


def test_fixmatch_loss_calculation(dummy_setup):
    torch.manual_seed(42)
    labeled_logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2]])
    labels = torch.tensor([0, 1], dtype=torch.long)

    unlabeled_w_logits = torch.tensor([[3.0, 0.1, 0.0], [0.0, 0.1, 3.0]])
    unlabeled_s_logits = torch.tensor([[2.5, 0.2, 0.1], [0.1, 0.2, 2.8]], requires_grad=True)

    results = compute_fixmatch_loss(
        labeled_logits=labeled_logits,
        labels=labels,
        unlabeled_weak_logits=unlabeled_w_logits,
        unlabeled_strong_logits=unlabeled_s_logits,
        tau=0.90,
        lambda_u=1.0,
    )

    assert "total_loss" in results
    assert "loss_s" in results
    assert "loss_u" in results
    assert results["mask_ratio"] == 1.0
    assert torch.equal(results["pseudo_labels"], torch.tensor([0, 2]))
    assert results["total_loss"].item() > 0.0


def test_fixmatch_zero_mask_handling():
    labeled_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1], dtype=torch.long)

    # Low max confidence < tau (0.99)
    unlabeled_w_logits = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    unlabeled_s_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)

    results = compute_fixmatch_loss(
        labeled_logits=labeled_logits,
        labels=labels,
        unlabeled_weak_logits=unlabeled_w_logits,
        unlabeled_strong_logits=unlabeled_s_logits,
        tau=0.99,
        lambda_u=1.0,
    )

    assert results["mask_ratio"] == 0.0
    assert results["loss_u"].item() == 0.0
    # Ensure gradients can still backprop without runtime error
    results["total_loss"].backward()
    assert unlabeled_s_logits.grad is not None


def test_fixmatch_gradient_isolation(dummy_setup):
    model = dummy_setup["model"]
    optimizer = dummy_setup["optimizer"]

    w_logits = model(dummy_setup["unlabeled_w"]).detach()
    w_logits.requires_grad = True

    s_logits = model(dummy_setup["unlabeled_s"])
    labeled_logits = model(dummy_setup["labeled_x"])

    results = compute_fixmatch_loss(
        labeled_logits=labeled_logits,
        labels=dummy_setup["labels"],
        unlabeled_weak_logits=w_logits,
        unlabeled_strong_logits=s_logits,
        tau=0.5,
    )

    results["total_loss"].backward()
    assert w_logits.grad is None, "Weak logits should not receive gradients!"


def test_flexmatch_adaptive_thresholds_and_counts(dummy_setup):
    num_classes = dummy_setup["num_classes"]
    flex_module = FlexMatchLoss(num_classes=num_classes, tau=0.95, mapping="convex")

    pseudo_labels = torch.tensor([0, 0, 0, 1])
    max_probs = torch.tensor([0.98, 0.97, 0.96, 0.99])

    flex_module.update_class_counts(pseudo_labels, max_probs)

    assert flex_module.class_counts[0].item() == 3.0
    assert flex_module.class_counts[1].item() == 1.0
    assert flex_module.class_counts[2].item() == 0.0

    thresholds = flex_module.compute_class_thresholds()
    # Class 0 max count (3.0) -> beta=1.0 -> M(1)=1/(2-1)=1.0 -> tau=0.95
    assert torch.isclose(thresholds[0], torch.tensor(0.95))
    # Class 1 count (1.0) -> beta=1/3 -> M(1/3)=(1/3)/(5/3)=0.2 -> tau=0.19
    assert torch.isclose(thresholds[1], torch.tensor(0.19))
    # Class 2 count (0.0) -> beta=0 -> threshold=0
    assert torch.isclose(thresholds[2], torch.tensor(0.0))


def test_train_fixmatch_and_flexmatch_steps(dummy_setup):
    model = dummy_setup["model"]
    optimizer = dummy_setup["optimizer"]

    # Test FixMatch step
    step_res = train_fixmatch_step(
        model,
        optimizer,
        dummy_setup["labeled_x"],
        dummy_setup["labels"],
        dummy_setup["unlabeled_w"],
        dummy_setup["unlabeled_s"],
        tau=0.5,
    )
    assert "total_loss" in step_res
    assert step_res["total_loss"] > 0

    # Test FlexMatch step
    flex_module = FlexMatchLoss(num_classes=3, tau=0.90)
    flex_res = train_flexmatch_step(
        model,
        flex_module,
        optimizer,
        dummy_setup["labeled_x"],
        dummy_setup["labels"],
        dummy_setup["unlabeled_w"],
        dummy_setup["unlabeled_s"],
    )
    assert "total_loss" in flex_res
    assert len(flex_res["class_thresholds"]) == 3


def test_evaluate_domain_performance(dummy_setup):
    model = dummy_setup["model"]
    device = torch.device("cpu")

    source_loader = [
        (dummy_setup["labeled_x"], dummy_setup["labels"])
    ]
    target_loader = [
        (dummy_setup["unlabeled_w"], torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long))
    ]

    metrics = evaluate_domain_performance(model, source_loader, target_loader, device)
    assert "source_loss" in metrics
    assert "source_accuracy" in metrics
    assert "target_loss" in metrics
    assert "target_accuracy" in metrics
