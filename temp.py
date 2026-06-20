from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from globals import device
from train_models import (
    ImageNetLogitLinearRouter,
    ImageNetLogitMLPRouter,
    ImageNetTransforms,
    LogitRouterPipeline,
    create_category_dataset,
    data,
    imagenetv2_dataset,
)


DEFAULT_CHECKPOINT = Path("models/logit_router/resnet34_logit_router.pth")


def load_logit_router_checkpoint(checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("backbone_name") != "resnet34":
        raise ValueError(f"Unsupported backbone: {checkpoint.get('backbone_name')}")

    backbone = torchvision.models.resnet34(weights=None).to(device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    backbone.eval()

    settings = checkpoint.get("settings", {})
    if checkpoint.get("router_name") == "ImageNetLogitLinearRouter":
        router = ImageNetLogitLinearRouter(checkpoint["num_categories"]).to(device)
    elif checkpoint.get("router_name") == "ImageNetLogitMLPRouter":
        router = ImageNetLogitMLPRouter(
            checkpoint["num_categories"],
            hidden_size=settings.get("hidden_size", 256),
            dropout=settings.get("dropout", 0.5),
        ).to(device)
    else:
        raise ValueError(f"Unsupported router: {checkpoint.get('router_name')}")

    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()

    model = LogitRouterPipeline(
        backbone,
        router,
        use_probabilities=settings.get("use_probabilities", True),
    ).to(device)
    model.eval()
    return checkpoint, model


def collect_confidences_and_correct(model, batch_size=64):
    loader = DataLoader(
        create_category_dataset(imagenetv2_dataset, data["groups"]),
        batch_size=batch_size,
        shuffle=False,
    )

    confidences = []
    correct = []
    total = 0
    raw_correct = 0

    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(device), y.to(device)
            probabilities = torch.softmax(model(X), dim=1)
            batch_confidences, predictions = probabilities.max(dim=1)

            batch_correct = predictions == y
            confidences.extend(batch_confidences.cpu().tolist())
            correct.extend(batch_correct.cpu().tolist())
            raw_correct += batch_correct.sum().item()
            total += y.numel()

    return confidences, correct, raw_correct / total


def metrics_at_threshold(confidences, correct, threshold):
    accepted = [
        is_correct
        for confidence, is_correct in zip(confidences, correct)
        if confidence >= threshold
    ]
    accepted_count = len(accepted)
    if accepted_count == 0:
        return {
            "threshold": threshold,
            "precision": 0.0,
            "recall": 0.0,
            "accepted": 0,
            "correct": 0,
        }

    correct_count = sum(accepted)
    return {
        "threshold": threshold,
        "precision": correct_count / accepted_count,
        "recall": accepted_count / len(confidences),
        "accepted": accepted_count,
        "correct": correct_count,
    }


def best_for_target_precision(confidences, correct, target_precision):
    candidates = [
        metrics_at_threshold(confidences, correct, threshold)
        for threshold in sorted(set(confidences))
    ]
    passing = [
        candidate
        for candidate in candidates
        if candidate["precision"] >= target_precision
    ]
    if passing:
        return max(passing, key=lambda item: (item["recall"], -item["threshold"]))

    return max(candidates, key=lambda item: (item["precision"], item["recall"]))


def print_metric_row(label, metrics):
    print(
        f"{label:<12} "
        f"threshold={metrics['threshold']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"accepted={metrics['accepted']} "
        f"correct={metrics['correct']}"
    )


def analyze_cutoffs(checkpoint_path=DEFAULT_CHECKPOINT, batch_size=64):
    checkpoint, model = load_logit_router_checkpoint(checkpoint_path)

    print("Checkpoint values:")
    for key, value in checkpoint.items():
        if key in {"backbone_state_dict", "router_state_dict"}:
            print(f"  {key}: {len(value)} tensors")
        else:
            print(f"  {key}: {value}")

    confidences, correct, raw_accuracy = collect_confidences_and_correct(
        model,
        batch_size=batch_size,
    )

    print(f"\nRaw accuracy: {raw_accuracy:.4f}")
    print(f"Total samples: {len(confidences)}")
    print("\nFixed thresholds:")
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print_metric_row(str(threshold), metrics_at_threshold(confidences, correct, threshold))

    print("\nBest thresholds by target precision:")
    for target_precision in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        metrics = best_for_target_precision(confidences, correct, target_precision)
        print_metric_row(f">={target_precision:.2f}", metrics)


if __name__ == "__main__":
    analyze_cutoffs()
