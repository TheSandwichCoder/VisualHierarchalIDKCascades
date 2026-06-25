from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from ._paths import ROOT as _ROOT
except ImportError:
    from _paths import ROOT as _ROOT

from globals import device
from model_trainer.get_skipper_dataset import (
    CASCADE_TYPE_TO_LABEL,
    LABEL_TO_CASCADE_TYPE,
    create_mlp_skipper_dataset,
)


class SkipperMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def _load_payload(dataset_path):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} does not exist. Create it with "
            "create_mlp_skipper_dataset() first, or call train_mlp_skipper(..., "
            "create_dataset_if_missing=True)."
        )
    return torch.load(dataset_path, map_location="cpu")


def _payload_inputs(payload):
    inputs = payload.get("inputs", payload.get("features"))
    if inputs is None:
        raise KeyError("Skipper dataset payload must contain 'inputs' or legacy 'features'")
    return inputs.to(torch.float32) if isinstance(inputs, torch.Tensor) else torch.tensor(inputs, dtype=torch.float32)


def _payload_targets(payload, binary_detector_label=True):
    targets = payload["targets"]
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    targets = targets.to(torch.long)

    if binary_detector_label:
        detector_label = CASCADE_TYPE_TO_LABEL["detector"]
        return (targets == detector_label).to(torch.long)

    return targets


def load_mlp_training_dataset(
    dataset_path="data/processed_data/skipper_dataset_mlp.pt",
    binary_detector_label=True,
):
    payload = _load_payload(dataset_path)
    inputs = _payload_inputs(payload)
    targets = _payload_targets(payload, binary_detector_label=binary_detector_label)
    return TensorDataset(inputs, targets), payload


def _class_weights(targets, output_size):
    counts = torch.bincount(targets, minlength=output_size).to(torch.float32)
    weights = torch.zeros(output_size, dtype=torch.float32)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / (output_size * counts[nonzero])
    return weights


def _evaluate(dataloader, model, loss_fn):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    confusion = torch.zeros(model.net[-1].out_features, model.net[-1].out_features, dtype=torch.long)

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item()
            correct += (predictions == y).sum().item()
            total += y.numel()

            for true_label, predicted_label in zip(y.cpu(), predictions.cpu()):
                confusion[int(true_label), int(predicted_label)] += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "average_loss": total_loss / len(dataloader) if dataloader else 0.0,
        "correct": correct,
        "total": total,
        "confusion_matrix": confusion.tolist(),
    }


def train_mlp_skipper(
    dataset_path="data/processed_data/skipper_dataset_mlp.pt",
    output_path="models/skipper/mlp_skipper.pth",
    create_dataset_if_missing=False,
    binary_detector_label=True,
    batch_size=64,
    epochs=20,
    hidden_size=256,
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    test_fraction=0.2,
    seed=42,
):
    if create_dataset_if_missing and not Path(dataset_path).exists():
        create_mlp_skipper_dataset(dataset_path=dataset_path)

    dataset, payload = load_mlp_training_dataset(
        dataset_path=dataset_path,
        binary_detector_label=binary_detector_label,
    )

    input_size = dataset.tensors[0].shape[1]
    output_size = 2 if binary_detector_label else int(dataset.tensors[1].max().item()) + 1

    train_size = int((1.0 - test_fraction) * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SkipperMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout=dropout,
    ).to(device)

    weights = _class_weights(dataset.tensors[1], output_size).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        metrics = _evaluate(test_loader, model, loss_fn)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total_loss / len(train_loader) if train_loader else 0.0
        history.append(metrics)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"test_loss={metrics['average_loss']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    final_metrics = history[-1] if history else _evaluate(test_loader, model, loss_fn)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "SkipperMLP",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "dropout": dropout,
            "binary_detector_label": binary_detector_label,
            "input_representation": payload.get("input_representation", "features"),
            "input_checkpoint_path": payload.get("input_checkpoint_path"),
            "label_to_cascade_type": payload.get("label_to_cascade_type", LABEL_TO_CASCADE_TYPE),
            "cascade_type_to_label": payload.get("cascade_type_to_label", CASCADE_TYPE_TO_LABEL),
            "history": history,
            "metrics": final_metrics,
        },
        output_path,
    )

    print(f"saved {output_path}")
    return model, final_metrics


if __name__ == "__main__":
    train_mlp_skipper(create_dataset_if_missing=True)
