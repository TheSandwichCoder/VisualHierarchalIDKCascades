from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from ._paths import ROOT as _ROOT
except ImportError:
    from _paths import ROOT as _ROOT

from empirical_cascade_optimizer.empirical_cascade_builder import construct_empirical_cascade
from globals import device
from model_trainer.get_models_stats import load_model_from_checkpoint
from model_trainer.train_models import imagenetv2_dataset


CASCADE_TYPE_TO_LABEL = {
    "detector": 0,
    "global": 1,
    "specialized": 2,
}
LABEL_TO_CASCADE_TYPE = {
    label: cascade_type
    for cascade_type, label in CASCADE_TYPE_TO_LABEL.items()
}
INPUT_REPRESENTATIONS = {"features", "logits", "probabilities"}


def _extract_features(model, sample):
    if hasattr(model, "features") and hasattr(model, "avgpool"):
        values = model.features(sample)
        values = model.avgpool(values)
        return torch.flatten(values, 1)

    if all(hasattr(model, name) for name in ("conv1", "bn1", "relu", "maxpool", "avgpool")):
        values = model.conv1(sample)
        values = model.bn1(values)
        values = model.relu(values)
        values = model.maxpool(values)
        values = model.layer1(values)
        values = model.layer2(values)
        values = model.layer3(values)
        values = model.layer4(values)
        values = model.avgpool(values)
        return torch.flatten(values, 1)

    if hasattr(model, "backbone"):
        return _extract_features(model.backbone, sample)

    raise ValueError(
        f"Cannot extract backbone features from {type(model).__name__}. "
        "Use input_representation='logits' or 'probabilities' for this model."
    )


def _extract_model_input(model, sample, input_representation):
    if input_representation not in INPUT_REPRESENTATIONS:
        raise ValueError(
            f"input_representation must be one of {sorted(INPUT_REPRESENTATIONS)}, "
            f"got {input_representation!r}"
        )

    with torch.no_grad():
        if input_representation == "features":
            values = _extract_features(model, sample)
        else:
            values = model(sample)
            if input_representation == "probabilities":
                values = torch.softmax(values, dim=1)

    return values.squeeze(0).detach().cpu()


def _create_skipper_dataset(
    path,
    dataset_path,
    input_checkpoint_path,
    input_representation,
    batch_size,
    max_samples,
    save,
):
    cascade = construct_empirical_cascade(path)
    input_model, _ = load_model_from_checkpoint(input_checkpoint_path)
    input_model.eval()

    loader = DataLoader(imagenetv2_dataset, batch_size=batch_size, shuffle=False)
    input_rows = []
    target_rows = []
    total = 0
    kept = 0

    for X, _ in tqdm(loader, desc=f"skipper {input_representation}"):
        if max_samples is not None and total >= max_samples:
            break

        X = X.to(device)

        for sample_index in range(X.shape[0]):
            if max_samples is not None and total >= max_samples:
                break

            sample = X[sample_index:sample_index + 1]

            trace = cascade.pred_with_type(sample)
            if trace.initial_returned_idk:
                input_rows.append(
                    _extract_model_input(
                        input_model,
                        sample,
                        input_representation=input_representation,
                    )
                )
                target_rows.append(CASCADE_TYPE_TO_LABEL[trace.ending_cascade_type])
                kept += 1

            total += 1

    if not input_rows:
        raise ValueError(
            "No samples reached the skipper condition. Try a larger max_samples "
            "or inspect whether the empirical cascade ever IDKs in its initial path."
        )

    X_tensor = torch.stack(input_rows).to(torch.float32)
    y_tensor = torch.tensor(target_rows, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    if save:
        output_path = Path(dataset_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "inputs": X_tensor,
                "targets": y_tensor,
                "input_representation": input_representation,
                "label_to_cascade_type": LABEL_TO_CASCADE_TYPE,
                "cascade_type_to_label": CASCADE_TYPE_TO_LABEL,
                "source_samples": total,
                "kept_samples": kept,
                "input_checkpoint_path": str(input_checkpoint_path),
            },
            output_path,
        )

    return dataset


def create_mlp_skipper_dataset(
    path="models/stats/empirical_outcomes.pkl",
    dataset_path="data/processed_data/skipper_dataset_mlp.pt",
    input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    batch_size=1,
    max_samples=None,
    input_representation="features",
    save=True,
):
    """Create an MLP skipper dataset.

    By default this stores true backbone features, not logits/probabilities.
    For MobileNetV3 small, the default feature vector has 576 values.
    """
    return _create_skipper_dataset(
        path=path,
        dataset_path=dataset_path,
        input_checkpoint_path=input_checkpoint_path,
        input_representation=input_representation,
        batch_size=batch_size,
        max_samples=max_samples,
        save=save,
    )


def create_rf_skipper_dataset(
    path="models/stats/empirical_outcomes.pkl",
    dataset_path="data/processed_data/skipper_dataset_rf.pt",
    input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    batch_size=1,
    max_samples=None,
    input_representation="probabilities",
    save=True,
):
    """Create an RF skipper dataset.

    By default this stores the identifier model's probability vector, not
    hidden features. Use input_representation='logits' for raw outputs or
    input_representation='features' for backbone features.
    """
    return _create_skipper_dataset(
        path=path,
        dataset_path=dataset_path,
        input_checkpoint_path=input_checkpoint_path,
        input_representation=input_representation,
        batch_size=batch_size,
        max_samples=max_samples,
        save=save,
    )


def create_skipper_dataset(*args, **kwargs):
    return create_mlp_skipper_dataset(*args, **kwargs)


def load_skipper_dataset(path="data/processed_data/skipper_dataset_mlp.pt"):
    payload = torch.load(path, map_location="cpu")
    inputs = payload.get("inputs", payload.get("features"))
    if inputs is None:
        raise KeyError(f"{path} does not contain 'inputs' or legacy 'features'")
    return TensorDataset(inputs, payload["targets"])


if __name__ == "__main__":
    dataset = create_mlp_skipper_dataset()
    print(f"created skipper dataset with {len(dataset)} samples")
