from pathlib import Path
from time import perf_counter
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from globals import device
from train_models import (
    ImageNetLogitLinearRouter,
    ImageNetLogitMLPRouter,
    LogitRouterPipeline,
    create_category_dataset,
    create_class_subset,
    data,
    get_mobilenet_v3_large,
    get_mobilenet_v3_large_mlp_identifier,
    get_mobilenet_v3_small,
    get_mobilenet_v3_small_mlp_identifier,
    get_resnet_18,
    get_resnet_34,
    imagenetv2_dataset,
)


def _load_logit_router_model(checkpoint):
    state_dict = checkpoint["backbone_state_dict"]
    backbone_name = _infer_resnet_backbone_name(state_dict, checkpoint.get("backbone_name"))
    if backbone_name == "resnet18":
        backbone = torchvision.models.resnet18(weights=None)
    elif backbone_name == "resnet34":
        backbone = torchvision.models.resnet34(weights=None)
    elif backbone_name == "resnet50":
        backbone = torchvision.models.resnet50(weights=None)
    elif backbone_name == "resnet101":
        backbone = torchvision.models.resnet101(weights=None)
    elif backbone_name == "resnet152":
        backbone = torchvision.models.resnet152(weights=None)
    else:
        raise ValueError(f"Unsupported logit-router backbone: {backbone_name}")

    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()

    settings = checkpoint.get("settings", {})
    router_name = checkpoint.get("router_name")
    num_categories = checkpoint["num_categories"]
    if router_name == "ImageNetLogitLinearRouter":
        router = ImageNetLogitLinearRouter(num_categories)
    elif router_name == "ImageNetLogitMLPRouter":
        router = ImageNetLogitMLPRouter(
            num_categories,
            hidden_size=settings.get("hidden_size", 256),
            dropout=settings.get("dropout", 0.5),
        )
    else:
        raise ValueError(f"Unsupported router: {router_name}")

    router.load_state_dict(checkpoint["router_state_dict"])
    router.to(device)
    router.eval()

    model = LogitRouterPipeline(
        backbone,
        router,
        use_probabilities=settings.get("use_probabilities", True),
    )
    model.to(device)
    model.eval()
    return model


def _infer_resnet_backbone_name(state_dict, fallback=None):
    fc_in_features = state_dict["fc.weight"].shape[1]
    layer_counts = {}
    for key in state_dict:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0].startswith("layer") and parts[1].isdigit():
            layer_counts.setdefault(parts[0], set()).add(int(parts[1]))

    block_counts = {
        layer: max(indices) + 1
        for layer, indices in layer_counts.items()
        if indices
    }

    if fc_in_features == 512:
        if block_counts.get("layer1") == 3:
            return "resnet34"
        return "resnet18"

    if fc_in_features == 2048:
        layer3_blocks = block_counts.get("layer3")
        if layer3_blocks == 6:
            return "resnet50"
        if layer3_blocks == 23:
            return "resnet101"
        if layer3_blocks == 36:
            return "resnet152"

    return fallback


def _load_category_state_dict_model(checkpoint):
    num_classes = checkpoint["num_classes"]
    state_dict = checkpoint["state_dict"]

    if any(key.startswith("layer4.") for key in state_dict):
        backbone_name = _infer_resnet_backbone_name(
            state_dict,
            checkpoint.get("model_type"),
        )
        if backbone_name == "resnet18":
            model = torchvision.models.resnet18(weights=None)
        elif backbone_name == "resnet34":
            model = torchvision.models.resnet34(weights=None)
        elif backbone_name == "resnet50":
            model = torchvision.models.resnet50(weights=None)
        elif backbone_name == "resnet101":
            model = torchvision.models.resnet101(weights=None)
        elif backbone_name == "resnet152":
            model = torchvision.models.resnet152(weights=None)
        else:
            raise ValueError(f"Could not infer ResNet architecture: {backbone_name}")

        if model.fc.out_features != num_classes:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif any(key.startswith("features.") for key in state_dict):
        model_type = checkpoint.get("model_type")
        if model_type == "mobilenet_v3_small_mlp_identifier":
            model = get_mobilenet_v3_small_mlp_identifier(
                num_classes,
                hidden_size=checkpoint.get("hidden_size") or 512,
                dropout=checkpoint.get("dropout") or 0.2,
            )
        elif model_type == "mobilenet_v3_large_mlp_identifier":
            model = get_mobilenet_v3_large_mlp_identifier(
                num_classes,
                hidden_size=checkpoint.get("hidden_size") or 512,
                dropout=checkpoint.get("dropout") or 0.2,
            )
        else:
            classifier_input_features = state_dict["classifier.0.weight"].shape[1]
            if classifier_input_features == 576:
                model = get_mobilenet_v3_small(num_classes)
            elif classifier_input_features == 960:
                model = get_mobilenet_v3_large(num_classes)
            else:
                raise ValueError(
                    "Could not infer MobileNetV3 architecture from "
                    f"classifier.0.weight shape {state_dict['classifier.0.weight'].shape}"
                )
    else:
        raise ValueError("Could not infer model architecture from checkpoint state_dict")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "backbone_state_dict" in checkpoint and "router_state_dict" in checkpoint:
        return _load_logit_router_model(checkpoint), checkpoint
    if "state_dict" in checkpoint:
        return _load_category_state_dict_model(checkpoint), checkpoint
    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def _subset_dataset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def evaluate_model_stats(model, dataloader, num_classes, confidence_threshold=0.0):
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    predicted_category_counts = torch.zeros(num_classes, dtype=torch.long)
    accepted_category_counts = torch.zeros(num_classes, dtype=torch.long)
    total = 0
    correct = 0
    accepted = 0
    accepted_correct = 0
    runtime_seconds = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            logits = model(X)
            if logits.shape[1] != num_classes:
                raise ValueError(
                    f"Model outputs {logits.shape[1]} classes, but the shared "
                    f"category test set has {num_classes} classes"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            runtime_seconds += perf_counter() - start

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            accepted_mask = confidences >= confidence_threshold

            total += y.numel()
            correct += (predictions == y).sum().item()
            accepted += accepted_mask.sum().item()
            accepted_correct += ((predictions == y) & accepted_mask).sum().item()
            predicted_category_counts += torch.bincount(
                predictions.cpu(),
                minlength=num_classes,
            )
            accepted_predictions = predictions[accepted_mask].cpu()
            accepted_category_counts += torch.bincount(
                accepted_predictions,
                minlength=num_classes,
            )

            for true_label, predicted_label in zip(y.cpu(), predictions.cpu()):
                confusion_matrix[int(true_label), int(predicted_label)] += 1

    precision = accepted_correct / accepted if accepted else 0.0
    recall = accepted / total if total else 0.0
    accuracy = correct / total if total else 0.0
    average_runtime = runtime_seconds / total if total else 0.0

    return {
        "average_runtime": average_runtime,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "accepted": accepted,
        "total": total,
        "predicted_category_counts": predicted_category_counts.tolist(),
        "predicted_category_probabilities": [
            count / total if total else 0.0
            for count in predicted_category_counts.tolist()
        ],
        "accepted_category_counts": accepted_category_counts.tolist(),
        "accepted_category_probabilities": [
            count / total if total else 0.0
            for count in accepted_category_counts.tolist()
        ],
        "accepted_category_distribution": [
            count / accepted if accepted else 0.0
            for count in accepted_category_counts.tolist()
        ],
        "confusion_matrix": confusion_matrix.tolist(),
    }


def collect_model_outputs(model, dataloader, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    predicted_category_counts = torch.zeros(num_classes, dtype=torch.long)
    confidences = []
    correct_flags = []
    predicted_labels = []
    total = 0
    correct = 0
    runtime_seconds = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            logits = model(X)
            if logits.shape[1] != num_classes:
                raise ValueError(
                    f"Model outputs {logits.shape[1]} classes, but the shared "
                    f"category test set has {num_classes} classes"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            runtime_seconds += perf_counter() - start

            probabilities = torch.softmax(logits, dim=1)
            batch_confidences, predictions = probabilities.max(dim=1)
            batch_correct = predictions == y

            confidences.extend(batch_confidences.cpu().tolist())
            correct_flags.extend(batch_correct.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            predicted_category_counts += torch.bincount(
                predictions.cpu(),
                minlength=num_classes,
            )
            total += y.numel()
            correct += batch_correct.sum().item()

            for true_label, predicted_label in zip(y.cpu(), predictions.cpu()):
                confusion_matrix[int(true_label), int(predicted_label)] += 1

    return {
        "confidences": confidences,
        "correct_flags": correct_flags,
        "predicted_labels": predicted_labels,
        "average_runtime": runtime_seconds / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "predicted_category_counts": predicted_category_counts.tolist(),
        "predicted_category_probabilities": [
            count / total if total else 0.0
            for count in predicted_category_counts.tolist()
        ],
        "confusion_matrix": confusion_matrix.tolist(),
    }


def stats_from_collected_outputs(outputs, confidence_threshold):
    num_classes = len(outputs["predicted_category_counts"])
    accepted_category_counts = [0 for _ in range(num_classes)]
    accepted_flags = []
    for confidence, is_correct, predicted_label in zip(
        outputs["confidences"],
        outputs["correct_flags"],
        outputs["predicted_labels"],
    ):
        if confidence >= confidence_threshold:
            accepted_flags.append(is_correct)
            accepted_category_counts[int(predicted_label)] += 1

    accepted = len(accepted_flags)
    accepted_correct = sum(accepted_flags)

    return {
        "average_runtime": outputs["average_runtime"],
        "accuracy": outputs["accuracy"],
        "recall": accepted / outputs["total"] if outputs["total"] else 0.0,
        "precision": accepted_correct / accepted if accepted else 0.0,
        "accepted": accepted,
        "total": outputs["total"],
        "predicted_category_counts": outputs["predicted_category_counts"],
        "predicted_category_probabilities": outputs["predicted_category_probabilities"],
        "accepted_category_counts": accepted_category_counts,
        "accepted_category_probabilities": [
            count / outputs["total"] if outputs["total"] else 0.0
            for count in accepted_category_counts
        ],
        "accepted_category_distribution": [
            count / accepted if accepted else 0.0
            for count in accepted_category_counts
        ],
        "confusion_matrix": outputs["confusion_matrix"],
    }


def threshold_for_target_precision(outputs, target_precision):
    thresholds = sorted(set(outputs["confidences"]))
    candidates = [
        (threshold, stats_from_collected_outputs(outputs, threshold))
        for threshold in thresholds
    ]
    passing = [
        (threshold, stats)
        for threshold, stats in candidates
        if stats["precision"] >= target_precision
    ]
    if passing:
        return max(passing, key=lambda item: (item[1]["recall"], -item[0]))
    return max(candidates, key=lambda item: (item[1]["precision"], item[1]["recall"]))


def create_specialized_model_stats_dataframe(
    checkpoint_paths=None,
    test_dataset=None,
    batch_size=64,
    confidence_thresholds=None,
    confidence_threshold=None,
    target_precisions=None,
    max_samples=None,
):
    if checkpoint_paths is None:
        directory = Path("models/specialized")
        checkpoint_paths = sorted(directory.glob("*.pth")) if directory.exists() else []
    else:
        checkpoint_paths = [Path(path) for path in checkpoint_paths]

    if confidence_thresholds is None:
        if confidence_threshold is None:
            confidence_thresholds = [0.0]
        else:
            confidence_thresholds = [confidence_threshold]
    if target_precisions is None:
        target_precisions = []

    rows = []

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating {checkpoint_path}")
        try:
            model, checkpoint = load_model_from_checkpoint(checkpoint_path)
            class_ids = [int(class_id) for class_id in checkpoint["class_ids"]]
            num_classes = int(checkpoint["num_classes"])
            if num_classes != len(class_ids):
                raise ValueError(
                    f"Checkpoint says num_classes={num_classes}, but has "
                    f"{len(class_ids)} class_ids"
                )

            if test_dataset is None:
                model_test_dataset = create_class_subset(
                    class_ids,
                    source_dataset=imagenetv2_dataset,
                    remap_labels=True,
                )
            else:
                model_test_dataset = test_dataset
            model_test_dataset = _subset_dataset(model_test_dataset, max_samples)
            dataloader = DataLoader(model_test_dataset, batch_size=batch_size, shuffle=False)

            outputs = collect_model_outputs(
                model,
                dataloader,
                num_classes=num_classes,
            )

            for threshold in confidence_thresholds:
                if threshold == "checkpoint":
                    threshold = checkpoint.get("confidence_threshold", 0.0)
                stats = stats_from_collected_outputs(outputs, float(threshold))
                rows.append({
                    "model_name": checkpoint_path.stem,
                    "checkpoint_path": str(checkpoint_path),
                    "class_ids": class_ids,
                    "num_classes": num_classes,
                    "threshold_kind": "fixed",
                    "target_precision": None,
                    "confidence_threshold": float(threshold),
                    **stats,
                })

            for target_precision in target_precisions:
                threshold, stats = threshold_for_target_precision(outputs, target_precision)
                rows.append({
                    "model_name": checkpoint_path.stem,
                    "checkpoint_path": str(checkpoint_path),
                    "class_ids": class_ids,
                    "num_classes": num_classes,
                    "threshold_kind": "target_precision",
                    "target_precision": target_precision,
                    "confidence_threshold": threshold,
                    **stats,
                })
        except Exception as error:
            rows.append({
                "model_name": checkpoint_path.stem,
                "checkpoint_path": str(checkpoint_path),
                "error": str(error),
            })

    return pd.DataFrame(rows)




def create_intermediate_model_stats_dataframe(
    checkpoint_paths=None,
    test_dataset=None,
    batch_size=64,
    confidence_threshold=None,
    confidence_thresholds=None,
    target_precisions=None,
    max_samples=None,
):
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "pandas is required to create the model stats DataFrame. "
            "Install it with: .\\.venv\\Scripts\\python.exe -m pip install pandas"
        ) from error

    if checkpoint_paths is None:
        checkpoint_paths = []
        for directory in [Path("models/logit_router"), Path("models/intermediate")]:
            if directory.exists():
                checkpoint_paths.extend(sorted(directory.glob("*.pth")))
    else:
        checkpoint_paths = [Path(path) for path in checkpoint_paths]

    if test_dataset is None:
        test_dataset = create_category_dataset(imagenetv2_dataset, data["groups"])
    test_dataset = _subset_dataset(test_dataset, max_samples)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rows = []
    num_classes = len(data["groups"])
    if confidence_thresholds is None:
        if confidence_threshold is None:
            confidence_thresholds = [0.0]
        else:
            confidence_thresholds = [confidence_threshold]

    if target_precisions is None:
        target_precisions = []

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating {checkpoint_path}")
        try:
            model, checkpoint = load_model_from_checkpoint(checkpoint_path)
            outputs = collect_model_outputs(
                model,
                dataloader,
                num_classes=num_classes,
            )

            for threshold in confidence_thresholds:
                if threshold == "checkpoint":
                    threshold = checkpoint.get("confidence_threshold", 0.0)
                stats = stats_from_collected_outputs(outputs, float(threshold))
                rows.append({
                    "model_name": checkpoint_path.stem,
                    "checkpoint_path": str(checkpoint_path),
                    "threshold_kind": "fixed",
                    "target_precision": None,
                    "confidence_threshold": float(threshold),
                    **stats,
                })

            for target_precision in target_precisions:
                threshold, stats = threshold_for_target_precision(outputs, target_precision)
                rows.append({
                    "model_name": checkpoint_path.stem,
                    "checkpoint_path": str(checkpoint_path),
                    "threshold_kind": "target_precision",
                    "target_precision": target_precision,
                    "confidence_threshold": threshold,
                    **stats,
                })
        except Exception as error:
            rows.append({
                "model_name": checkpoint_path.stem,
                "checkpoint_path": str(checkpoint_path),
                "error": str(error),
            })

    return pd.DataFrame(rows)


def create_intermediate_category_probability_dataframe(*args, **kwargs):
    stats_df = create_intermediate_model_stats_dataframe(*args, **kwargs)
    return category_probability_dataframe_from_stats(stats_df)


def category_probability_dataframe_from_stats(stats_df):
    rows = []

    for _, row in stats_df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            continue

        predicted_probabilities = row.get("predicted_category_probabilities")
        accepted_probabilities = row.get("accepted_category_probabilities")
        accepted_distribution = row.get("accepted_category_distribution")
        if not isinstance(predicted_probabilities, list):
            continue

        for category_id, group in enumerate(data["groups"]):
            rows.append({
                "model_name": row["model_name"],
                "checkpoint_path": row["checkpoint_path"],
                "threshold_kind": row["threshold_kind"],
                "target_precision": row["target_precision"],
                "confidence_threshold": row["confidence_threshold"],
                "category_id": category_id,
                "category_name": group["name"],
                "predicted_probability": predicted_probabilities[category_id],
                "accepted_probability": accepted_probabilities[category_id],
                "accepted_distribution": accepted_distribution[category_id],
            })

    return pd.DataFrame(rows)


def expand_probability_columns_for_csv(df, groups=None):
    if groups is None:
        groups = data["groups"]

    expanded_df = df.copy()
    probability_data = {}
    probability_columns = {
        "predicted_category_probabilities": "predicted_probability",
        "accepted_category_probabilities": "accepted_probability",
        "accepted_category_distribution": "accepted_distribution",
    }

    for source_column, prefix in probability_columns.items():
        if source_column not in expanded_df.columns:
            continue

        for category_id, group in enumerate(groups):
            column_name = f"{prefix}_{category_id:02d}"
            probability_data[column_name] = expanded_df[source_column].apply(
                lambda values: values[category_id]
                if isinstance(values, list) and category_id < len(values)
                else None
            )

    if probability_data:
        expanded_df = pd.concat([expanded_df, pd.DataFrame(probability_data)], axis=1)

    return expanded_df


def save_model_stats_csvs(df, output_prefix="model_stats"):
    compact_df = df.drop(columns=["confusion_matrix"], errors="ignore")
    expanded_df = expand_probability_columns_for_csv(compact_df)
    expanded_df.to_csv(f"{output_prefix}.csv", index=False)

    probability_df = category_probability_dataframe_from_stats(df)
    probability_df.to_csv(f"{output_prefix}_category_probabilities.csv", index=False)

    return expanded_df, probability_df


def create_model_stats_dataframe(*args, **kwargs):
    return create_intermediate_model_stats_dataframe(*args, **kwargs)


def create_mobilenet_identifier_stats_dataframe(
    checkpoint_paths=None,
    target_precisions=(0.75, 0.80, 0.85, 0.90, 0.95),
    **kwargs,
):
    if checkpoint_paths is None:
        checkpoint_paths = [
            path
            for path in sorted(Path("models/intermediate").glob("*.pth"))
            if "mobilenet" in path.stem.lower()
        ]

    kwargs.setdefault("confidence_thresholds", [])
    kwargs.setdefault("target_precisions", list(target_precisions))

    return create_intermediate_model_stats_dataframe(
        checkpoint_paths=checkpoint_paths,
        **kwargs,
    )


def save_mobilenet_identifier_stats(
    output_path="models/stats/mobilenet_identifier_stats.pkl",
    csv_path="mobilenet_identifier_stats.csv",
    **kwargs,
):
    df = create_mobilenet_identifier_stats_dataframe(**kwargs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)
    df.drop(columns=["confusion_matrix"], errors="ignore").to_csv(csv_path, index=False)
    return df


def create_global_model_stats_dataframe(
    checkpoint_paths=None,
    model_directory="models/global",
    test_dataset=None,
    batch_size=64,
    confidence_threshold=None,
    confidence_thresholds=None,
    target_precisions=None,
    max_samples=None,
):
    if checkpoint_paths is None:
        directory = Path(model_directory)
        checkpoint_paths = sorted(directory.glob("*.pth")) if directory.exists() else []
    else:
        checkpoint_paths = [Path(path) for path in checkpoint_paths]

    if test_dataset is None:
        test_dataset = imagenetv2_dataset
    test_dataset = _subset_dataset(test_dataset, max_samples)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if confidence_thresholds is None:
        if confidence_threshold is None:
            confidence_thresholds = [0.0]
        else:
            confidence_thresholds = [confidence_threshold]
    if target_precisions is None:
        target_precisions = []

    rows = []
    num_classes = 1000

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating {checkpoint_path}")
        try:
            model, checkpoint = load_model_from_checkpoint(checkpoint_path)
            outputs = collect_model_outputs(
                model,
                dataloader,
                num_classes=num_classes,
            )

            for threshold in confidence_thresholds:
                if threshold == "checkpoint":
                    threshold = checkpoint.get("confidence_threshold", 0.0)
                stats = stats_from_collected_outputs(outputs, float(threshold))
                rows.append({
                    "model_name": checkpoint_path.stem,
                    "checkpoint_path": str(checkpoint_path),
                    "threshold_kind": "fixed",
                    "target_precision": None,
                    "confidence_threshold": float(threshold),
                    **stats,
                })

            for target_precision in target_precisions:
                threshold, stats = threshold_for_target_precision(outputs, target_precision)
                rows.append({
                    "model_name": checkpoint_path.stem,
                    "checkpoint_path": str(checkpoint_path),
                    "threshold_kind": "target_precision",
                    "target_precision": target_precision,
                    "confidence_threshold": threshold,
                    **stats,
                })
        except Exception as error:
            rows.append({
                "model_name": checkpoint_path.stem,
                "checkpoint_path": str(checkpoint_path),
                "error": str(error),
            })

    return pd.DataFrame(rows)


def create_detector_model_stats_dataframe(*args, **kwargs):
    kwargs.setdefault("model_directory", "models/det")
    return create_global_model_stats_dataframe(*args, **kwargs)


def save_global_model_stats_csv(df, output_path="global_model_stats.csv"):
    df.drop(columns=["confusion_matrix"], errors="ignore").to_csv(output_path, index=False)

if __name__ == "__main__":
    df = create_intermediate_model_stats_dataframe(
        checkpoint_paths=sorted(Path("models/intermediate").glob("*.pth")),
        batch_size=64,
        confidence_thresholds=[],
        target_precisions=[0.75, 0.80, 0.85, 0.90, 0.95],
    )

    # df = create_global_model_stats_dataframe(
    #     checkpoint_paths=sorted(Path("models/det").glob("*.pth")),
    #     batch_size=64,
    #     target_precisions=[0.75, 0.80, 0.85, 0.90, 0.95],
    # )

    print(df.drop(columns=["confusion_matrix"], errors="ignore"))
    df.to_pickle("model_stats.pkl")
    save_model_stats_csvs(df)
