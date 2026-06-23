from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from get_models_stats import load_model_from_checkpoint
from globals import device
from optimize_cascade import (
    GlobClassifierNode,
    IdenClassifierNode,
    SpecClassifierNode,
    get_classifiers,
)
from train_models import data, imagenetv2_dataset


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    group_id: int | None
    name: str
    checkpoint_path: str
    threshold: float
    cost: float
    precision: float
    recall: float


def _candidate_id(kind, checkpoint_path, threshold, group_id=None):
    stem = Path(str(checkpoint_path)).stem
    threshold_text = f"{float(threshold):.8f}"
    if group_id is None:
        return f"{kind}:{stem}:{threshold_text}"
    return f"{kind}:{group_id}:{stem}:{threshold_text}"


def _candidate_from_node(node, kind, group_id=None):
    return Candidate(
        id=_candidate_id(kind, node.checkpoint_path, node.threshold, group_id),
        kind=kind,
        group_id=group_id,
        name=str(node.name),
        checkpoint_path=str(node.checkpoint_path),
        threshold=0.0 if node.threshold is None else float(node.threshold),
        cost=float(node.raw_cost),
        precision=float(node.p),
        recall=float(node.r),
    )


def load_candidates(
    specialized_path="models/stats/specialized_stats.pkl",
    identifier_path="models/stats/identifier_stats.pkl",
    globals_path="models/stats/global_stats.pkl",
    det_path="models/stats/det_stats.pkl",
    min_precision=0.75,
):
    specialized_nodes, identifier_nodes, global_nodes, det_node = get_classifiers(
        specialized_path=specialized_path,
        identifier_path=identifier_path,
        globals_path=globals_path,
        det_path=det_path,
        min_precision=min_precision,
    )

    candidates = []
    for node in global_nodes:
        candidates.append(_candidate_from_node(node, "global"))
    for node in identifier_nodes:
        candidates.append(_candidate_from_node(node, "identifier"))
    for group_id, nodes in enumerate(specialized_nodes):
        for node in nodes:
            candidates.append(_candidate_from_node(node, "specialized", group_id))

    det_candidate = _candidate_from_node(det_node, "detector")
    return candidates, det_candidate


def _dataset(max_samples=None):
    if max_samples is None:
        return imagenetv2_dataset
    return Subset(imagenetv2_dataset, list(range(min(max_samples, len(imagenetv2_dataset)))))


def _class_to_group():
    mapping = {}
    for group_id, group in enumerate(data["groups"]):
        for item in group["classes"]:
            mapping[int(item["index"])] = group_id
    return mapping


def _run_candidate(candidate, dataset, batch_size):
    model, checkpoint = load_model_from_checkpoint(candidate.checkpoint_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    accepted = []
    predictions = []
    true_labels = []

    class_ids = [int(class_id) for class_id in checkpoint.get("class_ids", [])]

    with torch.no_grad():
        for X, y in tqdm(loader, desc=candidate.id):
            X = X.to(device)
            logits = model(X)
            probabilities = torch.softmax(logits, dim=1)
            confidences, batch_predictions = probabilities.max(dim=1)
            batch_accepted = confidences >= candidate.threshold

            if candidate.kind == "specialized":
                mapped = [
                    class_ids[int(prediction)]
                    for prediction in batch_predictions.cpu().tolist()
                ]
            else:
                mapped = batch_predictions.cpu().tolist()

            accepted.extend(batch_accepted.cpu().tolist())
            predictions.extend(int(prediction) for prediction in mapped)
            true_labels.extend(int(label) for label in y.tolist())

    return accepted, predictions, true_labels


def collect_empirical_outcomes(
    output_path="models/stats/empirical_outcomes.pkl",
    min_precision=0.95,
    batch_size=64,
    max_samples=None,
    specialized_path="models/stats/specialized_stats.pkl",
    identifier_path="models/stats/identifier_stats.pkl",
    globals_path="models/stats/global_stats.pkl",
    det_path="models/stats/det_stats.pkl",
):
    candidates, det_candidate = load_candidates(
        specialized_path=specialized_path,
        identifier_path=identifier_path,
        globals_path=globals_path,
        det_path=det_path,
        min_precision=min_precision,
    )
    dataset = _dataset(max_samples)
    class_to_group = _class_to_group()

    metadata_rows = []
    outcome_frames = []
    true_labels = None

    for candidate in candidates:
        accepted, predictions, labels = _run_candidate(candidate, dataset, batch_size)
        if true_labels is None:
            true_labels = labels

        metadata_rows.append(candidate.__dict__)
        outcome_frames.append(pd.DataFrame({
            "sample_id": list(range(len(labels))),
            "candidate_id": candidate.id,
            "accepted": accepted,
            "prediction": predictions,
        }))

    labels_df = pd.DataFrame({
        "sample_id": list(range(len(true_labels or []))),
        "true_label": true_labels or [],
        "true_group": [class_to_group[int(label)] for label in (true_labels or [])],
    })

    payload = {
        "min_precision": min_precision,
        "labels": labels_df,
        "candidates": pd.DataFrame(metadata_rows),
        "detector": det_candidate.__dict__,
        "outcomes": pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, output_path)
    return payload


def load_empirical_outcomes(path="models/stats/empirical_outcomes.pkl"):
    return pd.read_pickle(path)


if __name__ == "__main__":
    collect_empirical_outcomes()
