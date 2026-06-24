from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ._paths import ROOT as _ROOT
except ImportError:
    from _paths import ROOT as _ROOT

from empirical_cascade_optimizer.empirical_hierarchy_optimizer import optimize_empirical_hierarchy
from model_trainer.get_models_stats import load_model_from_checkpoint
from globals import device
from model_trainer.train_models import imagenetv2_dataset


@dataclass
class RuntimeResult:
    accepted: bool
    prediction: int
    group_id: int | None = None


class RuntimeClassifier:
    def __init__(self, candidate_id, metadata, model_cache):
        self.candidate_id = candidate_id
        self.kind = metadata["kind"]
        self.threshold = float(metadata.get("threshold", 0.0))
        self.checkpoint_path = metadata["checkpoint_path"]
        self.model_cache = model_cache
        self.model = None
        self.checkpoint = None
        self.class_ids = None

    def _load(self):
        if self.model is not None:
            return
        if self.checkpoint_path in self.model_cache:
            self.model, self.checkpoint = self.model_cache[self.checkpoint_path]
        else:
            self.model, self.checkpoint = load_model_from_checkpoint(self.checkpoint_path)
            self.model_cache[self.checkpoint_path] = (self.model, self.checkpoint)
        self.class_ids = [
            int(class_id)
            for class_id in self.checkpoint.get("class_ids", [])
        ]

    def pred(self, x):
        self._load()
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

        confidence = float(confidences.item())
        prediction = int(predictions.item())
        accepted = confidence >= self.threshold

        if self.kind == "specialized":
            prediction = self.class_ids[prediction]
        if self.kind == "identifier":
            return RuntimeResult(
                accepted=accepted,
                prediction=prediction,
                group_id=prediction if accepted else None,
            )
        return RuntimeResult(accepted=accepted, prediction=prediction)


class DetectorClassifier:
    def __init__(self, metadata, model_cache):
        self.metadata = metadata
        self.checkpoint_path = metadata["checkpoint_path"]
        self.model_cache = model_cache
        self.model = None
        self.checkpoint = None

    def _load(self):
        if self.model is not None:
            return
        if self.checkpoint_path in self.model_cache:
            self.model, self.checkpoint = self.model_cache[self.checkpoint_path]
        else:
            self.model, self.checkpoint = load_model_from_checkpoint(self.checkpoint_path)
            self.model_cache[self.checkpoint_path] = (self.model, self.checkpoint)

    def pred(self, x):
        self._load()
        with torch.no_grad():
            logits = self.model(x)
            prediction = int(logits.argmax(dim=1).item())
        return RuntimeResult(accepted=True, prediction=prediction)


class EmpiricalRuntimeCascade:
    def __init__(self, optimizer, cascade):
        self.optimizer = optimizer
        self.cascade = cascade
        self.model_cache = {}
        self.detector = DetectorClassifier(optimizer.detector, self.model_cache)
        self.nodes = {}

        for candidate_id in optimizer.candidates.index:
            metadata = optimizer.describe_candidate(candidate_id)
            metadata["checkpoint_path"] = optimizer.candidates.loc[candidate_id, "checkpoint_path"]
            self.nodes[candidate_id] = RuntimeClassifier(
                candidate_id,
                metadata,
                self.model_cache,
            )

    def pred(self, x):
        for candidate_id in self.cascade.initial:
            if candidate_id == self.cascade.detector:
                return self.detector.pred(x).prediction

            node = self.nodes[candidate_id]
            result = node.pred(x)
            if not result.accepted:
                continue

            if node.kind == "identifier":
                chain = self.cascade.specialized.get(
                    (candidate_id, int(result.group_id)),
                    [self.cascade.detector],
                )
                return self._pred_specialized_chain(chain, x)

            return result.prediction

        return self.detector.pred(x).prediction

    def _pred_specialized_chain(self, chain, x):
        for candidate_id in chain:
            if candidate_id == self.cascade.detector:
                return self.detector.pred(x).prediction

            result = self.nodes[candidate_id].pred(x)
            if result.accepted:
                return result.prediction

        return self.detector.pred(x).prediction


def construct_empirical_cascade(path="models/stats/empirical_outcomes.pkl"):
    optimizer, cascade = optimize_empirical_hierarchy(path)
    return EmpiricalRuntimeCascade(optimizer, cascade)


def benchmark_empirical_cascade(
    path="models/stats/empirical_outcomes.pkl",
    batch_size=1,
    max_samples=None,
):
    cascade = construct_empirical_cascade(path)
    loader = DataLoader(imagenetv2_dataset, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    runtime_seconds = 0.0

    for X, y in tqdm(loader, desc="empirical cascade"):
        if max_samples is not None and total >= max_samples:
            break

        X = X.to(device)
        y = y.to(device)

        for sample_index in range(X.shape[0]):
            if max_samples is not None and total >= max_samples:
                break

            sample = X[sample_index:sample_index + 1]
            target = int(y[sample_index].item())

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            prediction = cascade.pred(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()

            runtime_seconds += perf_counter() - start
            correct += int(prediction == target)
            total += 1

    result = {
        "accuracy": correct / total if total else 0.0,
        "average_runtime": runtime_seconds / total if total else 0.0,
        "correct": correct,
        "total": total,
        "expected_cost": cascade.cascade.expected_cost,
    }
    print(result)
    return result


if __name__ == "__main__":
    benchmark_empirical_cascade()
