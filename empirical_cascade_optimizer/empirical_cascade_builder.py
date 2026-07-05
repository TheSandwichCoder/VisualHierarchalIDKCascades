from dataclasses import dataclass
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ._paths import ROOT as _ROOT
except ImportError:
    from _paths import ROOT as _ROOT

from empirical_cascade_optimizer.empirical_hierarchy_optimizer import optimize_empirical_hierarchy
from model_trainer.get_models_stats import load_model_from_checkpoint, resolve_repo_path
from globals import device
from model_trainer.train_models import imagenetv2_dataset


try:
    import joblib
except ModuleNotFoundError:
    joblib = None


@dataclass
class RuntimeResult:
    accepted: bool
    prediction: int
    group_id: int | None = None
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    features: torch.Tensor | None = None


@dataclass
class CascadeTrace:
    prediction: int
    ending_cascade_type: str
    initial_returned_idk: bool


# Temporary runtime-only threshold overrides. Remove this entry or set it back
# to 0.95640510 to restore the current stats-derived resnet18 threshold.
TEMP_CONFIDENCE_THRESHOLDS = {
    "resnet18": 0.65,
}


class RuntimeClassifier:
    def __init__(self, candidate_id, metadata, model_cache):
        self.candidate_id = candidate_id
        self.kind = metadata["kind"]
        self.threshold = float(
            TEMP_CONFIDENCE_THRESHOLDS.get(
                metadata.get("name"),
                metadata.get("threshold", 0.0),
            )
        )
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

    def _forward_with_features(self, x):
        self._load()

        if hasattr(self.model, "features") and hasattr(self.model, "avgpool"):
            features = self.model.features(x)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
            logits = self.model.classifier(features)
            return logits, features

        if all(hasattr(self.model, name) for name in ("conv1", "bn1", "relu", "maxpool", "avgpool", "fc")):
            features = self.model.conv1(x)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)
            features = self.model.layer1(features)
            features = self.model.layer2(features)
            features = self.model.layer3(features)
            features = self.model.layer4(features)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
            logits = self.model.fc(features)
            return logits, features

        logits = self.model(x)
        return logits, None

    def pred(self, x):
        with torch.no_grad():
            logits, features = self._forward_with_features(x)
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
                logits=logits.squeeze(0).detach().cpu(),
                probabilities=probabilities.squeeze(0).detach().cpu(),
                features=features.squeeze(0).detach().cpu() if features is not None else None,
            )
        return RuntimeResult(
            accepted=accepted,
            prediction=prediction,
            logits=logits.squeeze(0).detach().cpu(),
            probabilities=probabilities.squeeze(0).detach().cpu(),
            features=features.squeeze(0).detach().cpu() if features is not None else None,
        )


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


def _softmax(x, axis=1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _rf_summary_features(inputs, input_representation):
    if input_representation == "probabilities":
        probabilities = inputs
    elif input_representation == "logits":
        probabilities = _softmax(inputs)
    else:
        return inputs

    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12), axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    prediction = np.argmax(probabilities, axis=1)
    return np.column_stack([confidence, entropy, margin, prediction])


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
        "Use an RF skipper trained with logits or probabilities for this model."
    )


def _extract_model_input(model, sample, input_representation):
    with torch.no_grad():
        if input_representation == "features":
            values = _extract_features(model, sample)
        else:
            values = model(sample)
            if input_representation == "probabilities":
                values = torch.softmax(values, dim=1)

    return values.squeeze(0).detach().cpu()


def _same_checkpoint_path(left, right):
    try:
        return resolve_repo_path(left).resolve() == resolve_repo_path(right).resolve()
    except TypeError:
        return False


class RFSkipper:
    def __init__(
        self,
        skipper_path,
        input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    ):
        self.skipper_path = resolve_repo_path(skipper_path)
        self.payload = self._load_payload(self.skipper_path)
        self.model = self.payload["model"]
        self.input_representation = self.payload.get("input_representation", "probabilities")
        self.summary_features = bool(self.payload.get("summary_features", True))
        self.binary_detector_label = bool(self.payload.get("binary_detector_label", True))
        self.input_checkpoint_path = (
            self.payload.get("input_checkpoint_path")
            or input_checkpoint_path
        )
        self.input_model = None

    @staticmethod
    def _load_payload(path):
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                pass

        with path.open("rb") as handle:
            return pickle.load(handle)

    def _load_input_model(self):
        if self.input_model is None:
            self.input_model, _ = load_model_from_checkpoint(self.input_checkpoint_path)
            self.input_model.eval()

    def _inputs_for_sample(self, x, cached_result=None, cached_checkpoint_path=None):
        if (
            cached_result is not None
            and cached_checkpoint_path is not None
            and _same_checkpoint_path(cached_checkpoint_path, self.input_checkpoint_path)
            and self.input_representation in {"logits", "probabilities"}
        ):
            values = (
                cached_result.probabilities
                if self.input_representation == "probabilities"
                else cached_result.logits
            )
            if values is not None:
                inputs = values.unsqueeze(0).numpy()
                if self.summary_features:
                    inputs = _rf_summary_features(inputs, self.input_representation)
                return inputs, False

        self._load_input_model()
        values = _extract_model_input(
            self.input_model,
            x,
            input_representation=self.input_representation,
        )
        inputs = values.unsqueeze(0).numpy()
        if self.summary_features:
            inputs = _rf_summary_features(inputs, self.input_representation)
        return inputs, True

    def should_skip_to_detector(self, x, cached_result=None, cached_checkpoint_path=None):
        inputs, ran_input_model = self._inputs_for_sample(
            x,
            cached_result=cached_result,
            cached_checkpoint_path=cached_checkpoint_path,
        )
        prediction = int(self.model.predict(inputs)[0])
        should_skip = prediction == 1 if self.binary_detector_label else prediction == 0
        return should_skip, ran_input_model


class RuntimeSkipperMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class MLPSkipper:
    def __init__(
        self,
        skipper_path,
        input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    ):
        self.skipper_path = resolve_repo_path(skipper_path)
        self.payload = torch.load(self.skipper_path, map_location=device)
        self.input_representation = self.payload.get("input_representation", "features")
        self.binary_detector_label = bool(self.payload.get("binary_detector_label", True))
        self.input_checkpoint_path = (
            self.payload.get("input_checkpoint_path")
            or input_checkpoint_path
        )
        self.model = RuntimeSkipperMLP(
            self.payload["input_size"],
            self.payload["hidden_size"],
            self.payload["output_size"],
            self.payload.get("dropout", 0.0),
        ).to(device)
        self.model.load_state_dict(self.payload["model_state_dict"])
        self.model.eval()
        self.input_model = None

    def _load_input_model(self):
        if self.input_model is None:
            self.input_model, _ = load_model_from_checkpoint(self.input_checkpoint_path)
            self.input_model.eval()

    def _inputs_for_sample(self, x, cached_result=None, cached_checkpoint_path=None):
        if (
            cached_result is not None
            and cached_checkpoint_path is not None
            and _same_checkpoint_path(cached_checkpoint_path, self.input_checkpoint_path)
        ):
            values = None
            if self.input_representation == "probabilities":
                values = cached_result.probabilities
            elif self.input_representation == "logits":
                values = cached_result.logits
            elif self.input_representation == "features":
                values = cached_result.features

            if values is not None:
                return values.unsqueeze(0).to(device), False

        self._load_input_model()
        values = _extract_model_input(
            self.input_model,
            x,
            input_representation=self.input_representation,
        )
        return values.unsqueeze(0).to(device), True

    def should_skip_to_detector(self, x, cached_result=None, cached_checkpoint_path=None):
        inputs, ran_input_model = self._inputs_for_sample(
            x,
            cached_result=cached_result,
            cached_checkpoint_path=cached_checkpoint_path,
        )
        with torch.no_grad():
            prediction = int(self.model(inputs).argmax(dim=1).item())
        should_skip = prediction == 1 if self.binary_detector_label else prediction == 0
        return should_skip, ran_input_model


class EmpiricalRuntimeCascade:
    def __init__(
        self,
        optimizer,
        cascade,
        rf_skipper_path=None,
        mlp_skipper_path=None,
        rf_skipper_input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
        mlp_skipper_input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    ):
        if rf_skipper_path is not None and mlp_skipper_path is not None:
            raise ValueError("Use either rf_skipper_path or mlp_skipper_path, not both.")

        self.optimizer = optimizer
        self.cascade = cascade
        self.model_cache = {}
        self.detector = DetectorClassifier(optimizer.detector, self.model_cache)
        self.skipper = None
        self.skipper_kind = None
        if rf_skipper_path is not None:
            self.skipper = RFSkipper(
                rf_skipper_path,
                rf_skipper_input_checkpoint_path,
            )
            self.skipper_kind = "rf"
        elif mlp_skipper_path is not None:
            self.skipper = MLPSkipper(mlp_skipper_path, mlp_skipper_input_checkpoint_path)
            self.skipper_kind = "mlp"
        self.nodes = {}

        # Per-kind profiling. "runs" counts model invocations; "ends" counts
        # which kind produced the final prediction for each sample.
        self.stats = {
            "runs": {
                "identifier": 0,
                "specialized": 0,
                "global": 0,
                "detector": 0,
            },
            "ends": {
                "identifier": 0,
                "specialized": 0,
                "global": 0,
                "detector": 0,
            },
            "runtime_seconds": {
                "identifier": 0.0,
                "specialized": 0.0,
                "global": 0.0,
                "detector": 0.0,
            },
            "models": {},
            "skipper": {
                "enabled": self.skipper is not None,
                "kind": self.skipper_kind,
                "runs": 0,
                "input_model_runs": 0,
                "skips_to_detector": 0,
                "continues": 0,
                "runtime_seconds": 0.0,
            },
            "total": 0,
        }

        for candidate_id in optimizer.candidates.index:
            metadata = optimizer.describe_candidate(candidate_id)
            metadata["checkpoint_path"] = optimizer.candidates.loc[candidate_id, "checkpoint_path"]
            self.nodes[candidate_id] = RuntimeClassifier(
                candidate_id,
                metadata,
                self.model_cache,
            )
            self.stats["models"][candidate_id] = {
                "name": metadata["name"],
                "kind": metadata["kind"],
                "runs": 0,
                "ends": 0,
                "runtime_seconds": 0.0,
            }

        self.stats["models"][self.cascade.detector] = {
            "name": optimizer.detector.get("name"),
            "kind": "detector",
            "runs": 0,
            "ends": 0,
            "runtime_seconds": 0.0,
        }

    def pred(self, x):
        return self.pred_with_type(x).prediction

    def _run_node(self, node, x):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        result = node.pred(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        self.stats["runs"][node.kind] += 1
        elapsed = perf_counter() - start
        self.stats["runtime_seconds"][node.kind] += elapsed
        self.stats["models"][node.candidate_id]["runs"] += 1
        self.stats["models"][node.candidate_id]["runtime_seconds"] += elapsed
        return result

    def _run_detector(self, x):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        result = self.detector.pred(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        self.stats["runs"]["detector"] += 1
        elapsed = perf_counter() - start
        self.stats["runtime_seconds"]["detector"] += elapsed
        self.stats["models"][self.cascade.detector]["runs"] += 1
        self.stats["models"][self.cascade.detector]["runtime_seconds"] += elapsed
        return result

    def _run_skipper(self, x, cached_result=None, cached_checkpoint_path=None):
        if self.skipper is None:
            return False

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        should_skip, ran_input_model = self.skipper.should_skip_to_detector(
            x,
            cached_result=cached_result,
            cached_checkpoint_path=cached_checkpoint_path,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = perf_counter() - start
        self.stats["skipper"]["runs"] += 1
        self.stats["skipper"]["input_model_runs"] += int(ran_input_model)
        self.stats["skipper"]["runtime_seconds"] += elapsed
        if should_skip:
            self.stats["skipper"]["skips_to_detector"] += 1
        else:
            self.stats["skipper"]["continues"] += 1
        return should_skip

    def _trace(self, prediction, ending_cascade_type, initial_returned_idk, ending_model_id):
        self.stats["ends"][ending_cascade_type] += 1
        self.stats["models"][ending_model_id]["ends"] += 1
        self.stats["total"] += 1
        return CascadeTrace(
            prediction=prediction,
            ending_cascade_type=ending_cascade_type,
            initial_returned_idk=initial_returned_idk,
        )

    def profile(self):
        kind_profile = {}
        for kind, runs in self.stats["runs"].items():
            runtime_seconds = self.stats["runtime_seconds"][kind]
            kind_profile[kind] = {
                "runs": runs,
                "ends": self.stats["ends"][kind],
                "runtime_seconds": runtime_seconds,
                "average_runtime": runtime_seconds / runs if runs else 0.0,
            }

        model_profile = {}
        for candidate_id, stats in self.stats["models"].items():
            runs = stats["runs"]
            runtime_seconds = stats["runtime_seconds"]
            model_profile[candidate_id] = {
                **stats,
                "average_runtime": runtime_seconds / runs if runs else 0.0,
            }

        return {
            "total_predictions": self.stats["total"],
            "by_kind": kind_profile,
            "by_model": model_profile,
            "skipper": {
                **self.stats["skipper"],
                "average_runtime": (
                    self.stats["skipper"]["runtime_seconds"] / self.stats["skipper"]["runs"]
                    if self.stats["skipper"]["runs"]
                    else 0.0
                ),
            },
        }

    def pred_with_type(self, x):
        initial_returned_idk = False
        skipper_checked = False

        for candidate_id in self.cascade.initial:
            if candidate_id == self.cascade.detector:
                return self._trace(
                    prediction=self._run_detector(x).prediction,
                    ending_cascade_type="detector",
                    initial_returned_idk=initial_returned_idk,
                    ending_model_id=self.cascade.detector,
                )

            node = self.nodes[candidate_id]
            result = self._run_node(node, x)
            if not result.accepted:
                initial_returned_idk = True
                if not skipper_checked and self._run_skipper(
                    x,
                    cached_result=result,
                    cached_checkpoint_path=node.checkpoint_path,
                ):
                    return self._trace(
                        prediction=self._run_detector(x).prediction,
                        ending_cascade_type="detector",
                        initial_returned_idk=initial_returned_idk,
                        ending_model_id=self.cascade.detector,
                    )
                skipper_checked = True
                continue

            if node.kind == "identifier":
                chain = self.cascade.specialized.get(
                    (candidate_id, int(result.group_id)),
                    [self.cascade.detector],
                )
                return self._pred_specialized_chain(
                    chain,
                    x,
                    initial_returned_idk=initial_returned_idk,
                )

            return self._trace(
                prediction=result.prediction,
                ending_cascade_type=node.kind,
                initial_returned_idk=initial_returned_idk,
                ending_model_id=candidate_id,
            )

        return self._trace(
            prediction=self._run_detector(x).prediction,
            ending_cascade_type="detector",
            initial_returned_idk=initial_returned_idk,
            ending_model_id=self.cascade.detector,
        )

    def _pred_specialized_chain(self, chain, x, initial_returned_idk=False):
        for candidate_id in chain:
            if candidate_id == self.cascade.detector:
                return self._trace(
                    prediction=self._run_detector(x).prediction,
                    ending_cascade_type="detector",
                    initial_returned_idk=initial_returned_idk,
                    ending_model_id=self.cascade.detector,
                )

            nd = self.nodes[candidate_id]
            result = self._run_node(nd, x)
            if result.accepted:
                return self._trace(
                    prediction=result.prediction,
                    ending_cascade_type=nd.kind,
                    initial_returned_idk=initial_returned_idk,
                    ending_model_id=candidate_id,
                )

        return self._trace(
            prediction=self._run_detector(x).prediction,
            ending_cascade_type="detector",
            initial_returned_idk=initial_returned_idk,
            ending_model_id=self.cascade.detector,
        )


def construct_empirical_cascade(
    path="models/stats/empirical_outcomes.pkl",
    rf_skipper_path=None,
    mlp_skipper_path=None,
    rf_skipper_input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    mlp_skipper_input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
):
    optimizer, cascade = optimize_empirical_hierarchy(path)
    return EmpiricalRuntimeCascade(
        optimizer,
        cascade,
        rf_skipper_path=rf_skipper_path,
        mlp_skipper_path=mlp_skipper_path,
        rf_skipper_input_checkpoint_path=rf_skipper_input_checkpoint_path,
        mlp_skipper_input_checkpoint_path=mlp_skipper_input_checkpoint_path,
    )


def benchmark_empirical_cascade(
    path="models/stats/empirical_outcomes.pkl",
    batch_size=1,
    max_samples=None,
    rf_skipper_path=None,
    mlp_skipper_path=None,
    rf_skipper_input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
    mlp_skipper_input_checkpoint_path="models/intermediate/mobilenet_v3_small_identifier.pth",
):
    cascade = construct_empirical_cascade(
        path,
        rf_skipper_path=rf_skipper_path,
        mlp_skipper_path=mlp_skipper_path,
        rf_skipper_input_checkpoint_path=rf_skipper_input_checkpoint_path,
        mlp_skipper_input_checkpoint_path=mlp_skipper_input_checkpoint_path,
    )
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
        "profile": cascade.profile(),
    }

    print(result)
    return result


if __name__ == "__main__":
    benchmark_empirical_cascade()
