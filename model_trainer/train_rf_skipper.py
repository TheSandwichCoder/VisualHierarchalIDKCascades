from pathlib import Path
import pickle

import numpy as np
import torch

try:
    from ._paths import ROOT as _ROOT
except ImportError:
    from _paths import ROOT as _ROOT

from model_trainer.get_skipper_dataset import (
    CASCADE_TYPE_TO_LABEL,
    LABEL_TO_CASCADE_TYPE,
    create_rf_skipper_dataset,
    resolve_repo_path,
)


try:
    import joblib
except ModuleNotFoundError:
    joblib = None


def _load_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "scikit-learn is required to train the RF skipper. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install scikit-learn"
        ) from error

    return RandomForestClassifier, accuracy_score, classification_report, confusion_matrix, train_test_split


def _softmax(x, axis=1):
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def output_to_confidence(probabilities):
    return np.max(probabilities, axis=1)


def output_to_entropy(probabilities):
    eps = 1e-12
    return -np.sum(probabilities * np.log2(probabilities + eps), axis=1)


def output_to_margin(probabilities):
    sorted_probabilities = np.sort(probabilities, axis=1)
    return sorted_probabilities[:, -1] - sorted_probabilities[:, -2]


def output_to_pred(values):
    return np.argmax(values, axis=1)


def _load_payload(dataset_path):
    dataset_path = resolve_repo_path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} does not exist. Create it with "
            "create_rf_skipper_dataset() first, or call train_rf_skipper(..., "
            "create_dataset_if_missing=True)."
        )
    return torch.load(dataset_path, map_location="cpu")


def _payload_inputs(payload):
    inputs = payload.get("inputs", payload.get("features"))
    if inputs is None:
        raise KeyError("Skipper dataset payload must contain 'inputs' or legacy 'features'")
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    return np.asarray(inputs)


def _payload_targets(payload, binary_detector_label=True):
    targets = payload["targets"]
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    targets = np.asarray(targets)

    if binary_detector_label:
        detector_label = CASCADE_TYPE_TO_LABEL["detector"]
        return (targets == detector_label).astype(np.int64)

    return targets.astype(np.int64)


def inputs_to_rf_features(inputs, input_representation, summary_features=True):
    if not summary_features:
        return inputs

    if input_representation == "probabilities":
        probabilities = inputs
    elif input_representation == "logits":
        probabilities = _softmax(inputs)
    else:
        return inputs

    confidence = output_to_confidence(probabilities)
    entropy = output_to_entropy(probabilities)
    margin = output_to_margin(probabilities)
    prediction = output_to_pred(probabilities)

    return np.column_stack([confidence, entropy, margin, prediction])


def load_rf_training_data(
    dataset_path="data/processed_data/skipper_dataset_rf.pt",
    binary_detector_label=True,
    summary_features=True,
):
    payload = _load_payload(dataset_path)
    inputs = _payload_inputs(payload)
    targets = _payload_targets(payload, binary_detector_label=binary_detector_label)
    input_representation = payload.get("input_representation", "features")
    X = inputs_to_rf_features(
        inputs,
        input_representation=input_representation,
        summary_features=summary_features,
    )
    return X, targets, payload


def train_rf_skipper(
    dataset_path="data/processed_data/skipper_dataset_rf.pt",
    output_path="models/skipper/random_forest_skipper.pkl",
    create_dataset_if_missing=False,
    binary_detector_label=True,
    summary_features=True,
    test_size=0.2,
    random_state=42,
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=20,
):
    (
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        train_test_split,
    ) = _load_sklearn()

    if create_dataset_if_missing and not resolve_repo_path(dataset_path).exists():
        create_rf_skipper_dataset(dataset_path=dataset_path)

    X, y, payload = load_rf_training_data(
        dataset_path=dataset_path,
        binary_detector_label=binary_detector_label,
        summary_features=summary_features,
    )

    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    output_path = resolve_repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_payload = {
        "model": rf,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "binary_detector_label": binary_detector_label,
        "summary_features": summary_features,
        "input_representation": payload.get("input_representation", "features"),
        "input_checkpoint_path": payload.get("input_checkpoint_path"),
        "label_to_cascade_type": payload.get("label_to_cascade_type", LABEL_TO_CASCADE_TYPE),
        "cascade_type_to_label": payload.get("cascade_type_to_label", CASCADE_TYPE_TO_LABEL),
    }
    if joblib is not None:
        joblib.dump(saved_payload, output_path)
    else:
        with output_path.open("wb") as handle:
            pickle.dump(saved_payload, handle)

    print(f"RF accuracy: {accuracy:.4f}")
    print(report)
    print(f"saved {output_path}")

    return rf, {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    train_rf_skipper()
