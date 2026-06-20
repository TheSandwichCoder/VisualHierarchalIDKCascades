from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from get_models_stats import load_model_from_checkpoint
from globals import device
from optimize_cascade import (
    GlobClassifierNode as OptimizedGlobClassifierNode,
    IdenClassifierNode as OptimizedIdenClassifierNode,
    SpecClassifierNode as OptimizedSpecClassifierNode,
    optimize_cascade,
)
from train_models import imagenetv2_dataset


class ClassifierNode:
    def __init__(self, model_path, conf):
        self.fp = model_path
        self.model, self.checkpoint = load_model_from_checkpoint(model_path)
        self.idk_nd = None
        self.confidence_threshold = 0.0 if conf is None else float(conf)

    def pred(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

        confidence = float(confidences.item())
        prediction = int(predictions.item())

        if self.idk_nd is None or confidence >= self.confidence_threshold:
            return self.evaluate(x, prediction)

        return self.idk_nd.pred(x)

    def evaluate(self, x, y):
        return y

    def set_idk_nd(self, idk_nd):
        self.idk_nd = idk_nd


class SpecClassifierNode(ClassifierNode):
    def __init__(self, model_path, conf):
        super().__init__(model_path, conf)
        self.class_ids = [int(class_id) for class_id in self.checkpoint["class_ids"]]

    def evaluate(self, x, y):
        return self.class_ids[int(y)]


class IdenClassifierNode(ClassifierNode):
    def __init__(self, model_path, conf, c_spec):
        super().__init__(model_path, conf)
        self.c_spec = c_spec
    
    def evaluate(self, x, y):
        c_i = int(y)
        return self.c_spec[c_i].pred(x)


class GlobClassifierNode(ClassifierNode):
    pass


def _runtime_node_from_optimized_node(optimized_node, specialized_nodes=None):
    if isinstance(optimized_node, OptimizedIdenClassifierNode):
        return IdenClassifierNode(
            optimized_node.checkpoint_path,
            optimized_node.threshold,
            specialized_nodes,
        )

    if isinstance(optimized_node, OptimizedSpecClassifierNode):
        return SpecClassifierNode(
            optimized_node.checkpoint_path,
            optimized_node.threshold,
        )

    if isinstance(optimized_node, OptimizedGlobClassifierNode):
        return GlobClassifierNode(
            optimized_node.checkpoint_path,
            optimized_node.threshold,
        )

    raise TypeError(f"Unsupported optimized node type: {type(optimized_node).__name__}")


def _link_idk_chain(nodes):
    for node, next_node in zip(nodes, nodes[1:]):
        node.set_idk_nd(next_node)
    return nodes[0]


def construct_idk_cascade(min_precision=0.75):
    _, optimized_seq, optimized_specialized_nodes = optimize_cascade(
        min_precision=min_precision,
        return_specialized=True,
    )

    specialized_nodes = [
        _runtime_node_from_optimized_node(optimized_node)
        for optimized_node in optimized_specialized_nodes
    ]

    runtime_seq = [
        _runtime_node_from_optimized_node(optimized_node, specialized_nodes)
        for optimized_node in optimized_seq
    ]

    return _link_idk_chain(runtime_seq)


def benchmark(
    precision_thresholds=(0.75, 0.80, 0.85, 0.90, 0.95),
    batch_size=1,
    max_samples=None,
):
    results = []

    for precision_threshold in precision_thresholds:
        cascade = construct_idk_cascade(min_precision=precision_threshold)
        loader = DataLoader(imagenetv2_dataset, batch_size=batch_size, shuffle=False)

        total = 0
        correct = 0
        runtime_seconds = 0.0

        for X, y in tqdm(loader, desc=f"precision>={precision_threshold:.2f}"):
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
            "min_precision": precision_threshold,
            "accuracy": correct / total if total else 0.0,
            "average_runtime": runtime_seconds / total if total else 0.0,
            "correct": correct,
            "total": total,
        }
        results.append(result)
        print(result)

    return results


def benchmark_resnet152(
    checkpoint_path="models/det/resnet152.pth",
    batch_size=1,
    max_samples=None,
):
    model, _ = load_model_from_checkpoint(checkpoint_path)
    loader = DataLoader(imagenetv2_dataset, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    runtime_seconds = 0.0

    with torch.no_grad():
        for X, y in tqdm(loader, desc="resnet152"):
            if max_samples is not None and total >= max_samples:
                break

            if max_samples is not None:
                remaining = max_samples - total
                if remaining < X.shape[0]:
                    X = X[:remaining]
                    y = y[:remaining]

            X = X.to(device)
            y = y.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            logits = model(X)
            if device.type == "cuda":
                torch.cuda.synchronize()

            runtime_seconds += perf_counter() - start
            predictions = logits.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.numel()

    result = {
        "model": "resnet152",
        "accuracy": correct / total if total else 0.0,
        "average_runtime": runtime_seconds / total if total else 0.0,
        "correct": correct,
        "total": total,
    }
    print(result)
    return result


if __name__ == "__main__":
    benchmark_resnet152()
