import random
import torchvision
from tqdm import tqdm
from torch import nn
import torch
from pathlib import Path
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler, random_split
from globals import device
from read_json import load_groups
import torchvision.transforms as transforms
from ImageNetReader import *


ImageNetTransforms = preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

data = load_groups()
subtrain_dataset = ImageNetSubset(location="data", transform=ImageNetTransforms)
imagenetv2_dataset = ImageNetV2Dataset(location="data", transform=ImageNetTransforms)


def get_resnet_18(num_classes = None):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)

    if not num_classes is None:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    return model.to(device)

def get_resnet_34(num_classes = None):
    weights = torchvision.models.ResNet34_Weights.DEFAULT
    model = torchvision.models.resnet34(weights=weights)

    if not num_classes is None:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    return model.to(device)

def get_resnet_152():
    weights = torchvision.models.ResNet152_Weights.DEFAULT
    model = torchvision.models.resnet152(weights=weights)

    return model.to(device)


def get_partially_frozen_resnet_18(num_classes, trainable_layers=("fc",)):
    model = get_resnet_18(num_classes)
    trainable_layers = set(trainable_layers)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer_name in trainable_layers:
        layer = getattr(model, layer_name)
        for parameter in layer.parameters():
            parameter.requires_grad = True

    model.frozen_features = True
    model.trainable_layers = sorted(trainable_layers)
    return model

def get_partially_frozen_resnet_34(num_classes, trainable_layers=("fc",)):
    model = get_resnet_34(num_classes)
    trainable_layers = set(trainable_layers)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer_name in trainable_layers:
        if not hasattr(model, layer_name):
            raise ValueError(f"ResNet-34 has no layer named {layer_name}")
        layer = getattr(model, layer_name)
        for parameter in layer.parameters():
            parameter.requires_grad = True

    model.frozen_features = True
    model.trainable_layers = sorted(trainable_layers)
    return model

def get_mobilenet_v3_small(num_classes):
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    return model.to(device)

def get_mobilenet_v3_large(num_classes):
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    return model.to(device)


def _set_mobilenet_mlp_head(
    model,
    num_classes,
    model_type,
    hidden_size=512,
    dropout=0.2,
):
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, num_classes),
    )
    model.model_type = model_type
    model.hidden_size = hidden_size
    model.dropout = dropout
    return model


def get_mobilenet_v3_small_mlp_identifier(
    num_classes,
    hidden_size=512,
    dropout=0.2,
):
    weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    model = _set_mobilenet_mlp_head(
        model,
        num_classes,
        "mobilenet_v3_small_mlp_identifier",
        hidden_size=hidden_size,
        dropout=dropout,
    )
    return model.to(device)


def get_mobilenet_v3_large_mlp_identifier(
    num_classes,
    hidden_size=512,
    dropout=0.2,
):
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_large(weights=weights)
    model = _set_mobilenet_mlp_head(
        model,
        num_classes,
        "mobilenet_v3_large_mlp_identifier",
        hidden_size=hidden_size,
        dropout=dropout,
    )
    return model.to(device)


def get_frozen_mobilenet_v3_small(num_classes):
    model = get_mobilenet_v3_small(num_classes)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.frozen_features = True
    return model

def get_frozen_mobilenet_v3_large(num_classes):
    model = get_mobilenet_v3_large(num_classes)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.frozen_features = True
    return model


def get_frozen_mobilenet_v3_small_mlp_identifier(
    num_classes,
    hidden_size=512,
    dropout=0.2,
):
    model = get_mobilenet_v3_small_mlp_identifier(
        num_classes,
        hidden_size=hidden_size,
        dropout=dropout,
    )
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.frozen_features = True
    return model


def get_frozen_mobilenet_v3_large_mlp_identifier(
    num_classes,
    hidden_size=512,
    dropout=0.2,
):
    model = get_mobilenet_v3_large_mlp_identifier(
        num_classes,
        hidden_size=hidden_size,
        dropout=dropout,
    )
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.frozen_features = True
    return model



class ImageNetLogitMLPRouter(nn.Module):
    def __init__(self, num_categories, hidden_size=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1000),
            nn.Linear(1000, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_categories),
        )

    def forward(self, imagenet_logits):
        return self.net(imagenet_logits)


class ImageNetLogitLinearRouter(nn.Module):
    def __init__(self, num_categories):
        super().__init__()
        self.net = nn.Linear(1000, num_categories)

    def forward(self, imagenet_logits):
        return self.net(imagenet_logits)


class LogitRouterPipeline(nn.Module):
    def __init__(self, backbone, router, use_probabilities=True):
        super().__init__()
        self.backbone = backbone
        self.router = router
        self.use_probabilities = use_probabilities

    def forward(self, X):
        imagenet_features = self.backbone(X)
        if self.use_probabilities:
            imagenet_features = torch.softmax(imagenet_features, dim=1)
        return self.router(imagenet_features)

class CategoryGroupedDataset(Dataset):
    def __init__(self, source_dataset, groups):
        self.source_dataset = source_dataset
        self.groups = groups
        self.class_to_category = {}
        self.category_to_group = {}

        for category_id, group in enumerate(groups):
            self.category_to_group[category_id] = {
                "id": group["id"],
                "name": group["name"],
                "class_ids": [int(item["index"]) for item in group["classes"]],
            }
            for item in group["classes"]:
                self.class_to_category[int(item["index"])] = category_id

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, index):
        image, old_label = self.source_dataset[index]
        return image, self.class_to_category[old_label]

class RemappedSubset(Dataset):
    def __init__(self, subset, class_ids):
        self.subset = subset
        self.class_ids = [int(class_id) for class_id in class_ids]
        self.class_to_new_id = {
            old_id: new_id
            for new_id, old_id in enumerate(self.class_ids)
        }
        self.new_id_to_class = {
            new_id: old_id
            for old_id, new_id in self.class_to_new_id.items()
        }

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, old_label = self.subset[index]
        new_label = self.class_to_new_id[int(old_label)]
        return image, new_label


def test_helper(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def get_balanced_class_weights(labels, num_classes):
    label_counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes)
    total_count = label_counts.sum().item()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    nonzero = label_counts > 0
    weights[nonzero] = total_count / (num_classes * label_counts[nonzero].float())
    return weights


def get_balanced_sampler(labels):
    label_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    sample_weights = [
        1.0 / label_counts[label].item()
        for label in labels
    ]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_helper(dataloader, model, loss_fn, optimizer):

    model.train()
    if getattr(model, "frozen_features", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def get_subset_labels(dataset):
    if isinstance(dataset, CategoryGroupedDataset):
        source_labels = get_dataset_labels(dataset.source_dataset)
        return [
            dataset.class_to_category[int(label)]
            for label in source_labels
        ]
    if isinstance(dataset, Subset):
        parent_labels = get_subset_labels(dataset.dataset)
        return [parent_labels[index] for index in dataset.indices]
    return [int(dataset[index][1]) for index in range(len(dataset))]

def get_dataset_labels(source_dataset):
    if hasattr(source_dataset, "fnames"):
        return [int(path.parent.name) for path in source_dataset.fnames]
    if hasattr(source_dataset, "dataset") and hasattr(source_dataset.dataset, "samples"):
        return [int(label) for _, label in source_dataset.dataset.samples]
    return [int(source_dataset[index][1]) for index in range(len(source_dataset))]


def create_class_subset(class_ids, source_dataset=subtrain_dataset, remap_labels=False):
    """Return a dataset subset containing only the requested ImageNet class ids.

    Args:
        class_ids: Iterable of ImageNet class indices, such as [2, 3, 4].
        source_dataset: Dataset returning (sample, integer_label). Defaults to the
            ImageNetV2 dataset created above.

    By default the returned Subset keeps the original ImageNet labels. Pass
    remap_labels=True when training an n-class model head with CrossEntropyLoss.
    """
    ordered_class_ids = [int(class_id) for class_id in class_ids]
    class_ids = set(ordered_class_ids)
    if not ordered_class_ids:
        raise ValueError("class_ids must contain at least one ImageNet class id")

    invalid_ids = sorted(class_id for class_id in class_ids if class_id < 0 or class_id > 999)
    if invalid_ids:
        raise ValueError(f"ImageNet class ids must be in [0, 999], got {invalid_ids}")

    labels = get_dataset_labels(source_dataset)

    subset_indices = [index for index, label in enumerate(labels) if label in class_ids]

    if not subset_indices:
        raise ValueError(f"No samples found for ImageNet class ids {sorted(class_ids)}")

    subset = Subset(source_dataset, subset_indices)
    if remap_labels:
        return RemappedSubset(subset, ordered_class_ids)

    return subset

def create_category_dataset(source_dataset=subtrain_dataset, groups=None):
    """Return a dataset whose labels are visual group/category ids.

    For example, all classes in the "Fish, sharks, and rays" group become one
    label, all classes in the next group become another label, and so on.
    """
    if groups is None:
        groups = data["groups"]
    return CategoryGroupedDataset(source_dataset, groups)


def create_balanced_subset(dataset, max_per_class, seed=0):
    labels = get_subset_labels(dataset)
    indices_by_label = {}

    for index, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(index)

    rng = random.Random(seed)
    indices = []
    for label in sorted(indices_by_label):
        label_indices = indices_by_label[label]
        rng.shuffle(label_indices)
        indices.extend(label_indices[:max_per_class])

    rng.shuffle(indices)

    return Subset(dataset, indices)


def load_specialized_model(checkpoint, get_model=get_frozen_mobilenet_v3_small):
    model = get_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def find_precision_cutoff(dataloader, model, target_precision=0.95):
    """Find a confidence cutoff that reaches target precision, or maximizes it.

    Precision is measured only on predictions whose max softmax confidence is at
    least the cutoff. Returned recall is the accepted fraction of the dataset.
    """
    model.eval()
    confidences = []
    correct = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            probabilities = torch.softmax(model(X), dim=1)
            batch_confidences, predictions = probabilities.max(dim=1)

            confidences.extend(batch_confidences.cpu().tolist())
            correct.extend((predictions == y).cpu().tolist())

    if not confidences:
        raise ValueError("Cannot choose a precision cutoff from an empty dataloader")

    candidates = []
    for threshold in sorted(set(confidences), reverse=True):
        accepted = [
            is_correct
            for confidence, is_correct in zip(confidences, correct)
            if confidence >= threshold
        ]
        if not accepted:
            continue

        accepted_count = len(accepted)
        true_positive_count = sum(accepted)
        precision = true_positive_count / accepted_count
        recall = accepted_count / len(confidences)
        candidates.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "accepted": accepted_count,
            "total": len(confidences),
            "target_met": precision >= target_precision,
        })

    target_candidates = [
        candidate
        for candidate in candidates
        if candidate["target_met"]
    ]
    if target_candidates:
        return max(
            target_candidates,
            key=lambda candidate: (
                candidate["recall"],
                -candidate["threshold"],
            ),
        )

    return max(
        candidates,
        key=lambda candidate: (
            candidate["precision"],
            candidate["recall"],
            -candidate["threshold"],
        ),
    )


def predict_with_cutoff(model, X, threshold):
    probabilities = torch.softmax(model(X), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    predictions = predictions.clone()
    predictions[confidences < threshold] = -1
    return predictions, confidences


def save_model(model, cutoff_info, path, class_ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": getattr(model, "model_type", None),
        "hidden_size": getattr(model, "hidden_size", None),
        "dropout": getattr(model, "dropout", None),
        "num_classes": len(class_ids),
        "class_ids": list(class_ids),
        "class_to_new_id": {
            class_id: new_id
            for new_id, class_id in enumerate(class_ids)
        },
        "new_id_to_class": {
            new_id: class_id
            for new_id, class_id in enumerate(class_ids)
        },
        "confidence_threshold": cutoff_info["threshold"],
        "precision": cutoff_info["precision"],
        "recall": cutoff_info["recall"],
        "accepted": cutoff_info["accepted"],
        "total": cutoff_info["total"],
        "target_met": cutoff_info["target_met"],
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def save_pretrained_model_checkpoint(model, path, model_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": model_name,
        "num_classes": 1000,
        "class_ids": list(range(1000)),
        "confidence_threshold": 0.0,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved {model_name} checkpoint to {path}")


def save_logit_router_checkpoint(
    backbone,
    router,
    path,
    accuracy,
    settings,
    cutoff_info=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": "resnet34_logit_router",
        "backbone_name": "resnet34",
        "router_name": type(router).__name__,
        "num_categories": len(data["groups"]),
        "groups": [
            {
                "id": group["id"],
                "name": group["name"],
                "class_ids": [int(item["index"]) for item in group["classes"]],
            }
            for group in data["groups"]
        ],
        "settings": settings,
        "accuracy": accuracy,
        "backbone_state_dict": backbone.state_dict(),
        "router_state_dict": router.state_dict(),
    }

    if cutoff_info is not None:
        checkpoint.update({
            "confidence_threshold": cutoff_info["threshold"],
            "precision": cutoff_info["precision"],
            "recall": cutoff_info["recall"],
            "accepted": cutoff_info["accepted"],
            "total": cutoff_info["total"],
            "target_met": cutoff_info["target_met"],
        })

    torch.save(checkpoint, path)
    print(f"Saved logit router checkpoint to {path}")


def print_checkpoint_values(checkpoint):
    for key, value in checkpoint.items():
        if key == "state_dict":
            parameter_count = sum(tensor.numel() for tensor in value.values())
            print(f"  state_dict: {len(value)} tensors, {parameter_count} parameters")
        else:
            print(f"  {key}: {value}")


def test_pretrained_imagenet_group_baseline(batch_size=32):
    class_to_category = {}
    for category_id, group in enumerate(data["groups"]):
        for item in group["classes"]:
            class_to_category[int(item["index"])] = category_id

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    test_set = CategoryGroupedDataset(imagenetv2_dataset, data["groups"])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            imagenet_predictions = model(X).argmax(1).cpu().tolist()
            group_predictions = torch.tensor(
                [class_to_category[int(prediction)] for prediction in imagenet_predictions],
                device=device,
            )
            correct += (group_predictions == y).sum().item()
            total += y.numel()

    accuracy = correct / total if total else 0.0
    print(f"Pretrained ImageNet -> visual group baseline accuracy: {100 * accuracy:.1f}%")
    return accuracy


def build_class_to_category_tensor(groups=None):
    if groups is None:
        groups = data["groups"]

    class_to_category = torch.empty(1000, dtype=torch.long)
    for category_id, group in enumerate(groups):
        for item in group["classes"]:
            class_to_category[int(item["index"])] = category_id
    return class_to_category


def test_group_probability_baseline(batch_size=64):
    num_categories = len(data["groups"])
    class_to_category = build_class_to_category_tensor().to(device)
    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    ).to(device)
    backbone.eval()

    test_set = create_category_dataset(imagenetv2_dataset)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            probabilities = torch.softmax(backbone(X), dim=1)
            group_scores = torch.zeros(
                probabilities.shape[0],
                num_categories,
                device=device,
            )
            group_scores.scatter_add_(
                1,
                class_to_category.unsqueeze(0).expand(probabilities.shape[0], -1),
                probabilities,
            )
            pred = group_scores.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()

    accuracy = correct / total if total else 0.0
    print(f"Pretrained ImageNet probability-sum group accuracy: {100 * accuracy:.1f}%")
    return accuracy


def train_logit_mlp_router(
    batch_size=64,
    epochs=8,
    hidden_size=256,
    dropout=0.5,
    lr=3e-4,
    weight_decay=1e-2,
    max_train_per_category=100,
    max_test_per_category=None,
    linear=False,
    use_probabilities=True,
    finetune_backbone=False,
    save_path="models/logit_router/resnet18_logit_router.pth",
):
    num_categories = len(data["groups"])
    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    ).to(device)
    if finetune_backbone:
        backbone.train()
        for parameter in backbone.parameters():
            parameter.requires_grad = True
    else:
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False

    if linear:
        router = ImageNetLogitLinearRouter(num_categories).to(device)
    else:
        router = ImageNetLogitMLPRouter(
            num_categories,
            hidden_size=hidden_size,
            dropout=dropout,
        ).to(device)

    if finetune_backbone:
        optimizer = torch.optim.AdamW([
            {"params": router.parameters(), "lr": lr},
            {"params": backbone.fc.parameters(), "lr": lr * 0.3},
            {"params": backbone.layer4.parameters(), "lr": lr * 0.1},
            {"params": backbone.layer3.parameters(), "lr": lr * 0.03},
            {
                "params": list(backbone.conv1.parameters())
                + list(backbone.bn1.parameters())
                + list(backbone.layer1.parameters())
                + list(backbone.layer2.parameters()),
                "lr": lr * 0.01,
            },
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_set = create_category_dataset(subtrain_dataset)
    test_set = create_category_dataset(imagenetv2_dataset)
    if max_train_per_category is not None:
        train_set = create_balanced_subset(train_set, max_train_per_category)
    if max_test_per_category is not None:
        test_set = create_balanced_subset(test_set, max_test_per_category)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        router.train()
        if finetune_backbone:
            backbone.train()
        else:
            backbone.eval()

        total = 0
        correct = 0
        total_loss = 0

        print(f"Logit MLP epoch {epoch + 1}/{epochs}")
        for X, y in tqdm(train_loader):
            X, y = X.to(device), y.to(device)

            if finetune_backbone:
                imagenet_features = backbone(X)
                if use_probabilities:
                    imagenet_features = torch.softmax(imagenet_features, dim=1)
            else:
                with torch.no_grad():
                    imagenet_features = backbone(X)
                    if use_probabilities:
                        imagenet_features = torch.softmax(imagenet_features, dim=1)

            pred = router(imagenet_features)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.numel()

        print(
            f"  train accuracy: {100 * correct / total:.1f}% "
            f"loss: {total_loss / len(train_loader):.4f}"
        )

    router.eval()
    backbone.eval()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            imagenet_features = backbone(X)
            if use_probabilities:
                imagenet_features = torch.softmax(imagenet_features, dim=1)
            pred = router(imagenet_features)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.numel()

    accuracy = correct / total if total else 0
    print(
        f"Logit MLP router test accuracy: {100 * accuracy:.1f}% "
        f"loss: {total_loss / len(test_loader):.4f} "
        f"samples: {total}"
    )

    pipeline = LogitRouterPipeline(
        backbone,
        router,
        use_probabilities=use_probabilities,
    ).to(device)
    pipeline.eval()
    cutoff_info = find_precision_cutoff(test_loader, pipeline)
    print(f"precision: {cutoff_info['precision']} recall: {cutoff_info['recall']}")

    save_logit_router_checkpoint(
        backbone=backbone,
        router=router,
        path=save_path,
        accuracy=accuracy,
        cutoff_info=cutoff_info,
        settings={
            "batch_size": batch_size,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_train_per_category": max_train_per_category,
            "max_test_per_category": max_test_per_category,
            "linear": linear,
            "use_probabilities": use_probabilities,
            "finetune_backbone": finetune_backbone,
        },
    )

    return backbone, router, accuracy


def train_specialized_models(get_model=get_frozen_mobilenet_v3_small, batch_size=32, epochs=5):
    loss_fn = nn.CrossEntropyLoss()

    for group in data["groups"]:
        print(f"TRAINING GROUP {group['id']}")
        class_ids = [item["index"] for item in group["classes"]]

        model = get_model(group["count"])
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        train_set = create_class_subset(class_ids, remap_labels=True)
        test_set = create_class_subset(class_ids, source_dataset=imagenetv2_dataset, remap_labels=True)


        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_helper(train_loader, model, loss_fn, optimizer)
        test_helper(test_loader, model, loss_fn)
        cutoff_info = find_precision_cutoff(test_loader, model)
        print(f"precision: {cutoff_info['precision']} recall: {cutoff_info['recall']}")

        m_name = group["name"] + "_m2.pth"
        fp = "models/specialized/" + m_name

        save_model(
            model,
            cutoff_info,
            fp,
            class_ids,
        )

def train_intermediate_models(get_model=get_partially_frozen_resnet_18, batch_size=32, epochs=3, name = "intermediate1"):
    groups = data["groups"]
    num_classes = len(groups)

    model = get_model(num_classes)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )

    train_set = create_category_dataset(subtrain_dataset)
    test_set = create_category_dataset(imagenetv2_dataset)
    # train_labels = get_subset_labels(train_set)
    # class_weights = get_balanced_class_weights(train_labels, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_helper(train_loader, model, loss_fn, optimizer)
        test_helper(test_loader, model, loss_fn)
        cutoff_info = find_precision_cutoff(test_loader, model)
        print(f"precision: {cutoff_info['precision']} recall: {cutoff_info['recall']}")

        m_name = f"{name}.pth"
        fp = f"models/intermediate/{m_name}"

        save_model(
            model,
            cutoff_info,
            fp,
            list(range(num_classes)),
        )


def train_mobilenet_identifier_models(batch_size=32, epochs=3):
    return [
        train_intermediate_models(
            get_model=get_frozen_mobilenet_v3_small,
            batch_size=batch_size,
            epochs=epochs,
            name="mobilenet_v3_small_identifier",
        ),
        train_intermediate_models(
            get_model=get_frozen_mobilenet_v3_large,
            batch_size=batch_size,
            epochs=epochs,
            name="mobilenet_v3_large_identifier",
        ),
    ]


def train_mobilenet_mlp_identifier_models(
    batch_size=32,
    epochs=3,
    hidden_size=512,
    dropout=0.2,
):
    return [
        train_intermediate_models(
            get_model=lambda num_classes: get_frozen_mobilenet_v3_small_mlp_identifier(
                num_classes,
                hidden_size=hidden_size,
                dropout=dropout,
            ),
            batch_size=batch_size,
            epochs=epochs,
            name="mobilenet_v3_small_mlp_identifier",
        ),
        train_intermediate_models(
            get_model=lambda num_classes: get_frozen_mobilenet_v3_large_mlp_identifier(
                num_classes,
                hidden_size=hidden_size,
                dropout=dropout,
            ),
            batch_size=batch_size,
            epochs=epochs,
            name="mobilenet_v3_large_mlp_identifier",
        ),
    ]

def get_global_models():
    m1 = get_resnet_18()
    m2 = get_resnet_34()

    save_pretrained_model_checkpoint(m1, "models/global/resnet18.pth", "resnet18")
    save_pretrained_model_checkpoint(m2, "models/global/resnet34.pth", "resnet34")
    return m1, m2

def get_det_model():
    m1 = get_resnet_152()
    save_pretrained_model_checkpoint(m1, "models/det/resnet152.pth", "resnet152")
    return m1




if __name__ == "__main__":
    # train_specialized_models(get_model=get_frozen_mobilenet_v3_large)
    # get_global_models()
    train_mobilenet_mlp_identifier_models()
    # get_det_model()
    # train_logit_mlp_router(batch_size=32, epochs = 3)
    # train_intermediate_models(name = "intermediate1")
    # train_intermediate_models(get_model = get_partially_frozen_resnet_34, name = "intermediate2")
    # test_first_5_groups_on_imagenetv2()
