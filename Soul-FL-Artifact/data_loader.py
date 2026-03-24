
import os
import logging
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

CIFAR10_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

FEMNIST_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples: int = 10,
    seed: int = 42,
) -> List[List[int]]:

    rng = np.random.default_rng(seed)
    num_classes = int(targets.max()) + 1
    class_indices: Dict[int, List[int]] = {
        c: np.where(targets == c)[0].tolist() for c in range(num_classes)
    }
    for c in class_indices:
        rng.shuffle(class_indices[c])

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idxs = class_indices[c]
        proportions = rng.dirichlet([alpha] * num_clients)
        # Distribute indices proportionally
        splits = (proportions * len(idxs)).astype(int)
        # Fix rounding
        splits[-1] = len(idxs) - splits[:-1].sum()
        cumsum = np.concatenate([[0], np.cumsum(splits)])
        for i in range(num_clients):
            client_indices[i].extend(idxs[cumsum[i]: cumsum[i + 1]])

    # Ensure minimum samples
    for i in range(num_clients):
        if len(client_indices[i]) < min_samples:
            logger.warning(
                "Client %d has only %d samples; consider increasing dataset or reducing num_clients.",
                i, len(client_indices[i]),
            )

    return client_indices


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def load_cifar10_federated(
    data_dir: str,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Tuple[List[Subset], datasets.CIFAR10]:
    os.makedirs(data_dir, exist_ok=True)

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=CIFAR10_TRAIN_TRANSFORM
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=CIFAR10_TEST_TRANSFORM
    )

    targets = np.array(train_set.targets)
    partition = dirichlet_partition(targets, num_clients, alpha, seed=seed)

    client_datasets = [Subset(train_set, idxs) for idxs in partition]
    sizes = [len(d) for d in client_datasets]
    logger.info(
        "CIFAR-10 federated: %d clients | α=%.2f | sizes min=%d max=%d mean=%.0f",
        num_clients, alpha, min(sizes), max(sizes), np.mean(sizes),
    )
    return client_datasets, test_set


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def load_femnist_federated(
    data_dir: str,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Tuple[List[Subset], Dataset]:

    os.makedirs(data_dir, exist_ok=True)

    train_set = datasets.EMNIST(
        root=data_dir, split="byclass", train=True, download=True,
        transform=FEMNIST_TRANSFORM,
    )
    test_set = datasets.EMNIST(
        root=data_dir, split="byclass", train=False, download=True,
        transform=FEMNIST_TRANSFORM,
    )

    targets = np.array(train_set.targets)
    partition = dirichlet_partition(targets, num_clients, alpha, seed=seed)
    client_datasets = [Subset(train_set, idxs) for idxs in partition]
    sizes = [len(d) for d in client_datasets]
    logger.info(
        "FEMNIST federated: %d clients | α=%.2f | sizes min=%d max=%d mean=%.0f",
        num_clients, alpha, min(sizes), max(sizes), np.mean(sizes),
    )
    return client_datasets, test_set


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
class LabelFlipDataset(Dataset):


    def __init__(self, base: Dataset, source_class: int, target_class: int):
        self.base = base
        self.source = source_class
        self.target = target_class

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if y == self.source:
            y = self.target
        return x, y


class BackdoorDataset(Dataset):


    def __init__(
        self,
        base: Dataset,
        source_class: int,
        target_class: int,
        trigger_size: int = 4,
        trigger_value: float = 1.0,
        poison_fraction: float = 0.2,
        seed: int = 0,
    ):
        self.base = base
        self.source = source_class
        self.target = target_class
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        rng = np.random.default_rng(seed)

        # Select indices to poison
        source_idxs = [
            i for i in range(len(base))  # type: ignore[arg-type]
            if base[i][1] == source_class
        ]
        n_poison = int(len(source_idxs) * poison_fraction)
        self.poison_set = set(rng.choice(source_idxs, n_poison, replace=False).tolist())

    def _inject_trigger(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return x

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poison_set:
            x = self._inject_trigger(x)
            y = self.target
        return x, y


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def get_client_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def get_test_loader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def compute_class_histogram(dataset: Dataset, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in dataset:  # type: ignore[union-attr]
        counts[int(label)] += 1
    total = counts.sum()
    return counts / (total + 1e-12)


def federated_data_factory(cfg) -> Tuple[List[Dataset], Dataset]:

    if cfg.fl.dataset == "cifar10":
        return load_cifar10_federated(
            cfg.fl.data_dir, cfg.fl.num_clients, cfg.fl.dirichlet_alpha, cfg.fl.seed
        )
    elif cfg.fl.dataset == "femnist":
        return load_femnist_federated(
            cfg.fl.data_dir, cfg.fl.num_clients, cfg.fl.dirichlet_alpha, cfg.fl.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.fl.dataset}")
