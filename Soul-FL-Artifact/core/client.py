"""
core/client.py — Federated learning client (Section IV, Algorithm 1).

Each client performs local SGD training and returns a pseudo-gradient update.
Adversarial clients (sleeper, free-rider, manifold) inject poisoned updates
after the pivot round T_switch.
"""

import copy
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from core.models import get_flat_params, set_flat_params, compute_gradient_update
from config import FLConfig, AttackConfig

logger = logging.getLogger(__name__)


class FederatedClient:

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        cfg_fl: FLConfig,
        device: torch.device,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.cfg = cfg_fl
        self.device = device

        self._loader: Optional[DataLoader] = None

    # ------------------------------------------------------------------
    # Data loader 
    # ------------------------------------------------------------------
    @property
    def loader(self) -> DataLoader:
        if self._loader is None:
            self._loader = DataLoader(
                self.dataset,
                batch_size=self.cfg.local_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )
        return self._loader

    @property
    def num_samples(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def train(self, global_model: nn.Module) -> Tuple[torch.Tensor, float]:
        model = copy.deepcopy(global_model).to(self.device)
        global_flat = get_flat_params(model).clone()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.cfg.local_lr,
            momentum=self.cfg.momentum,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        model.train()

        total_loss = 0.0
        num_batches = 0

        for _ in range(self.cfg.local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        local_flat = get_flat_params(model)
        delta_w = compute_gradient_update(global_flat, local_flat)
        avg_loss = total_loss / max(num_batches, 1)
        return delta_w.cpu(), avg_loss


# ---------------------------------------------------------------------------
# Adversarial client variants
# ---------------------------------------------------------------------------
class SleeperSybilClient(FederatedClient):

    def __init__(
        self,
        client_id: int,
        honest_dataset: Dataset,
        poisoned_dataset: Dataset,
        cfg_fl: FLConfig,
        cfg_atk: AttackConfig,
        device: torch.device,
    ):
        super().__init__(client_id, honest_dataset, cfg_fl, device)
        self._poisoned_dataset = poisoned_dataset
        self._poisoned_loader: Optional[DataLoader] = None
        self.pivot_round = cfg_atk.pivot_round
        self._has_pivoted = False

    @property
    def poisoned_loader(self) -> DataLoader:
        if self._poisoned_loader is None:
            self._poisoned_loader = DataLoader(
                self._poisoned_dataset,
                batch_size=self.cfg.local_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
        return self._poisoned_loader

    def train(self, global_model: nn.Module, current_round: int = 0) -> Tuple[torch.Tensor, float]:
        if current_round >= self.pivot_round:
            if not self._has_pivoted:
                logger.debug("Client %d pivoting at round %d", self.client_id, current_round)
                self._has_pivoted = True
            # Switch to poisoned data
            original_loader = self._loader
            self._loader = self.poisoned_loader
            result = super().train(global_model)
            self._loader = original_loader
            return result
        return super().train(global_model)


class FreeRiderSybilClient(FederatedClient):

    def train(self, global_model: nn.Module, **kwargs) -> Tuple[torch.Tensor, float]:
        num_params = sum(p.numel() for p in global_model.parameters())
        # Tiny random noise — mimics a free-rider with no real data
        noise = torch.randn(num_params) * 1e-5
        return noise.cpu(), 0.0


class LazyHoardClient(FederatedClient):

    def __init__(self, *args, hoard_rounds: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_count = 0
        self.hoard_rounds = hoard_rounds
        self._cached_delta: Optional[torch.Tensor] = None

    def train(self, global_model: nn.Module, **kwargs) -> Tuple[torch.Tensor, float]:
        self._call_count += 1
        if self._call_count <= self.hoard_rounds:
            # Honest phase — cache update for later replay
            delta, loss = super().train(global_model)
            self._cached_delta = delta.clone()
            return delta, loss
        else:
            # Lazy phase — replay stale update or return scaled noise
            if self._cached_delta is not None:
                return self._cached_delta * 0.1, 0.0
            return torch.zeros(sum(p.numel() for p in global_model.parameters())), 0.0


class AdaptiveManifoldClient(FederatedClient):

    def __init__(
        self,
        client_id: int,
        honest_dataset: Dataset,
        poisoned_dataset: Dataset,
        cfg_fl: FLConfig,
        cfg_atk: AttackConfig,
        device: torch.device,
    ):
        super().__init__(client_id, honest_dataset, cfg_fl, device)
        self._poisoned_dataset = poisoned_dataset
        self._poisoned_loader: Optional[DataLoader] = None
        self.manifold_budget = cfg_atk.manifold_budget
        self.pivot_round = cfg_atk.pivot_round

    @property
    def poisoned_loader(self) -> DataLoader:
        if self._poisoned_loader is None:
            self._poisoned_loader = DataLoader(
                self._poisoned_dataset,
                batch_size=self.cfg.local_batch_size,
                shuffle=True,
            )
        return self._poisoned_loader

    def train(self, global_model: nn.Module, current_round: int = 0,
              fingerprint_fn=None) -> Tuple[torch.Tensor, float]:

        if current_round < self.pivot_round:
            return super().train(global_model)

        # Step 1: compute unconstrained poisoned update
        original = self._loader
        self._loader = self.poisoned_loader
        delta_poison, loss = super().train(global_model)
        self._loader = original

        # Step 2: project onto ‖Δw‖_2 ≤ manifold_budget
        norm = delta_poison.norm(p=2).item()
        if norm > self.manifold_budget:
            delta_poison = delta_poison * (self.manifold_budget / (norm + 1e-12))

        return delta_poison.cpu(), loss


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def build_clients(
    client_datasets,
    poisoned_datasets,
    malicious_ids: set,
    cfg,
    device: torch.device,
):

    clients = []
    attack_type = cfg.attack.attack_type

    for cid, ds in enumerate(client_datasets):
        if cid not in malicious_ids:
            clients.append(FederatedClient(cid, ds, cfg.fl, device))
        else:
            if attack_type == "sleeper_label_flip":
                pds = poisoned_datasets[cid] if poisoned_datasets else ds
                clients.append(
                    SleeperSybilClient(cid, ds, pds, cfg.fl, cfg.attack, device)
                )
            elif attack_type == "free_rider_sybil":
                clients.append(FreeRiderSybilClient(cid, ds, cfg.fl, device))
            elif attack_type == "lazy_hoard":
                clients.append(LazyHoardClient(cid, ds, cfg.fl, device))
            elif attack_type == "adaptive_manifold":
                pds = poisoned_datasets[cid] if poisoned_datasets else ds
                clients.append(
                    AdaptiveManifoldClient(cid, ds, pds, cfg.fl, cfg.attack, device)
                )
            else:
                clients.append(FederatedClient(cid, ds, cfg.fl, device))

    return clients
