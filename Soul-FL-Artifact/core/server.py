import copy
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import SoulFLConfig
from core.aggregation import AggregationEngine
from core.client import FederatedClient
from core.models import (
    build_model, get_flat_params, set_flat_params, model_size_mb
)
from data_loader import (
    get_test_loader, compute_class_histogram,
    LabelFlipDataset, BackdoorDataset,
)
from security.blockchain_sim import SoulFLContract, create_chain
from security.cvae import GradientFingerprintEngine
from security.zk import ZKEnrollmentEngine

logger = logging.getLogger(__name__)


class SoulFLServer:

    def __init__(
        self,
        cfg: SoulFLConfig,
        clients: List[FederatedClient],
        test_dataset,
        malicious_ids: set,
        method: str = "soul_fl",
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.clients = clients
        self.malicious_ids = malicious_ids
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global model
        self.global_model = build_model(cfg.fl.dataset, cfg.fl.num_classes).to(self.device)
        logger.info(
            "Global model: %s | %.2f MB | %d params",
            cfg.fl.dataset, model_size_mb(self.global_model),
            sum(p.numel() for p in self.global_model.parameters()),
        )

        # Test loader
        self.test_loader = get_test_loader(test_dataset, batch_size=256)

        # Aggregation engine
        self.agg_engine = AggregationEngine(method=method)

        # Soul-FL components (only initialized in soul_fl mode)
        self.zk_engine: Optional[ZKEnrollmentEngine] = None
        self.fingerprint_engine: Optional[GradientFingerprintEngine] = None
        self.chain: Optional[SoulFLContract] = None

        if method == "soul_fl":
            self.zk_engine = ZKEnrollmentEngine(cfg.zk, seed=cfg.fl.seed)
            self.fingerprint_engine = GradientFingerprintEngine(
                cfg.cvae, self.device, seed=cfg.fl.seed
            )
            self.chain = create_chain(cfg.trust)

        # Metrics history
        self.history: Dict[str, List] = {
            "round": [],
            "test_acc": [],
            "asr": [],
            "num_eligible": [],
            "num_accepted": [],
            "mean_trust": [],
            "cvae_threshold": [],
        }

        # Baseline state (for stateful methods)
        self._sample_counts = {c.client_id: c.num_samples for c in clients}

    # ------------------------------------------------------------------
    # Phase I: Enrollment
    # ------------------------------------------------------------------
    def enroll_all_clients(self) -> None:
        """
        Run ZK enrollment for all clients.  Called once before training.
        """
        if self.method != "soul_fl":
            return

        logger.info("=== Phase I: ZK Enrollment (%d clients) ===", len(self.clients))
        enrolled = 0
        rejected = 0

        for client in self.clients:
            # Extract labels for anchor
            labels = self._extract_labels(client)
            if labels is None or len(labels) == 0:
                logger.warning("Client %d: no labels available, skipping enrollment.", client.client_id)
                rejected += 1
                continue

            # Build anchor + proof
            anchor, true_hist = self.zk_engine.prepare_anchor(
                client.client_id,
                labels,
                num_classes=self.cfg.fl.num_classes,
            )
            proof = self.zk_engine.generate_proof(anchor, true_hist)
            record = self.zk_engine.verify_and_enroll(proof, anchor)

            if record is None:
                logger.warning("Client %d enrollment REJECTED.", client.client_id)
                rejected += 1
                continue

            # Mint SBT on-chain
            self.chain.mint_sbt(
                client.client_id,
                record.commitment,
                sender="0xAGGREGATOR",
            )
            enrolled += 1

        logger.info("Enrollment complete: %d enrolled, %d rejected.", enrolled, rejected)

        # Mine block for all mints
        self.chain._mine_block()

    def _extract_labels(self, client: FederatedClient) -> Optional[np.ndarray]:
        try:
            ds = client.dataset
            labels = []
            for _, y in ds:
                labels.append(int(y))
                if len(labels) >= 500:  # cap to save time
                    break
            return np.array(labels, dtype=np.int32)
        except Exception as e:
            logger.warning("Label extraction failed for client %d: %s", client.client_id, e)
            return None

    # ------------------------------------------------------------------
    # Main training loop 
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, List]:
        """
        Execute T rounds of Soul-FL training and return metrics history.
        """
        if self.method == "soul_fl":
            self.enroll_all_clients()

        logger.info(
            "=== Starting %s training | %d rounds | device=%s ===",
            self.method.upper(), self.cfg.fl.num_rounds, self.device,
        )

        for t in range(1, self.cfg.fl.num_rounds + 1):
            metrics = self._run_round(t)
            self._record_metrics(t, metrics)

            if t % 10 == 0:
                logger.info(
                    "Round %3d/%d | acc=%.2f%% | ASR=%.2f%% | eligible=%d | accepted=%d",
                    t, self.cfg.fl.num_rounds,
                    metrics["test_acc"] * 100,
                    metrics["asr"] * 100,
                    metrics.get("num_eligible", 0),
                    metrics.get("num_accepted", 0),
                )

        if self.method == "soul_fl":
            gas = self.chain.gas_report()
            logger.info("Gas report: %d total gas | %.6f ETH", gas["total_gas"], gas["total_eth"])

        return self.history

    # ------------------------------------------------------------------
    def _run_round(self, t: int) -> dict:
        """Execute a single training round."""
        # ---- Client selection ----------------------------------------
        if self.method == "soul_fl":
            # Algorithm 1 lines 2-5: filter by trust balance
            self.chain.advance_round(sender="0xAGGREGATOR")
            eligible = self.chain.get_eligible_clients()
        else:
            eligible = [c.client_id for c in self.clients]

        if len(eligible) == 0:
            logger.warning("Round %d: no eligible clients!", t)
            return {"test_acc": 0.0, "asr": 0.0}

        # Sample n clients
        n = min(self.cfg.fl.clients_per_round, len(eligible))
        selected_ids = random.sample(eligible, n)
        selected_clients = {c.client_id: c for c in self.clients if c.client_id in selected_ids}

        # ---- Broadcast global model ----------------------------------
        global_flat = get_flat_params(self.global_model).clone()

        # ---- Local training -----------------------------------------
        raw_updates: List[Tuple[int, np.ndarray]] = []
        for cid in selected_ids:
            client = selected_clients[cid]
            # Pass current_round to clients that need it (adversarial)
            try:
                delta_w, _ = client.train(self.global_model, current_round=t)
            except TypeError:
                delta_w, _ = client.train(self.global_model)
            raw_updates.append((cid, delta_w.numpy() if torch.is_tensor(delta_w) else delta_w))

        # ---- Phase II: C-VAE Fingerprinting -------------------------
        accepted_updates, vouchers = self._phase2_filter(raw_updates, t)

        # ---- Phase III: On-chain voucher redemption -----------------
        if self.method == "soul_fl" and vouchers:
            self.chain.batch_redeem(vouchers)

        if not accepted_updates:
            logger.warning("Round %d: all updates rejected by fingerprinting.", t)
            return self._eval_round(t, num_eligible=len(eligible), num_accepted=0)

        # ---- Phase IV: Trust-weighted aggregation -------------------
        weights = {}
        if self.method == "soul_fl":
            weights, _ = self.chain.compute_weights([cid for cid, _ in accepted_updates])

        agg_delta = self.agg_engine.aggregate(
            accepted_updates,
            weights=weights if self.method == "soul_fl" else None,
            sample_counts=self._sample_counts,
            num_malicious=int(len(self.malicious_ids) * n / len(self.clients)),
        )

        # ---- Update global model ------------------------------------
        eta = self.cfg.fl.local_lr
        new_flat = global_flat - eta * torch.tensor(agg_delta, dtype=torch.float32)
        set_flat_params(self.global_model, new_flat.to(self.device))

        # ---- C-VAE online adaptation --------------------------------
        if self.method == "soul_fl" and self.fingerprint_engine:
            accepted_dws = [dw for _, dw in accepted_updates]
            accepted_anchors = [
                self.zk_engine.get_anchor_vector(cid)
                for cid, _ in accepted_updates
                if self.zk_engine.get_anchor_vector(cid) is not None
            ]
            if accepted_anchors:
                self.fingerprint_engine.adapt(accepted_dws[:len(accepted_anchors)], accepted_anchors)

        # ---- Evaluation ---------------------------------------------
        trust_stats = self.chain.trust_stats() if self.method == "soul_fl" else {}
        return self._eval_round(
            t,
            num_eligible=len(eligible),
            num_accepted=len(accepted_updates),
            mean_trust=trust_stats.get("mean_balance", 0.0),
        )

    # ------------------------------------------------------------------
    def _phase2_filter(
        self,
        raw_updates: List[Tuple[int, np.ndarray]],
        t: int,
    ) -> Tuple[List[Tuple[int, np.ndarray]], list]:

        if self.method != "soul_fl":
            return raw_updates, []

        accepted = []
        vouchers = []

        for cid, dw in raw_updates:
            anchor = self.zk_engine.get_anchor_vector(cid)
            if anchor is None:
                # Client not enrolled → skip
                v = self.chain.issue_voucher(cid, accepted=False)
                vouchers.append(v)
                continue

            re, accept = self.fingerprint_engine.score(dw, anchor)

            if accept:
                accepted.append((cid, dw))

            v = self.chain.issue_voucher(cid, accepted=accept)
            vouchers.append(v)

        return accepted, vouchers

    # ------------------------------------------------------------------
    def _eval_round(
        self,
        t: int,
        num_eligible: int = 0,
        num_accepted: int = 0,
        mean_trust: float = 0.0,
    ) -> dict:
        acc = self._evaluate_accuracy()
        asr = self._evaluate_asr()
        tau = self.fingerprint_engine.get_threshold() if self.fingerprint_engine else 0.0
        return {
            "test_acc": acc,
            "asr": asr,
            "num_eligible": num_eligible,
            "num_accepted": num_accepted,
            "mean_trust": mean_trust,
            "cvae_threshold": tau,
        }

    @torch.no_grad()
    def _evaluate_accuracy(self) -> float:
        self.global_model.eval()
        correct = total = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.global_model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
        return correct / (total + 1e-12)

    @torch.no_grad()
    def _evaluate_asr(self) -> float:

        if self.cfg.attack.attack_type == "none":
            return 0.0

        src = self.cfg.attack.source_class
        tgt = self.cfg.attack.target_class

        self.global_model.eval()
        total = attack_success = 0

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            src_mask = (y == src)
            if src_mask.sum() == 0:
                continue

            x_src = x[src_mask]
            preds = self.global_model(x_src).argmax(dim=1)
            attack_success += (preds == tgt).sum().item()
            total += src_mask.sum().item()

        return attack_success / (total + 1e-12)

    def _record_metrics(self, t: int, metrics: dict) -> None:
        self.history["round"].append(t)
        self.history["test_acc"].append(metrics["test_acc"])
        self.history["asr"].append(metrics["asr"])
        self.history["num_eligible"].append(metrics.get("num_eligible", 0))
        self.history["num_accepted"].append(metrics.get("num_accepted", 0))
        self.history["mean_trust"].append(metrics.get("mean_trust", 0.0))
        self.history["cvae_threshold"].append(metrics.get("cvae_threshold", 0.0))

    # ------------------------------------------------------------------
    # Checkpoint 
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "global_model": self.global_model.state_dict(),
                "history": self.history,
                "method": self.method,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(ck["global_model"])
        self.history = ck["history"]
        logger.info("Checkpoint loaded: %s", path)
