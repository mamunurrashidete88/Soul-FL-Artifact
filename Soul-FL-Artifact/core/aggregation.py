"""
core/aggregation.py — Aggregation rules: Soul-FL + baselines (Section V-A).

Baselines implemented
---------------------
  • FedAvg     — McMahan et al. 2017
  • Krum        — Blanchard et al. 2017
  • FLAME       — Nguyen et al. 2022
  • RoFL        — Lycklama et al. 2023
  • Aion        — Liu et al. 2025
  • DP-BREM     — Gu et al. 2025

"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias
Update = Tuple[int, np.ndarray]   # (client_id, delta_w)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def _clamp_norm(v: np.ndarray, clip: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > clip:
        return v * (clip / n)
    return v


# ---------------------------------------------------------------------------
# Soul-FL 
# ---------------------------------------------------------------------------
def soul_fl_aggregate(
    updates: List[Update],
    weights: Dict[int, float],
) -> np.ndarray:
    if not updates:
        raise ValueError("No updates to aggregate.")

    dim = updates[0][1].shape[0]
    agg = np.zeros(dim, dtype=np.float64)
    total_w = 0.0

    for cid, dw in updates:
        w = weights.get(cid, 0.0)
        if w <= 0:
            continue
        agg += w * dw.astype(np.float64)
        total_w += w

    if total_w < 1e-12:
        logger.warning("soul_fl_aggregate: all weights zero, returning zeros.")
        return np.zeros(dim)

    # Re-normalize in case floating-point drift
    return (agg / (total_w + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# FedAvg  
# ---------------------------------------------------------------------------
def fedavg_aggregate(
    updates: List[Update],
    sample_counts: Optional[Dict[int, int]] = None,
) -> np.ndarray:

    if not updates:
        raise ValueError("No updates.")

    if sample_counts is None:
        weights = {cid: 1.0 / len(updates) for cid, _ in updates}
    else:
        total = sum(sample_counts.get(cid, 1) for cid, _ in updates)
        weights = {cid: sample_counts.get(cid, 1) / (total + 1e-12) for cid, _ in updates}

    dim = updates[0][1].shape[0]
    agg = np.zeros(dim, dtype=np.float64)
    for cid, dw in updates:
        agg += weights[cid] * dw.astype(np.float64)
    return agg.astype(np.float32)


# ---------------------------------------------------------------------------
# Krum  
# ---------------------------------------------------------------------------
def krum_aggregate(
    updates: List[Update],
    num_malicious: int = 0,
    multi: bool = False,
) -> np.ndarray:

    n = len(updates)
    if n < 3:
        return fedavg_aggregate(updates)

    dws = np.stack([dw for _, dw in updates], axis=0)  # (n, d)
    f = min(num_malicious, n // 2 - 1)
    k = n - f - 2

    # Pairwise distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(dws[i] - dws[j]) ** 2
            dist[i, j] = dist[j, i] = d

    # Krum score = sum of k smallest distances
    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(dist[i])
        scores[i] = sorted_dists[1: k + 2].sum()  # exclude self (0)

    if multi:
        # Multi-Krum: select top-(n-f) by score
        selected = np.argsort(scores)[: n - f]
        return dws[selected].mean(axis=0).astype(np.float32)
    else:
        best = int(np.argmin(scores))
        return dws[best].astype(np.float32)


# ---------------------------------------------------------------------------
# FLAME  
# ---------------------------------------------------------------------------
def flame_aggregate(
    updates: List[Update],
    noise_sigma: float = 0.001,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    from sklearn.cluster import KMeans

    if not updates:
        raise ValueError("No updates.")
    if rng is None:
        rng = np.random.default_rng(0)

    dws = np.stack([dw for _, dw in updates], axis=0).astype(np.float64)
    n, d = dws.shape

    if n < 3:
        return fedavg_aggregate(updates)

    # Normalize for cosine clustering
    norms = np.linalg.norm(dws, axis=1, keepdims=True) + 1e-12
    dws_norm = dws / norms

    # K-means cosine clustering (k=2)
    km = KMeans(n_clusters=min(2, n), n_init=5, random_state=0)
    km.fit(dws_norm)
    labels = km.labels_

    # Select majority cluster
    unique, counts = np.unique(labels, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    selected_mask = labels == majority_label
    selected = dws[selected_mask]

    # Clip to median norm
    orig_norms = np.linalg.norm(dws, axis=1)
    clip_thresh = float(np.median(orig_norms))
    clipped = np.stack([_clamp_norm(v, clip_thresh) for v in selected], axis=0)

    # Average
    avg = clipped.mean(axis=0)

    # Add Gaussian DP noise
    noise = rng.normal(0, noise_sigma * clip_thresh, size=d)
    return (avg + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# RoFL  
# ---------------------------------------------------------------------------
def rofl_aggregate(
    updates: List[Update],
    norm_bound: float = 10.0,
) -> np.ndarray:

    if not updates:
        raise ValueError("No updates.")

    clipped = [(cid, _clamp_norm(dw, norm_bound)) for cid, dw in updates]
    return fedavg_aggregate(clipped)


# ---------------------------------------------------------------------------
# Aion  
# ---------------------------------------------------------------------------
def aion_aggregate(
    updates: List[Update],
    history: Optional[Dict[int, np.ndarray]] = None,
    momentum: float = 0.9,
    anomaly_threshold: float = 3.0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:

    if history is None:
        history = {}

    accepted = []
    new_history = dict(history)

    for cid, dw in updates:
        if cid in history:
            ema = history[cid]
            # Cosine similarity
            cos = float(
                np.dot(dw, ema) / (np.linalg.norm(dw) * np.linalg.norm(ema) + 1e-12)
            )
            dissimilarity = 1.0 - cos
            if dissimilarity < anomaly_threshold * 0.3:  # normalized threshold
                accepted.append((cid, dw))
        else:
            accepted.append((cid, dw))

        # Update EMA
        prev = history.get(cid, dw)
        new_history[cid] = momentum * prev + (1 - momentum) * dw

    if not accepted:
        logger.warning("Aion: all updates rejected, falling back to FedAvg.")
        accepted = updates

    return fedavg_aggregate(accepted), new_history


# ---------------------------------------------------------------------------
# DP-BREM  
# ---------------------------------------------------------------------------
def dp_brem_aggregate(
    updates: List[Update],
    history: Optional[Dict[int, float]] = None,
    clip_norm: float = 5.0,
    dp_sigma: float = 0.1,
    momentum_decay: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    if history is None:
        history = {}
    if rng is None:
        rng = np.random.default_rng(0)

    if not updates:
        raise ValueError("No updates.")

    dws = {cid: dw.astype(np.float64) for cid, dw in updates}

    # Global mean direction for trust scoring
    global_dir = np.stack(list(dws.values()), axis=0).mean(axis=0)
    global_norm = np.linalg.norm(global_dir) + 1e-12

    # Trust score update
    new_history = dict(history)
    weights = {}
    for cid, dw in dws.items():
        cos = float(np.dot(dw, global_dir) / (np.linalg.norm(dw) * global_norm + 1e-12))
        prev_score = history.get(cid, 0.5)
        new_score = momentum_decay * prev_score + (1 - momentum_decay) * max(0, cos)
        new_history[cid] = new_score
        weights[cid] = max(new_score, 0.05)  # floor weight to avoid starvation

    total_w = sum(weights.values()) + 1e-12

    # Weighted average with norm clipping
    dim = next(iter(dws.values())).shape[0]
    agg = np.zeros(dim, dtype=np.float64)
    for cid, dw in dws.items():
        clipped = _clamp_norm(dw, clip_norm)
        agg += (weights[cid] / total_w) * clipped

    # Gaussian DP noise
    noise = rng.normal(0, dp_sigma * clip_norm, size=dim)
    return (agg + noise).astype(np.float32), new_history


# ---------------------------------------------------------------------------
# Aggregation dispatcher
# ---------------------------------------------------------------------------
class AggregationEngine:


    def __init__(self, method: str = "soul_fl"):
        self.method = method
        self._aion_history: Dict[int, np.ndarray] = {}
        self._dpbrem_history: Dict[int, float] = {}
        self._rng = np.random.default_rng(42)

    def aggregate(
        self,
        updates: List[Update],
        weights: Optional[Dict[int, float]] = None,
        sample_counts: Optional[Dict[int, int]] = None,
        num_malicious: int = 0,
    ) -> np.ndarray:
        if not updates:
            raise ValueError("No updates provided to aggregate.")

        m = self.method

        if m == "soul_fl":
            assert weights is not None, "Soul-FL requires trust weights."
            return soul_fl_aggregate(updates, weights)

        elif m == "fedavg":
            return fedavg_aggregate(updates, sample_counts)

        elif m == "krum":
            return krum_aggregate(updates, num_malicious=num_malicious)

        elif m == "multi_krum":
            return krum_aggregate(updates, num_malicious=num_malicious, multi=True)

        elif m == "flame":
            return flame_aggregate(updates, rng=self._rng)

        elif m == "rofl":
            return rofl_aggregate(updates)

        elif m == "aion":
            result, self._aion_history = aion_aggregate(updates, self._aion_history)
            return result

        elif m == "dp_brem":
            result, self._dpbrem_history = dp_brem_aggregate(
                updates, self._dpbrem_history, rng=self._rng
            )
            return result

        else:
            raise ValueError(f"Unknown aggregation method: {m}")
