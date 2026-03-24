
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

from config import ZKConfig

logger = logging.getLogger(__name__)

# Deterministic secret used to simulate circuit-based SNARK soundness.
# In production, this would be the trusted setup verification key.
_CIRCUIT_KEY = os.environ.get("SOUL_FL_ZK_KEY", "soul-fl-zk-circuit-v1").encode()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StatisticalAnchor:

    client_id: int
    bins: np.ndarray           # K-bin noisy histogram (unnormalized counts)
    num_bins: int
    ldp_epsilon: float
    declared_size: int         # claimed |D_i| (subject to ZK verification)

    def to_bytes(self) -> bytes:
        """Canonical serialization for commitment hashing."""
        payload = {
            "client_id": self.client_id,
            "bins": self.bins.round(6).tolist(),
            "num_bins": self.num_bins,
            "ldp_epsilon": self.ldp_epsilon,
            "declared_size": self.declared_size,
        }
        return json.dumps(payload, sort_keys=True).encode()

    def as_conditioning_vector(self, target_dim: int = 32) -> np.ndarray:

        from scipy.ndimage import zoom
        bins = self.bins.astype(np.float32)
        # Normalize
        total = bins.sum() + 1e-12
        bins = bins / total
        # Resize to target_dim
        if len(bins) != target_dim:
            factor = target_dim / len(bins)
            bins = zoom(bins, factor, order=1)
            bins = bins / (bins.sum() + 1e-12)  # renormalize
        return bins.astype(np.float32)


@dataclass
class ZKProof:

    client_id: int
    commitment: str            # C_i = Hash(pk_i || S_i)
    proof_bytes: bytes         # simulated circuit proof
    public_inputs: dict        # τ_size, ε, num_bins  (public witness)
    is_valid: bool = True


@dataclass
class EnrollmentRecord:
    """Stored on-chain after successful enrollment."""
    client_id: int
    public_key: str
    commitment: str            # C_i
    anchor_vector: np.ndarray  # cached S_i for C-VAE conditioning
    declared_size: int
    ldp_epsilon: float
    enrolled: bool = True


# ---------------------------------------------------------------------------
# LDP (Laplace mechanism)
# ---------------------------------------------------------------------------
def _laplace_noise(scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Laplace noise with scale = Δf / ε."""
    return rng.laplace(loc=0.0, scale=scale, size=size)


def apply_ldp_to_histogram(
    histogram: np.ndarray,
    ldp_epsilon: float,
    sensitivity: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()
    scale = sensitivity / (ldp_epsilon + 1e-12)
    noise = _laplace_noise(scale, len(histogram), rng)
    noisy = histogram.astype(np.float64) + noise
    # Clip to non-negative (histograms cannot be negative)
    noisy = np.clip(noisy, a_min=0.0, a_max=None)
    return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Histogram construction
# ---------------------------------------------------------------------------
def build_histogram_from_labels(
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:

    hist = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    return hist


def build_histogram_from_pixels(
    images: np.ndarray,
    num_bins: int = 32,
) -> np.ndarray:

    flat = images.reshape(len(images), -1).mean(axis=-1)
    hist, _ = np.histogram(flat, bins=num_bins, range=(0.0, 1.0))
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# ZK Proof construction & verification  
# ---------------------------------------------------------------------------
def _make_public_key(client_id: int) -> str:
    """Deterministic public key for simulation."""
    return hashlib.sha256(f"pk_{client_id}".encode()).hexdigest()


def _simulate_proof(
    client_id: int,
    true_histogram: np.ndarray,
    noisy_anchor: np.ndarray,
    dataset_size: int,
    ldp_epsilon: float,
    min_size: int,
    circuit_key: bytes = _CIRCUIT_KEY,
) -> Tuple[bytes, bool]:

    # Validity checks (these would be enforced by the circuit constraints)
    is_valid = (
        dataset_size >= min_size
        and ldp_epsilon > 0
        and len(true_histogram) == len(noisy_anchor)
    )
    if not is_valid:
        return b"", False

    # Private witness:  (D_i histogram, randomness used in LDP)
    witness_payload = json.dumps({
        "cid":  client_id,
        "hist": true_histogram.round(6).tolist(),
        "size": dataset_size,
        "eps":  ldp_epsilon,
    }, sort_keys=True).encode()

    proof = hmac.new(circuit_key, witness_payload, "sha256").digest()
    return proof, True


def _verify_proof(
    proof: bytes,
    client_id: int,
    noisy_anchor: StatisticalAnchor,
    min_size: int,
    circuit_key: bytes = _CIRCUIT_KEY,
) -> bool:

    if not proof:
        return False
    # Public inputs only → we verify the structural constraints
    public_ok = (
        noisy_anchor.declared_size >= min_size
        and noisy_anchor.ldp_epsilon > 0
    )
    return public_ok


# ---------------------------------------------------------------------------
# On-chain commitment
# ---------------------------------------------------------------------------
def compute_commitment(public_key: str, anchor: StatisticalAnchor) -> str:
    """C_i = Hash(pk_i || S_i)  (Section IV Phase I)."""
    data = public_key.encode() + b"||" + anchor.to_bytes()
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# ZK Enrollment Engine
# ---------------------------------------------------------------------------
class ZKEnrollmentEngine:


    def __init__(self, cfg: ZKConfig, seed: int = 0):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._registry: dict[int, EnrollmentRecord] = {}

    # ------------------------------------------------------------------
    # Client-side
    # ------------------------------------------------------------------
    def prepare_anchor(
        self,
        client_id: int,
        labels: np.ndarray,
        num_classes: int,
        images: Optional[np.ndarray] = None,
    ) -> Tuple[StatisticalAnchor, np.ndarray]:

        # True histogram (private witness)
        true_hist = build_histogram_from_labels(labels, num_classes).astype(np.float32)
        # Normalize
        total = true_hist.sum() + 1e-12
        true_hist_norm = true_hist / total

        # Apply LDP
        noisy = apply_ldp_to_histogram(
            true_hist_norm,
            ldp_epsilon=self.cfg.ldp_epsilon,
            sensitivity=1.0,
            rng=self._rng,
        )

        # Optionally supplement with pixel histogram (for richer anchor)
        if images is not None:
            pixel_hist = build_histogram_from_pixels(images, self.cfg.num_bins)
            pixel_hist = pixel_hist / (pixel_hist.sum() + 1e-12)
            pixel_noisy = apply_ldp_to_histogram(pixel_hist, self.cfg.ldp_epsilon, rng=self._rng)
            # Concatenate: label histogram (num_classes) + pixel histogram (num_bins)
            noisy = np.concatenate([noisy, pixel_noisy])

        anchor = StatisticalAnchor(
            client_id=client_id,
            bins=noisy,
            num_bins=len(noisy),
            ldp_epsilon=self.cfg.ldp_epsilon,
            declared_size=len(labels),
        )
        return anchor, true_hist_norm

    def generate_proof(
        self,
        anchor: StatisticalAnchor,
        true_hist: np.ndarray,
    ) -> ZKProof:
        """
        Generate a simulated zk-SNARK proof for the enrollment relation.
        """
        pk = _make_public_key(anchor.client_id)
        commitment = compute_commitment(pk, anchor)

        proof_bytes, is_valid = _simulate_proof(
            client_id=anchor.client_id,
            true_histogram=true_hist,
            noisy_anchor=anchor.bins,
            dataset_size=anchor.declared_size,
            ldp_epsilon=anchor.ldp_epsilon,
            min_size=self.cfg.min_dataset_size,
        )

        return ZKProof(
            client_id=anchor.client_id,
            commitment=commitment,
            proof_bytes=proof_bytes,
            public_inputs={
                "tau_size":  self.cfg.min_dataset_size,
                "epsilon":   anchor.ldp_epsilon,
                "num_bins":  anchor.num_bins,
            },
            is_valid=is_valid,
        )

    # ------------------------------------------------------------------
    # Server / contract side
    # ------------------------------------------------------------------
    def verify_and_enroll(
        self,
        proof: ZKProof,
        anchor: StatisticalAnchor,
    ) -> Optional[EnrollmentRecord]:
        """
        Verify ZK proof and issue an enrollment record.
        Returns None if verification fails.
        """
        if not proof.is_valid:
            logger.warning("Client %d: ZK proof invalid (is_valid=False).", anchor.client_id)
            return None

        ok = _verify_proof(
            proof.proof_bytes,
            anchor.client_id,
            anchor,
            self.cfg.min_dataset_size,
        )
        if not ok:
            logger.warning("Client %d: ZK proof verification FAILED.", anchor.client_id)
            return None

        pk = _make_public_key(anchor.client_id)
        record = EnrollmentRecord(
            client_id=anchor.client_id,
            public_key=pk,
            commitment=proof.commitment,
            anchor_vector=anchor.as_conditioning_vector(target_dim=self.cfg.num_bins),
            declared_size=anchor.declared_size,
            ldp_epsilon=anchor.ldp_epsilon,
            enrolled=True,
        )
        self._registry[anchor.client_id] = record
        logger.debug("Client %d enrolled.  Commitment: %s", anchor.client_id, proof.commitment[:16])
        return record

    def is_enrolled(self, client_id: int) -> bool:
        return client_id in self._registry and self._registry[client_id].enrolled

    def get_anchor_vector(self, client_id: int) -> Optional[np.ndarray]:
        rec = self._registry.get(client_id)
        return rec.anchor_vector if rec else None

    def get_commitment(self, client_id: int) -> Optional[str]:
        rec = self._registry.get(client_id)
        return rec.commitment if rec else None

    @property
    def enrolled_ids(self) -> list:
        return [cid for cid, r in self._registry.items() if r.enrolled]

    def revoke(self, client_id: int) -> None:
        """Mark a client as unenrolled (e.g., after repeated fingerprint failures)."""
        if client_id in self._registry:
            self._registry[client_id].enrolled = False
            logger.info("Client %d enrollment revoked.", client_id)
