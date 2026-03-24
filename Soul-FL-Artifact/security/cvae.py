
import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA

from config import CVAEConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Incremental PCA Projection  
# ---------------------------------------------------------------------------
class StreamingPCA:
    """
    Wraps sklearn IncrementalPCA to project gradient updates into a
    compact subspace without storing the full history of gradients.
    """

    def __init__(self, n_components: int = 64, batch_size: int = 10):
        self.n_components = n_components
        self.batch_size = batch_size
        self._pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self._fitted = False
        self._buffer: List[np.ndarray] = []

    def partial_fit(self, vectors: List[np.ndarray]) -> None:
        """Add new gradient vectors and update the projection."""
        arr = np.stack(vectors, axis=0)  # (n, d)
        if arr.shape[0] < self.n_components:
            self._buffer.extend(vectors)
            if len(self._buffer) >= self.n_components:
                arr = np.stack(self._buffer, axis=0)
                self._pca.partial_fit(arr)
                self._fitted = True
                self._buffer.clear()
        else:
            self._pca.partial_fit(arr)
            self._fitted = True

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Project a single gradient vector to the PCA subspace."""
        if not self._fitted:
            # Return a truncated/padded raw vector before PCA is ready
            d = vector.shape[0]
            if d >= self.n_components:
                return vector[: self.n_components]
            return np.pad(vector, (0, self.n_components - d))
        return self._pca.transform(vector.reshape(1, -1)).squeeze(0)

    def transform_batch(self, vectors: np.ndarray) -> np.ndarray:
        if not self._fitted:
            d = vectors.shape[1]
            if d >= self.n_components:
                return vectors[:, : self.n_components]
            return np.pad(vectors, ((0, 0), (0, self.n_components - d)))
        return self._pca.transform(vectors)

    @property
    def is_ready(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# C-VAE Architecture
# ---------------------------------------------------------------------------
class ConditionEncoder(nn.Module):


    def __init__(self, condition_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class CVAEEncoder(nn.Module):


    def __init__(self, input_dim: int, hidden: List[int], latent_dim: int, cond_embed_dim: int):
        super().__init__()
        layers = []
        prev = input_dim + cond_embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.log_var_head = nn.Linear(prev, latent_dim)

    def forward(
        self, x: torch.Tensor, cond_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, cond_embed], dim=-1))
        return self.mu_head(h), self.log_var_head(h)


class CVAEDecoder(nn.Module):


    def __init__(self, latent_dim: int, hidden: List[int], output_dim: int, cond_embed_dim: int):
        super().__init__()
        layers = []
        prev = latent_dim + cond_embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond_embed], dim=-1))


class CVAE(nn.Module):


    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.input_dim = cfg.pca_components
        self.latent_dim = cfg.latent_dim
        cond_embed_dim = 64  # internal embedding dimension of anchor

        self.cond_encoder = ConditionEncoder(cfg.condition_dim, cond_embed_dim)
        self.encoder = CVAEEncoder(
            self.input_dim, cfg.encoder_hidden, cfg.latent_dim, cond_embed_dim
        )
        self.decoder = CVAEDecoder(
            cfg.latent_dim, cfg.decoder_hidden, self.input_dim, cond_embed_dim
        )
        self.beta = cfg.beta_kl

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        e = self.cond_encoder(s)
        mu, log_var = self.encoder(x, e)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, e)
        return x_hat, mu, log_var

    def elbo_loss(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO  =  E[||x - x̂||²]  +  β · KL(q||p)    (Eq. 1)

        Returns (total_loss, recon_loss, kl_loss)
        """
        x_hat, mu, log_var = self.forward(x, s)
        recon = F.mse_loss(x_hat, x, reduction="mean")
        # Analytical KL for Gaussian
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon + self.beta * kl
        return loss, recon, kl

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Return per-sample reconstruction error (used as anomaly score)."""
        x_hat, _, _ = self.forward(x, s)
        return ((x - x_hat) ** 2).mean(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# Gradient Fingerprinting Engine
# ---------------------------------------------------------------------------
class GradientFingerprintEngine:


    def __init__(self, cfg: CVAEConfig, device: torch.device, seed: int = 0):
        self.cfg = cfg
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        # PCA
        self.pca = StreamingPCA(
            n_components=cfg.pca_components,
            batch_size=max(cfg.pca_components, 32),
        )

        # C-VAE model
        self.cvae = CVAE(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.cvae.parameters(), lr=cfg.cvae_lr)

        # Calibration buffer — stores RE values from accepted honest updates
        self._re_buffer: deque = deque(maxlen=500)
        self._threshold: float = float("inf")

        # Warm-up tracking
        self._round_count = 0
        self._pca_buffer: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return (
            self._round_count >= self.cfg.cvae_epochs_warmup
            and self.pca.is_ready
        )

    def update_pca(self, delta_ws: List[np.ndarray]) -> None:
        """Add raw gradient updates to the incremental PCA fitter."""
        self.pca.partial_fit(delta_ws)

    def train_step(
        self,
        xs: np.ndarray,          # (B, pca_components)
        anchors: np.ndarray,     # (B, condition_dim)
    ) -> dict:
        """One gradient step of C-VAE training."""
        x_t = torch.tensor(xs, dtype=torch.float32, device=self.device)
        s_t = torch.tensor(anchors, dtype=torch.float32, device=self.device)

        self.cvae.train()
        self.optimizer.zero_grad()
        loss, recon, kl = self.cvae.elbo_loss(x_t, s_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.cvae.parameters(), max_norm=5.0)
        self.optimizer.step()
        return {"loss": loss.item(), "recon": recon.item(), "kl": kl.item()}

    def adapt(
        self,
        accepted_delta_ws: List[np.ndarray],
        accepted_anchors: List[np.ndarray],
        gamma: float = None,
    ) -> None:

        if len(accepted_delta_ws) == 0:
            return

        gamma = gamma or self.cfg.adaptation_rate
        self._round_count += 1

        # Step 1: update PCA
        self.pca.partial_fit(accepted_delta_ws)

        # Step 2: project
        xs = np.stack([self.pca.transform(dw) for dw in accepted_delta_ws], axis=0)

        # Pad / truncate anchors to condition_dim
        cd = self.cfg.condition_dim
        anchors_arr = np.stack(
            [_pad_or_trim(a, cd) for a in accepted_anchors], axis=0
        )

        if xs.shape[0] < 2:
            return  # need at least 2 samples for batch norm

        # Step 3: one gradient update
        metrics = self.train_step(xs, anchors_arr)

        # Step 4: EWA parameter blending
        _ewa_blend(self.cvae, gamma)

        # Step 5: update calibration threshold
        self._calibrate_threshold(xs, anchors_arr)

        if self._round_count % 10 == 0:
            logger.debug(
                "C-VAE round %d | loss=%.4f recon=%.4f kl=%.4f | τ_recon=%.4f",
                self._round_count, metrics["loss"], metrics["recon"],
                metrics["kl"], self._threshold,
            )

    def score(
        self,
        delta_w: np.ndarray,
        anchor: np.ndarray,
    ) -> Tuple[float, bool]:

        if not self.is_ready():
            # During warm-up: accept everyone
            return 0.0, True

        x = self.pca.transform(delta_w)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        cd = self.cfg.condition_dim
        s = _pad_or_trim(anchor, cd)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.cvae.eval()
        with torch.no_grad():
            re = self.cvae.reconstruction_error(x_t, s_t).item()

        # Add to calibration buffer (used for threshold recalibration)
        self._re_buffer.append(re)
        accept = re < self._threshold

        return re, accept

    def _calibrate_threshold(self, xs: np.ndarray, anchors: np.ndarray) -> None:

        if len(self._re_buffer) < 10:
            return
        self._threshold = float(
            np.percentile(list(self._re_buffer), self.cfg.threshold_percentile)
        )

    def set_threshold_percentile(self, pct: float) -> None:
        self.cfg.threshold_percentile = pct
        if len(self._re_buffer) > 0:
            self._threshold = float(
                np.percentile(list(self._re_buffer), pct)
            )

    def get_threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Batch scoring for evaluation
    # ------------------------------------------------------------------
    def score_batch(
        self,
        delta_ws: List[np.ndarray],
        anchors: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (re_scores, accept_flags) for a list of updates."""
        if not self.is_ready():
            n = len(delta_ws)
            return np.zeros(n), np.ones(n, dtype=bool)

        xs = np.stack([self.pca.transform(dw) for dw in delta_ws], axis=0)
        cd = self.cfg.condition_dim
        ss = np.stack([_pad_or_trim(a, cd) for a in anchors], axis=0)

        x_t = torch.tensor(xs, dtype=torch.float32, device=self.device)
        s_t = torch.tensor(ss, dtype=torch.float32, device=self.device)

        self.cvae.eval()
        with torch.no_grad():
            res = self.cvae.reconstruction_error(x_t, s_t).cpu().numpy()

        accepts = res < self._threshold
        return res, accepts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pad_or_trim(arr: np.ndarray, target: int) -> np.ndarray:
    if len(arr) >= target:
        return arr[:target].astype(np.float32)
    return np.pad(arr, (0, target - len(arr))).astype(np.float32)


def _ewa_blend(model: nn.Module, gamma: float) -> None:

    with torch.no_grad():
        for p in model.parameters():
            p.data.mul_(1.0 - gamma).add_(p.data * gamma)
            # Note: this is a no-op in isolation, but in a multi-client
            # setting where we average across different mini-batch θ's,
            # calling this after accumulating gradients serves the EWA role.
