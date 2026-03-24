
import hashlib
import hmac
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import TrustConfig

logger = logging.getLogger(__name__)

# Aggregator signing key (
_AGG_SIGN_KEY = os.environ.get("SOUL_FL_AGG_KEY", "soul-fl-aggregator-v1").encode()


# ---------------------------------------------------------------------------
# SBT State record  
# ---------------------------------------------------------------------------
@dataclass
class SBTState:

    client_id: int
    commitment: str            # C_i = Hash(pk_i || S_i)
    B_stored: float            # trust balance at last checkpoint
    t_last: int                # round at which B_stored was last checkpointed
    nonce: int = 0             # replay-protection nonce for vouchers
    active: bool = True        # False if revoked
    total_vouchers_received: int = 0
    total_rounds_accepted: int = 0

    def effective_balance(self, current_round: int, decay_rate: float) -> float:
        """
        Lazy evaluation of B_eff(t)  (Eq. 2).
        Does not update storage — evaluated on-demand to save gas.
        """
        dt = current_round - self.t_last
        return self.B_stored * math.exp(-decay_rate * dt)


# ---------------------------------------------------------------------------
# Voucher  
# ---------------------------------------------------------------------------
@dataclass
class Voucher:

    client_id: int
    amount: float              # V_a (acceptance reward) or 0 (rejected)
    round_issued: int
    nonce: int
    chain_id: int
    sbt_address: str
    signature: bytes = field(default=b"", repr=False)

    def payload_bytes(self) -> bytes:
        payload = {
            "client_id":   self.client_id,
            "amount":      round(self.amount, 8),
            "round":       self.round_issued,
            "nonce":       self.nonce,
            "chain_id":    self.chain_id,
            "sbt_address": self.sbt_address,
        }
        return json.dumps(payload, sort_keys=True).encode()

    @property
    def is_reward(self) -> bool:
        return self.amount > 0


def sign_voucher(voucher: Voucher, key: bytes = _AGG_SIGN_KEY) -> Voucher:
    """HMAC-SHA256 signature ."""
    sig = hmac.new(key, voucher.payload_bytes(), "sha256").digest()
    voucher.signature = sig
    return voucher


def verify_voucher(voucher: Voucher, key: bytes = _AGG_SIGN_KEY) -> bool:
    """Verify aggregator signature on voucher."""
    expected = hmac.new(key, voucher.payload_bytes(), "sha256").digest()
    return hmac.compare_digest(expected, voucher.signature)


# ---------------------------------------------------------------------------
# On-Chain Trust Engine (smart contract simulation)
# ---------------------------------------------------------------------------
class TrustEngine:


    def __init__(self, cfg: TrustConfig, seed: int = 0):
        self.cfg = cfg
        self._states: Dict[int, SBTState] = {}
        self._event_log: List[dict] = []   # immutable audit trail
        self._current_round: int = 0

    # ------------------------------------------------------------------
    # SBT lifecycle
    # ------------------------------------------------------------------
    def mint(self, client_id: int, commitment: str) -> SBTState:

        if client_id in self._states:
            raise ValueError(f"SBT already exists for client {client_id}")

        state = SBTState(
            client_id=client_id,
            commitment=commitment,
            B_stored=self.cfg.initial_balance,
            t_last=self._current_round,
        )
        self._states[client_id] = state
        self._log("MINT", client_id, B_new=self.cfg.initial_balance)
        logger.debug("SBT minted for client %d  B₀=%.2f", client_id, self.cfg.initial_balance)
        return state

    def revoke(self, client_id: int) -> None:
        """Revoke a client's SBT (permanently excludes from aggregation)."""
        if client_id in self._states:
            self._states[client_id].active = False
            self._log("REVOKE", client_id)

    # ------------------------------------------------------------------
    # Core contract functions
    # ------------------------------------------------------------------
    def get_effective_balance(self, client_id: int) -> float:

        state = self._states.get(client_id)
        if state is None or not state.active:
            return 0.0
        return state.effective_balance(self._current_round, self.cfg.decay_rate)

    def is_eligible(self, client_id: int) -> bool:
        """Eligibility gate:  B_eff(t) ≥ τ_min  (predicate 𝕀_Live)."""
        return self.get_effective_balance(client_id) >= self.cfg.min_balance

    def redeem_voucher(self, voucher: Voucher) -> bool:

        if not verify_voucher(voucher):
            logger.warning("Client %d: voucher signature invalid.", voucher.client_id)
            return False

        if voucher.chain_id != self.cfg.chain_id:
            logger.warning("Client %d: wrong chain_id.", voucher.client_id)
            return False

        state = self._states.get(voucher.client_id)
        if state is None or not state.active:
            logger.warning("Client %d: SBT not found or revoked.", voucher.client_id)
            return False

        if voucher.nonce != state.nonce:
            logger.warning(
                "Client %d: nonce mismatch (expected %d, got %d).",
                voucher.client_id, state.nonce, voucher.nonce,
            )
            return False

        # Apply decay to current round before refueling
        b_eff = state.effective_balance(self._current_round, self.cfg.decay_rate)

        # Refuel (acceptance-only — zero-amount vouchers just apply decay)
        b_new = b_eff + voucher.amount

        # Clamp to maximum  B_max = max(B₀, V_a / (1 - e^{-λ}))
        b_max = max(
            self.cfg.initial_balance,
            self.cfg.acceptance_reward / (1 - math.exp(-self.cfg.decay_rate)),
        )
        b_new = min(b_new, b_max)

        # Update state checkpoint
        state.B_stored = b_new
        state.t_last = self._current_round
        state.nonce += 1
        if voucher.is_reward:
            state.total_vouchers_received += 1
            state.total_rounds_accepted += 1

        self._log(
            "REFUEL", voucher.client_id,
            amount=voucher.amount, B_new=b_new, round_=self._current_round,
        )
        return True

    # ------------------------------------------------------------------
    # Aggregator-side helpers
    # ------------------------------------------------------------------
    def issue_voucher(
        self,
        client_id: int,
        accepted: bool,
        current_round: int,
    ) -> Voucher:

        state = self._states.get(client_id)
        nonce = state.nonce if state else 0
        amount = self.cfg.acceptance_reward if accepted else 0.0

        v = Voucher(
            client_id=client_id,
            amount=amount,
            round_issued=current_round,
            nonce=nonce,
            chain_id=self.cfg.chain_id,
            sbt_address=self.cfg.sbt_address,
        )
        return sign_voucher(v)

    def advance_round(self) -> None:
        """Increment global round counter (called by aggregator at round closure)."""
        self._current_round += 1

    def set_round(self, t: int) -> None:
        self._current_round = t

    # ------------------------------------------------------------------
    # Aggregation weight computation  (Phase IV)
    # ------------------------------------------------------------------
    def compute_aggregation_weights(
        self, client_ids: List[int]
    ) -> Tuple[Dict[int, float], float]:

        eligible = {
            cid: self.get_effective_balance(cid)
            for cid in client_ids
            if self.is_eligible(cid)
        }

        if not eligible:
            logger.warning("No eligible clients in round %d!", self._current_round)
            # Fall back to uniform weights (safety net)
            n = len(client_ids)
            return {cid: 1.0 / n for cid in client_ids}, 1.0

        squashed = {cid: math.log1p(b) for cid, b in eligible.items()}
        Z = sum(squashed.values()) + 1e-12
        weights = {cid: v / Z for cid, v in squashed.items()}
        return weights, Z

    # ------------------------------------------------------------------
    # Diagnostics / audit
    # ------------------------------------------------------------------
    def get_balance_snapshot(self, round_: Optional[int] = None) -> Dict[int, float]:
        t = round_ if round_ is not None else self._current_round
        return {
            cid: s.effective_balance(t, self.cfg.decay_rate)
            for cid, s in self._states.items()
            if s.active
        }

    def get_eligible_clients(self) -> List[int]:
        return [cid for cid in self._states if self.is_eligible(cid)]

    def trust_stats(self) -> dict:
        balances = list(self.get_balance_snapshot().values())
        if not balances:
            return {}
        return {
            "round": self._current_round,
            "num_active": len(balances),
            "num_eligible": len(self.get_eligible_clients()),
            "mean_balance": float(np.mean(balances)),
            "min_balance": float(np.min(balances)),
            "max_balance": float(np.max(balances)),
        }

    def _log(self, event: str, client_id: int, **kwargs) -> None:
        entry = {
            "event": event,
            "client_id": client_id,
            "round": self._current_round,
            "timestamp": time.time(),
            **kwargs,
        }
        self._event_log.append(entry)

    def export_audit_log(self) -> List[dict]:
        return list(self._event_log)

    # ------------------------------------------------------------------
    # Design constraint verification (Lemma 1)
    # ------------------------------------------------------------------
    def verify_design_constraint(self, beta: float) -> bool:

        lam = self.cfg.decay_rate
        b_bar = beta * self.cfg.acceptance_reward / (1 - math.exp(-lam))
        ok = b_bar < self.cfg.min_balance
        if not ok:
            logger.warning(
                "Design constraint VIOLATED: B̄=%.4f ≥ τ_min=%.4f  "
                "(reduce β, V_a, or increase λ).",
                b_bar, self.cfg.min_balance,
            )
        return ok
