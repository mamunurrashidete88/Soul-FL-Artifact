"""
security/blockchain_sim.py — Lightweight Blockchain Simulation Layer .

Simulates the on-chain smart contract responsible for:
  • Immutable SBT state storage  (identity registry)
  • Deterministic trust-decay enforcement  (no aggregator discretion)
  • Auditable, non-repudiable event history
  • Gas metering for deployment feasibility analysis

"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from security.trust_engine import TrustEngine, SBTState, Voucher
from config import TrustConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block / Transaction primitives
# ---------------------------------------------------------------------------
@dataclass
class Transaction:
    tx_hash: str
    block_number: int
    sender: str
    function: str
    args: dict
    gas_used: int
    success: bool
    return_value: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Block:
    number: int
    timestamp: float
    transactions: List[Transaction] = field(default_factory=list)
    parent_hash: str = "0x" + "0" * 64

    @property
    def block_hash(self) -> str:
        payload = json.dumps(
            {
                "number": self.number,
                "timestamp": self.timestamp,
                "txs": [t.tx_hash for t in self.transactions],
                "parent": self.parent_hash,
            },
            sort_keys=True,
        ).encode()
        return "0x" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Gas cost constants 
# ---------------------------------------------------------------------------
GAS = {
    "MINT_SBT":        55_000,
    "REDEEM_VOUCHER":  35_000,
    "REVOKE_SBT":      22_000,
    "GET_BALANCE":      3_000,   # view function — no state change
    "ADVANCE_ROUND":   10_000,
}

# Wei per gas (1 gwei = 1e-9 ETH)
GAS_PRICE_GWEI = 1.0


def gas_cost_eth(gas: int) -> float:
    return gas * GAS_PRICE_GWEI * 1e-9


# ---------------------------------------------------------------------------
# SoulFL Smart Contract Simulation
# ---------------------------------------------------------------------------
class SoulFLContract:

    def __init__(self, cfg: TrustConfig, aggregator_address: str = "0xAGGREGATOR"):
        self.cfg = cfg
        self.engine = TrustEngine(cfg)
        self.aggregator = aggregator_address

        # Chain state
        self._blocks: List[Block] = []
        self._pending_txs: List[Transaction] = []
        self._block_number: int = 0
        self._current_round: int = 0

        # Cumulative gas tracking
        self._total_gas: int = 0
        self._gas_by_function: Dict[str, int] = {}

        # Mine genesis block
        self._mine_block()

    # ------------------------------------------------------------------
    # Contract functions  (each mines a tx)
    # ------------------------------------------------------------------
    def mint_sbt(self, client_id: int, commitment: str, sender: str) -> Transaction:
        """
        Mint a non-transferable SBT for an enrolled client.
        Only callable once per client_id (reverts on duplicate).
        """
        try:
            state = self.engine.mint(client_id, commitment)
            success = True
            ret = state.B_stored
        except ValueError as e:
            logger.warning("mintSBT reverted: %s", e)
            success = False
            ret = None

        tx = self._record_tx("mintSBT", sender, GAS["MINT_SBT"], success, ret,
                             args={"client_id": client_id, "commitment": commitment})
        return tx

    def redeem_voucher(self, voucher: Voucher, sender: str) -> Transaction:
        """
        Redeem an aggregator-signed voucher to refuel trust balance.
        Verifies signature and nonce; applies lazy decay before refueling.
        """
        self.engine.set_round(self._current_round)
        ok = self.engine.redeem_voucher(voucher)
        tx = self._record_tx(
            "redeemVoucher", sender, GAS["REDEEM_VOUCHER"], ok,
            args={"client_id": voucher.client_id, "amount": voucher.amount,
                  "round": voucher.round_issued},
        )
        return tx

    def revoke_sbt(self, client_id: int, sender: str) -> Transaction:
        """Revoke a client's SBT.  Only callable by aggregator."""
        if sender != self.aggregator:
            tx = self._record_tx("revokeSBT", sender, GAS["REVOKE_SBT"], False,
                                 args={"client_id": client_id, "error": "not aggregator"})
            return tx
        self.engine.revoke(client_id)
        tx = self._record_tx("revokeSBT", sender, GAS["REVOKE_SBT"], True,
                             args={"client_id": client_id})
        return tx

    def get_effective_balance(self, client_id: int) -> float:
        """View function — returns B_eff(t) via lazy evaluation."""
        self.engine.set_round(self._current_round)
        return self.engine.get_effective_balance(client_id)

    def advance_round(self, sender: str) -> Transaction:
        """Increment the on-chain round counter.  Only aggregator."""
        if sender != self.aggregator:
            return self._record_tx("advanceRound", sender, GAS["ADVANCE_ROUND"], False,
                                   args={"error": "not aggregator"})
        self._current_round += 1
        self.engine.set_round(self._current_round)
        tx = self._record_tx("advanceRound", sender, GAS["ADVANCE_ROUND"], True,
                             args={"new_round": self._current_round})
        self._mine_block()
        return tx

    # ------------------------------------------------------------------
    # Batch helpers 
    # ------------------------------------------------------------------
    def batch_redeem(self, vouchers: List[Voucher]) -> List[Transaction]:
        """Redeem multiple vouchers in a single block."""
        txs = []
        for v in vouchers:
            tx = self.redeem_voucher(v, sender=f"0xCLIENT{v.client_id}")
            txs.append(tx)
        self._mine_block()
        return txs

    def batch_mint(self, enrollments: List[dict]) -> List[Transaction]:
        """Mint SBTs for all enrolled clients."""
        txs = []
        for e in enrollments:
            tx = self.mint_sbt(e["client_id"], e["commitment"], sender=self.aggregator)
            txs.append(tx)
        self._mine_block()
        return txs

    # ------------------------------------------------------------------
    # Gas metering
    # ------------------------------------------------------------------
    def gas_report(self) -> dict:
        return {
            "total_gas":      self._total_gas,
            "total_eth":      gas_cost_eth(self._total_gas),
            "by_function":    {k: {"gas": v, "eth": gas_cost_eth(v)}
                               for k, v in self._gas_by_function.items()},
            "blocks_mined":   self._block_number,
            "txs_processed":  sum(len(b.transactions) for b in self._blocks),
        }

    # ------------------------------------------------------------------
    # Trust query wrappers
    # ------------------------------------------------------------------
    def get_eligible_clients(self) -> List[int]:
        self.engine.set_round(self._current_round)
        return self.engine.get_eligible_clients()

    def compute_weights(self, client_ids: List[int]):
        self.engine.set_round(self._current_round)
        return self.engine.compute_aggregation_weights(client_ids)

    def issue_voucher(self, client_id: int, accepted: bool) -> Voucher:
        return self.engine.issue_voucher(client_id, accepted, self._current_round)

    def trust_stats(self) -> dict:
        self.engine.set_round(self._current_round)
        return self.engine.trust_stats()

    def export_audit_log(self) -> List[dict]:
        return self.engine.export_audit_log()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _record_tx(
        self,
        function: str,
        sender: str,
        gas: int,
        success: bool,
        return_value: Any = None,
        args: dict = None,
    ) -> Transaction:
        tx_hash = "0x" + hashlib.sha256(
            f"{function}{sender}{self._block_number}{time.time_ns()}".encode()
        ).hexdigest()[:40]

        tx = Transaction(
            tx_hash=tx_hash,
            block_number=self._block_number,
            sender=sender,
            function=function,
            args=args or {},
            gas_used=gas if success else gas // 2,
            success=success,
            return_value=return_value,
        )
        self._pending_txs.append(tx)
        self._total_gas += tx.gas_used
        self._gas_by_function[function] = (
            self._gas_by_function.get(function, 0) + tx.gas_used
        )
        return tx

    def _mine_block(self) -> Block:
        parent = self._blocks[-1].block_hash if self._blocks else "0x" + "0" * 64
        block = Block(
            number=self._block_number,
            timestamp=time.time(),
            transactions=list(self._pending_txs),
            parent_hash=parent,
        )
        self._blocks.append(block)
        self._pending_txs.clear()
        self._block_number += 1
        return block

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def chain_length(self) -> int:
        return self._block_number


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_chain(cfg: TrustConfig) -> SoulFLContract:
    """Instantiate a fresh simulated blockchain for a Soul-FL experiment."""
    contract = SoulFLContract(cfg, aggregator_address="0xAGGREGATOR")
    logger.info(
        "Simulated blockchain initialized | chain_id=%d | decay=%.3f | τ_min=%.2f",
        cfg.chain_id, cfg.decay_rate, cfg.min_balance,
    )
    return contract
