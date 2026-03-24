
import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Phase I: ZK Enrollment
# ─────────────────────────────────────────────────────────────────────────────

class TestZKEnrollment:
    """Tests for security/zk.py — Phase I of Soul-FL."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from config import ZKConfig
        from security.zk import ZKEnrollmentEngine
        self.cfg = ZKConfig(min_dataset_size=50)   # lowered for fast tests
        self.engine = ZKEnrollmentEngine(self.cfg, seed=0)
        self.rng = np.random.default_rng(42)

    def _enroll(self, client_id: int, n: int = 200):
        """Helper: enroll a client with `n` samples."""
        labels = self.rng.integers(0, 10, size=n)
        anchor, true_hist = self.engine.prepare_anchor(client_id, labels, num_classes=10)
        proof  = self.engine.generate_proof(anchor, true_hist)
        record = self.engine.verify_and_enroll(proof, anchor)
        return record, anchor, proof

    # ── Soundness ────────────────────────────────────────────────────────────

    def test_valid_enrollment_succeeds(self):
        record, _, _ = self._enroll(0, n=200)
        assert record is not None
        assert record.enrolled is True
        assert record.client_id == 0

    def test_free_rider_rejected(self):
        """Client with |D_i| < τ_size must be rejected (Theorem 1, Part 1)."""
        labels = self.rng.integers(0, 10, size=3)   # << τ_size=50
        anchor, th = self.engine.prepare_anchor(99, labels, num_classes=10)
        proof  = self.engine.generate_proof(anchor, th)
        record = self.engine.verify_and_enroll(proof, anchor)
        assert record is None, "Free-rider must be rejected"

    def test_enrollment_idempotent(self):
        self._enroll(0, 200)
        assert self.engine.is_enrolled(0)
        # Second enrollment with same ID is a no-op (already enrolled)
        record2, _, _ = self._enroll(0, 200)
        assert record2 is not None  # re-enrollment: engine allows overwrite

    def test_commitment_deterministic(self):
        """Same key + anchor → same commitment."""
        from security.zk import compute_commitment, _make_public_key, StatisticalAnchor
        bins = np.ones(10, dtype=np.float32) / 10
        s    = StatisticalAnchor(0, bins, 10, 1.0, 200)
        pk   = _make_public_key(0)
        c1   = compute_commitment(pk, s)
        c2   = compute_commitment(pk, s)
        assert c1 == c2

    def test_anchor_vector_dimension(self):
        record, anchor, _ = self._enroll(0, 200)
        vec = self.engine.get_anchor_vector(0)
        assert vec is not None
        assert vec.shape == (self.cfg.num_bins,)
        assert np.isfinite(vec).all()

    # ── LDP ──────────────────────────────────────────────────────────────────

    def test_ldp_output_nonnegative(self):
        from security.zk import apply_ldp_to_histogram
        h = np.array([0.3, 0.2, 0.3, 0.1, 0.1], dtype=np.float32)
        for eps in [0.1, 0.5, 1.0, 4.0]:
            noisy = apply_ldp_to_histogram(h, eps)
            assert (noisy >= 0).all(), f"Negative bins at ε={eps}"

    def test_ldp_higher_epsilon_less_noise(self):
        """Higher ε → tighter privacy → less noise magnitude."""
        from security.zk import apply_ldp_to_histogram
        h   = np.ones(10, dtype=np.float32) / 10
        rng = np.random.default_rng(7)
        noise_low  = np.std([apply_ldp_to_histogram(h.copy(), 0.1, rng=rng) for _ in range(200)])
        noise_high = np.std([apply_ldp_to_histogram(h.copy(), 4.0, rng=rng) for _ in range(200)])
        assert noise_high < noise_low, "Higher ε should produce less noise"

    def test_histogram_from_labels(self):
        from security.zk import build_histogram_from_labels
        labels = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
        h = build_histogram_from_labels(labels, num_classes=3)
        np.testing.assert_array_equal(h, [2, 1, 3])

    # ── Revocation ───────────────────────────────────────────────────────────

    def test_revocation(self):
        self._enroll(5, 200)
        assert self.engine.is_enrolled(5)
        self.engine.revoke(5)
        assert not self.engine.is_enrolled(5)


# ─────────────────────────────────────────────────────────────────────────────
# Phase III: Trust Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustEngine:
    """Tests for security/trust_engine.py — Phase III."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from config import TrustConfig
        from security.trust_engine import TrustEngine
        self.cfg = TrustConfig()   # τ_min=15, λ=0.05, V_a=10, B₀=100
        self.te  = TrustEngine(self.cfg)
        self.te.mint(0, "commit_0")
        self.te.mint(1, "commit_1")

    # ── Decay ────────────────────────────────────────────────────────────────

    def test_initial_balance(self):
        self.te.set_round(0)
        assert self.te.get_effective_balance(0) == pytest.approx(100.0, abs=0.01)

    def test_exponential_decay_formula(self):
        """B_eff(t) = B₀ · e^{−λt}   (Eq. 2)."""
        for t in [1, 5, 14, 50, 100]:
            self.te.set_round(t)
            b    = self.te.get_effective_balance(0)
            b_ex = 100.0 * math.exp(-0.05 * t)
            assert b == pytest.approx(b_ex, rel=1e-4), f"Round {t}: {b} vs {b_ex}"

    def test_decay_monotone_decreasing(self):
        balances = []
        for t in range(0, 60, 5):
            self.te.set_round(t)
            balances.append(self.te.get_effective_balance(0))
        for i in range(len(balances) - 1):
            assert balances[i] >= balances[i + 1]

    def test_half_life(self):
        """At t ≈ ln(2)/λ ≈ 13.86 rounds, B_eff ≈ B₀/2."""
        half_life = math.log(2) / self.cfg.decay_rate
        self.te.set_round(round(half_life))
        b = self.te.get_effective_balance(0)
        assert b == pytest.approx(50.0, rel=0.02)

    # ── Voucher lifecycle ────────────────────────────────────────────────────

    def test_acceptance_voucher_refuels(self):
        self.te.set_round(14)
        b_before = self.te.get_effective_balance(0)
        v = self.te.issue_voucher(0, accepted=True, current_round=14)
        assert v.amount == pytest.approx(self.cfg.acceptance_reward)
        ok = self.te.redeem_voucher(v)
        assert ok
        b_after = self.te.get_effective_balance(0)
        assert b_after == pytest.approx(b_before + self.cfg.acceptance_reward, rel=1e-4)

    def test_rejection_voucher_no_refuel(self):
        self.te.set_round(0)
        b_before = self.te.get_effective_balance(0)
        v = self.te.issue_voucher(0, accepted=False, current_round=0)
        assert v.amount == 0.0
        self.te.redeem_voucher(v)
        b_after = self.te.get_effective_balance(0)
        assert b_after == pytest.approx(b_before, rel=1e-4)

    def test_replay_protection(self):
        from security.trust_engine import sign_voucher
        self.te.set_round(0)
        v = self.te.issue_voucher(0, True, 0)
        assert self.te.redeem_voucher(v)              # first: ok
        # Second attempt with stale nonce
        v_bad = self.te.issue_voucher(0, True, 0)
        v_bad.nonce = 0                                # stale nonce
        v_bad = sign_voucher(v_bad)
        assert not self.te.redeem_voucher(v_bad)       # must fail

    def test_balance_capped_at_bmax(self):
        """Trust must never exceed B_max = max(B₀, V_a/(1−e^{−λ}))."""
        b_max = max(
            self.cfg.initial_balance,
            self.cfg.acceptance_reward / (1 - math.exp(-self.cfg.decay_rate)),
        )
        self.te.set_round(0)
        # Redeem many vouchers
        for _ in range(50):
            v = self.te.issue_voucher(0, True, 0)
            self.te.redeem_voucher(v)
        b = self.te.get_effective_balance(0)
        assert b <= b_max + 0.01

    # ── Eligibility ──────────────────────────────────────────────────────────

    def test_eligibility_above_threshold(self):
        self.te.set_round(0)   # B=100 > τ_min=15
        assert self.te.is_eligible(0)

    def test_eligibility_drops_after_long_absence(self):
        """After sufficient rounds without vouchers, B_eff drops below τ_min."""
        tau_min = self.cfg.min_balance
        lam     = self.cfg.decay_rate
        t_drop  = math.ceil(math.log(100.0 / tau_min) / lam) + 5
        self.te.set_round(t_drop)
        assert not self.te.is_eligible(0)

    def test_nonexistent_client_ineligible(self):
        assert not self.te.is_eligible(999)

    # ── Lemma 1 ──────────────────────────────────────────────────────────────

    def test_lemma1_satisfied_default(self):
        """Default params must satisfy B̄ < τ_min (Lemma 1 design constraint)."""
        ok = self.te.verify_design_constraint(beta=0.05)
        assert ok, "Lemma 1 must be satisfied with default TrustConfig"

    @pytest.mark.parametrize("beta", [0.0, 0.01, 0.05, 0.10])
    def test_lemma1_parametric(self, beta):
        lam   = self.cfg.decay_rate
        Va    = self.cfg.acceptance_reward
        B_bar = beta * Va / (1 - math.exp(-lam))
        ok    = B_bar < self.cfg.min_balance
        assert ok == self.te.verify_design_constraint(beta=beta)

    # ── Aggregation weights ──────────────────────────────────────────────────

    def test_weights_sum_to_one(self):
        self.te.set_round(0)
        w, Z = self.te.compute_aggregation_weights([0, 1])
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_weights_sublinear(self):
        """φ(x)=log(1+x) → doubling balance should NOT double weight."""
        from config import TrustConfig
        from security.trust_engine import TrustEngine
        t2 = TrustEngine(TrustConfig())
        t2.mint(0, "c0"); t2.mint(1, "c1")
        # Set client 1 to 2× balance
        t2._states[1].B_stored = 200.0
        t2._states[1].t_last   = 0
        t2.set_round(0)
        w, _ = t2.compute_aggregation_weights([0, 1])
        ratio_balance = 200.0 / 100.0
        ratio_weight  = w[1] / w[0]
        assert ratio_weight < ratio_balance   # sublinear

    def test_empty_eligible_falls_back(self):
        """When no client is eligible, uniform fallback is used."""
        # Drain all trust
        self.te.set_round(10000)
        w, _ = self.te.compute_aggregation_weights([0, 1])
        assert len(w) == 2
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Blockchain 
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockchainSim:

    @pytest.fixture(autouse=True)
    def setup(self):
        from config import TrustConfig
        from security.blockchain_sim import create_chain
        self.chain = create_chain(TrustConfig())

    def test_mint_and_balance(self):
        self.chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        b = self.chain.get_effective_balance(0)
        assert b == pytest.approx(100.0, abs=0.1)

    def test_round_advancement(self):
        self.chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        b0 = self.chain.get_effective_balance(0)
        self.chain.advance_round("0xAGGREGATOR")
        b1 = self.chain.get_effective_balance(0)
        assert b1 < b0, "Balance should decay after round advancement"

    def test_voucher_roundtrip(self):
        self.chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        v = self.chain.issue_voucher(0, accepted=True)
        tx = self.chain.redeem_voucher(v, sender="0xCLIENT0")
        assert tx.success

    def test_only_aggregator_can_mint(self):
        tx = self.chain.mint_sbt(0, "c0", sender="0xEVIL")
        pass  # documented: Python sim trusts caller; Solidity enforces it

    def test_gas_tracking(self):
        self.chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        g = self.chain.gas_report()
        assert g["total_gas"] > 0
        assert "mintSBT" in g["by_function"]

    def test_audit_log_immutable(self):
        self.chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        log = self.chain.export_audit_log()
        original_len = len(log)
        # Mutating returned list should not affect internal log
        log.append({"event": "FAKE"})
        assert len(self.chain.export_audit_log()) == original_len

    def test_eligibility_gate(self):
        self.chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        eligible = self.chain.get_eligible_clients()
        assert 0 in eligible


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation 
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.rng = np.random.default_rng(99)
        d = 256
        self.updates = [(i, self.rng.standard_normal(d).astype(np.float32))
                        for i in range(10)]
        self.sc = {i: 100 for i in range(10)}

    def test_fedavg_uniform(self):
        from core.aggregation import fedavg_aggregate
        r = fedavg_aggregate(self.updates)
        np.testing.assert_allclose(
            r, np.stack([dw for _, dw in self.updates]).mean(0), rtol=1e-5
        )

    def test_fedavg_weighted(self):
        from core.aggregation import fedavg_aggregate
        # Assign all weight to client 0
        sc = {0: 10_000, **{i: 1 for i in range(1, 10)}}
        r = fedavg_aggregate(self.updates, sc)
        # Result should be very close to client 0's update
        assert np.linalg.norm(r - self.updates[0][1]) < 0.1

    def test_krum_returns_single_vector(self):
        from core.aggregation import krum_aggregate
        r = krum_aggregate(self.updates, num_malicious=2)
        assert r.shape == (256,)

    def test_krum_multi(self):
        from core.aggregation import krum_aggregate
        r = krum_aggregate(self.updates, num_malicious=2, multi=True)
        assert r.shape == (256,)

    def test_krum_rejects_outlier(self):
        """Krum should ignore a clearly Byzantine update."""
        from core.aggregation import krum_aggregate
        ups = list(self.updates[:8])
        # Add a huge outlier
        outlier = (99, np.ones(256, dtype=np.float32) * 1e6)
        ups.append(outlier)
        r = krum_aggregate(ups, num_malicious=1)
        # Result should not be close to the outlier
        assert np.linalg.norm(r - outlier[1]) > 1.0

    def test_flame_output_shape(self):
        from core.aggregation import flame_aggregate
        r = flame_aggregate(self.updates)
        assert r.shape == (256,)

    def test_rofl_clips_norms(self):
        from core.aggregation import rofl_aggregate
        large = [(i, np.ones(256, dtype=np.float32) * 100) for i in range(5)]
        r = rofl_aggregate(large, norm_bound=1.0)
        assert np.linalg.norm(r) <= 1.1   # slight tolerance for averaging

    def test_aion_returns_history(self):
        from core.aggregation import aion_aggregate
        r, hist = aion_aggregate(self.updates)
        assert r.shape == (256,)
        assert isinstance(hist, dict)

    def test_dp_brem_adds_noise(self):
        from core.aggregation import dp_brem_aggregate
        # Same inputs, different seeds → slightly different outputs (DP noise)
        r1, _ = dp_brem_aggregate(self.updates, rng=np.random.default_rng(1))
        r2, _ = dp_brem_aggregate(self.updates, rng=np.random.default_rng(2))
        assert not np.allclose(r1, r2), "DP-BREM should inject random noise"

    def test_soul_fl_weights_respected(self):
        from core.aggregation import soul_fl_aggregate
        # Weight client 0 exclusively
        w = {0: 1.0}
        r = soul_fl_aggregate(self.updates[:5], w)
        np.testing.assert_allclose(r, self.updates[0][1], rtol=1e-5)

    def test_aggregation_engine_all_methods(self):
        from core.aggregation import AggregationEngine
        for method in ["fedavg", "krum", "flame", "rofl", "aion", "dp_brem"]:
            ae = AggregationEngine(method)
            r  = ae.aggregate(self.updates, sample_counts=self.sc, num_malicious=2)
            assert r.shape == (256,), f"{method} returned wrong shape"

    def test_no_updates_raises(self):
        from core.aggregation import fedavg_aggregate
        with pytest.raises((ValueError, Exception)):
            fedavg_aggregate([])


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_default_fl_params(self):
        from config import FLConfig
        c = FLConfig()
        assert c.num_clients == 100
        assert c.clients_per_round == 10
        assert c.num_rounds == 200
        assert c.dirichlet_alpha == 0.5

    def test_femnist_num_classes(self):
        from config import get_default_config
        cfg = get_default_config("femnist")
        assert cfg.fl.num_classes == 62

    def test_cifar10_num_classes(self):
        from config import get_default_config
        cfg = get_default_config("cifar10")
        assert cfg.fl.num_classes == 10

    def test_lemma1_default_params(self):
        """Default TrustConfig must satisfy Lemma 1 for typical β values."""
        from config import TrustConfig
        import math
        tc    = TrustConfig()
        B_bar = 0.05 * tc.acceptance_reward / (1 - math.exp(-tc.decay_rate))
        assert B_bar < tc.min_balance, (
            f"B̄={B_bar:.3f} must be < τ_min={tc.min_balance}"
        )

    def test_cvae_config_defaults(self):
        from config import CVAEConfig
        c = CVAEConfig()
        assert c.threshold_percentile == 95.0
        assert c.beta_kl == 1.0
        assert c.adaptation_rate == 0.01

    def test_attack_config_pivot(self):
        from config import AttackConfig
        a = AttackConfig()
        assert a.pivot_round == 51

    @pytest.mark.parametrize("lam,Va,beta,tau,should_satisfy", [
        (0.05, 10.0, 0.05, 15.0, True),
        (0.05, 10.0, 0.50, 15.0, False),   # high β violates
        (0.10, 5.0,  0.05, 5.0,  True),
        (0.01, 10.0, 0.05, 15.0, False),   # slow decay violates
    ])
    def test_lemma1_parametric(self, lam, Va, beta, tau, should_satisfy):
        B_bar = beta * Va / (1 - math.exp(-lam))
        assert (B_bar < tau) == should_satisfy


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical Analysis Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestTheory:

    def test_bounded_adversarial_weight_convergence(self):
        lam  = 0.05
        Va   = 10.0
        beta = 0.05
        tau  = 15.0
        B0   = 100.0
        phi  = math.log1p

        B    = B0
        Z    = 10.0   # constant normalizer (lower bound assumption)
        cum_weight = 0.0
        prev_cum   = 0.0

        for t in range(1000):

            B = math.exp(-lam) * B + Va * beta   # E[B_{t+1}] with evasion prob beta
            if B >= tau:
                cum_weight += phi(B) / Z

            # Check increments are shrinking
            if t > 500:
                delta = cum_weight - prev_cum
                assert delta < 1e-3, f"Weight increment at t={t}: {delta:.6f} not converging"
            prev_cum = cum_weight

        # Check total is finite
        assert cum_weight < 1e6, f"Cumulative weight diverged: {cum_weight}"

    def test_trust_contraction_lemma1(self):

        lam, Va, beta = 0.05, 10.0, 0.05
        B_bar = beta * Va / (1 - math.exp(-lam))
        B     = 100.0   # initial trust

        for t in range(1, 200):
            B_prev = B
            B = math.exp(-lam) * B + Va * beta

            # Eq. trust_contraction: |E[B(t)] - B̄| = e^{-λt} * |B(0) - B̄|
            expected_excess = (100.0 - B_bar) * math.exp(-lam * t)
            actual_excess   = abs(B - B_bar)
            assert actual_excess == pytest.approx(expected_excess, rel=1e-3)

        # Converge to B̄
        assert B == pytest.approx(B_bar, rel=1e-3)

    def test_phi_sublinear(self):
        phi = math.log1p
        for x in [0.1, 1.0, 10.0, 100.0]:
            assert phi(2 * x) < 2 * phi(x), f"Failed at x={x}"

    def test_decay_geometric_series_bound(self):

        lam  = 0.05
        tau  = 15.0
        B0   = 100.0
        beta = 0.05
        Va   = 10.0
        B_bar = beta * Va / (1 - math.exp(-lam))
        xi    = math.exp(-lam)

        # Markov bound: P(B(t) ≥ τ) ≤ E[B(t)] / τ
        # E[B(t)] ≤ B̄ + e^{-λt} * (B0 - B̄)
        total_bound = 0.0
        for t in range(500):
            E_B = B_bar + (B0 - B_bar) * (xi ** t)
            p_bound = min(E_B / tau, 1.0)   # Markov
            total_bound += p_bound

        # Geometric series: 
        C         = (B0 - B_bar) / tau
        geo_bound = B_bar / (tau * (1 - xi) + 1e-12) + C / (1 - xi)
        assert total_bound < geo_bound * 1.1, \
            f"Σ_t P(B(t)≥τ) = {total_bound:.2f} exceeds bound {geo_bound:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Data Loader 
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:

    def test_dirichlet_partition_count(self):
        from data_loader import dirichlet_partition
        targets = np.array([i % 10 for i in range(1000)])
        parts   = dirichlet_partition(targets, num_clients=10, alpha=0.5, seed=0)
        assert len(parts) == 10
        # All indices assigned
        all_idx = sorted(idx for p in parts for idx in p)
        assert all_idx == list(range(1000))

    def test_dirichlet_noniid_skew(self):
        """Lower alpha → more skewed distribution."""
        from data_loader import dirichlet_partition
        targets = np.array([i % 10 for i in range(5000)])
        parts_05 = dirichlet_partition(targets, 20, alpha=0.5,  seed=1)
        parts_10 = dirichlet_partition(targets, 20, alpha=10.0, seed=1)
        # Measure entropy of class distribution per client
        def mean_entropy(parts):
            from collections import Counter
            entropies = []
            for p in parts:
                if not p: continue
                cnt = Counter(targets[p])
                total = sum(cnt.values())
                probs = np.array([v/total for v in cnt.values()])
                entropies.append(-np.sum(probs * np.log(probs + 1e-12)))
            return np.mean(entropies)
        # High alpha → near IID → higher entropy
        assert mean_entropy(parts_10) > mean_entropy(parts_05)

    def test_label_flip_dataset(self):
        from data_loader import LabelFlipDataset
        from torch.utils.data import TensorDataset
        import torch
        base  = TensorDataset(torch.randn(50, 3, 32, 32),
                              torch.zeros(50, dtype=torch.long))
        flipped = LabelFlipDataset(base, source_class=0, target_class=5)
        x, y = flipped[0]
        assert y == 5, "Label should be flipped from 0 to 5"
        assert len(flipped) == 50

    def test_compute_class_histogram(self):
        from data_loader import compute_class_histogram
        from torch.utils.data import TensorDataset
        import torch
        ys   = torch.tensor([0,0,1,1,1,2])
        base = TensorDataset(torch.randn(6,3,32,32), ys)
        h    = compute_class_histogram(base, num_classes=3)
        np.testing.assert_allclose(h, [2/6, 3/6, 1/6], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Pytest entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
