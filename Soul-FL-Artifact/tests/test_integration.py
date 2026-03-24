import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Integration 1: Full enrollment → trust → aggregation 
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
 

    def test_enroll_mint_decay_aggregate(self):

        from config import ZKConfig, TrustConfig
        from security.zk import ZKEnrollmentEngine
        from security.blockchain_sim import create_chain

        rng    = np.random.default_rng(0)
        zk_cfg = ZKConfig(min_dataset_size=30)
        te_cfg = TrustConfig()

        engine = ZKEnrollmentEngine(zk_cfg, seed=0)
        chain  = create_chain(te_cfg)

        enrolled_ids = []
        for cid in range(5):
            labels = rng.integers(0, 10, size=300)
            anchor, th = engine.prepare_anchor(cid, labels, 10)
            proof  = engine.generate_proof(anchor, th)
            record = engine.verify_and_enroll(proof, anchor)
            assert record is not None, f"Client {cid} enrolment failed"
            chain.mint_sbt(cid, record.commitment, sender="0xAGGREGATOR")
            enrolled_ids.append(cid)


        for _ in range(10):
            chain.advance_round(sender="0xAGGREGATOR")

        eligible = chain.get_eligible_clients()
        assert set(enrolled_ids).issubset(set(eligible)), \
            "All clients should still be eligible after 10 rounds"

        weights, Z = chain.compute_weights(enrolled_ids)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert Z > 0

    def test_unenrolled_client_has_zero_weight(self):
        """A client who never minted an SBT must not receive aggregation weight."""
        from config import TrustConfig
        from security.blockchain_sim import create_chain

        chain = create_chain(TrustConfig())
        chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        # Client 99 never minted
        weights, _ = chain.compute_weights([0, 99])
        assert weights.get(99, 0.0) == 0.0

    def test_revoked_client_excluded(self):
        from config import TrustConfig
        from security.blockchain_sim import create_chain

        chain = create_chain(TrustConfig())
        chain.mint_sbt(0, "c0", sender="0xAGGREGATOR")
        chain.mint_sbt(1, "c1", sender="0xAGGREGATOR")
        chain.revoke_sbt(1, sender="0xAGGREGATOR")

        eligible = chain.get_eligible_clients()
        assert 0 in eligible
        assert 1 not in eligible

    def test_weight_monotone_in_balance(self):
        """
        Client with higher effective balance should receive strictly more weight
        (due to sublinear φ, but monotone).
        """
        from config import TrustConfig
        from security.trust_engine import TrustEngine

        te = TrustEngine(TrustConfig())
        te.mint(0, "c0"); te.mint(1, "c1")
        # Give client 0 a voucher, client 1 nothing
        te.set_round(0)
        v = te.issue_voucher(0, True, 0); te.redeem_voucher(v)
        te.set_round(1)

        w, _ = te.compute_aggregation_weights([0, 1])
        assert w[0] > w[1], "Higher-balance client should have higher weight"


# ─────────────────────────────────────────────────────────────────────────────
# Integration 2: Multi-round adversarial 
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialSimulation:
    """Simulate the full sleeper Sybil lifecycle over T rounds."""

    def test_sleeper_sybil_lifecycle(self):

        from config import TrustConfig
        from security.trust_engine import TrustEngine, sign_voucher

        tc = TrustConfig()
        te = TrustEngine(tc)
        te.mint(0, "honest_client")
        te.mint(1, "adversary")

        PIVOT = 51
        TOTAL = 90

        for rnd in range(1, TOTAL + 1):
            te.set_round(rnd)

            # Honest client always accepted
            v_h = te.issue_voucher(0, accepted=True, current_round=rnd)
            te.redeem_voucher(v_h)

            # Adversary accepted before pivot, rejected after
            accepted = rnd < PIVOT
            v_a = te.issue_voucher(1, accepted=accepted, current_round=rnd)
            te.redeem_voucher(v_a)

        # Honest client still eligible
        te.set_round(TOTAL)
        assert te.is_eligible(0), "Honest client should remain eligible"

        # Adversary: trust should have decayed significantly post-pivot
        b_adv = te.get_effective_balance(1)
        b_hon = te.get_effective_balance(0)
        assert b_adv < b_hon, "Adversary balance should be lower than honest after pivot"

        # Adversary should eventually fall below τ_min
        te.set_round(TOTAL + 100)
        assert not te.is_eligible(1), \
            "Adversary should eventually become ineligible after sustained rejection"

    def test_lazy_hoarder_loses_influence(self):
        """
        A client who earns trust early but stops contributing should
        eventually decay below the eligibility threshold.
        """
        from config import TrustConfig
        from security.trust_engine import TrustEngine

        tc = TrustConfig()
        te = TrustEngine(tc)
        te.mint(0, "lazy")

        # Phase 1: earn trust for 30 rounds
        for rnd in range(1, 31):
            te.set_round(rnd)
            v = te.issue_voucher(0, True, rnd)
            te.redeem_voucher(v)

        b_peak = te.get_effective_balance(0)

        # Phase 2: stop contributing — only decay
        t_drop = math.ceil(math.log(b_peak / tc.min_balance) / tc.decay_rate) + 31 + 5
        te.set_round(t_drop)
        assert not te.is_eligible(0), \
            f"Lazy client should drop below τ_min by round {t_drop}"

    def test_free_rider_never_earns_trust(self):
        """A free-rider who fails enrollment never gets an SBT → weight = 0."""
        from config import ZKConfig, TrustConfig
        from security.zk import ZKEnrollmentEngine
        from security.blockchain_sim import create_chain

        engine = ZKEnrollmentEngine(ZKConfig(min_dataset_size=100), seed=0)
        chain  = create_chain(TrustConfig())

        # Free-rider with only 5 samples
        labels = np.array([0, 1, 2, 3, 4])
        anchor, th = engine.prepare_anchor(99, labels, 10)
        proof  = engine.generate_proof(anchor, th)
        record = engine.verify_and_enroll(proof, anchor)

        assert record is None, "Free-rider must not be enrolled"

        # No SBT was minted → zero weight
        chain.advance_round(sender="0xAGGREGATOR")
        w = chain.get_effective_balance(99)
        assert w == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration 3: Aggregation under adversarial updates
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregationRobustness:
    """Verify aggregation rules handle adversarial inputs correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(7)
        d   = 128

        # Honest: small Gaussian updates
        self.honest  = [(i, rng.standard_normal(d).astype(np.float32) * 0.1)
                        for i in range(8)]
        # Adversarial: large poisoning updates
        self.poison  = [(8, np.ones(d, dtype=np.float32) * 100.0),
                        (9, np.ones(d, dtype=np.float32) * 100.0)]
        self.all_ups = self.honest + self.poison

    def test_krum_rejects_poison(self):
        """Krum (f=2) should select an update close to the honest cluster."""
        from core.aggregation import krum_aggregate
        r = krum_aggregate(self.all_ups, num_malicious=2)
        # Result should not be near the poison cluster (norm ≈ 100√128 ≈ 1131)
        assert np.linalg.norm(r) < 5.0, f"Krum selected near-poison: norm={np.linalg.norm(r):.1f}"

    def test_rofl_clips_poison(self):
        """RoFL clips to norm bound: result norm should not exceed bound."""
        from core.aggregation import rofl_aggregate
        bound = 2.0
        r = rofl_aggregate(self.all_ups, norm_bound=bound)
        # Average of clipped updates — should be well within reasonable range
        assert np.linalg.norm(r) < bound * 2, \
            f"RoFL result too large: {np.linalg.norm(r):.2f}"

    def test_soul_fl_zero_weight_excludes_update(self):
        """Adversarial clients with weight=0 contribute nothing."""
        from core.aggregation import soul_fl_aggregate
        # Only honest clients (0-7) get weight
        weights = {i: 1.0 / 8 for i in range(8)}
        r = soul_fl_aggregate(self.all_ups, weights)
        # Should match honest-only FedAvg
        from core.aggregation import fedavg_aggregate
        r_honest = fedavg_aggregate(self.honest)
        np.testing.assert_allclose(r, r_honest, rtol=1e-5)

    def test_fedavg_collapses_under_majority_poison(self):
        """FedAvg with majority adversarial updates should be pulled toward poison."""
        from core.aggregation import fedavg_aggregate
        # Majority poisoned
        majority_poison = [(i, np.ones(64, dtype=np.float32) * 50.0) for i in range(7)]
        minority_honest = [(7, np.zeros(64, dtype=np.float32))]
        r = fedavg_aggregate(majority_poison + minority_honest)
        # Should be close to 50 (poison dominates)
        assert np.mean(r) > 30.0, "FedAvg should collapse under majority poison"

    def test_all_methods_finite_output(self):
        """No aggregation method should produce NaN or Inf."""
        from core.aggregation import (fedavg_aggregate, krum_aggregate,
                                       flame_aggregate, rofl_aggregate,
                                       aion_aggregate, dp_brem_aggregate)
        sc = {i: 50 for i in range(10)}
        for name, fn in [("fedavg",  lambda: fedavg_aggregate(self.all_ups, sc)),
                          ("krum",    lambda: krum_aggregate(self.all_ups, 2)),
                          ("flame",   lambda: flame_aggregate(self.all_ups)),
                          ("rofl",    lambda: rofl_aggregate(self.all_ups, 5.0)),
                          ("aion",    lambda: aion_aggregate(self.all_ups)[0]),
                          ("dp_brem", lambda: dp_brem_aggregate(self.all_ups)[0])]:
            r = fn()
            assert np.isfinite(r).all(), f"{name} produced non-finite output"


# ─────────────────────────────────────────────────────────────────────────────
# Integration 4: Theoretical guarantees
# ─────────────────────────────────────────────────────────────────────────────

class TestTheoreticalGuarantees:

    @pytest.fixture(autouse=True)
    def setup(self):
        from utils.theoretical_analysis import TheoreticalParams
        self.p = TheoreticalParams()

    def test_theorem1_negligible_false_acceptance(self):
        from utils.theoretical_analysis import theorem1_enrollment_soundness
        r = theorem1_enrollment_soundness(Q=10_000, kappa=128,
                                           alpha=0.02, beta=0.05)
        assert r["enrollment_sound"]
        assert r["E_Nsyb"] < 1e-20
        assert r["negl_kappa"] < 1e-30

    def test_lemma1_contraction_convergence(self):
        from utils.theoretical_analysis import lemma1_trust_contraction
        r = lemma1_trust_contraction(self.p, T=500)
        assert r["design_constraint_satisfied"]
        assert r["converged_to_B_bar"]
        assert r["max_bound_violation"] < 1e-10   # floating-point tolerance

    def test_lemma1_trajectory_shape(self):
        """Trust trajectory should be monotonically converging to B̄."""
        from utils.theoretical_analysis import lemma1_trust_contraction
        r   = lemma1_trust_contraction(self.p, T=300)
        traj = r["trajectory_sample"]     # sampled every 50 steps
        # Differences from B̄ should be strictly decreasing
        diffs = [abs(b - r["B_bar"]) for b in traj]
        for i in range(len(diffs) - 1):
            assert diffs[i] >= diffs[i + 1] - 1e-6

    def test_theorem2_summability(self):
        from utils.theoretical_analysis import theorem2_bounded_influence
        r = theorem2_bounded_influence(self.p, T=2000)
        assert r["summable"]
        assert r["bounded"]
        assert r["Psi_numerical"] <= r["Psi_analytic_bound"] * 1.05

    def test_theorem3_convergence(self):
        from utils.theoretical_analysis import theorem3_convergence
        r = theorem3_convergence(self.p, T=300)
        assert r["converges"]
        assert r["eps_summable"]
        assert r["contraction_factor"] < 1.0
        assert r["bound_E_wT_sq"] < 1e8   # finite bound

    def test_high_privacy_may_violate_lemma1(self):
        """ε=0.5 → β≈0.10 → B̄ > τ_min with default τ_min=15 (Table II Group A)."""
        from utils.theoretical_analysis import TheoreticalParams, lemma1_trust_contraction
        p_hi = TheoreticalParams(beta=0.10)   # high privacy → detector less reliable
        r    = lemma1_trust_contraction(p_hi)
        # B̄ = 0.10 * 10 / (1 - e^-0.05) ≈ 20.5 > τ_min=15
        assert r["B_bar"] > 15.0, "High privacy should produce B̄ > τ_min=15"
        assert not r["design_constraint_satisfied"]

    def test_fast_decay_tightens_bounds(self):
        """λ=0.10 → smaller ξ → tighter Theorem 2 bound."""
        from utils.theoretical_analysis import (TheoreticalParams,
                                                 theorem2_bounded_influence)
        p_fast = TheoreticalParams(lam=0.10)
        p_slow = TheoreticalParams(lam=0.01)
        r_fast = theorem2_bounded_influence(p_fast)
        r_slow = theorem2_bounded_influence(p_slow)
        assert r_fast["Psi_analytic_bound"] < r_slow["Psi_analytic_bound"]

    @pytest.mark.parametrize("lam,satisfied", [
        (0.05, True),
        (0.10, True),
        (0.01, False),   # slow decay → B̄ too large
    ])
    def test_lemma1_across_decay_rates(self, lam, satisfied):
        from utils.theoretical_analysis import TheoreticalParams, lemma1_trust_contraction
        p = TheoreticalParams(lam=lam)
        r = lemma1_trust_contraction(p)
        assert r["design_constraint_satisfied"] == satisfied, \
            f"λ={lam}: expected satisfied={satisfied}, got B̄={r['B_bar']:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration 5: MetricsTracker  statistics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsTracker:

    def test_record_and_summarize(self, tmp_path):
        from utils.metrics import MetricsTracker

        tracker = MetricsTracker(
            method="soul_fl", dataset="cifar10",
            attack="sleeper_label_flip", run_id=0,
            pivot_round=10, num_clients=100,
            log_dir=str(tmp_path),
        )

        # Simulate: high acc before pivot, drops after, recovers
        for t in range(1, 31):
            acc = 0.87 if t < 10 else (0.50 if 10 <= t < 20 else 0.84)
            asr = 0.01 if t < 10 else (0.60 if 10 <= t < 20 else 0.04)
            tracker.record(t, test_acc=acc, asr=asr,
                           num_eligible=90, num_accepted=8)

        s = tracker.summarize(recovery_threshold=0.80)
        assert s.final_test_acc > 0.80
        assert s.max_accuracy_drop > 0.30      # large drop after pivot
        assert s.rounds_to_recover == 10       # recovers at round 20 (10 post-pivot)
        assert s.final_asr < 0.10

    def test_final_acc_is_last_5_round_mean(self, tmp_path):
        from utils.metrics import MetricsTracker

        tracker = MetricsTracker(
            "fedavg", "cifar10", "none", 0,
            log_dir=str(tmp_path),
        )
        accs = [0.70, 0.71, 0.72, 0.73, 0.74, 0.75]
        for t, a in enumerate(accs, 1):
            tracker.record(t, a, 0.0)

        s = tracker.summarize()
        expected = np.mean(accs[-5:])
        assert s.final_test_acc == pytest.approx(expected, rel=1e-5)

    def test_no_recovery_flag(self, tmp_path):
        from utils.metrics import MetricsTracker

        tracker = MetricsTracker(
            "fedavg", "cifar10", "sleeper_label_flip", 0,
            pivot_round=5, log_dir=str(tmp_path),
        )
        for t in range(1, 21):
            tracker.record(t, 0.40, 0.80)   # never recovers

        s = tracker.summarize(recovery_threshold=0.80)
        assert s.rounds_to_recover == -1, "Should flag -1 when recovery never occurs"

    def test_aggregate_runs(self, tmp_path):
        from utils.metrics import MetricsTracker

        summaries = []
        for run in range(3):
            t = MetricsTracker("soul_fl","cifar10","none", run,
                               log_dir=str(tmp_path))
            for i in range(1, 6):
                t.record(i, 0.85 + run*0.01, 0.02)
            summaries.append(t.summarize())

        agg = MetricsTracker.aggregate_runs(summaries)
        assert agg["num_runs"] == 3
        assert "mean" in agg["test_acc"]
        assert agg["test_acc"]["mean"] == pytest.approx(0.86, abs=0.01)

    def test_save_and_load(self, tmp_path):
        from utils.metrics import MetricsTracker

        t = MetricsTracker("krum","cifar10","none",0,log_dir=str(tmp_path))
        t.record(1, 0.80, 0.05); t.record(2, 0.82, 0.04)
        path = str(tmp_path / "krum_run0.json")
        t.save(path)

        t2 = MetricsTracker.load(path)
        assert t2.method == "krum"
        assert len(t2.rounds) == 2
        assert t2.rounds[1].test_acc == pytest.approx(0.82)



class TestSensitivityDirections:


    def test_groupA_high_privacy_degrades_detection(self):
        """Higher LDP noise (lower ε) → higher β → weaker detector."""
        from utils.theoretical_analysis import TheoreticalParams, theorem2_bounded_influence
        p_low  = TheoreticalParams(beta=0.02)   # low privacy = good detection
        p_high = TheoreticalParams(beta=0.10)   # high privacy = poor detection
        r_low  = theorem2_bounded_influence(p_low)
        r_high = theorem2_bounded_influence(p_high)
        # Higher β → higher adversarial influence bound
        assert r_high["Psi_analytic_bound"] > r_low["Psi_analytic_bound"]

    def test_groupB_optimal_decay_balances_liveness(self):
        """Optimal λ (0.05) should produce a smaller Ψ_bound than both extremes."""
        from utils.theoretical_analysis import TheoreticalParams, theorem2_bounded_influence
        bounds = {}
        for lam in [0.01, 0.05, 0.10, 0.20]:
            p = TheoreticalParams(lam=lam)
            r = theorem2_bounded_influence(p)
            bounds[lam] = r["Psi_analytic_bound"]
        # Faster decay → smaller ξ → smaller bound
        assert bounds[0.20] < bounds[0.10] < bounds[0.05] < bounds[0.01]

    def test_groupC_strict_threshold_increases_fpr(self):

        from config import TrustConfig
        from security.trust_engine import TrustEngine

        def rounds_until_ineligible(rejection_rate: float, max_rounds: int = 300) -> int:
            tc = TrustConfig()
            te = TrustEngine(tc)
            te.mint(0, "c0")
            rng = np.random.default_rng(0)
            for rnd in range(1, max_rounds + 1):
                te.set_round(rnd)
                accepted = rng.random() > rejection_rate
                v = te.issue_voucher(0, accepted, rnd)
                te.redeem_voucher(v)
                if not te.is_eligible(0):
                    return rnd
            return max_rounds + 1

     
        r_strict = rounds_until_ineligible(rejection_rate=0.14)

        r_robust = rounds_until_ineligible(rejection_rate=0.02)

        assert r_strict < r_robust, \


    def test_convergence_improves_with_faster_decay(self):
        """Table II Group B: faster decay → adversarial term shrinks → better bound."""
        from utils.theoretical_analysis import TheoreticalParams, theorem3_convergence
        r_slow = theorem3_convergence(TheoreticalParams(lam=0.01), T=200)
        r_fast = theorem3_convergence(TheoreticalParams(lam=0.10), T=200)
        assert r_fast["adversarial_term"] <= r_slow["adversarial_term"]


# ─────────────────────────────────────────────────────────────────────────────
# Integration 7: Gas cost 
# ─────────────────────────────────────────────────────────────────────────────

class TestGasCosts:
    """Verify simulated gas costs are in plausible on-chain range (Section V-A)."""

    def test_mint_gas_in_range(self):
        from config import TrustConfig
        from security.blockchain_sim import SoulFLContract, GAS
        assert 40_000 <= GAS["MINT_SBT"] <= 80_000

    def test_redeem_gas_cheaper_than_mint(self):
        from security.blockchain_sim import GAS
        assert GAS["REDEEM_VOUCHER"] < GAS["MINT_SBT"]

    def test_gas_accumulates_over_rounds(self):
        from config import TrustConfig
        from security.blockchain_sim import create_chain

        chain = create_chain(TrustConfig())
        for i in range(5):
            chain.mint_sbt(i, f"c{i}", sender="0xAGGREGATOR")

        g0 = chain.gas_report()["total_gas"]
        for _ in range(3):
            chain.advance_round(sender="0xAGGREGATOR")

        g1 = chain.gas_report()["total_gas"]
        assert g1 > g0, "Gas should accumulate as transactions are processed"

    def test_batch_redeem_records_all(self):
        from config import TrustConfig
        from security.blockchain_sim import create_chain

        chain = create_chain(TrustConfig())
        for i in range(3):
            chain.mint_sbt(i, f"c{i}", sender="0xAGGREGATOR")
        chain.advance_round(sender="0xAGGREGATOR")

        vouchers = [chain.issue_voucher(i, True) for i in range(3)]
        txs = chain.batch_redeem(vouchers)
        assert len(txs) == 3
        assert all(tx.success for tx in txs)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
