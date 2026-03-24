import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

---------------------------------------------------------------------
@dataclass
class RoundMetrics:
    round_idx: int
    test_acc: float
    asr: float
    num_eligible: int = 0
    num_accepted: int = 0
    mean_trust: float = 0.0
    cvae_threshold: float = 0.0
    train_loss: float = 0.0
    wall_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Experiment 
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSummary:
    method: str
    dataset: str
    attack: str
    run_id: int
    num_rounds: int

    # Primary metrics (last 5 rounds)
    final_test_acc: float = 0.0
    final_asr: float = 0.0

    # Recovery analysis (pivot at T_switch)
    max_acc_before_pivot: float = 0.0
    min_acc_after_pivot: float = 1.0
    max_accuracy_drop: float = 0.0    # Table I: Max Drop
    rounds_to_recover: int = -1       # Table I: Recov. Rnds  (-1 = did not recover)
    recovery_threshold: float = 0.80  # 80% accuracy = "recovered"

    # Trust system stats
    mean_eligible_frac: float = 0.0
    mean_accept_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics 
# ---------------------------------------------------------------------------
class MetricsTracker:
    

    def __init__(
        self,
        method: str,
        dataset: str,
        attack: str,
        run_id: int = 0,
        pivot_round: int = 51,
        num_clients: int = 100,
        log_dir: str = "experiments/logs",
    ):
        self.method = method
        self.dataset = dataset
        self.attack = attack
        self.run_id = run_id
        self.pivot_round = pivot_round
        self.num_clients = num_clients

        self.rounds: List[RoundMetrics] = []
        self._start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(
            log_dir,
            f"{method}_{dataset}_{attack}_run{run_id}.jsonl",
        )

    # ------------------------------------------------------------------
    def record(
        self,
        round_idx: int,
        test_acc: float,
        asr: float,
        num_eligible: int = 0,
        num_accepted: int = 0,
        mean_trust: float = 0.0,
        cvae_threshold: float = 0.0,
        train_loss: float = 0.0,
    ) -> RoundMetrics:
        m = RoundMetrics(
            round_idx=round_idx,
            test_acc=test_acc,
            asr=asr,
            num_eligible=num_eligible,
            num_accepted=num_accepted,
            mean_trust=mean_trust,
            cvae_threshold=cvae_threshold,
            train_loss=train_loss,
        )
        self.rounds.append(m)

        # Append to JSONL log
        with open(self._log_path, "a") as f:
            f.write(json.dumps(asdict(m)) + "\n")

        return m

    # ------------------------------------------------------------------
    #   
    # ------------------------------------------------------------------
    def summarize(self, recovery_threshold: float = 0.80) -> ExperimentSummary:
        if not self.rounds:
            return ExperimentSummary(
                self.method, self.dataset, self.attack, self.run_id,
                num_rounds=0,
            )

        accs  = [m.test_acc  for m in self.rounds]
        asrs  = [m.asr       for m in self.rounds]
        n     = len(self.rounds)

        # Split at pivot
        pre  = [accs[i] for i, m in enumerate(self.rounds) if m.round_idx < self.pivot_round]
        post = [accs[i] for i, m in enumerate(self.rounds) if m.round_idx >= self.pivot_round]

        final_acc  = float(np.mean(accs[-5:])) if n >= 5 else accs[-1]
        final_asr  = float(np.mean(asrs[-5:])) if n >= 5 else asrs[-1]

        max_pre   = max(pre)  if pre  else final_acc
        min_post  = min(post) if post else final_acc
        drop      = max(0.0, max_pre - min_post)

        # Rounds to recover: 
        rtr = -1
        if post:
            post_rounds = [(m.round_idx, m.test_acc)
                           for m in self.rounds if m.round_idx >= self.pivot_round]
            for r, a in post_rounds:
                if a >= recovery_threshold:
                    rtr = r - self.pivot_round
                    break

        # Trust stats
        elig = [m.num_eligible for m in self.rounds]
        acc_rate = [
            m.num_accepted / max(m.num_eligible, 1)
            for m in self.rounds if m.num_eligible > 0
        ]

        summary = ExperimentSummary(
            method=self.method,
            dataset=self.dataset,
            attack=self.attack,
            run_id=self.run_id,
            num_rounds=n,
            final_test_acc=final_acc,
            final_asr=final_asr,
            max_acc_before_pivot=max_pre,
            min_acc_after_pivot=min_post,
            max_accuracy_drop=drop,
            rounds_to_recover=rtr,
            recovery_threshold=recovery_threshold,
            mean_eligible_frac=float(np.mean(elig)) / self.num_clients if elig else 0.0,
            mean_accept_rate=float(np.mean(acc_rate)) if acc_rate else 0.0,
        )
        return summary

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_runs(summaries: List[ExperimentSummary]) -> dict:
        """Compute mean ± std over multiple runs (Table I / II format)."""
        if not summaries:
            return {}

        def stats(vals: List[float]) -> dict:
            arr = np.array(vals)
            return {"mean": float(arr.mean()), "std": float(arr.std())}

        return {
            "method":            summaries[0].method,
            "dataset":           summaries[0].dataset,
            "attack":            summaries[0].attack,
            "num_runs":          len(summaries),
            "test_acc":          stats([s.final_test_acc   for s in summaries]),
            "asr":               stats([s.final_asr         for s in summaries]),
            "max_drop":          stats([s.max_accuracy_drop for s in summaries]),
            "rounds_to_recover": stats([s.rounds_to_recover for s in summaries
                                        if s.rounds_to_recover >= 0]),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "method":   self.method,
            "dataset":  self.dataset,
            "attack":   self.attack,
            "run_id":   self.run_id,
            "rounds":   [asdict(m) for m in self.rounds],
            "summary":  asdict(self.summarize()),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug("Metrics saved: %s", path)

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        with open(path) as f:
            d = json.load(f)
        tracker = cls(
            method=d["method"], dataset=d["dataset"],
            attack=d["attack"], run_id=d["run_id"],
        )
        for r in d["rounds"]:
            tracker.rounds.append(RoundMetrics(**r))
        return tracker

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------
    @staticmethod
    def print_comparison_table(results: List[dict]) -> None:
        """Print a Table-I-style comparison."""
        header = (
            f"{'Method':<12} | {'Acc (%)':>12} | {'ASR (%)':>12} | "
            f"{'Max Drop':>10} | {'Recov Rnds':>12}"
        )
        sep = "─" * len(header)
        print("\n" + sep)
        print(header)
        print(sep)
        for r in results:
            acc  = r.get("test_acc",  {})
            asr  = r.get("asr",       {})
            drop = r.get("max_drop",  {})
            rtr  = r.get("rounds_to_recover", {})
            def fmt(d):
                if not d: return "N/A"
                return f"{d['mean']*100:.1f}±{d['std']*100:.1f}"
            def fmt_raw(d):
                if not d: return "N/A"
                m = d.get('mean', -1)
                return "Failed" if m < 0 else f"{m:.0f}±{d.get('std', 0):.0f}"
            print(
                f"{r['method']:<12} | {fmt(acc):>12} | {fmt(asr):>12} | "
                f"{fmt(drop):>10} | {fmt_raw(rtr):>12}"
            )
        print(sep + "\n")
