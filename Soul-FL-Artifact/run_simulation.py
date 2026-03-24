
import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch

from config import SoulFLConfig, get_default_config
from core.client import build_clients
from core.server import SoulFLServer
from data_loader import (
    federated_data_factory,
    LabelFlipDataset,
    BackdoorDataset,
    compute_class_histogram,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("soul_fl.main")


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def make_poisoned_datasets(client_datasets, cfg: SoulFLConfig):
    """Build per-client poisoned datasets for adversarial clients."""
    poisoned = {}
    for i, ds in enumerate(client_datasets):
        if cfg.attack.attack_type in ("sleeper_label_flip", "adaptive_manifold"):
            poisoned[i] = LabelFlipDataset(
                ds,
                source_class=cfg.attack.source_class,
                target_class=cfg.attack.target_class,
            )
        elif cfg.attack.attack_type == "backdoor":
            poisoned[i] = BackdoorDataset(
                ds,
                source_class=cfg.attack.source_class,
                target_class=cfg.attack.target_class,
                trigger_size=cfg.attack.backdoor_trigger_size,
            )
        else:
            poisoned[i] = ds
    return poisoned


# ---------------------------------------------------------------------------

def run_experiment(cfg: SoulFLConfig, method: str, run_id: int = 0) -> dict:
    set_seed(cfg.fl.seed + run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "=== Experiment | method=%s | dataset=%s | attack=%s | run=%d ===",
        method, cfg.fl.dataset, cfg.attack.attack_type, run_id,
    )

    # Data
    client_datasets, test_dataset = federated_data_factory(cfg)
    poisoned_datasets = make_poisoned_datasets(client_datasets, cfg)

    # Malicious client selection
    num_malicious = int(cfg.fl.num_clients * cfg.attack.malicious_fraction)
    all_ids = list(range(cfg.fl.num_clients))
    malicious_ids = set(random.sample(all_ids, num_malicious))
    logger.info("Malicious clients (%d): %s", num_malicious, sorted(malicious_ids)[:10])

    # Build clients
    clients = build_clients(
        client_datasets,
        poisoned_datasets,
        malicious_ids,
        cfg,
        device,
    )

    # Add Sybil clients if configured
    if cfg.attack.num_sybils > 0:
        from core.client import FreeRiderSybilClient
        for k in range(cfg.attack.num_sybils):
            sybil_id = cfg.fl.num_clients + k
            # Sybils get tiny random dataset (< τ_size → blocked by ZK)
            tiny_ds = _make_tiny_dataset(1, cfg.fl.num_classes)
            sybil = FreeRiderSybilClient(sybil_id, tiny_ds, cfg.fl, device)
            clients.append(sybil)

    # Server
    server = SoulFLServer(
        cfg=cfg,
        clients=clients,
        test_dataset=test_dataset,
        malicious_ids=malicious_ids,
        method=method,
        device=device,
    )

    # Train
    history = server.train()

    # 
    os.makedirs(cfg.exp.output_dir, exist_ok=True)
    result_path = os.path.join(
        cfg.exp.output_dir,
        f"{method}_{cfg.fl.dataset}_{cfg.attack.attack_type}_run{run_id}.json",
    )
    result = {
        "method": method,
        "dataset": cfg.fl.dataset,
        "attack": cfg.attack.attack_type,
        "run_id": run_id,
        "final_test_acc": history["test_acc"][-1] if history["test_acc"] else 0,
        "final_asr": history["asr"][-1] if history["asr"] else 0,
        "max_acc_drop": (
            max(history["test_acc"][:cfg.attack.pivot_round - 1] or [0])
            - min(history["test_acc"][cfg.attack.pivot_round:] or [0])
        ),
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved: %s", result_path)

    return result


def _make_tiny_dataset(size: int, num_classes: int):
    from torch.utils.data import TensorDataset
    x = torch.randn(size, 3, 32, 32)
    y = torch.zeros(size, dtype=torch.long)
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def run_multi(cfg, method, num_runs=5):
    results = [run_experiment(deepcopy(cfg), method, run_id=i) for i in range(num_runs)]
    accs = [r["final_test_acc"] for r in results]
    asrs = [r["final_asr"] for r in results]
    logger.info(
        "method=%s | acc=%.2f±%.2f%% | ASR=%.2f±%.2f%%",
        method,
        np.mean(accs) * 100, np.std(accs) * 100,
        np.mean(asrs) * 100, np.std(asrs) * 100,
    )
    return results


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
BASELINE_METHODS = ["soul_fl", "fedavg", "krum", "flame", "rofl", "aion", "dp_brem"]


def sweep_baselines(dataset: str = "cifar10", attack: str = "sleeper_label_flip"):
    """Reproduce Table / Figure comparisons against all baselines."""
    cfg = get_default_config(dataset, attack)
    summary = {}
    for method in BASELINE_METHODS:
        res = run_multi(cfg, method, num_runs=cfg.fl.num_runs)
        accs = [r["final_test_acc"] for r in res]
        asrs = [r["final_asr"] for r in res]
        summary[method] = {
            "acc_mean": np.mean(accs),
            "acc_std":  np.std(accs),
            "asr_mean": np.mean(asrs),
            "asr_std":  np.std(asrs),
        }
    _print_summary_table(summary)
    return summary


def sybil_sweep(dataset: str = "cifar10"):
    """Fig. 1c: Accuracy vs number of Sybil identities K."""
    sybil_counts = [0, 10, 20, 50, 100]
    for method in ["soul_fl", "fedavg", "flame"]:
        for K in sybil_counts:
            cfg = get_default_config(dataset, "free_rider_sybil")
            cfg.attack.num_sybils = K
            run_experiment(cfg, method)


def noniid_sweep(dataset: str = "cifar10"):
    """Fig. 3: Robustness under increasing data heterogeneity."""
    alphas = [1.0, 0.5, 0.3]
    for alpha in alphas:
        cfg = get_default_config(dataset, "sleeper_label_flip")
        cfg.fl.dirichlet_alpha = alpha
        for method in ["soul_fl", "fedavg", "aion", "dp_brem"]:
            run_experiment(cfg, method)


def ablation_study(dataset: str = "cifar10"):
    """Table I: Component synergy ablation."""
    variants = {
        "soul_fl_full":      {"zk": True,  "fp": True,  "decay": True},
        "soul_fl_no_zk":     {"zk": False, "fp": True,  "decay": True},
        "soul_fl_no_fp":     {"zk": True,  "fp": False, "decay": True},
        "soul_fl_no_decay":  {"zk": True,  "fp": True,  "decay": False},
    }
    for name, flags in variants.items():
        cfg = get_default_config(dataset, "sleeper_label_flip")
        if not flags["decay"]:
            cfg.trust.decay_rate = 1e-9   # effectively no decay
        if not flags["fp"]:
            cfg.cvae.cvae_epochs_warmup = 9999  # disable fingerprinting
        # Note: disabling ZK enrollment would allow free-riders through;
        # simulated here by setting min_dataset_size = 0
        if not flags["zk"]:
            cfg.zk.min_dataset_size = 0

        run_experiment(cfg, "soul_fl")


def sensitivity_analysis(dataset: str = "cifar10"):
    """Table II: Sensitivity of security-critical parameters."""
    # Group A: LDP epsilon
    for eps in [0.5, 1.0, 4.0]:
        cfg = get_default_config(dataset, "sleeper_label_flip")
        cfg.zk.ldp_epsilon = eps
        run_experiment(cfg, "soul_fl")

    # Group B: Decay rate
    for lam in [0.01, 0.05, 0.10]:
        cfg = get_default_config(dataset, "sleeper_label_flip")
        cfg.trust.decay_rate = lam
        run_experiment(cfg, "soul_fl")

    # Group C: C-VAE threshold percentile
    for pct in [90.0, 95.0, 99.0]:
        cfg = get_default_config(dataset, "sleeper_label_flip")
        cfg.cvae.threshold_percentile = pct
        run_experiment(cfg, "soul_fl")


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def _print_summary_table(summary: Dict):
    header = f"{'Method':<12} | {'Acc (%)':>10} | {'ASR (%)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for method, v in summary.items():
        print(
            f"{method:<12} | "
            f"{v['acc_mean']*100:>7.2f}±{v['acc_std']*100:.2f} | "
            f"{v['asr_mean']*100:>7.2f}±{v['asr_std']*100:.2f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Soul-FL Simulation Runner")
    p.add_argument("--dataset",  default="cifar10",
                   choices=["cifar10", "femnist"], help="FL dataset")
    p.add_argument("--attack",   default="sleeper_label_flip",
                   choices=["none", "sleeper_label_flip", "free_rider_sybil",
                             "lazy_hoard", "adaptive_manifold"],
                   help="Attack type")
    p.add_argument("--method",   default="soul_fl",
                   choices=BASELINE_METHODS, help="Aggregation method")
    p.add_argument("--malicious_fraction", type=float, default=0.20)
    p.add_argument("--pivot_round",        type=int,   default=51)
    p.add_argument("--num_rounds",         type=int,   default=200)
    p.add_argument("--num_clients",        type=int,   default=100)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--num_runs",           type=int,   default=1)

    # Experiment modes
    p.add_argument("--sweep_baselines", action="store_true")
    p.add_argument("--sybil_sweep",     action="store_true")
    p.add_argument("--noniid_sweep",    action="store_true")
    p.add_argument("--ablation",        action="store_true")
    p.add_argument("--sensitivity",     action="store_true")
    p.add_argument("--output_dir",      default="experiments")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_default_config(args.dataset, args.attack)
    cfg.fl.num_rounds         = args.num_rounds
    cfg.fl.num_clients        = args.num_clients
    cfg.fl.seed               = args.seed
    cfg.attack.malicious_fraction = args.malicious_fraction
    cfg.attack.pivot_round    = args.pivot_round
    cfg.exp.output_dir        = args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep_baselines:
        sweep_baselines(args.dataset, args.attack)
    elif args.sybil_sweep:
        sybil_sweep(args.dataset)
    elif args.noniid_sweep:
        noniid_sweep(args.dataset)
    elif args.ablation:
        ablation_study(args.dataset)
    elif args.sensitivity:
        sensitivity_analysis(args.dataset)
    else:
        if args.num_runs > 1:
            run_multi(cfg, args.method, args.num_runs)
        else:
            run_experiment(cfg, args.method, run_id=0)


if __name__ == "__main__":
    main()
