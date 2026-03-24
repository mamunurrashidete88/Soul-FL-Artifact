

import json
import os
from typing import Dict, List, Optional

import numpy as np

# Matplotlib is optional — used only when plot() is called
try:
    import matplotlib
    matplotlib.use("Agg")   # headless / non-interactive
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Style constants 
# ─────────────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "soul_fl":  "#1f77b4",
    "fedavg":   "#d62728",
    "krum":     "#ff7f0e",
    "flame":    "#2ca02c",
    "rofl":     "#9467bd",
    "aion":     "#8c564b",
    "dp_brem":  "#e377c2",
}
METHOD_LABELS = {
    "soul_fl":  "Soul-FL",
    "fedavg":   "FedAvg",
    "krum":     "Krum",
    "flame":    "FLAME",
    "rofl":     "RoFL",
    "aion":     "Aion",
    "dp_brem":  "DP-BREM",
}
METHOD_STYLES = {
    "soul_fl":  ("solid",  "o", 2.0),
    "fedavg":   ("dashed", "s", 1.5),
    "krum":     ("dotted", "^", 1.5),
    "flame":    ("dashdot","D", 1.5),
    "rofl":     ("dashed", "v", 1.5),
    "aion":     ("dotted", "P", 1.5),
    "dp_brem":  ("dashed", "X", 1.5),
}

PIVOT_COLOR     = "#666666"
PIVOT_STYLE     = "--"
FONT_SIZE       = 10
TITLE_SIZE      = 11
LEGEND_SIZE     = 8
FIGURE_DPI      = 150


def _require_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for plotting. Install via: pip install matplotlib")


def _setup_axes(ax, xlabel: str = "Round", ylabel: str = ""):
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading 
# ─────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: str, method: str = None, dataset: str = None,
                 attack: str = None) -> List[dict]:
    """Load all JSON result files matching optional filters."""
    results = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        if method  and not fname.startswith(method):  continue
        if dataset and dataset not in fname:           continue
        if attack  and attack  not in fname:           continue
        try:
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


def _smooth(values: List[float], window: int = 3) -> np.ndarray:
    """Simple moving-average smoothing for cleaner plots."""
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_recovery(
    results_by_method: Dict[str, dict],
    pivot_round: int = 51,
    dataset: str = "CIFAR-10",
    ax=None,
    smooth: int = 3,
):

    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for method, res in results_by_method.items():
        hist   = res.get("history", {})
        rounds = hist.get("round",    [])
        accs   = hist.get("test_acc", [])
        if not rounds: continue

        ls, mk, lw = METHOD_STYLES.get(method, ("solid", "o", 1.5))
        ax.plot(
            rounds, _smooth(accs, smooth) * 100,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            linestyle=ls, linewidth=lw,
            marker=mk, markersize=0,    # markers would clutter per-round plots
        )

    ax.axvline(x=pivot_round, color=PIVOT_COLOR, linestyle=PIVOT_STYLE,
               linewidth=1.2, label=f"Pivot (Rnd {pivot_round})")
    ax.set_ylim(0, 105)
    _setup_axes(ax, "Round", "Test Accuracy (%)")
    ax.set_title(f"Recovery After Sleeper Pivot — {dataset}", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2, framealpha=0.8)
    return ax


def plot_accuracy_vs_malicious_fraction(
    results_by_fraction: Dict[float, Dict[str, float]],
    dataset: str = "CIFAR-10",
    ax=None,
):
    """Fig. 1(b/e): Final accuracy vs fraction of malicious clients."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    fracs = sorted(results_by_fraction.keys())
    for method in METHOD_COLORS:
        accs = [results_by_fraction[f].get(method, None) for f in fracs]
        if all(a is None for a in accs): continue
        ls, mk, lw = METHOD_STYLES.get(method, ("solid","o",1.5))
        y = [a * 100 if a is not None else np.nan for a in accs]
        ax.plot(
            [f * 100 for f in fracs], y,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            linestyle=ls, linewidth=lw, marker=mk, markersize=5,
        )

    ax.set_ylim(0, 105)
    _setup_axes(ax, "Malicious Fraction (%)", "Test Accuracy (%)")
    ax.set_title(f"Robustness vs Malicious Fraction — {dataset}", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    return ax


def plot_accuracy_vs_sybils(
    results_by_k: Dict[int, Dict[str, float]],
    dataset: str = "CIFAR-10",
    ax=None,
):
    """Fig. 1(c/f): Accuracy vs number of Sybil identities K."""
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    ks = sorted(results_by_k.keys())
    for method in METHOD_COLORS:
        accs = [results_by_k[k].get(method, None) for k in ks]
        if all(a is None for a in accs): continue
        ls, mk, lw = METHOD_STYLES.get(method, ("solid","o",1.5))
        y = [a * 100 if a is not None else np.nan for a in accs]
        ax.plot(ks, y,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "gray"),
                linestyle=ls, linewidth=lw, marker=mk, markersize=5)

    ax.set_ylim(0, 105)
    _setup_axes(ax, "Sybil Count K", "Test Accuracy (%)")
    ax.set_title(f"Invariance to Sybil Swarms — {dataset}", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def plot_asr_over_rounds(
    results_by_method: Dict[str, dict],
    pivot_round: int = 51,
    dataset: str = "CIFAR-10",
    ax=None,
    smooth: int = 3,
):
 
    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    for method, res in results_by_method.items():
        hist   = res.get("history", {})
        rounds = hist.get("round", [])
        asrs   = hist.get("asr",   [])
        if not rounds: continue

        ls, mk, lw = METHOD_STYLES.get(method, ("solid","o",1.5))
        ax.plot(
            rounds, _smooth(asrs, smooth) * 100,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "gray"),
            linestyle=ls, linewidth=lw,
        )

    ax.axvline(x=pivot_round, color=PIVOT_COLOR, linestyle=PIVOT_STYLE,
               linewidth=1.2, label=f"Pivot (Rnd {pivot_round})")
    ax.set_ylim(-2, 105)
    ax.axhline(y=5, color="#999999", linestyle=":", linewidth=0.8)
    _setup_axes(ax, "Round", "Attack Success Rate (%)")
    ax.set_title(f"ASR — Sleeper Attack — {dataset}", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def plot_noniid_stress(
    results_by_alpha: Dict[float, Dict[str, dict]],
    dataset: str = "CIFAR-10",
):

    _require_mpl()
    alphas = sorted(results_by_alpha.keys(), reverse=True)
    fig, axes = plt.subplots(2, len(alphas), figsize=(14, 7),
                             sharex=True, sharey="row")
    fig.suptitle(f"Non-IID Stress Test — {dataset}", fontsize=TITLE_SIZE + 1, y=1.01)

    for col, alpha in enumerate(alphas):
        methods_data = results_by_alpha.get(alpha, {})
        for method, res in methods_data.items():
            hist   = res.get("history", {})
            rounds = hist.get("round",    [])
            asrs   = hist.get("asr",      [])
            accs   = hist.get("test_acc", [])
            ls, _, lw = METHOD_STYLES.get(method, ("solid","o",1.5))
            c = METHOD_COLORS.get(method, "gray")
            lbl = METHOD_LABELS.get(method, method)
            if rounds:
                axes[0, col].plot(rounds, _smooth(asrs, 3)*100, label=lbl,
                                  color=c, linestyle=ls, linewidth=lw)
                axes[1, col].plot(rounds, _smooth(accs, 3)*100, label=lbl,
                                  color=c, linestyle=ls, linewidth=lw)

        for row in range(2):
            _setup_axes(axes[row, col],
                        "Round",
                        "ASR (%)" if row == 0 else "Test Accuracy (%)")
            axes[row, col].set_title(f"α={alpha}", fontsize=TITLE_SIZE)
            axes[row, col].set_ylim(-2, 105)
            if col == 0:
                axes[row, col].legend(fontsize=LEGEND_SIZE - 1)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def plot_trust_dynamics(
    history: Dict[str, list],
    pivot_round: int = 51,
    ax=None,
):
    """Plot mean_trust and num_eligible over training rounds (Soul-FL only)."""
    _require_mpl()
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    else:
        axes = [ax] * 2

    rounds = history.get("round", [])
    trust  = history.get("mean_trust", [])
    elig   = history.get("num_eligible", [])

    if rounds:
        axes[0].plot(rounds, trust, color=METHOD_COLORS["soul_fl"], linewidth=1.8)
        axes[0].axvline(pivot_round, color=PIVOT_COLOR, linestyle=PIVOT_STYLE)
        _setup_axes(axes[0], "", "Mean Trust Balance")
        axes[0].set_title("Soul-FL Trust Dynamics", fontsize=TITLE_SIZE)

        axes[1].plot(rounds, elig, color="#2ca02c", linewidth=1.8)
        axes[1].axvline(pivot_round, color=PIVOT_COLOR, linestyle=PIVOT_STYLE,
                        label=f"Pivot (Rnd {pivot_round})")
        _setup_axes(axes[1], "Round", "Eligible Clients")
        axes[1].legend(fontsize=LEGEND_SIZE)

    return axes


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_bar(
    group_data: Dict[str, List[dict]],
    metric: str = "test_acc",
    ylabel: str = "Test Accuracy (%)",
    title: str = "Sensitivity Analysis",
    ax=None,
):

    _require_mpl()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))

    all_bars  = [(g, b) for g, bars in group_data.items() for b in bars]
    x         = np.arange(len(all_bars))
    means     = np.array([b["mean"] for _, b in all_bars]) * 100
    stds      = np.array([b["std"]  for _, b in all_bars]) * 100
    labels    = [b["label"] for _, b in all_bars]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#1f77b4",
                  edgecolor="black", linewidth=0.7, alpha=0.85, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right",
                                          fontsize=FONT_SIZE - 1)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Group separators
    offsets = [0]
    for g, bars_list in group_data.items():
        offsets.append(offsets[-1] + len(bars_list))
    for sep in offsets[1:-1]:
        ax.axvline(x=sep - 0.5, color="gray", linestyle="--", linewidth=0.8)

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def make_all_figures(results_dir: str, output_dir: str = "figures") -> None:

    _require_mpl()
    os.makedirs(output_dir, exist_ok=True)

    for dataset, dataset_label in [("cifar10", "CIFAR-10"), ("femnist", "FEMNIST")]:
        results = load_results(results_dir, dataset=dataset)
        if not results:
            print(f"No results found for {dataset} in {results_dir}")
            continue

        by_method = {r["method"]: r for r in results}

        # Figure 1: Accuracy
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_accuracy_recovery(by_method, ax=axes[0], dataset=dataset_label)
        axes[1].set_title("Accuracy vs Malicious Fraction (load from sweep)")
        axes[2].set_title("Accuracy vs Sybil Count K (load from sweep)")
        fig.tight_layout()
        _save_fig(fig, output_dir, f"fig1_accuracy_{dataset}")

        # Figure 2: ASR
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_asr_over_rounds(by_method, ax=axes[0], dataset=dataset_label)
        axes[1].set_title("ASR — Manifold Attack (load from sweep)")
        axes[2].set_title("ASR vs Malicious Fraction (load from sweep)")
        fig.tight_layout()
        _save_fig(fig, output_dir, f"fig2_asr_{dataset}")

        # Trust dynamics (Soul-FL only)
        sf_results = [r for r in results if r.get("method") == "soul_fl"]
        if sf_results:
            fig, _ = plt.subplots(2, 1, figsize=(6, 5))
            plot_trust_dynamics(sf_results[0].get("history", {}))
            fig.tight_layout()
            _save_fig(fig, output_dir, f"trust_dynamics_{dataset}")

    print(f"Figures saved to {output_dir}/")


def _save_fig(fig, output_dir: str, name: str) -> None:
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Soul-FL Plotting Utility")
    p.add_argument("--results_dir", default="experiments",
                   help="Directory containing experiment JSON files")
    p.add_argument("--output_dir",  default="figures",
                   help="Directory to save generated figures")
    args = p.parse_args()
    make_all_figures(args.results_dir, args.output_dir)
