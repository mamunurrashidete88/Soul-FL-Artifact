from dataclasses import dataclass, field
from typing import Literal, List


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class FLConfig:
    # Participants
    num_clients: int = 100          # N  total clients
    clients_per_round: int = 10     # n  sampled per round
    num_rounds: int = 200           # T  global rounds

    # Data
    dataset: Literal["cifar10", "femnist"] = "cifar10"
    data_dir: str = "data"
    dirichlet_alpha: float = 0.5    # Non-IID concentration (0.5 = moderate, 0.3 = extreme)
    num_classes: int = 10

    # Local training (Section V-A)
    local_epochs: int = 5
    local_lr: float = 0.01
    local_batch_size: int = 32
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Reproducibility
    seed: int = 42
    num_runs: int = 5


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class TrustConfig:
    decay_rate: float = 0.05        # 
    initial_balance: float = 100.0  # 

    min_balance: float = 15.0       # 
    acceptance_reward: float = 10.0 # 

    squash_fn: str = "log1p"

    # Blockchain simulation
    chain_id: int = 31337           # Hardhat local chain
    sbt_address: str = "0xSBTADDR"
    gas_per_decay_update: int = 21_000   # 


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class ZKConfig:
    # Statistical anchor
    num_bins: int = 32              # 
    ldp_epsilon: float = 1.0        # 
    min_dataset_size: int = 200     # 

    # Security
    snark_soundness_bits: int = 128
    commitment_hash: str = "sha256" # Hash(pk_i || S_i)

    # LDP budget presets
    ldp_presets: dict = field(default_factory=lambda: {
        "high_privacy": 0.5,
        "balanced":     1.0,
        "low_privacy":  4.0,
    })


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class CVAEConfig:
    # PCA projection
    pca_components: int = 64        # 
    pca_incremental: bool = True    # 

    # C-VAE architecture
    latent_dim: int = 32
    encoder_hidden: List[int] = field(default_factory=lambda: [256, 128])
    decoder_hidden: List[int] = field(default_factory=lambda: [128, 256])
    condition_dim: int = 32         # 

    # Training
    cvae_lr: float = 1e-3
    cvae_batch_size: int = 64
    cvae_epochs_warmup: int = 10    # 
    beta_kl: float = 1.0            # 

    # Detection thresholds (percentile of reconstruction error on honest updates)
    threshold_percentile: float = 95.0   # 
    threshold_presets: dict = field(default_factory=lambda: {
        "strict": 90.0,
        "robust": 95.0,
        "loose":  99.0,
    })

    # 
    adaptation_rate: float = 0.01   # 


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class AttackConfig:
    attack_type: Literal[
        "none",
        "sleeper_label_flip",
        "free_rider_sybil",
        "lazy_hoard",
        "adaptive_manifold",
    ] = "sleeper_label_flip"

    malicious_fraction: float = 0.20   # 
    num_sybils: int = 0                # 

    # Sleeper: pivot round
    pivot_round: int = 51              # 

    # Label-flip
    source_class: int = 0
    target_class: int = 1

    # Manifold adversary
    manifold_budget: float = 5.0       # 

    # Backdoor
    backdoor_trigger_size: int = 4     # 


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    output_dir: str = "experiments"
    log_dir: str = "experiments/logs"
    checkpoint_dir: str = "experiments/checkpoints"
    save_every: int = 10            # save checkpoint every N rounds
    log_level: str = "INFO"
    wandb_project: str = "soul-fl"
    use_wandb: bool = False


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
@dataclass
class SoulFLConfig:
    fl: FLConfig = field(default_factory=FLConfig)
    trust: TrustConfig = field(default_factory=TrustConfig)
    zk: ZKConfig = field(default_factory=ZKConfig)
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Comparison baselines (Section V-A)
    baselines: List[str] = field(default_factory=lambda: [
        "fedavg", "krum", "flame", "rofl", "aion", "dp_brem"
    ])

    def __post_init__(self):
        # Adjust num_classes for FEMNIST (62 classes)
        if self.fl.dataset == "femnist":
            self.fl.num_classes = 62


# Convenience factory
def get_default_config(dataset: str = "cifar10", attack: str = "sleeper_label_flip") -> SoulFLConfig:
    cfg = SoulFLConfig()
    cfg.fl.dataset = dataset
    cfg.attack.attack_type = attack
    cfg.__post_init__()
    return cfg
