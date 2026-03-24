
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
#  
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def default_cfg():
    from config import get_default_config
    return get_default_config("cifar10", "sleeper_label_flip")


@pytest.fixture(scope="session")
def trust_cfg():
    from config import TrustConfig
    return TrustConfig()


@pytest.fixture(scope="session")
def zk_cfg():
    from config import ZKConfig
    return ZKConfig(min_dataset_size=50)


@pytest.fixture(scope="session")
def cvae_cfg():
    from config import CVAEConfig
    return CVAEConfig(pca_components=32, latent_dim=16,
                      encoder_hidden=[64, 32], decoder_hidden=[32, 64])


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def zk_engine(zk_cfg):
    from security.zk import ZKEnrollmentEngine
    return ZKEnrollmentEngine(zk_cfg, seed=0)


@pytest.fixture
def enrolled_engine(zk_engine):
    """ZK engine with clients 0-9 already enrolled (300 samples each)."""
    rng = np.random.default_rng(42)
    for cid in range(10):
        labels = rng.integers(0, 10, size=300)
        anchor, th = zk_engine.prepare_anchor(cid, labels, num_classes=10)
        proof = zk_engine.generate_proof(anchor, th)
        zk_engine.verify_and_enroll(proof, anchor)
    return zk_engine


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def trust_engine(trust_cfg):
    from security.trust_engine import TrustEngine
    te = TrustEngine(trust_cfg)
    for i in range(10):
        te.mint(i, f"commit_{i}")
    return te


@pytest.fixture
def chain(trust_cfg):
    from security.blockchain_sim import create_chain
    c = create_chain(trust_cfg)
    for i in range(10):
        c.mint_sbt(i, f"commit_{i}", sender="0xAGGREGATOR")
    c.advance_round(sender="0xAGGREGATOR")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def random_updates_128():
    rng = np.random.default_rng(99)
    return [(i, rng.standard_normal(128).astype(np.float32)) for i in range(10)]


@pytest.fixture
def random_updates_256():
    rng = np.random.default_rng(7)
    return [(i, rng.standard_normal(256).astype(np.float32)) for i in range(10)]


@pytest.fixture
def sample_counts_uniform():
    return {i: 100 for i in range(10)}


@pytest.fixture
def uniform_weights():
    return {i: 0.1 for i in range(10)}


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def theory_params():
    from utils.theoretical_analysis import TheoreticalParams
    return TheoreticalParams()


@pytest.fixture(scope="session")
def theory_params_high_privacy():
    """High privacy (ε=0.5): higher β → may violate Lemma 1."""
    from utils.theoretical_analysis import TheoreticalParams
    return TheoreticalParams(beta=0.10)


@pytest.fixture(scope="session")
def theory_params_fast_decay():
    from utils.theoretical_analysis import TheoreticalParams
    return TheoreticalParams(lam=0.10)


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests needing GPU/large data")
    config.addinivalue_line("markers", "torch: tests requiring PyTorch")


def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False

    skip_torch = pytest.mark.skip(reason="PyTorch not installed")
    for item in items:
        if "torch" in item.keywords and not torch_available:
            item.add_marker(skip_torch)
