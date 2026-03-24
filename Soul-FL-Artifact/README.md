# Soul-FL: Mitigating Sleeper Sybil Attacks in Federated Learning via Statistically Anchored Identities and Decaying Trust

> **Paper:** *Soul-FL: Mitigating Sleeper Sybil Attacks in Federated Learning via Statistically Anchored Identities and Decaying Trust*
> IEEE Transactions on Dependable and Secure Computing (under review)

---

## Overview

Soul-FL is a **trust-minimized federated learning framework** that defends against **adaptive sleeper Sybil attacks** — adversaries that behave honestly to accumulate trust before pivoting to model poisoning.

Unlike existing defenses, Soul-FL treats **trust as a perishable resource** rather than a static credential.  It does so through three tightly integrated mechanisms:

| Phase | Mechanism | What it prevents |
|-------|-----------|-----------------|
| **I**   | ZK-SNARK statistical eligibility anchors | Free-rider Sybils with no real data |
| **II**  | Anchor-conditioned C-VAE gradient fingerprinting | Sleeper pivots & manifold-constrained attacks |
| **III** | On-chain exponential trust decay (Soulbound Tokens) | Trust hoarding & lazy Sybils |
| **IV**  | φ-squashed trust-weighted aggregation | Undue influence from high-trust adversaries |

---

## Repository Structure

```
Soul-FL-Artifact/
├── config.py                    # All hyper-parameters 
├── data_loader.py               # CIFAR-10 / FEMNIST with Dirichlet non-IID partitioning
├── run_simulation.py            # Main entry point + full experiment suite
├── requirements.txt
├── package.json                 # Hardhat / Node.js dependencies
├── hardhat.config.js
│
├── core/
│   ├── models.py                # 
│   ├── client.py                # Honest + adversarial client variants
│   ├── aggregation.py           # 
│   └── server.py                # 
│
├── security/
│   ├── zk.py                    # ZK-SNARK anchor + LDP 
│   ├── cvae.py                  # C-VAE fingerprinting + PCA 
│   ├── trust_engine.py          # SBT trust decay + vouchers 
│   └── blockchain_sim.py        # On-chain smart contract 
│
├── contracts/
│   └── SoulFLToken.sol          # Solidity 0.8.19 SBT contract
│
├── scripts/
│   └── deploy_and_test.js       # Hardhat deployment 
│
└── experiments/
    ├── checkpoints/
    └── logs/
```

---

## Quick Start

### Python Setup

```bash
# install Python dependencies
pip install -r requirements.txt


### Solidity Setup

```bash
# Install Node.js dependencies
npm install

# Compile contracts
npm run compile

# Start local Hardhat node (in separate terminal)
npm run node

# Deploy and run functional tests
npm run deploy
```

---


---

## Attack Types

| Attack | Flag | Section |
|--------|------|---------|
| Sleeper label-flip | `--attack sleeper_label_flip` | III-B (ii) |
| Free-rider Sybil | `--attack free_rider_sybil` | III-B (i) |
| Lazy hoarding | `--attack lazy_hoard` | III-B (iii) |
| Adaptive manifold | `--attack adaptive_manifold` | III-B (iv) |

---



---

## Baselines

| Method | Reference | `--method` flag |
|--------|-----------|-----------------|
| FedAvg | McMahan et al. 2017 | `fedavg` |
| Krum | Blanchard et al. 2017 | `krum` |
| FLAME | Nguyen et al. 2022 | `flame` |
| RoFL | Lycklama et al. 2023 | `rofl` |
| Aion | Liu et al. 2025 | `aion` |
| DP-BREM | Gu et al. 2025 | `dp_brem` |

---


