# security package — lazy imports so non-torch submodules can be used standalone


def __getattr__(name):
    """Lazy attribute resolution to avoid importing torch at package load time."""
    _no_torch = {"CVAE", "GradientFingerprintEngine", "StreamingPCA"}
    _trust = {"TrustEngine", "SBTState", "Voucher", "sign_voucher", "verify_voucher"}
    _zk    = {"ZKEnrollmentEngine", "StatisticalAnchor", "ZKProof", "EnrollmentRecord"}
    _chain = {"SoulFLContract", "create_chain"}

    if name in _no_torch:
        from security import cvae as _m
        return getattr(_m, name)
    if name in _trust:
        from security import trust_engine as _m
        return getattr(_m, name)
    if name in _zk:
        from security import zk as _m
        return getattr(_m, name)
    if name in _chain:
        from security import blockchain_sim as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'security' has no attribute {name!r}")


__all__ = [
    # zk
    "ZKEnrollmentEngine", "StatisticalAnchor", "ZKProof", "EnrollmentRecord",
    # cvae (torch required)
    "CVAE", "GradientFingerprintEngine", "StreamingPCA",
    # trust
    "TrustEngine", "SBTState", "Voucher", "sign_voucher", "verify_voucher",
    # blockchain
    "SoulFLContract", "create_chain",
]
